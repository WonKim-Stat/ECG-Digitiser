"""Unit tests for --column_mapping lines in src/run/digitize.py.

measure_column_mapping() reads the time axis off the printed grid lines: the 1 mm
lines give where every millimetre is, the bold 5 mm lines which 1 mm line a window is
on. No model and no data files: pages of vertical grid lines are synthesised with
numpy, undistorted and with the distortions of a scanned sheet (a compressed last
column, a step of about half a line inside it, a stretched first column), and the
map is compared with the lines that were drawn.
"""
import csv
import warnings

import numpy as np
import pytest
import torch
import wfdb
from torchvision.io.image import write_png

from config import (
    FREQUENCY,
    LEAD_LABEL_MAPPING,
    LONG_SIGNAL_LENGTH_SEC,
    SHORT_SIGNAL_LENGTH_SEC,
    Y_SHIFT_RATIO,
)
from src.run import digitize

WIDTH = 2200
LINES = digitize.GRID_LINES_PER_COLUMN
# A scanned page: 0.954 x 200 dpi, a 7.5 px line period, where a 4 px step of the
# lines is more than half a line.
P_SCAN = 469.0
LINE0_SCAN = 140.3
# Darkness of a thin and of a bold printed line, and their width.
THIN, BOLD, LINE_SIGMA = 90.0, 190.0, 0.6
SHORT_SAMPLES = int(SHORT_SIGNAL_LENGTH_SEC * FREQUENCY)
LONG_SAMPLES = int(LONG_SIGNAL_LENGTH_SEC * FREQUENCY)


def identity(x0):
    return np.asarray(x0, dtype=float)


def compressed(start, factor):
    """The sheet beyond start printed factor times as wide, and its inverse."""

    def forward(x0):
        x0 = np.asarray(x0, dtype=float)
        return np.where(x0 < start, x0, start + factor * (x0 - start))

    def inverse(x):
        x = np.asarray(x, dtype=float)
        return np.where(x < start, x, start + (x - start) / factor)

    return forward, inverse


def stepped(at, step):
    """Everything right of at moved by step pixels, as a scanned column 4 is."""

    def forward(x0):
        x0 = np.asarray(x0, dtype=float)
        return np.where(x0 < at, x0, x0 + step)

    return forward


def stretched_start(line0, P, displacement):
    """The first column pushed by displacement px at its start, fading out by 0.8 P."""

    def forward(x0):
        x0 = np.asarray(x0, dtype=float)
        return x0 + displacement * np.clip(1 - (x0 - line0) / (0.8 * P), 0, 1)

    return forward


def grid_page(line0, P, warp=identity, height=300, bold=True, width=WIDTH):
    """A page of vertical grid lines as run() reads it, [3, H, W] uint8.

    The ideal lines are at line0 + k P / 62.5, every fifth one bold (the one at line0
    among them), and each is drawn where warp puts it, an anti-aliased stroke whose
    darkness is read at the pixel centres c + 0.5.
    """
    period = P / LINES
    k = np.arange(np.floor(-line0 / period) - 2, np.ceil((width - line0) / period) + 2)
    positions = warp(line0 + k * period)
    darkness = np.zeros(width)
    x = np.arange(width) + 0.5
    for position, index in zip(positions, k):
        near = np.abs(x - position) < 5 * LINE_SIGMA
        weight = BOLD if bold and index % digitize.COLUMN_MAPPING_BOLD_EVERY == 0 else THIN
        darkness[near] += weight * np.exp(-0.5 * ((x[near] - position) / LINE_SIGMA) ** 2)
    row = np.round(255 - np.clip(darkness, 0, 255)).astype(np.uint8)
    page = np.broadcast_to(row, (height, width)).copy()
    # Red lines, as on a printed page: darkest in the two lower channels.
    image = np.stack([np.full_like(page, 255), page, page])
    return torch.from_numpy(image)


def pixel_start(x):
    """First pixel whose centre is at or right of the continuous x: the mask x1."""
    return float(np.ceil(x - 0.5))


def column_edges(line0, P, warp=identity, snap=0.5, late=None):
    """The mask edges of a full 3x4 page with rhythm strip whose traces follow warp.

    A trace starts snap px left of its grid line. late adds pixels to the start of
    the column 1 leads, as a model that starts a mask a few pixels late.
    """
    late = late or {}
    edges = []
    for column in range(digitize.NUM_COLUMNS):
        start = pixel_start(warp(line0 + column * P) - snap)
        end = pixel_start(warp(line0 + (column + 1) * P) - snap)
        for lead in range(3):
            edges.append((start + late.get((column, lead), 0), column * LINES))
            edges.append((end, (column + 1) * LINES))
    start = pixel_start(warp(line0) - snap)
    end = pixel_start(warp(line0 + digitize.NUM_COLUMNS * P) - snap)
    edges += [(start, 0.0), (end, digitize.NUM_COLUMNS * LINES)]
    return edges


def mapped_lines(info, P, snap):
    """Where the map puts every whole millimetre line 0 .. 250, in page x."""
    m = np.arange(digitize.NUM_COLUMNS * LINES + 1)
    x = digitize._column_map_x(info["knots_m"], info["knots_x"], P / LINES, m)
    return m, x + snap


def dense_error(info, line0, P, snap, warp=identity):
    """Largest distance between the map and the drawn lines over 0 .. 250 mm, every 0.05
    mm (every sample of the rhythm strip), in pixels."""
    period = P / LINES
    m = np.arange(0.0, digitize.NUM_COLUMNS * LINES, 0.05)
    x = digitize._column_map_x(info["knots_m"], info["knots_x"], period, m) + snap
    return float(np.max(np.abs(x - warp(line0 + m * period))))


def edge_rounding(line0, P, warp=identity, snap=0.5):
    """How far right of their grid lines the synthetic mask edges start, in the median:
    the origin offset the map has to report."""
    period = P / LINES
    edges = column_edges(line0, P, warp, snap)
    return float(np.median([x1 + snap - warp(line0 + mm * period) for x1, mm in edges]))


def measure(image, line0, P, warp=identity, snap=0.5, median_mm=None, **kwargs):
    edges = column_edges(line0, P, warp, snap, **kwargs)
    if median_mm is None:
        median_mm = digitize.COLUMN_MAPPING_MEDIAN_MM
    return digitize.measure_column_mapping(
        image, line0 - snap, P, edges, snap, median_mm=median_mm
    )


def grid_profile(image, period):
    """The coherent band profile measure_column_mapping() reads the windows from."""
    profiles, _ = digitize._band_profiles(digitize._darkness(image))
    weights = np.abs(digitize._grid_line_comb(profiles, period))
    keep = weights >= digitize.GRID_LINE_MIN_BAND_AMPLITUDE * np.median(weights)
    return profiles[keep].sum(axis=0)


# ------------------------------------------------------------ (i) uniform page
P_GEN = 1700 * digitize.PAGE_PITCH_RATIO
LINE0_GEN = 118.62


def test_a_uniform_page_is_in_the_dead_band_and_its_map_is_the_uniform_grid():
    image = grid_page(LINE0_GEN, P_GEN)
    x_map, info = measure(image, LINE0_GEN, P_GEN)
    assert info["reason"] == ""
    assert info["dead_band"] and x_map is None
    assert info["shift"] < digitize.COLUMN_MAPPING_MIN_SHIFT_PX
    assert info["agreement"] == 1.0 and info["coverage"] == 1.0
    assert abs(info["origin_move"]) < 0.1
    # The mask edges start where their pixels do, -0.25 px off the lines in the median:
    # the edge x1 + the snap, each on its own line.
    rounding = edge_rounding(LINE0_GEN, P_GEN)
    assert rounding == pytest.approx(-0.25, abs=0.02)
    assert info["origin_offset"] == pytest.approx(rounding, abs=0.1)
    # The knots themselves are the uniform grid, far below a pixel.
    m, x = mapped_lines(info, P_GEN, 0.5)
    assert np.max(np.abs(x - (LINE0_GEN + m * P_GEN / LINES))) < 0.1


def _one_lead(g0, P, height=300):
    """The mask and position of a short lead of column 1 (a sine), for vectorise_grid."""
    columns = np.arange(int(np.ceil(g0 + P - 0.5)), int(np.ceil(g0 + 2 * P - 0.5)))
    t = (columns + 0.5 - g0 - P) / P * SHORT_SIGNAL_LENGTH_SEC
    rows = np.floor(150 - 30 * np.sin(2 * np.pi * 1.3 * t)).astype(int)
    label = np.zeros((height, WIDTH), dtype=np.uint8)
    label[rows, columns] = LEAD_LABEL_MAPPING["aVR"]
    masks, positions, _ = digitize.cut_binary(torch.from_numpy(label)[None], torch.zeros(3, height, WIDTH))
    return masks["aVR"], positions["aVR"]


@pytest.mark.parametrize("sharpen", ["none", "bandlimited"])
def test_vectorise_grid_without_a_map_is_bitwise_the_uniform_grid(sharpen):
    g0 = LINE0_GEN - 0.5
    mask, position = _one_lead(g0, P_GEN)
    image = torch.zeros(3, 300, WIDTH, dtype=torch.uint8)
    args = (image, mask, position, g0, P_GEN, 1, False, Y_SHIFT_RATIO, "aVR")
    before = digitize.vectorise_grid(*args, sharpen=sharpen).numpy()
    none = digitize.vectorise_grid(*args, sharpen=sharpen, x_map=None).numpy()
    assert before.tobytes() == none.tobytes()
    # The uniform grid as a map gives the same trace to floating point accuracy.

    def uniform(m):
        return g0 + np.asarray(m) * P_GEN / LINES

    mapped = digitize.vectorise_grid(*args, sharpen=sharpen, x_map=uniform).numpy()
    assert np.max(np.abs(mapped - before)) < 1e-5


def test_vectorise_grid_reads_the_samples_where_the_map_puts_them():
    g0 = LINE0_GEN - 0.5
    mask, position = _one_lead(g0, P_GEN)
    image = torch.zeros(3, 300, WIDTH, dtype=torch.uint8)
    args = (image, mask, position, g0, P_GEN, 1, False, Y_SHIFT_RATIO, "aVR")
    # A map a whole number of samples (4 px = 10.16 samples, so 3 samples) later.
    dx = 3 * P_GEN / SHORT_SAMPLES

    def later(m):
        return g0 + np.asarray(m) * P_GEN / LINES + dx

    base = digitize.vectorise_grid(*args, x_map=None).numpy()
    moved = digitize.vectorise_grid(*args, x_map=later).numpy()
    assert np.max(np.abs(moved[:-3] - base[3:])) < 1e-5


# ------------------------------------------------------- (ii) smooth compression
def test_a_compressed_last_column_is_followed_line_by_line():
    start = LINE0_SCAN + 3 * P_SCAN
    warp, _ = compressed(start, 0.98)
    image = grid_page(LINE0_SCAN, P_SCAN, warp)
    x_map, info = measure(image, LINE0_SCAN, P_SCAN, warp, snap=0.0)
    assert info["reason"] == ""
    assert x_map is not None
    m, x = mapped_lines(info, P_SCAN, 0.0)
    truth = warp(LINE0_SCAN + m * P_SCAN / LINES)
    assert np.max(np.abs(x - truth)) < 0.25
    assert dense_error(info, LINE0_SCAN, P_SCAN, 0.0, warp) < 0.25
    # The compression only ever moves the lines one way, so the running median keeps it.
    assert info["smoothing"] < 0.05
    assert info["origin_offset"] == pytest.approx(
        edge_rounding(LINE0_SCAN, P_SCAN, warp, 0.0), abs=0.1
    )
    # The largest move is at the end of the page: 2 % of a column.
    assert info["shift"] == pytest.approx(0.02 * P_SCAN, abs=0.3)
    # x_map is the same map, in the coordinate of the traces.
    assert np.allclose(x_map(m.astype(float)), x, atol=1e-9)


# ------------------------------------------------------------- (iii) a step
STEP_AT = LINE0_SCAN + 3.34 * P_SCAN
BOLD_SCAN = digitize.COLUMN_MAPPING_BOLD_EVERY * P_SCAN / LINES


# The phase of the bold lines on the page, LINE0_SCAN's among them: at 2 px the -4 px
# step carries the bold offset across its wrap from 0 to one bold period, which only the
# unwrapping of the bold offset along x gets right.
@pytest.mark.parametrize("bold_phase", [2.0, 10.0, 18.0, LINE0_SCAN - 3 * BOLD_SCAN, 35.0])
def test_a_step_of_half_a_line_stays_on_its_line(bold_phase):
    line0 = 3 * BOLD_SCAN + bold_phase
    step_at = line0 + 3.34 * P_SCAN
    warp = stepped(step_at, -4.0)
    image = grid_page(line0, P_SCAN, warp)
    period = P_SCAN / LINES
    assert 4.0 > period / 2  # the step is more than half a line
    x_map, info = measure(image, line0, P_SCAN, warp, snap=0.0)
    assert info["reason"] == ""
    m, x = mapped_lines(info, P_SCAN, 0.0)
    truth = warp(line0 + m * P_SCAN / LINES)
    # Every line more than one 1 mm window away from the step is found where it is.
    window = digitize.COLUMN_MAPPING_FINE_LINES * period
    away = np.abs(truth - step_at) > window
    assert np.max(np.abs(x - truth)[away]) < 0.5
    beyond = truth > step_at + window
    assert np.count_nonzero(beyond) > 30
    assert np.max(np.abs(x - truth)[beyond]) < 0.5


def test_the_1_mm_phase_alone_slips_a_line_on_that_step():
    """The power of the step test: unwrapped by continuity alone the 1 mm offset
    takes the short way round the -4 px step and ends up a whole line off."""
    warp = stepped(STEP_AT, -4.0)
    image = grid_page(LINE0_SCAN, P_SCAN, warp)
    period = P_SCAN / LINES
    g0 = LINE0_SCAN
    centres, fine, coarse = digitize._column_mapping_windows(
        grid_profile(image, period), period, g0 - 0.25 * P_SCAN, g0 + 4.25 * P_SCAN
    )
    strong = np.abs(fine) >= 0.3 * np.median(np.abs(fine))
    phase = np.unwrap(-np.angle(fine[strong]))
    offset = phase / (2 * np.pi) * period
    x = centres[strong]
    window = digitize.COLUMN_MAPPING_FINE_LINES * period
    before = np.median(offset[(x > g0) & (x < STEP_AT - window)])
    after = np.median(offset[x > STEP_AT + window])
    # The truth is -4 px; the 1 mm phase alone says about -4 + one line.
    assert abs((after - before) - (-4.0)) > period / 2
    assert (after - before) == pytest.approx(-4.0 + period, abs=0.5)
    # The column map, with the bold lines, gets the -4 px.
    _, info = measure(image, LINE0_SCAN, P_SCAN, warp, snap=0.0)
    knots_x = info["knots_x"]
    lattice = LINE0_SCAN + info["knots_m"] * period
    moved = knots_x - lattice
    change = np.median(moved[knots_x > STEP_AT + window]) - np.median(
        moved[(knots_x > g0) & (knots_x < STEP_AT - window)]
    )
    assert change == pytest.approx(-4.0, abs=0.3)


def test_a_short_stretch_a_line_off_is_dropped(monkeypatch):
    # Windows whose bold phase names the next line, on 40 px of a clean page, as a
    # smudge does: the map leaves them out instead of reading 20 samples wrong there.
    image = grid_page(LINE0_GEN, P_GEN)
    period = P_GEN / LINES
    windows = digitize._column_mapping_windows

    def one_line_off(profile, period_, low, high):
        centres, fine, coarse = windows(profile, period_, low, high)
        island = (centres > 1700) & (centres < 1740)
        # One line is a fifth of the bold period.
        shift = np.exp(-2j * np.pi / digitize.COLUMN_MAPPING_BOLD_EVERY)
        return centres, fine, np.where(island, coarse * shift, coarse)

    monkeypatch.setattr(digitize, "_column_mapping_windows", one_line_off)
    x_map, info = measure(image, LINE0_GEN, P_GEN)
    assert info["islands"] >= 1
    assert info["jump"] < digitize.COLUMN_MAPPING_RUN_BREAK
    assert info["reason"] == ""
    m, x = mapped_lines(info, P_GEN, 0.5)
    assert np.max(np.abs(x - (LINE0_GEN + m * period))) < 0.25
    assert x_map is None and info["dead_band"]


def _patched_windows(monkeypatch, where, fine_lines=0.0, bold_lines=0.0):
    """Move the 1 mm and the bold offset of the windows centred in where = (low, high)
    by that many lines: a window's phase against the absolute x turns by one period
    per period its lines move."""
    windows = digitize._column_mapping_windows

    def patched(profile, period, low, high):
        centres, fine, coarse = windows(profile, period, low, high)
        inside = (centres > where[0]) & (centres < where[1])
        fine = np.where(inside, fine * np.exp(-2j * np.pi * fine_lines), fine)
        bold = digitize.COLUMN_MAPPING_BOLD_EVERY
        coarse = np.where(inside, coarse * np.exp(-2j * np.pi * bold_lines / bold), coarse)
        return centres, fine, coarse

    monkeypatch.setattr(digitize, "_column_mapping_windows", patched)


def test_a_long_stretch_a_line_off_is_refused(monkeypatch):
    # 200 px whose bold offset names the next line: too long for an island, so the map
    # jumps by a whole line there, which no printed page does.
    image = grid_page(LINE0_GEN, P_GEN)
    _patched_windows(monkeypatch, (1500, 1700), bold_lines=1.0)
    x_map, info = measure(image, LINE0_GEN, P_GEN)
    assert x_map is None
    assert info["reason"].startswith("grid lines jump by")
    assert info["jump"] > digitize.COLUMN_MAPPING_MAX_JUMP


def test_a_single_window_off_its_neighbours_is_dropped(monkeypatch):
    # Both offsets of one window 0.4 lines off, as a smudge on a few lines does: they
    # agree with each other and the step is too small for a run break, so only the
    # comparison with the four neighbours drops it. Without the running median, which
    # would hide it as well.
    image = grid_page(LINE0_GEN, P_GEN)
    period = P_GEN / LINES
    _patched_windows(monkeypatch, (1200, 1204), fine_lines=0.4, bold_lines=0.4)
    x_map, info = measure(image, LINE0_GEN, P_GEN, median_mm=0)
    assert info["reason"] == "" and info["spikes"] >= 1
    # The window would have put its line 0.4 lines, 3.1 px, off.
    assert dense_error(info, LINE0_GEN, P_GEN, 0.5) < 0.25 < 0.4 * period


def test_windows_whose_1_mm_and_5_mm_offsets_disagree_are_dropped(monkeypatch):
    # 80 px whose 1 mm phase is 0.4 lines off and whose bold lines are right: a run of
    # windows the neighbours and the running median would follow, so only the
    # disagreement of the two offsets drops it.
    image = grid_page(LINE0_GEN, P_GEN)
    _patched_windows(monkeypatch, (1200, 1280), fine_lines=0.4)
    x_map, info = measure(image, LINE0_GEN, P_GEN)
    assert info["reason"] == ""
    assert 0.9 < info["agreement"] < 1.0
    assert dense_error(info, LINE0_GEN, P_GEN, 0.5) < 0.25


# --------------------------------------------------- (iv) a stretched first column
def test_the_origin_is_the_local_line_where_column_1_starts():
    warp = stretched_start(LINE0_SCAN, P_SCAN, -3.3)
    image = grid_page(LINE0_SCAN, P_SCAN, warp)
    x_map, info = measure(image, LINE0_SCAN, P_SCAN, warp, snap=0.0)
    assert info["reason"] == ""
    assert x_map is not None
    # The first sample is read on the displaced line, not on the uniform lattice.
    assert float(x_map(0.0)) == pytest.approx(LINE0_SCAN - 3.3, abs=0.25)
    assert info["origin_move"] == pytest.approx(-3.3, abs=0.25)
    m, x = mapped_lines(info, P_SCAN, 0.0)
    assert np.max(np.abs(x - warp(LINE0_SCAN + m * P_SCAN / LINES))) < 0.25


def test_late_column_1_masks_do_not_move_the_origin_to_the_next_line():
    # Column 1 masks that start 3.9 px late, about half a line: all the other edges
    # of the page agree on the line, so the origin stays where the traces start.
    warp = stretched_start(LINE0_SCAN, P_SCAN, -3.3)
    image = grid_page(LINE0_SCAN, P_SCAN, warp)
    late = {(0, lead): 3.9 for lead in range(3)}
    x_map, info = measure(image, LINE0_SCAN, P_SCAN, warp, snap=0.0, late=late)
    assert info["reason"] == ""
    assert float(x_map(0.0)) == pytest.approx(LINE0_SCAN - 3.3, abs=0.25)


def test_mask_edges_between_two_lines_leave_the_uniform_grid():
    # Traces that do not start on a grid line: the lines say nothing about the origin.
    image = grid_page(LINE0_SCAN, P_SCAN)
    period = P_SCAN / LINES
    edges = column_edges(LINE0_SCAN + period / 2, P_SCAN, snap=0.0)
    x_map, info = digitize.measure_column_mapping(image, LINE0_SCAN, P_SCAN, edges, 0.0)
    assert x_map is None
    assert "off the next grid line" in info["reason"]


# --------------------------------------------------------------- the running median
def test_the_running_median_keeps_steps_and_stretches_and_takes_out_the_snap_ripple():
    m = np.arange(-11.0, 260.0, 0.53)  # knots about half a line apart, as on a scan
    inner = (m >= 0) & (m <= digitize.NUM_COLUMNS * LINES)
    width = digitize.COLUMN_MAPPING_MEDIAN_MM
    # A run that only falls is its own median, up to the outer knots: a step at 3.34
    # columns and a slope after it, on a page a little narrower than the uniform
    # columns. So is a run that only rises, as a stretched first column does.
    falls = (
        -4 * np.clip((m - 208.75) / 3, 0, 1)
        - 4 * np.clip((m - 230) / 20, 0, 1)
        - 0.01 * m
    )
    rises = -3.3 * np.clip(1 - m / 50, 0, 1) + 0.01 * m
    for moves in (falls, rises):
        assert np.array_equal(digitize._running_median(m, moves, width), moves)
    # The ripple of the snap of the bold lines, one period per window, is taken out, and
    # a step under it stays a step.
    ripple = 0.3 * np.sin(2 * np.pi * m / 13.5)
    assert np.max(np.abs(digitize._running_median(m, ripple, width)[inner])) < 0.05
    step = np.where(m < 208.75, 0.0, -4.0)
    smoothed = digitize._running_median(m, step + ripple, width)
    assert np.max(np.abs(smoothed - step)[inner]) < 0.1


def snapped(x0):
    """A line drawn at 200 dpi, snapped to the centre of the pixel it falls in."""
    return np.floor(np.asarray(x0, dtype=float)) + 0.5


@pytest.mark.parametrize("line0", [LINE0_GEN, 118.3])
def test_the_pixel_snap_of_drawn_lines_is_taken_out_of_the_map(line0):
    # The lines of a drawn page sit up to half a pixel off their millimetre, its traces
    # do not: the map as measured follows the lines, the median puts it back on the
    # millimetres. Both stay in the dead band.
    image = grid_page(line0, P_GEN, snapped)
    _, measured = measure(image, line0, P_GEN, median_mm=0)
    x_map, smoothed = measure(image, line0, P_GEN)
    assert x_map is None and smoothed["dead_band"] and measured["dead_band"]
    assert dense_error(measured, line0, P_GEN, 0.5) > 0.2
    assert dense_error(smoothed, line0, P_GEN, 0.5) < 0.15
    assert smoothed["shift"] < measured["shift"]


# ---------------------------------------------------------- (v), (vi) fall backs
def test_parser_column_mapping_defaults_to_lines():
    parser = digitize.get_parser()
    args = parser.parse_args(["-d", "in", "-o", "out"])
    assert args.column_mapping == "lines"
    assert args.column_mapping_median == digitize.COLUMN_MAPPING_MEDIAN_MM == 13.5
    # The uniform columns of the pages before the default stay one flag away.
    args = parser.parse_args(["-d", "in", "-o", "out", "--column_mapping", "uniform"])
    assert args.column_mapping == "uniform"
    args = parser.parse_args(["-d", "in", "-o", "out", "--column_mapping", "lines"])
    assert args.column_mapping == "lines"
    args = parser.parse_args(["-d", "in", "-o", "out", "--column_mapping_median", "0"])
    assert args.column_mapping_median == 0.0
    with pytest.raises(SystemExit):
        parser.parse_args(["-d", "in", "-o", "out", "--column_mapping", "columns"])


def test_a_gap_in_the_lines_is_refused():
    # 0.3 columns of paper without lines inside column 3: a step there would go unseen.
    image = grid_page(LINE0_SCAN, P_SCAN).clone()
    start = int(LINE0_SCAN + 2.3 * P_SCAN)
    image[:, :, start : start + int(0.3 * P_SCAN)] = 255
    x_map, info = measure(image, LINE0_SCAN, P_SCAN, snap=0.0)
    assert x_map is None
    assert info["reason"].startswith("map has a gap of")
    assert info["gap"] > digitize.COLUMN_MAPPING_MAX_GAP


def test_a_page_with_holes_all_over_is_refused():
    # Holes of 0.15 columns every half column: no gap is long enough to refuse, but the
    # map keeps too few of its windows.
    image = grid_page(LINE0_SCAN, P_SCAN).clone()
    hole, every = int(0.15 * P_SCAN), int(0.5 * P_SCAN)
    for start in range(int(LINE0_SCAN) + 20, int(LINE0_SCAN + 4 * P_SCAN), every):
        image[:, :, start : start + hole] = 255
    x_map, info = measure(image, LINE0_SCAN, P_SCAN, snap=0.0)
    assert x_map is None
    assert info["reason"].startswith("map keeps")
    assert info["coverage"] < digitize.COLUMN_MAPPING_MIN_COVERAGE
    assert info["gap"] <= digitize.COLUMN_MAPPING_MAX_GAP


def test_a_page_without_grid_lines_keeps_the_uniform_grid():
    blank = torch.full((3, 300, WIDTH), 255, dtype=torch.uint8)
    x_map, info = measure(blank, LINE0_GEN, P_GEN)
    assert x_map is None
    assert info["reason"] == "no grid lines along the columns"


def test_a_page_without_bold_lines_keeps_the_uniform_grid():
    # The 1 mm lines alone cannot name their line: the bold offset is noise, it
    # agrees with the 1 mm one by chance only.
    rng = np.random.default_rng(0)
    image = grid_page(LINE0_GEN, P_GEN, bold=False).numpy().astype(int)
    image = np.clip(image + rng.integers(-3, 4, image.shape), 0, 255).astype(np.uint8)
    x_map, info = measure(torch.from_numpy(image), LINE0_GEN, P_GEN)
    assert x_map is None
    assert info["agreement"] < digitize.COLUMN_MAPPING_MIN_AGREEMENT
    assert "5 mm and 1 mm lines agree" in info["reason"]


def test_column_mapping_edges_name_the_millimetre_of_every_edge():
    label = np.zeros((300, WIDTH), dtype=np.uint8)
    label[100, 118:610] = LEAD_LABEL_MAPPING["I"]
    label[150, 610:1102] = LEAD_LABEL_MAPPING["aVL"]
    label[250, 118:2087] = LEAD_LABEL_MAPPING["II"]
    masks, positions, _ = digitize.cut_binary(
        torch.from_numpy(label)[None], torch.zeros(3, 300, WIDTH)
    )
    edges = digitize.column_mapping_edges(masks, positions, ["II"])
    assert sorted(edges) == sorted(
        [(118.0, 0.0), (610.0, 62.5), (610.0, 62.5), (1102.0, 125.0),
         (118.0, 0.0), (2087.0, 250.0)]
    )


# ------------------------------------------------------------------- end to end
HEIGHT = 1700
LEADS = {
    "I": 0, "III": 0, "aVR": 1, "aVL": 1, "aVF": 1,
    "V1": 2, "V2": 2, "V3": 2, "V4": 3, "V5": 3, "V6": 3,
}


def analytic(t, k):
    """A slow sine and one narrow bump per second, which a timing error shows up on."""
    return 0.3 * np.sin(2 * np.pi * (0.7 * t + k / 13.0)) + 0.6 * np.exp(
        -0.5 * (((t % 1.0) - 0.4 - 0.03 * k) / 0.03) ** 2
    )


def label_page(line0, P, inverse, snap):
    """A 3x4 label mask with rhythm strip whose traces are printed through the warp.

    inverse maps a page x back to the sheet, where the time axis is uniform.
    """
    label = np.zeros((HEIGHT, WIDTH), dtype=np.uint8)
    names = list(LEADS) + ["II"]
    for k, lead in enumerate(names):
        is_long = lead == "II"
        column = 0 if is_long else LEADS[lead]
        n_columns = digitize.NUM_COLUMNS if is_long else 1
        start = line0 - snap + column * P
        sheet_x = np.arange(WIDTH) + 0.5
        sheet = inverse(sheet_x + snap) - snap
        inside = (sheet >= start) & (sheet < start + n_columns * P)
        columns = np.flatnonzero(inside)
        # Page time: a lead of column c shows 2.5 c to 2.5 (c + 1) s.
        t = (sheet[columns] - (line0 - snap)) / P * SHORT_SIGNAL_LENGTH_SEC
        ratio = Y_SHIFT_RATIO["full" if is_long else lead]
        rows = np.floor(
            digitize.baseline_row(ratio, HEIGHT) - analytic(t, k) / (6.25 / P)
        ).astype(int)
        label[rows, columns] = LEAD_LABEL_MAPPING[lead]
    return label, names


def run_page(tmp_path, name, image, label, column_mapping, capsys=None, extra=()):
    # column_mapping None passes no flag at all, which is how the default is tested.
    mode = "default" if column_mapping is None else column_mapping
    data = tmp_path / f"{name}_data"
    masks = tmp_path / f"{name}_masks"
    out = tmp_path / f"{name}_{mode}{'_'.join(('',) + tuple(extra))}_out"
    for folder in (data, masks):
        folder.mkdir(exist_ok=True)
    write_png(image, str(data / f"{name}.png"))
    write_png(torch.from_numpy(label)[None], str(masks / f"{name}_mask.png"))
    # The runs of this file are about the map of the page alone, and their numbers were
    # fixed with it: no second try of a failed grid check and no row maps, which were
    # the defaults then. On the compressed page the rows would get maps of their own.
    argv = [
        "-d", str(data), "-o", str(out), "--mask_folder", str(masks),
        "--rotation", "hough", "--perspective", "off", "--resolution", "keep",
        "--trace_estimator", "mask", "--trace_shift", "off", "--grid_line_offset", "0.5",
        "--grid_rescue", "off", "--row_mapping", "off",
        "--verbose", *extra,
    ]
    if column_mapping is not None:
        argv += ["--column_mapping", column_mapping]
    digitize.run(digitize.get_parser().parse_args(argv))
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        record = wfdb.rdrecord(str(out / name))
    with open(out / "qc.csv", newline="") as f:
        row = next(csv.DictReader(f))
    with open(out / f"{name}.dat", "rb") as f:
        raw = f.read()
    return record, row, raw


def test_run_keeps_a_uniform_page_byte_identical(tmp_path):
    image = grid_page(LINE0_GEN, P_GEN, height=HEIGHT)
    label, _ = label_page(LINE0_GEN, P_GEN, identity, 0.5)
    _, row_u, raw_u = run_page(tmp_path, "flat", image, label, "uniform")
    _, row_l, raw_l = run_page(tmp_path, "flat", image, label, "lines")
    assert raw_l == raw_u
    assert row_u["column_mapping"] == "uniform"
    assert row_u["column_mapping_shift_px"] == "nan"
    assert row_l["column_mapping"] == "uniform: dead band"
    assert float(row_l["column_mapping_shift_px"]) < digitize.COLUMN_MAPPING_MIN_SHIFT_PX


def test_run_measures_a_drawn_page_with_the_median_it_is_given(tmp_path):
    # Lines snapped to the pixels: the median halves how far the map would move the
    # samples, and with or without it the page stays byte identical in the dead band.
    image = grid_page(LINE0_GEN, P_GEN, snapped, height=HEIGHT)
    label, _ = label_page(LINE0_GEN, P_GEN, identity, 0.5)
    _, _, raw_u = run_page(tmp_path, "drawn", image, label, "uniform")
    _, row_med, raw_med = run_page(tmp_path, "drawn", image, label, "lines")
    _, row_raw, raw_raw = run_page(
        tmp_path, "drawn", image, label, "lines", extra=("--column_mapping_median", "0")
    )
    assert raw_med == raw_u and raw_raw == raw_u
    assert row_med["column_mapping"] == row_raw["column_mapping"] == "uniform: dead band"
    shift_med = float(row_med["column_mapping_shift_px"])
    shift_raw = float(row_raw["column_mapping_shift_px"])
    assert shift_med < 0.15 < 0.2 < shift_raw


def test_run_without_grid_lines_keeps_the_uniform_grid(tmp_path, capsys):
    image = torch.full((3, HEIGHT, WIDTH), 255, dtype=torch.uint8)
    label, _ = label_page(LINE0_GEN, P_GEN, identity, 0.5)
    _, _, raw_u = run_page(tmp_path, "blank", image, label, "uniform")
    assert "Column mapping for record" not in capsys.readouterr().out
    _, row, raw_l = run_page(tmp_path, "blank", image, label, "lines")
    out = capsys.readouterr().out
    # The grid line stage warned already; the map only says why it was not measured.
    assert out.count("Column mapping for record blank: uniform: grid lines not used") == 1
    assert "WARNING: column mapping" not in out
    assert raw_l == raw_u
    assert row["column_mapping"] == "uniform: grid lines not used"
    assert row["column_mapping_shift_px"] == "nan"


def test_run_reports_a_map_it_cannot_trust(tmp_path, capsys):
    image = grid_page(LINE0_GEN, P_GEN, height=HEIGHT, bold=False)
    label, _ = label_page(LINE0_GEN, P_GEN, identity, 0.5)
    _, _, raw_u = run_page(tmp_path, "thin", image, label, "uniform")
    capsys.readouterr()
    _, row, raw_l = run_page(tmp_path, "thin", image, label, "lines")
    out = capsys.readouterr().out
    assert raw_l == raw_u
    assert "WARNING: column mapping not used for record thin" in out
    assert "keeping the uniform grid" in out
    assert row["column_mapping"].startswith("uniform: 5 mm and 1 mm lines agree")


def test_run_reads_a_compressed_last_column_on_its_own_lines(tmp_path, capsys):
    # A 1 % narrower last column, grid lines and traces alike: the uniform columns
    # read column 4 up to 12 samples late, the lines map reads it where it is.
    start = LINE0_GEN + 3 * P_GEN
    forward, inverse = compressed(start, 0.99)
    image = grid_page(LINE0_GEN, P_GEN, forward, height=HEIGHT)
    label, names = label_page(LINE0_GEN, P_GEN, inverse, 0.5)
    uniform, _, _ = run_page(tmp_path, "squeezed", image, label, "uniform")
    capsys.readouterr()
    lines, row, _ = run_page(tmp_path, "squeezed", image, label, "lines")
    out = capsys.readouterr().out
    assert "Column mapping for record squeezed: lines" in out
    assert row["column_mapping"] == "lines"
    # The pitch of the uniform grid is fitted to the whole comb, compressed column
    # included, so it takes up part of the 4.9 px itself: 4.24 px are left, and 3.74
    # would say that the map was built without the snap of --grid_line_offset.
    assert float(row["column_mapping_shift_px"]) == pytest.approx(4.24, abs=0.2)

    t = np.arange(LONG_SAMPLES) / FREQUENCY
    for k, lead in enumerate(names):
        channel = names.index(lead)
        expected = analytic(t, k)
        column = 0 if lead == "II" else LEADS[lead]
        window = slice(0, LONG_SAMPLES) if lead == "II" else slice(
            column * SHORT_SAMPLES, (column + 1) * SHORT_SAMPLES
        )
        got_lines = lines.p_signal[window, lines.sig_name.index(lead)]
        got_uniform = uniform.p_signal[window, uniform.sig_name.index(lead)]
        # 0.013 mV at most; a map half a pixel off, as without the snap, gives 0.044.
        error_lines = np.max(np.abs(got_lines - expected[window]))
        assert error_lines < 0.025, lead
        if column == 3 or lead == "II":
            assert np.max(np.abs(got_uniform - expected[window])) > 0.15, lead


def test_run_maps_the_columns_by_default_and_uniform_still_turns_it_off(
    tmp_path, capsys
):
    # The page of the test above, run with no --column_mapping flag at all and with
    # --column_mapping uniform, the time axis before lines became the default.
    start = LINE0_GEN + 3 * P_GEN
    forward, inverse = compressed(start, 0.99)
    image = grid_page(LINE0_GEN, P_GEN, forward, height=HEIGHT)
    label, _ = label_page(LINE0_GEN, P_GEN, inverse, 0.5)
    _, row_d, raw_d = run_page(tmp_path, "squeezed", image, label, None)
    out = capsys.readouterr().out
    assert "Column mapping for record squeezed: lines" in out
    assert row_d["column_mapping"] == "lines"
    assert float(row_d["column_mapping_shift_px"]) == pytest.approx(4.24, abs=0.2)

    _, row_u, raw_u = run_page(tmp_path, "squeezed", image, label, "uniform")
    # uniform is still accepted: nothing is measured and the columns stay uniform.
    assert "Column mapping for record" not in capsys.readouterr().out
    assert row_u["column_mapping"] == "uniform"
    assert row_u["column_mapping_shift_px"] == "nan"
    assert raw_u != raw_d
    # The default is --column_mapping lines byte for byte.
    _, _, raw_l = run_page(tmp_path, "squeezed", image, label, "lines")
    assert raw_l == raw_d
