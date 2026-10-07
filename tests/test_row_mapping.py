"""Unit tests for --row_mapping lines, the default, in src/run/digitize.py.

measure_row_mapping() measures the printed grid lines once more in the band of each
of the four layout rows, against the column map of the page, and run() then reads the
leads of a row on the map of that row; a row whose lines carry no map of its own keeps
the page map. No model and no data files: the pages are drawn here, a 1 mm / 5 mm grid
at 200 dpi with thin dark traces and their label mask, whose rows are displaced
against each other, lines and traces alike, as on a printed and scanned sheet.
"""
import contextlib
import csv
import io
import re
import warnings
from functools import lru_cache

import numpy as np
import pytest
import torch
import wfdb
from torchvision.io.image import write_png

from config import FREQUENCY, LEAD_LABEL_MAPPING, SHORT_SIGNAL_LENGTH_SEC, Y_SHIFT_RATIO
from src.run import digitize

# ------------------------------------------------------------------ the drawn page
WIDTH, HEIGHT = 2200, 1700
PERIOD = 200 / 25.4  # px per printed millimetre at 200 dpi
ORIGIN_MM = 15  # the first column starts on the 15th line of the page
LINES = digitize.GRID_LINES_PER_COLUMN
SPAN = digitize.NUM_COLUMNS * LINES
RHYTHM_LEAD = "II"
SHORT_SAMPLES = int(SHORT_SIGNAL_LENGTH_SEC * FREQUENCY)
GAIN = {
    "I": 0.6, "II": 1.0, "III": 0.4, "aVR": -0.8, "aVL": 0.1, "aVF": 0.7,
    "V1": -0.5, "V2": 0.9, "V3": 1.1, "V4": 1.0, "V5": 0.8, "V6": 0.6,
}


def layout_rows():
    """{short lead: layout row} from the baselines config gives the leads.

    The oracle of digitize.LAYOUT_ROW: the pages are drawn with the rows from here.
    """
    leads = digitize.STANDARD_LEAD_OFFSETS_SEC
    ratios = sorted({Y_SHIFT_RATIO[lead] for lead in leads}, reverse=True)
    return {lead: ratios.index(Y_SHIFT_RATIO[lead]) for lead in leads}


def beats(t):
    """An ECG-like signal in mV at the times t in seconds."""
    value = np.zeros_like(t)
    for beat in np.arange(0.4, 10.5, 0.83):
        value += np.exp(-(((t - beat) / 0.018) ** 2))
        value += 0.25 * np.exp(-(((t - beat - 0.25) / 0.06) ** 2))
        value += 0.1 * np.exp(-(((t - beat + 0.16) / 0.03) ** 2))
    return value


def _strokes(positions, bold, length):
    """Darkness of lines at the positions along one axis, area sampled per pixel."""
    profile = np.zeros(length + 8)
    for x, is_bold in zip(positions, bold):
        half, dark = (1.0, 170.0) if is_bold else (0.5, 90.0)
        first = int(np.floor(x - half))
        for pixel in range(first, first + 4):
            cover = min(pixel + 1, x + half) - max(pixel, x - half)
            if cover > 0 and 0 <= pixel < profile.size:
                profile[pixel] = max(profile[pixel], dark * min(cover, 1.0))
    return profile[:length]


@lru_cache(maxsize=None)
def drawn_page(
    compress=0.0,
    rows=(0.0, 0.0, 0.0, 0.0),
    grow=(False, False, False, False),
    snap=0.0,
    blank=None,
    blocks=None,
    leave_out=(),
    stray=None,
):
    """A drawn page: (image [3, H, W] uint8, label mask [1, H, W] uint8, trace_x).

    The 1 mm grid (every fifth line bold) has its fourth column narrower by the share
    compress, lines and traces alike. rows are the displacements in px of the grid
    lines and the traces inside the band of each layout row, constant or, with grow,
    rising linearly from 0 at the start of the first column to the value at the end
    of the last. snap is how far right of the traces the lines are drawn. blank =
    (row, first_mm, last_mm) leaves the grid, not the traces, out in the band of that
    row between the x of those grid millimetres, or draws it there at the share of its
    darkness given as a fourth number, and blocks = (row, shift_px, block_px) moves the
    vertical lines of that row's band by shift_px inside every second block of block_px
    pixels. stray = (row, top, bottom, share, shift_px) draws the grid in the pixel rows
    top to bottom at that share of its darkness, with the vertical lines of that row
    shift_px to the right. leave_out names leads that are not drawn. trace_x(row, mm)
    is the x of the traces of a row at a grid millimetre after the start of the first
    column; row None is the page without the displacement of a row.
    """
    g0 = ORIGIN_MM * PERIOD
    layout = layout_rows()

    def trace_x(row, mm):
        mm = np.asarray(mm, dtype=float)
        x = g0 + mm * PERIOD - compress * PERIOD * np.clip(mm - SPAN + LINES, 0, None)
        if row is None:
            return x
        shift = rows[row]
        if grow[row]:
            shift = shift * np.clip(mm / SPAN, 0.0, 1.0)
        return x + shift

    ratios = sorted({Y_SHIFT_RATIO[lead] for lead in layout}, reverse=True)
    ratios.append(Y_SHIFT_RATIO["full"])
    baselines = np.array([digitize.baseline_row(ratio, HEIGHT) for ratio in ratios])
    # The band of a row: halfway to its neighbours, half a row beyond the outer ones.
    band_edges = np.r_[
        baselines[0] - (baselines[1] - baselines[0]) / 2,
        (baselines[1:] + baselines[:-1]) / 2,
        baselines[-1] + (baselines[-1] - baselines[-2]) / 2,
    ]
    cuts = [0] + [int(round(edge)) for edge in band_edges] + [HEIGHT]

    k = np.arange(-ORIGIN_MM - 2, int(WIDTH / PERIOD) + 3)
    bold = k % 5 == 0
    lines_y = np.arange(0, int(HEIGHT / PERIOD) + 1)
    horizontal = _strokes(lines_y * PERIOD + 0.5 * PERIOD, lines_y % 5 == 0, HEIGHT)
    darkness = np.zeros((HEIGHT, WIDTH))
    for top, bottom, row in zip(cuts[:-1], cuts[1:], [None, 0, 1, 2, 3, None]):
        vertical = _strokes(trace_x(row, k) + snap, bold, WIDTH)
        if blocks is not None and row == blocks[0]:
            moved = _strokes(trace_x(row, k) + snap + blocks[1], bold, WIDTH)
            for start in range(100, WIDTH - 100, 2 * blocks[2]):
                vertical[start : start + blocks[2]] = moved[start : start + blocks[2]]
        band = np.maximum(vertical[None, :], horizontal[top:bottom, None])
        if blank is not None and row == blank[0]:
            share = blank[3] if len(blank) > 3 else 0.0
            band[:, int(trace_x(None, blank[1])) : int(trace_x(None, blank[2]))] *= share
        darkness[top:bottom] = band
    if stray is not None:
        row, top, bottom, share, shift = stray
        moved = _strokes(trace_x(row, k) + snap + shift, bold, WIDTH)
        darkness[top:bottom] = share * np.maximum(
            moved[None, :], horizontal[top:bottom, None]
        )
    image = np.full((3, HEIGHT, WIDTH), 255, dtype=np.uint8)
    # Red lines, as on a printed page: darkest in the two lower channels.
    image[1] = image[2] = np.round(255 - darkness).astype(np.uint8)

    mask = np.zeros((1, HEIGHT, WIDTH), dtype=np.uint8)
    fine_mm = np.linspace(-5, SPAN + 5, 20 * int(SPAN + 10) + 1)
    for lead, label in LEAD_LABEL_MAPPING.items():
        if lead in leave_out:
            continue
        # II is drawn as the rhythm strip only, as the label masks of the model have it.
        if lead == RHYTHM_LEAD:
            row, first_mm, last_mm = digitize.RHYTHM_ROW, 0.0, SPAN
        else:
            row = layout[lead]
            column = int(
                digitize.STANDARD_LEAD_OFFSETS_SEC[lead] / SHORT_SIGNAL_LENGTH_SEC
            )
            first_mm, last_mm = column * LINES, (column + 1) * LINES
        first = int(np.ceil(trace_x(row, first_mm) - 0.5))
        last = int(np.ceil(trace_x(row, last_mm) - 0.5)) - 1
        mm_at = np.interp(
            np.arange(first, last + 2, dtype=float), trace_x(row, fine_mm), fine_mm
        )
        y = baselines[row] - GAIN[lead] * beats(mm_at / 25.0) * 10 * PERIOD
        for index, pixel in enumerate(range(first, last + 1)):
            top = int(np.floor(min(y[index], y[index + 1]) - 0.5))
            bottom = int(np.floor(max(y[index], y[index + 1]) - 0.5)) + 1
            image[:, top : bottom + 1, pixel] = 0
            mask[0, top - 1 : bottom + 2, pixel] = label
    return torch.from_numpy(image), torch.from_numpy(mask), trace_x


# Rows 1 and 2 displaced against rows 0 and 3: +0.6 px, and -0.4 px growing along x.
DISPLACED = {
    "compress": 0.015,
    "rows": (0.0, 0.6, -0.4, 0.0),
    "grow": (False, False, True, False),
}
PAGES = {
    # In the dead band of the column map: no page map, so no row maps.
    "even": {},
    # A page map in use and no row off it.
    "squeezed": {"compress": 0.015},
    "displaced": DISPLACED,
    # No grid lines over 0.35 column of row 2: the gap rule.
    "rowgap": dict(DISPLACED, blank=(2, 1.3 * LINES, 1.65 * LINES)),
    # The lines are there at a tenth of their darkness: too weak a comb for a window.
    "rowfaint": dict(DISPLACED, blank=(2, 1.3 * LINES, 1.65 * LINES, 0.1)),
    # The lines of row 2 are 3 px off in every second block of 80 px: the coverage rule.
    "rowcov": {"compress": 0.015, "blocks": (2, 3.0, 80)},
    # Row 1 is 0.4 line off the page map: beyond the branch tolerance.
    "rowfar": {"compress": 0.015, "rows": (0.0, 0.4 * PERIOD, 0.0, 0.0)},
}


@lru_cache(maxsize=None)
def page_map(name, snap=0.0, median=13.5, **changes):
    """The grid stages of run() on a drawn page, up to the column map."""
    image, mask, trace_x = drawn_page(**{**PAGES[name], "snap": snap, **changes})
    masks, positions, _ = digitize.cut_binary(mask, image)
    with contextlib.redirect_stdout(io.StringIO()):
        g0, P, long_leads, reason = digitize.fit_column_grid(
            masks, positions, HEIGHT, "page", name
        )
    assert reason == "", reason
    g0, P, lines = digitize.refine_grid_from_lines(image, g0, P, snap_offset=snap)
    assert lines["reason"] == "", lines["reason"]
    x_map, info = digitize.measure_column_mapping(
        image,
        g0,
        P,
        digitize.column_mapping_edges(masks, positions, long_leads),
        snap_offset=snap,
        median_mm=median,
    )
    return {
        "image": image,
        "masks": masks,
        "positions": positions,
        "long_leads": long_leads,
        "g0": g0,
        "P": P,
        "x_map": x_map,
        "info": info,
        "trace_x": trace_x,
        "snap": snap,
    }


def row_maps(page, median=13.5):
    return digitize.measure_row_mapping(
        page["image"],
        page["g0"],
        page["P"],
        page["masks"],
        page["positions"],
        page["long_leads"],
        page["info"],
        snap_offset=page["snap"],
        median_mm=median,
    )


def read_mm(out):
    """The grid millimetre of every sample the leads of a row read."""
    return np.concatenate(
        [(c + np.arange(SHORT_SAMPLES) / SHORT_SAMPLES) * LINES for c in out["columns"]]
    )


def row_centres(rows):
    """The centres of the four rows, from their bands: the edge between two bands is
    halfway between their centres, and the first band ends as far below its centre as
    it starts above it."""
    edges = [rows[0]["band"][0]] + [out["band"][1] for out in rows]
    centres = [(edges[0] + edges[1]) / 2]
    for edge in edges[1:-1]:
        centres.append(2 * edge - centres[-1])
    return centres


def turn_the_lines(patch, delta):
    """Move the 1 mm lines of every window of measure_row_mapping() by delta(x) px.

    The phase of the 1 mm comb of the window at x is turned by what a displacement of
    its lines by delta(x) pixels to the right is. A drawn page cannot do that from one
    window to the next: the windows are half a line apart and eight lines long.
    """
    windows = digitize._column_mapping_windows

    def turned(profile, period, low, high):
        centres, fine, coarse = windows(profile, period, low, high)
        return centres, fine * np.exp(-2j * np.pi * delta(centres) / period), coarse

    patch.setattr(digitize, "_column_mapping_windows", turned)


# ------------------------------------------------------------- flags and constants
def test_the_flag_is_lines_unless_off_is_asked_for(capsys):
    parse = digitize.get_parser().parse_args
    folders = ["-d", "in", "-o", "out"]
    assert parse(folders).row_mapping == "lines"
    assert parse(folders + ["--row_mapping", "lines"]).row_mapping == "lines"
    assert parse(folders + ["--row_mapping", "off"]).row_mapping == "off"
    with pytest.raises(SystemExit):
        parse(folders + ["--row_mapping", "page"])
    capsys.readouterr()


def test_the_median_is_two_periods_of_the_ripple_unless_given():
    parse = digitize.get_parser().parse_args
    folders = ["-d", "in", "-o", "out"]
    assert digitize.ROW_MAPPING_MEDIAN_MM == 27.0 == 2 * digitize.COLUMN_MAPPING_MEDIAN_MM
    assert parse(folders).row_mapping_median == 27.0
    assert parse(folders + ["--row_mapping_median", "13.5"]).row_mapping_median == 13.5
    assert parse(folders + ["--row_mapping_median", "0"]).row_mapping_median == 0.0


def test_the_layout_rows_are_those_of_the_baselines_in_config():
    assert digitize.LAYOUT_ROW == layout_rows()
    assert digitize.NUM_ROWS == len(set(digitize.LAYOUT_ROW.values())) + 1 == 4
    assert digitize.RHYTHM_ROW == digitize.NUM_ROWS - 1
    # The rhythm strip is drawn below every short lead.
    assert Y_SHIFT_RATIO["full"] < min(
        Y_SHIFT_RATIO[lead] for lead in digitize.LAYOUT_ROW
    )


def test_lead_row_puts_a_rhythm_strip_in_the_last_row():
    assert digitize.lead_row("II", ["II"]) == digitize.RHYTHM_ROW
    assert digitize.lead_row("II", []) == 1
    assert digitize.lead_row("V1", ["II"]) == 0
    assert digitize.lead_row("V6", ["II"]) == 2
    # A rhythm strip of any lead, and a lead that is in no row.
    assert digitize.lead_row("V1", ["V1"]) == digitize.RHYTHM_ROW
    assert digitize.lead_row("X", ["II"]) is None


def test_column_map_m_is_the_inverse_of_column_map_x():
    knots_m = np.array([0.0, 10.0, 20.0, 40.0])
    knots_x = np.array([100.0, 178.0, 260.0, 415.0])
    period = 7.874
    # Between the knots and beyond them, where the uniform grid is carried on.
    m = np.array([-12.5, -0.1, 0.0, 3.3, 10.0, 19.9, 33.0, 40.0, 47.25])
    x = digitize._column_map_x(knots_m, knots_x, period, m)
    assert np.allclose(digitize._column_map_m(knots_m, knots_x, period, x), m, atol=1e-12)
    assert digitize._column_map_m(
        knots_m, knots_x, period, 100.0 - period
    ) == pytest.approx(-1)


# ------------------------------------------------------------------- the row maps
@pytest.mark.parametrize("snap", [0.0, 0.5])
def test_displaced_rows_are_measured_within_a_tenth_of_a_pixel(snap):
    page = page_map("displaced", snap=snap)
    assert page["x_map"] is not None
    rows = row_maps(page)
    assert [out["row"] for out in rows] == [0, 1, 2, 3]
    assert all(out["accepted"] and out["why"] == "" for out in rows)
    # Lead II of the second row is the rhythm strip, so that row reads three columns.
    assert [out["columns"] for out in rows] == [[0, 1, 2, 3], [1, 2, 3]] + [
        [0, 1, 2, 3]
    ] * 2
    trace_x = page["trace_x"]
    for out in rows:
        mm = read_mm(out)
        # Against row 0: +0.6 px for row 1, 0 to -0.4 px along x for row 2, 0 for row 3.
        against_row0 = out["x_map"](mm) - rows[0]["x_map"](mm)
        drawn = trace_x(out["row"], mm) - trace_x(0, mm)
        assert np.max(np.abs(against_row0 - drawn)) <= 0.1, out["row"]
        # Against the drawn traces, where the page map is the mean of the four rows.
        error = np.max(np.abs(out["x_map"](mm) - trace_x(out["row"], mm)))
        assert error <= 0.15, (out["row"], error)
    page_error = max(
        np.max(np.abs(page["x_map"](read_mm(out)) - trace_x(out["row"], read_mm(out))))
        for out in rows
    )
    assert page_error > 0.3
    # The numbers of the --verbose line: the move of row 1 against the page map.
    assert rows[1]["shift"] == pytest.approx(0.6 - 0.1, abs=0.15)
    assert all(low <= median <= high for median, low, high in rows[1]["d"])
    assert rows[1]["windows"] == rows[1]["inside"] > 300 and rows[1]["beyond"] == 0


def test_the_band_of_a_row_reaches_halfway_to_its_neighbours():
    page = page_map("displaced")
    rows = row_maps(page)
    ratios = sorted(
        set(Y_SHIFT_RATIO[lead] for lead in digitize.LAYOUT_ROW), reverse=True
    )
    baselines = [
        digitize.baseline_row(r, HEIGHT) for r in ratios + [Y_SHIFT_RATIO["full"]]
    ]
    for out, above, below in zip(rows[1:3], baselines[:2], baselines[2:]):
        middle = baselines[out["row"]]
        # The centre of a row is where its masks are, a few px off the baseline.
        assert out["band"][0] == pytest.approx((above + middle) / 2, abs=25)
        assert out["band"][1] == pytest.approx((middle + below) / 2, abs=25)
    assert all(a["band"][1] == b["band"][0] for a, b in zip(rows[:-1], rows[1:]))
    assert rows[0]["band"][0] >= 0 and rows[-1]["band"][1] <= HEIGHT


def test_a_page_without_displaced_rows_stays_on_the_page_map():
    rows = row_maps(page_map("squeezed"))
    assert all(out["accepted"] for out in rows)
    assert max(out["shift"] for out in rows) <= 0.05


@pytest.mark.parametrize(
    "name, kept, why",
    [
        ("rowgap", 2, r"row has a gap of 0\.\d\d columns"),
        ("rowfaint", 2, r"row has a gap of 0\.\d\d columns"),
        ("rowcov", 2, r"row keeps \d+ % of its windows, \d+ beyond the branch tolerance"),
        ("rowfar", 1, r"\d+ of \d+ windows beyond the branch tolerance"),
    ],
)
def test_a_row_without_a_map_of_its_own_keeps_the_page_map(name, kept, why):
    rows = row_maps(page_map(name))
    assert not rows[kept]["accepted"] and rows[kept]["x_map"] is None
    assert re.fullmatch(why, rows[kept]["why"]), rows[kept]["why"]
    assert all(
        out["accepted"] and out["x_map"] is not None for out in rows if out["row"] != kept
    )
    if name == "rowfar":
        # The lines are there, further off the page map than a line can be told.
        assert rows[kept]["beyond"] > 0.9 * rows[kept]["inside"]
        assert rows[kept]["windows"] == 0


def test_a_row_is_judged_on_the_columns_its_leads_read():
    # Lead II is the rhythm strip, so the second row reads no first column: 0.95 column
    # without grid lines there is neither a gap of that row nor windows it has lost.
    page = page_map("displaced", blank=(1, 0.0, 0.95 * LINES))
    assert page["x_map"] is not None
    rows = row_maps(page)
    assert rows[1]["columns"] == [1, 2, 3]
    assert rows[1]["accepted"] and rows[1]["windows"] == rows[1]["inside"] > 300
    # The same stretch in the row below, which reads the first column, is its own.
    rows = row_maps(page_map("displaced", blank=(2, 0.0, 0.95 * LINES)))
    assert rows[2]["columns"] == [0, 1, 2, 3]
    assert not rows[2]["accepted"]
    assert re.fullmatch(r"row keeps 7\d % of its windows", rows[2]["why"]), rows[2]["why"]
    assert rows[1]["accepted"]


def test_the_centre_of_a_row_is_the_median_over_its_leads():
    # One lead of the first row with its mask 100 px lower, each of its leads in turn:
    # the median over the four stays with the other three, and so do the bands of the
    # page, within the few pixels the leads of a row differ by.
    page = page_map("displaced")
    plain = row_maps(page)
    for lead in ("I", "aVR", "V1", "V4"):
        positions = {name: dict(at) for name, at in page["positions"].items()}
        positions[lead]["y1"] += 100
        rows = row_maps({**page, "positions": positions})
        for out, before in zip(rows, plain):
            assert out["band"] == pytest.approx(before["band"], abs=5), (lead, out["row"])


def test_the_displacement_of_a_row_is_held_beyond_its_outer_windows():
    # No grid lines over the last 8 mm of row 1 and beyond them: its last window is an
    # eighth of a column before the end of the columns, and what the row is displaced
    # by there holds for the samples after it. The page map, the mean of the other
    # rows over that stretch, is more than half a pixel off the traces of row 1 there.
    page = page_map("displaced", blank=(1, SPAN - 8.0, SPAN + 20.0))
    out = row_maps(page)[1]
    assert out["accepted"] and out["windows"] < out["inside"]
    assert 0.1 < out["gap"] < 0.15
    mm = np.arange(SPAN - 8.0, SPAN, 0.05)
    drawn = page["trace_x"](1, mm)
    assert np.max(np.abs(out["x_map"](mm) - drawn)) < 0.3
    assert np.min(np.abs(page["x_map"](mm) - drawn)) > 0.5


def test_a_row_band_too_low_for_a_profile_keeps_the_page_map():
    # The masks of row 1 moved to 12 px above those of row 2 and the rhythm strip to
    # 40 px below them: the band of row 2 is 25 pixel rows high, too few for a median
    # profile of the lines, and that of the rhythm strip, 40 rows, is still measured.
    page = page_map("displaced")
    centres = row_centres(row_maps(page))
    moves = {1: centres[2] - 12 - centres[1], 3: centres[2] + 40 - centres[3]}
    positions = {}
    for lead, position in page["positions"].items():
        row = digitize.lead_row(lead, page["long_leads"])
        positions[lead] = dict(position, y1=position["y1"] + moves.get(row, 0.0))
    rows = row_maps({**page, "positions": positions})
    low = rows[2]
    height = int(np.floor(low["band"][1])) - int(np.ceil(low["band"][0]))
    assert height == 25 < digitize.ROW_MAPPING_MIN_BAND_ROWS
    assert not low["accepted"] and low["x_map"] is None
    assert low["why"] == "row band of 25 rows"
    assert all(out["accepted"] for out in rows if out["row"] != 2)


def test_a_sub_band_with_a_weak_comb_is_left_out_of_the_profile_of_a_row():
    # The lower of the two sub-bands of row 2 has its lines at a fifth of their darkness
    # and 2 px to the right. That is a weaker comb than GRID_LINE_MIN_BAND_AMPLITUDE of
    # the page's bands, so the row is measured on its upper sub-band alone and its map
    # is the one of the page without those lines; with them in its profile the row
    # would move by a quarter of a pixel.
    page = page_map("displaced")
    plain = row_maps(page)
    top, bottom = int(np.ceil(plain[2]["band"][0])), int(np.floor(plain[2]["band"][1]))
    assert (bottom - top) // digitize.GRID_LINE_BAND_HEIGHT == 2
    stray = (2, (top + bottom) // 2, bottom, 0.2, 2.0)
    image, _, _ = drawn_page(**{**PAGES["displaced"], "stray": stray})
    assert not torch.equal(image, page["image"])
    rows = row_maps({**page, "image": image})
    for out, before in zip(rows, plain):
        assert out["accepted"], out["row"]
        assert np.max(np.abs(out["knots_x"] - before["knots_x"])) < 0.01, out["row"]


def test_a_row_whose_map_would_not_grow_along_x_keeps_the_page_map(monkeypatch):
    # Lines 2.2 px right of the page map up to the middle of the page and 2.2 px left of
    # it from there on, both within the branch tolerance: between the two windows at
    # the step, 4 px apart, the knots of the row would fall back by 0.4 px.
    page = page_map("squeezed", median=0.0)
    middle = page["g0"] + 2 * page["P"]
    with monkeypatch.context() as patch:
        turn_the_lines(patch, lambda x: np.where(x < middle, 2.2, -2.2))
        rows = row_maps(page, median=0.0)
    assert all(out["windows"] == out["inside"] and out["beyond"] == 0 for out in rows)
    assert all(not out["accepted"] and out["x_map"] is None for out in rows)
    assert {out["why"] for out in rows} == {"row map does not grow along x"}
    # The same step the other way is a map that grows, and a row reads on it.
    with monkeypatch.context() as patch:
        turn_the_lines(patch, lambda x: np.where(x < middle, -2.2, 2.2))
        rows = row_maps(page, median=0.0)
    assert all(out["accepted"] and out["why"] == "" for out in rows)
    for out in rows:
        medians = [median for median, _, _ in out["d"]]
        assert medians == pytest.approx([-2.2, -2.2, 2.2, 2.2], abs=0.05), out["row"]


def test_a_step_of_the_lines_of_a_row_is_a_step_of_its_map_at_the_same_knots(
    monkeypatch,
):
    # What a window measures belongs to the grid millimetre of the traces under its
    # centre, which are snap_offset left of its lines. On a page with the lines half a
    # pixel right of the traces and a page map as measured, a knot is the trace x under
    # a window, so lines that step by 2 px between two windows move the knots of the
    # row by the same step between the same two knots.
    page = page_map("squeezed", snap=0.5, median=0.0)
    plain = row_maps(page, median=0.0)
    middle = page["g0"] + 2 * page["P"]
    with monkeypatch.context() as patch:
        turn_the_lines(patch, lambda x: np.where(x < middle, 0.0, 2.0))
        rows = row_maps(page, median=0.0)
    step = np.where(page["info"]["knots_x"] + 0.5 < middle, 0.0, 2.0)
    assert 0.0 < step.mean() < 2.0
    for out, before in zip(rows, plain):
        assert out["accepted"] and before["accepted"]
        moved = out["knots_x"] - before["knots_x"]
        assert np.max(np.abs(moved - step)) < 1e-6, out["row"]


def test_a_page_without_a_lead_in_a_row_keeps_the_page_map_in_every_row():
    page = page_map("displaced", leave_out=("III", "aVF", "V3", "V6"))
    rows = row_maps(page)
    assert len(rows) == digitize.NUM_ROWS
    assert all(not out["accepted"] and out["x_map"] is None for out in rows)
    assert {out["why"] for out in rows} == {"row bands: no lead of row 2"}


@pytest.mark.parametrize("snap", [0.0, 0.5])
def test_the_running_median_takes_the_ripple_out_of_a_row(snap):
    # The traces of a drawn page do not share the pixel snap of its lines: as
    # measured, a row of an even page (compressed fourth column only) moves by the
    # ripple of that snap, with the median it stays where the page map is.
    page = page_map("squeezed", snap=snap)
    raw, one, two = (row_maps(page, median=m) for m in (0.0, 13.5, 27.0))
    for out_raw, out_one, out_two in zip(raw, one, two):
        assert out_raw["accepted"] and out_one["accepted"] and out_two["accepted"]
        assert out_two["shift"] <= out_one["shift"] + 1e-9 <= out_raw["shift"] + 2e-9
    # 0 and anything below it is the displacement as measured.
    below = row_maps(page, median=-1.0)
    assert all(np.array_equal(a["knots_x"], b["knots_x"]) for a, b in zip(raw, below))
    assert any(not np.array_equal(a["knots_x"], b["knots_x"]) for a, b in zip(raw, two))
    # The default is ROW_MAPPING_MEDIAN_MM.
    default = digitize.measure_row_mapping(
        page["image"],
        page["g0"],
        page["P"],
        page["masks"],
        page["positions"],
        page["long_leads"],
        page["info"],
        snap_offset=snap,
    )
    assert all(np.array_equal(a["knots_x"], b["knots_x"]) for a, b in zip(default, two))


def test_measure_row_mapping_leaves_the_page_map_as_it_is():
    page = page_map("displaced")
    knots_m, knots_x = page["info"]["knots_m"].copy(), page["info"]["knots_x"].copy()
    mm = np.arange(0.0, SPAN, 0.05)
    before = page["x_map"](mm)
    rows = row_maps(page)
    assert np.array_equal(page["info"]["knots_m"], knots_m)
    assert np.array_equal(page["info"]["knots_x"], knots_x)
    assert np.array_equal(page["x_map"](mm), before)
    # Every row has its own knots: the map of one row is not the map of the next.
    assert not np.array_equal(rows[0]["x_map"](mm), rows[1]["x_map"](mm))
    assert rows[0]["knots_x"] is not rows[1]["knots_x"]


def test_row_mapping_line_says_what_a_row_is_read_on():
    rows = row_maps(page_map("rowgap"))
    own = digitize.row_mapping_line("rec-0_0000", rows[1])
    assert own.startswith("Row mapping for record rec-0_0000: row 1 own map, windows ")
    assert f"windows {rows[1]['windows']}/{rows[1]['inside']}" in own
    assert re.search(r"c1 \+0\.\d\d/[+-]0\.\d\d/\+0\.\d\d c2 ", own), own
    kept = digitize.row_mapping_line("rec", rows[2])
    assert kept.startswith(
        "Row mapping for record rec: row 2 page map kept (row has a gap of"
    )
    assert "WARNING" not in own + kept and "\n" not in own + kept
    # A row that was not measured has no moves to report.
    none = row_maps(page_map("displaced", leave_out=("III", "aVF", "V3", "V6")))
    assert digitize.row_mapping_line("rec", none[0]).endswith("c1 - c2 - c3 - c4 - px")


# ------------------------------------------------------------------- end to end
STAGES = (
    "--rotation", "hough", "--perspective", "off", "--resolution", "keep",
    "--paper_normalisation", "off", "-f",
)
# The runs below are about the row maps alone, and their numbers were fixed with no
# second try of a page that fails a check of the grid: --grid_rescue off, which was the
# default then. The runs of the two defaults together take STAGES.
FLAGS = (*STAGES, "--grid_rescue", "off")
# The pages of the runs, drawn with the lines 0.5 px right of the traces and read with
# the default --grid_line_offset 0.5, the setting of the real pages.
RUN_PAGES = ("even", "squeezed", "displaced", "rowgap")
ROWS_ON = ("--row_mapping", "lines")
# lines is the default, so the run that reads every lead on the page map asks for off.
ROWS_OFF = ("--row_mapping", "off")


@pytest.fixture(scope="module")
def runs(tmp_path_factory):
    """run() on drawn pages with the flags given, each set of pages and flags once.

    Returns a function: (*flags, pages=RUN_PAGES, base=FLAGS) -> (output folder, what
    run() printed); base are the flags every run of this file has.
    """
    root = tmp_path_factory.mktemp("row_mapping")
    folders, done = {}, {}

    def run(*flags, pages=RUN_PAGES, base=FLAGS):
        if pages not in folders:
            data, masks = root / f"data{len(folders)}", root / f"masks{len(folders)}"
            data.mkdir()
            masks.mkdir()
            for name in pages:
                image, mask, _ = drawn_page(**{**PAGES[name], "snap": 0.5})
                write_png(image, str(data / f"{name}.png"))
                write_png(mask, str(masks / f"{name}_mask.png"))
            folders[pages] = (data, masks)
        if (pages, base, flags) not in done:
            data, masks = folders[pages]
            out = root / f"out{len(done)}"
            argv = ["-d", str(data), "-o", str(out), "--mask_folder", str(masks)]
            printed = io.StringIO()
            with contextlib.redirect_stdout(printed):
                digitize.run(digitize.get_parser().parse_args(argv + [*base, *flags]))
            done[pages, base, flags] = (out, printed.getvalue())
        return done[pages, base, flags]

    return run


def signals(folder, name):
    with open(folder / f"{name}.dat", "rb") as dat, open(
        folder / f"{name}.hea", "rb"
    ) as hea:
        return dat.read(), hea.read()


def qc_rows(folder):
    with open(folder / "qc.csv", newline="") as f:
        return {row["record"]: row for row in csv.DictReader(f)}


def read_record(folder, name):
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        return wfdb.rdrecord(str(folder / name))


def drawn_snr(folder, name):
    """SNR in dB of every lead of a digitised page against the signal it was drawn from
    (sample n is 1/20 mm at 25 mm/s and 500 Hz; means removed)."""
    record = read_record(folder, name)
    snr = {}
    for index, lead in enumerate(record.sig_name):
        read = record.p_signal[:, index]
        drawn = GAIN[lead] * beats(np.arange(read.size) / FREQUENCY)
        valid = np.isfinite(read)
        drawn = drawn[valid] - drawn[valid].mean()
        error = (read[valid] - read[valid].mean()) - drawn
        snr[lead] = float(10 * np.log10(np.sum(drawn ** 2) / np.sum(error ** 2)))
    return snr


def row_lines(printed, name):
    return [
        line
        for line in printed.splitlines()
        if line.startswith(f"Row mapping for record {name}:")
    ]


def test_run_with_the_flag_off_writes_off(runs):
    out, printed = runs(*ROWS_OFF)
    rows = qc_rows(out)
    assert sorted(rows) == sorted(RUN_PAGES)
    assert all(float(row["rotation_angle"]) == 0.0 for row in rows.values())
    assert all(row["row_mapping"] == "off" for row in rows.values())
    assert "Row mapping" not in printed
    assert rows["even"]["column_mapping"] == "uniform: dead band"
    assert all(rows[name]["column_mapping"] == "lines" for name in RUN_PAGES[1:])


def test_a_run_that_does_not_name_the_flag_is_the_run_with_lines(runs):
    # lines is the default: --row_mapping lines is the run without the flag, to the
    # lines it prints, and --row_mapping off is not.
    few = ("displaced",)
    plain, printed_plain = runs(pages=few)
    named, printed_named = runs(*ROWS_ON, pages=few)
    off, _ = runs(*ROWS_OFF, pages=few)
    assert printed_named == printed_plain
    assert signals(named, "displaced") == signals(plain, "displaced")
    assert qc_rows(named) == qc_rows(plain)
    assert qc_rows(plain)["displaced"]["row_mapping"] == "lines 4/4"
    assert len(row_lines(printed_plain, "displaced")) == digitize.NUM_ROWS
    assert signals(off, "displaced") != signals(plain, "displaced")
    assert qc_rows(off)["displaced"]["row_mapping"] == "off"


def test_a_run_that_names_neither_flag_is_the_run_with_both_written_out(runs):
    # lines and map are the defaults. On a page whose rows are displaced, the run that
    # names neither --row_mapping nor --grid_rescue is the run with both written out, to
    # the signals, the QC row and the lines it prints: every row is read on its own
    # map, and the page, which fails no check of the grid, is not tried a second time.
    # With both off it is another run.
    few = ("displaced",)
    default, printed = runs(pages=few, base=STAGES)
    both = ("--grid_rescue", "map", *ROWS_ON)
    named, printed_named = runs(*both, pages=few, base=STAGES)
    off, _ = runs("--grid_rescue", "off", *ROWS_OFF, pages=few, base=STAGES)
    assert printed == printed_named
    assert qc_rows(default) == qc_rows(named)
    assert signals(default, "displaced") == signals(named, "displaced")
    assert signals(default, "displaced") != signals(off, "displaced")
    row, row_off = qc_rows(default)["displaced"], qc_rows(off)["displaced"]
    assert (row["grid_rescue"], row["row_mapping"]) == ("", "lines 4/4")
    assert (row_off["grid_rescue"], row_off["row_mapping"]) == ("off", "off")
    assert len(row_lines(printed, "displaced")) == digitize.NUM_ROWS
    assert "Grid rescue" not in printed
    # The rescue has nothing to do on this page: but for its QC cell, the run is the
    # one with the row maps alone.
    rows_only, _ = runs(*ROWS_ON)
    assert signals(default, "displaced") == signals(rows_only, "displaced")
    assert {**row, "grid_rescue": "off"} == qc_rows(rows_only)["displaced"]


def test_run_reads_a_page_without_a_page_map_as_with_the_flag_off(runs):
    plain, _ = runs(*ROWS_OFF)
    out, printed = runs(*ROWS_ON)
    assert signals(out, "even") == signals(plain, "even")
    row = qc_rows(out)["even"]
    assert row["row_mapping"] == "page map not used"
    assert {**row, "row_mapping": "off"} == qc_rows(plain)["even"]
    assert row_lines(printed, "even") == [
        "Row mapping for record even: page map not used"
    ]


def test_run_reads_every_row_on_its_own_map(runs):
    plain, _ = runs(*ROWS_OFF)
    out, printed = runs(*ROWS_ON)
    rows, rows_plain = qc_rows(out), qc_rows(plain)
    assert rows["displaced"]["row_mapping"] == "lines 4/4"
    assert rows["squeezed"]["row_mapping"] == "lines 4/4"
    assert rows["rowgap"]["row_mapping"] == "lines 3/4"
    for name in ("displaced", "rowgap"):
        assert signals(out, name) != signals(plain, name), name
        # The page map is the one of the run with the flag off.
        for column in ("column_mapping", "column_mapping_shift_px", "grid_rescue"):
            assert rows[name][column] == rows_plain[name][column]
    # --verbose: one line per row, and a row that keeps the page map says why.
    said = row_lines(printed, "displaced")
    assert len(said) == digitize.NUM_ROWS
    assert all(f": row {row} own map, " in line for row, line in enumerate(said))
    said = row_lines(printed, "rowgap")
    assert [(": row 2 page map kept (row has a gap of" in line) for line in said] == [
        False,
        False,
        True,
        False,
    ]
    assert "WARNING: column mapping" not in printed


def test_run_reads_the_displaced_rows_as_they_were_drawn(runs):
    plain, _ = runs(*ROWS_OFF)
    out, _ = runs(*ROWS_ON)
    before, after = drawn_snr(plain, "displaced"), drawn_snr(out, "displaced")
    gain = {lead: after[lead] - before[lead] for lead in before}
    layout = layout_rows()
    displaced = [
        lead for lead in before if lead != RHYTHM_LEAD and layout[lead] in (1, 2)
    ]
    # 25.0 -> 32.5 dB in the median: no lead loses a decibel, and the leads of the two
    # displaced rows, read half a pixel off on the page map, gain.
    assert np.median(list(before.values())) < 27.0
    assert np.median(list(after.values())) > 29.0
    assert min(gain.values()) > -1.0, gain
    assert np.median([gain[lead] for lead in displaced]) > 1.0, gain
    # A page whose rows are not displaced is read as before, within a decibel a lead.
    before, after = drawn_snr(plain, "squeezed"), drawn_snr(out, "squeezed")
    assert all(abs(after[lead] - before[lead]) < 1.0 for lead in before)


def test_run_reads_a_lead_on_the_map_of_its_own_row(runs):
    # With --baseline page no lead depends on another one, so the leads of the row
    # that keeps the page map are the samples of the run with the flag off, and the
    # leads of every other row are not.
    few = ("rowgap",)
    plain, _ = runs("--baseline", "page", *ROWS_OFF, pages=few)
    out, _ = runs("--baseline", "page", *ROWS_ON, pages=few)
    assert qc_rows(out)["rowgap"]["row_mapping"] == "lines 3/4"
    before, after = read_record(plain, "rowgap"), read_record(out, "rowgap")
    assert before.sig_name == after.sig_name
    for index, lead in enumerate(before.sig_name):
        same = np.array_equal(
            before.p_signal[:, index], after.p_signal[:, index], equal_nan=True
        )
        row = digitize.lead_row(lead, [RHYTHM_LEAD])
        assert same == (row == 2), (lead, row)


def test_run_hands_the_median_it_is_given_to_the_rows(runs):
    few = ("displaced",)
    default, _ = runs(*ROWS_ON)
    two, _ = runs(*ROWS_ON, "--row_mapping_median", "27", pages=few)
    one, _ = runs(*ROWS_ON, "--row_mapping_median", "13.5", pages=few)
    assert signals(two, "displaced") == signals(default, "displaced")
    assert signals(one, "displaced") != signals(default, "displaced")
    assert qc_rows(one)["displaced"]["row_mapping"] == "lines 4/4"


def test_run_says_nothing_of_the_rows_without_verbose(runs):
    # Neither of a page that is read on row maps nor of one without a page map.
    few = ("even", "displaced")
    loud, _ = runs(*ROWS_ON)
    out, printed = runs(*ROWS_ON, "--no-verbose", pages=few)
    assert "Row mapping" not in printed
    rows, rows_loud = qc_rows(out), qc_rows(loud)
    assert [rows[name]["row_mapping"] for name in few] == [
        "page map not used",
        "lines 4/4",
    ]
    assert all(signals(out, name) == signals(loud, name) for name in few)
    assert rows == {name: rows_loud[name] for name in few}


def test_run_has_no_rows_to_map_on_uniform_columns(runs):
    few = ("displaced",)
    plain, _ = runs("--column_mapping", "uniform", *ROWS_OFF, pages=few)
    out, printed = runs("--column_mapping", "uniform", *ROWS_ON, pages=few)
    assert signals(out, "displaced") == signals(plain, "displaced")
    row = qc_rows(out)["displaced"]
    assert (
        row["column_mapping"] == "uniform" and row["row_mapping"] == "page map not used"
    )
    assert row_lines(printed, "displaced") == [
        "Row mapping for record displaced: page map not used"
    ]


def test_a_rescued_page_gets_row_maps_too(tmp_path):
    # A fourth column 2.6 % narrower fails the fit; rescued on its column map, the
    # page is one with a page map, and its rows are measured against that map.
    image, mask, _ = drawn_page(
        compress=0.026, rows=DISPLACED["rows"], grow=DISPLACED["grow"]
    )
    data, masks = tmp_path / "data", tmp_path / "masks"
    data.mkdir()
    masks.mkdir()
    write_png(image, str(data / "narrow.png"))
    write_png(mask, str(masks / "narrow_mask.png"))
    rows = {}
    for tag, flags in (
        ("plain", ("--grid_rescue", "off", *ROWS_OFF)),
        ("rows", ("--grid_rescue", "off", *ROWS_ON)),
        ("rescue", ("--grid_rescue", "map", *ROWS_OFF)),
        ("both", ("--grid_rescue", "map", *ROWS_ON)),
        # map and lines are the defaults: no flag is the run with both.
        ("default", ()),
    ):
        argv = ["-d", str(data), "-o", str(tmp_path / tag), "--mask_folder", str(masks)]
        argv += [*STAGES, "--grid_line_offset", "0", "--no-verbose", *flags]
        with contextlib.redirect_stdout(io.StringIO()):
            digitize.run(digitize.get_parser().parse_args(argv))
        rows[tag] = qc_rows(tmp_path / tag)["narrow"]
    assert rows["plain"]["column_mapping"] == "uniform: no column grid"
    assert (rows["plain"]["grid_rescue"], rows["plain"]["row_mapping"]) == ("off", "off")
    # Without the rescue there is no page map to measure the rows against.
    assert rows["rows"]["row_mapping"] == "page map not used"
    assert signals(tmp_path / "rows", "narrow") == signals(tmp_path / "plain", "narrow")
    assert (rows["rescue"]["grid_rescue"], rows["rescue"]["row_mapping"]) == (
        "fit",
        "off",
    )
    assert (rows["both"]["grid_rescue"], rows["both"]["row_mapping"]) == (
        "fit",
        "lines 4/4",
    )
    assert rows["both"]["column_mapping"] == "lines"
    assert signals(tmp_path / "both", "narrow") != signals(tmp_path / "rescue", "narrow")
    assert rows["default"] == rows["both"]
    default = signals(tmp_path / "default", "narrow")
    assert default == signals(tmp_path / "both", "narrow")
    assert default != signals(tmp_path / "plain", "narrow")
    # The rows are measured with the --grid_line_offset of the run, 0 on this page as
    # for a scan. On their own maps the leads are read as they were drawn, 31.7 dB in
    # the median where the page map alone gives 25.0 dB; measured half a pixel off,
    # with the default offset, the rows would be read at 16.8 dB.
    rescued = np.median(list(drawn_snr(tmp_path / "rescue", "narrow").values()))
    both = np.median(list(drawn_snr(tmp_path / "both", "narrow").values()))
    assert both > 29.0 and both > rescued + 3.0, (rescued, both)
