"""Unit tests for --grid_rescue map in src/run/digitize.py.

A page whose mask edges are off one uniform column grid (fit_column_grid() gives no
grid) or whose grid lines are off the origin of the mask grid (refine_grid_from_lines()
refuses them) is tried a second time, rescue_grid_fit() and rescue_grid_phase(), and
kept on that second grid only if measure_column_mapping() returns a map on it; a page
whose map does not stand is what it is without the flag. No model and no data files:
the pages are drawn here, a 1 mm / 5 mm grid at 200 dpi with thin dark traces and
their label mask, undistorted and with the distortions of a printed and scanned sheet
(a narrower fourth column, masks that start early, grid lines off the traces, a stretch
without grid lines).
"""
import contextlib
import csv
import inspect
import io
import re
import warnings
from functools import lru_cache

import numpy as np
import pytest
import torch
import wfdb
from torchvision.io.image import write_png

from config import FREQUENCY, LEAD_LABEL_MAPPING, Y_SHIFT_RATIO
from src.run import digitize

# ------------------------------------------------------------------ the drawn page
WIDTH, HEIGHT = 2200, 1700
PERIOD = 200 / 25.4  # px per printed millimetre at 200 dpi
ORIGIN_MM = 15  # the first column starts on the 15th line of the page
LINES = digitize.GRID_LINES_PER_COLUMN
SPAN = digitize.NUM_COLUMNS * LINES
G0_DRAWN = ORIGIN_MM * PERIOD
P_DRAWN = LINES * PERIOD
RHYTHM_LEAD = "II"
GAIN = {
    "I": 0.6, "II": 1.0, "III": 0.4, "aVR": -0.8, "aVL": 0.1, "aVF": 0.7,
    "V1": -0.5, "V2": 0.9, "V3": 1.1, "V4": 1.0, "V5": 0.8, "V6": 0.6,
}


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
def drawn_page(compress=0.0, line_phase=0.0, early=0, snap=0.0, blank=None):
    """A drawn page: (image [3, H, W] uint8, label mask [1, H, W] uint8, trace_x).

    The 1 mm grid (every fifth line bold) has its fourth column narrower by the share
    compress, lines and traces alike. line_phase moves the lines alone, snap is how far
    right of the traces the lines are drawn, early extends the masks of the first row's
    leads by that many pixels to the left, and blank = (first_mm, last_mm) leaves the
    grid, not the traces, out between the x of those grid millimetres. trace_x(mm) is
    the x of the traces at a grid millimetre after the start of the first column.
    """

    def trace_x(mm):
        mm = np.asarray(mm, dtype=float)
        return (
            G0_DRAWN
            + mm * PERIOD
            - compress * PERIOD * np.clip(mm - SPAN + LINES, 0, None)
        )

    k = np.arange(-ORIGIN_MM - 2, int(WIDTH / PERIOD) + 3)
    vertical = _strokes(trace_x(k) + snap + line_phase, k % 5 == 0, WIDTH)
    rows = np.arange(0, int(HEIGHT / PERIOD) + 1)
    horizontal = _strokes(rows * PERIOD + 0.5 * PERIOD, rows % 5 == 0, HEIGHT)
    darkness = np.maximum(vertical[None, :], horizontal[:, None])
    if blank is not None:
        darkness[:, int(trace_x(blank[0])) : int(trace_x(blank[1]))] = 0.0
    image = np.full((3, HEIGHT, WIDTH), 255, dtype=np.uint8)
    # Red lines, as on a printed page: darkest in the two lower channels.
    image[1] = image[2] = np.round(255 - darkness).astype(np.uint8)

    mask = np.zeros((1, HEIGHT, WIDTH), dtype=np.uint8)
    fine_mm = np.linspace(-5, SPAN + 5, 20 * int(SPAN + 10) + 1)
    fine_x = trace_x(fine_mm)
    for lead, label in LEAD_LABEL_MAPPING.items():
        # II is drawn as the rhythm strip only, as the label masks of the model have it.
        if lead == RHYTHM_LEAD:
            first_mm, last_mm, ratio = 0.0, SPAN, Y_SHIFT_RATIO["full"]
        else:
            column = int(
                digitize.STANDARD_LEAD_OFFSETS_SEC[lead]
                / digitize.SHORT_SIGNAL_LENGTH_SEC
            )
            first_mm, last_mm, ratio = (
                column * LINES,
                (column + 1) * LINES,
                Y_SHIFT_RATIO[lead],
            )
        first = int(np.ceil(trace_x(first_mm) - 0.5))
        last = int(np.ceil(trace_x(last_mm) - 0.5)) - 1
        mm_at = np.interp(np.arange(first, last + 2, dtype=float), fine_x, fine_mm)
        y = (
            digitize.baseline_row(ratio, HEIGHT)
            - GAIN[lead] * beats(mm_at / 25.0) * 10 * PERIOD
        )
        for index, pixel in enumerate(range(first, last + 1)):
            top = int(np.floor(min(y[index], y[index + 1]) - 0.5))
            bottom = int(np.floor(max(y[index], y[index + 1]) - 0.5)) + 1
            image[:, top : bottom + 1, pixel] = 0
            mask[0, top - 1 : bottom + 2, pixel] = label
        if early and ratio == Y_SHIFT_RATIO["I"]:
            top = int(np.floor(y[0] - 0.5))
            mask[0, top - 1 : top + 3, first - early : first] = label
    return torch.from_numpy(image), torch.from_numpy(mask), trace_x


# name -> the arguments of drawn_page(). What the checks of the uniform grid make of
# each without the flag is pinned in test_the_pages_fail_the_checks_they_are_drawn_for.
PAGES = {
    # Nothing to rescue: an even page, and one the column map reads without help.
    "even": {},
    "squeezed": {"compress": 0.015},
    # A fourth column 2.6 % narrower: its mask edges are off the grid of the others.
    "fit": {"compress": 0.026},
    # The same with masks that start early: the lines are refused as well.
    "fitphase": {"compress": 0.026, "early": 7},
    # ... and with 0.32 column without grid lines, which the map does not bridge.
    "fitgap": {"compress": 0.026, "blank": (1.3 * LINES, 1.62 * LINES)},
    # Masks that start 7 px early move the mask origin off the lines.
    "phase": {"compress": 0.015, "early": 7},
    # Lines half a line off the traces: no line is where the mask edges are.
    "phasebad": {"line_phase": PERIOD / 2},
}
NO_GRID = "a column has all its edges off the grid"
RESCUED = {"fit": "fit", "fitphase": "fit+phase", "phase": "phase"}
REFUSED = {
    "fitgap": r"refused: map has a gap of 0\.\d\d columns",
    "phasebad": r"refused: mask edges are [+-]\d+\.\d px off the next grid line",
}
UNTOUCHED = ("even", "squeezed")


@lru_cache(maxsize=None)
def cut(name, **changes):
    """(image, masks, positions) of a drawn page, as run() hands them to the grid."""
    image, mask, _ = drawn_page(**{**PAGES[name], **changes})
    masks, positions, _ = digitize.cut_binary(mask, image)
    return image, masks, positions


def fit(masks, positions, **kwargs):
    return digitize.fit_column_grid(masks, positions, HEIGHT, "page", "rec", **kwargs)


def same_map(a, b):
    """Two results of measure_column_mapping() hold the same map, bit for bit."""
    return (
        (a[0] is None) == (b[0] is None)
        and all(
            a[1][key] == b[1][key] or (a[1][key] != a[1][key] and b[1][key] != b[1][key])
            for key in (
                "shift",
                "windows",
                "agreement",
                "origin_offset",
                "coverage",
                "gap",
            )
        )
        and all(np.array_equal(a[1][key], b[1][key]) for key in ("knots_m", "knots_x"))
    )


# ------------------------------------------------------------- flag and constants
def test_the_flag_is_off_unless_map_is_asked_for(capsys):
    parse = digitize.get_parser().parse_args
    folders = ["-d", "in", "-o", "out"]
    assert parse(folders).grid_rescue == "off"
    assert parse(folders + ["--grid_rescue", "map"]).grid_rescue == "map"
    with pytest.raises(SystemExit):
        parse(folders + ["--grid_rescue", "lines"])
    capsys.readouterr()


def test_the_second_fit_allows_twice_the_residual_of_the_first():
    assert digitize.GRID_RESCUE_RESIDUAL_TOLERANCE == 0.04
    assert digitize.GRID_RESCUE_RESIDUAL_TOLERANCE == 2 * digitize.GRID_RESIDUAL_TOLERANCE


def test_the_pages_fail_the_checks_they_are_drawn_for():
    state = {}
    for name in PAGES:
        image, masks, positions = cut(name)
        g0, P, _, reason = fit(masks, positions, quiet=True)
        if g0 is None:
            state[name] = f"no grid: {reason}"
            continue
        _, _, info = digitize.refine_grid_from_lines(image, g0, P, snap_offset=0.0)
        state[name] = f"lines refused: {info['reason']}" if info["reason"] else "grid"
    assert state["even"] == state["squeezed"] == "grid"
    assert state["fit"] == state["fitphase"] == state["fitgap"] == f"no grid: {NO_GRID}"
    for name in ("phase", "phasebad"):
        assert state[name].startswith("lines refused: next grid line is "), name


# ------------------------------------------------- the parameters of the two checks
def test_fit_column_grid_takes_another_tolerance():
    _, masks, positions = cut("fit")
    plain = fit(masks, positions)
    assert plain[:2] == (None, None) and plain[3] == NO_GRID
    # The default is GRID_RESIDUAL_TOLERANCE, and a tolerance just above it is no help.
    assert fit(masks, positions, tolerance=digitize.GRID_RESIDUAL_TOLERANCE) == plain
    assert fit(masks, positions, tolerance=0.021) == plain
    g0, P, long_leads, reason = fit(
        masks, positions, tolerance=digitize.GRID_RESCUE_RESIDUAL_TOLERANCE
    )
    assert reason == "" and long_leads == [RHYTHM_LEAD]
    # One grid through all four columns: the last is 12.8 px short, so the fitted
    # pitch is a little under the drawn one and the origin a little right of it.
    assert P_DRAWN - 3.0 < P < P_DRAWN - 1.0
    assert G0_DRAWN < g0 < G0_DRAWN + 4.0
    # A column 4 % narrower: its outer edge is 10.3 px off the grid of all four, 2.1 %
    # of the pitch, which both checks of the fit allow at the second tolerance, the
    # one of the inliers and the last one of the residual, and not at the first.
    _, masks, positions = cut("fit", compress=0.04)
    assert fit(masks, positions)[3] == NO_GRID
    g0, P, _, reason = fit(masks, positions, tolerance=0.04)
    assert reason == ""
    edges = digitize.column_mapping_edges(masks, positions, [RHYTHM_LEAD])
    residual = max(abs(x - (g0 + mm / LINES * P)) for x, mm in edges)
    assert digitize.GRID_RESIDUAL_TOLERANCE * P < residual < 0.04 * P
    # A column 5 % narrower is off the second grid as well.
    _, masks, positions = cut("fit", compress=0.05)
    assert fit(masks, positions, tolerance=0.04)[3] == NO_GRID


def test_fit_column_grid_reads_the_module_tolerance_when_it_is_called(monkeypatch):
    # Not when it is defined: a changed GRID_RESIDUAL_TOLERANCE is the default.
    _, masks, positions = cut("fit")
    relaxed = fit(masks, positions, tolerance=0.04)
    monkeypatch.setattr(digitize, "GRID_RESIDUAL_TOLERANCE", 0.04)
    assert fit(masks, positions) == relaxed


def test_grid_inliers_take_the_tolerance_of_the_fit():
    origin, pitch = 100.0, 500.0
    lead_edges = [(c, origin + c * pitch, True) for c in range(4)]
    lead_edges += [(c, origin + c * pitch, False) for c in range(1, 5)]
    lead_edges.append((4, origin + 4 * pitch - 15, False))
    assert list(digitize._grid_inliers(lead_edges)) == [True] * 8 + [False]
    assert list(digitize._grid_inliers(lead_edges, 0.04)) == [True] * 9


def test_fit_column_grid_quiet_leaves_out_the_pitch_line(capsys):
    _, masks, positions = cut("squeezed")
    loud = fit(masks, positions)
    assert capsys.readouterr().out.count("does not match the fitted pitch") == 1
    quiet = fit(masks, positions, quiet=True)
    assert capsys.readouterr().out == ""
    assert quiet == loud and quiet[3] == ""


def test_refine_grid_from_lines_reports_the_pitch_of_the_lines():
    image, masks, positions = cut("even")
    g0, P, _, _ = fit(masks, positions)
    _, P_lines, info = digitize.refine_grid_from_lines(image, g0, P, snap_offset=0.0)
    assert info["reason"] == "" and info["pitch"] == P_lines
    assert P_lines == pytest.approx(P_DRAWN, abs=0.05)

    # Lines that are refused by their origin still have their pitch.
    image, masks, positions = cut("phase")
    g0, P, _, _ = fit(masks, positions)
    g0_out, P_out, info = digitize.refine_grid_from_lines(image, g0, P, snap_offset=0.0)
    assert info["reason"].startswith("next grid line is ")
    assert (g0_out, P_out) == (g0, P)
    assert np.isfinite(info["pitch"]) and info["pitch"] != P
    # 1.5 % of one column in four: the comb of the whole page is 0.2 px narrower.
    assert info["pitch"] == pytest.approx(P_DRAWN - 0.18, abs=0.1)

    blank = torch.full((3, HEIGHT, WIDTH), 255, dtype=torch.uint8)
    _, _, info = digitize.refine_grid_from_lines(blank, g0, P, snap_offset=0.0)
    assert info["reason"].startswith("no grid lines found")
    assert np.isnan(info["pitch"])


# ---------------------------------------------------- which failures are tried again
def test_the_failures_tried_again_are_those_of_a_distorted_column(monkeypatch):
    again = digitize._fit_failed_on_edges
    _, masks, positions = cut("fit")
    assert again(fit(masks, positions)[3])
    # The other failure of the edges, with the outlier check out of the way.
    _, wide_masks, wide_positions = cut("fit", compress=0.06)
    with monkeypatch.context() as patch:
        patch.setattr(
            digitize,
            "_grid_inliers",
            lambda lead_edges, tolerance=None: np.ones(len(lead_edges), dtype=bool),
        )
        reason = fit(wide_masks, wide_positions)[3]
        # 15.7 px, 3.2 % of the pitch: the second tolerance is the one of this check.
        relaxed = fit(wide_masks, wide_positions, tolerance=0.04)[3]
    assert reason.startswith("column edges are ") and reason.endswith(" px off the grid")
    assert again(reason)
    assert relaxed == ""
    # A page that is not the 3x4 layout with rhythm strip is not tried again.
    no_strip = {lead: mask for lead, mask in masks.items() if lead != RHYTHM_LEAD}
    assert fit(no_strip, positions)[3] == "no rhythm strip found"
    fewer = {
        lead: (None if lead in ("V4", "V5", "V6") else m) for lead, m in masks.items()
    }
    assert fit(fewer, positions)[3] == "not every column has a short lead"
    for reason in (
        "no leads found",
        "no rhythm strip found",
        "rhythm strip does not span all columns",
        "unknown lead X",
        "not every column has a short lead",
        "",
    ):
        assert not again(reason), reason


def test_only_lines_refused_by_their_origin_are_tried_again():
    again = digitize._lines_refused_by_phase
    image, masks, positions = cut("phase")
    g0, P, _, _ = fit(masks, positions)
    assert again(
        digitize.refine_grid_from_lines(image, g0, P, snap_offset=0.0)[2]["reason"]
    )
    blank = torch.full((3, HEIGHT, WIDTH), 255, dtype=torch.uint8)
    none_found = digitize.refine_grid_from_lines(blank, g0, P)[2]["reason"]
    assert none_found.startswith("no grid lines found") and not again(none_found)
    assert not again("image too small for the grid line profile")
    assert not again("")


# ------------------------------------------------------------------ the second try
@pytest.mark.parametrize("snap, median", [(0.0, 13.5), (0.5, 27.0)])
def test_rescue_grid_fit_keeps_a_narrow_column_on_its_map(snap, median, capsys):
    image, masks, positions = cut("fit", snap=snap)
    _, _, trace_x = drawn_page(**{**PAGES["fit"], "snap": snap})
    rescue = digitize.rescue_grid_fit(
        image,
        masks,
        positions,
        "page",
        "rec",
        NO_GRID,
        snap_offset=snap,
        median_mm=median,
    )
    # The second fit prints nothing: its grid is in the line of the rescue.
    assert capsys.readouterr().out == ""
    assert rescue["how"] == "fit" and rescue["refused"] == "" and rescue["was"] == NO_GRID
    # Its tolerance is its own: the first fit of the next page is what it was.
    assert digitize.GRID_RESIDUAL_TOLERANCE == 0.02
    assert fit(masks, positions)[3] == NO_GRID
    assert rescue["long_leads"] == [RHYTHM_LEAD]
    # The grid is the second fit refined by the lines, the map is measured on it.
    g0, P, _, _ = fit(masks, positions, tolerance=0.04, quiet=True)
    g0, P, lines = digitize.refine_grid_from_lines(image, g0, P, snap_offset=snap)
    assert lines["reason"] == ""
    assert (rescue["g0"], rescue["P"]) == (g0, P) and rescue["grid_lines"] == lines
    again = digitize.measure_column_mapping(
        image,
        g0,
        P,
        digitize.column_mapping_edges(masks, positions, [RHYTHM_LEAD]),
        snap_offset=snap,
        median_mm=median,
    )
    assert rescue["x_map"] is not None
    assert same_map((rescue["x_map"], rescue["map_info"]), again)
    # The map is on the drawn traces, the uniform grid of the second fit is not.
    mm = np.arange(0.0, SPAN, 0.05)
    assert np.max(np.abs(rescue["x_map"](mm) - trace_x(mm))) < 0.2
    assert np.max(np.abs(g0 + mm * P / LINES - trace_x(mm))) > 10.0


@pytest.mark.parametrize("median", [13.5, 27.0])
def test_rescue_grid_fit_goes_on_through_lines_refused_by_their_origin(median):
    image, masks, positions = cut("fitphase")
    _, _, trace_x = drawn_page(**PAGES["fitphase"])
    rescue = digitize.rescue_grid_fit(
        image,
        masks,
        positions,
        "page",
        "rec",
        NO_GRID,
        snap_offset=0.0,
        median_mm=median,
    )
    assert rescue["how"] == "fit+phase" and rescue["refused"] == ""
    assert rescue["was"].startswith(f"{NO_GRID}; next grid line is ")
    assert rescue["long_leads"] == [RHYTHM_LEAD]
    g0, P, _, _ = fit(masks, positions, tolerance=0.04, quiet=True)
    _, _, lines = digitize.refine_grid_from_lines(image, g0, P, snap_offset=0.0)
    assert digitize._lines_refused_by_phase(lines["reason"])
    # The pitch of the lines, with the second fit turned about its centre.
    assert rescue["P"] == lines["pitch"]
    assert rescue["g0"] + 2 * rescue["P"] == pytest.approx(g0 + 2 * P, abs=1e-9)
    # The map of that grid, with the snap offset and the median the rescue was given.
    again = digitize.measure_column_mapping(
        image,
        rescue["g0"],
        rescue["P"],
        digitize.column_mapping_edges(masks, positions, [RHYTHM_LEAD]),
        snap_offset=0.0,
        median_mm=median,
    )
    assert same_map((rescue["x_map"], rescue["map_info"]), again)
    mm = np.arange(0.0, SPAN, 0.05)
    assert np.max(np.abs(rescue["x_map"](mm) - trace_x(mm))) < 0.2


@pytest.mark.parametrize(
    "changes, how, refused",
    [
        # 0.32 column without grid lines: the lines refine the second fit, the map
        # does not bridge the gap.
        (
            {"blank": (1.3 * LINES, 1.62 * LINES)},
            "fit",
            r"map has a gap of 0\.\d\d columns",
        ),
        # Lines half a line off the traces: neither their origin nor the map's stands.
        (
            {"line_phase": PERIOD / 2},
            "fit+phase",
            r"mask edges are [+-]\d\.\d px off the next grid line",
        ),
        # A column 5 % narrower is off the second fit as well.
        ({"compress": 0.05}, "fit", f"second fit: {NO_GRID}"),
    ],
)
def test_rescue_grid_fit_refuses_a_page_without_a_map(changes, how, refused):
    image, masks, positions = cut("fit", **changes)
    rescue = digitize.rescue_grid_fit(
        image, masks, positions, "page", "rec", NO_GRID, snap_offset=0.0
    )
    assert rescue["how"] == how
    assert re.fullmatch(refused, rescue["refused"]), rescue["refused"]
    assert rescue.get("x_map") is None


def test_rescue_grid_fit_refuses_a_page_without_grid_lines():
    _, masks, positions = cut("fit")
    blank = torch.full((3, HEIGHT, WIDTH), 255, dtype=torch.uint8)
    rescue = digitize.rescue_grid_fit(blank, masks, positions, "page", "rec", NO_GRID)
    assert rescue["how"] == "fit"
    assert rescue["refused"].startswith("grid lines: no grid lines found (contrast ")


@pytest.mark.parametrize("snap, median", [(0.0, 13.5), (0.5, 27.0)])
def test_rescue_grid_phase_reads_the_page_on_the_pitch_of_the_lines(snap, median):
    image, masks, positions = cut("phase", snap=snap)
    _, _, trace_x = drawn_page(**{**PAGES["phase"], "snap": snap})
    g0, P, long_leads, _ = fit(masks, positions)
    _, _, lines = digitize.refine_grid_from_lines(image, g0, P, snap_offset=snap)
    before = dict(lines)
    edges = digitize.column_mapping_edges(masks, positions, long_leads)
    rescue = digitize.rescue_grid_phase(
        image, g0, P, lines, edges, snap_offset=snap, median_mm=median
    )
    assert rescue["how"] == "phase" and rescue["refused"] == ""
    assert rescue["was"] == lines["reason"]
    # The pitch the lines have, the mask grid turned about its centre, and none of
    # the origin that was refused.
    assert rescue["P"] == lines["pitch"] != P
    assert rescue["g0"] + 2 * rescue["P"] == pytest.approx(g0 + 2 * P, abs=1e-9)
    # The info of the lines as measured, without its reason; the refusal is not edited.
    assert lines == before
    assert rescue["grid_lines"] == {**lines, "reason": ""}
    again = digitize.measure_column_mapping(
        image, rescue["g0"], rescue["P"], edges, snap_offset=snap, median_mm=median
    )
    assert rescue["x_map"] is not None
    assert same_map((rescue["x_map"], rescue["map_info"]), again)
    mm = np.arange(0.0, SPAN, 0.05)
    assert np.max(np.abs(rescue["x_map"](mm) - trace_x(mm))) < 0.2
    # The mask grid, which the page keeps without the flag, starts 3 px early.
    assert np.max(np.abs(g0 + mm * P / LINES - trace_x(mm))) > 3.0


def test_rescue_grid_phase_refuses_lines_that_are_off_the_traces():
    image, masks, positions = cut("phasebad")
    g0, P, long_leads, _ = fit(masks, positions)
    _, _, lines = digitize.refine_grid_from_lines(image, g0, P, snap_offset=0.0)
    rescue = digitize.rescue_grid_phase(
        image,
        g0,
        P,
        lines,
        digitize.column_mapping_edges(masks, positions, long_leads),
        snap_offset=0.0,
    )
    assert rescue["how"] == "phase" and rescue["x_map"] is None
    assert rescue["refused"].startswith("mask edges are ")


def test_a_map_inside_the_dead_band_is_no_rescue():
    assert digitize._map_refusal(lambda m: m, {"reason": ""}) == ""
    assert digitize._map_refusal(None, {"reason": "", "dead_band": True}) == "dead band"
    assert digitize._map_refusal(None, {"reason": "no mask edges"}) == "no mask edges"


def test_grid_rescue_line_names_the_record_the_check_and_the_map():
    image, masks, positions = cut("fit")
    rescue = digitize.rescue_grid_fit(
        image, masks, positions, "page", "rec", NO_GRID, snap_offset=0.0
    )
    line = digitize.grid_rescue_line("rec-0_0000", rescue)
    assert line.startswith(
        "Grid rescue for record rec-0_0000: fit, read on the column map"
    )
    info = rescue["map_info"]
    for number in (
        f"g0 {rescue['g0']:.2f} px",
        f"P {rescue['P']:.2f} px",
        f"shift {info['shift']:.2f} px",
        f"windows {info['windows']}",
        f"agreement {info['agreement']:.3f}",
        f"origin offset {info['origin_offset']:+.2f} px",
        f"was: {NO_GRID}",
    ):
        assert number in line, number
    assert "WARNING" not in line and "\n" not in line

    image, masks, positions = cut("fitgap")
    refused = digitize.rescue_grid_fit(
        image, masks, positions, "page", "rec", NO_GRID, snap_offset=0.0
    )
    line = digitize.grid_rescue_line("rec", refused)
    assert line.startswith("Grid rescue for record rec: refused (map has a gap of ")
    assert "page left as it is" in line and f"was: {NO_GRID}" in line
    assert "WARNING" not in line


# ------------------------------------------------------------------- end to end
FLAGS = (
    "--rotation", "hough", "--perspective", "off", "--resolution", "keep",
    "--grid_line_offset", "0", "--paper_normalisation", "off", "-f",
)


# The pages of the runs that only compare a flag set with another one: a page for each
# of the two checks that are tried again, or the one whose fit fails.
FEW = ("fit", "phase")
ONE = ("fit",)


@pytest.fixture(scope="module")
def runs(tmp_path_factory):
    """run() on drawn pages with the flags given, each set of pages and flags once.

    Returns a function: (*flags, pages=every page) -> (output folder, what run()
    printed).
    """
    root = tmp_path_factory.mktemp("grid_rescue")
    folders, done = {}, {}

    def run(*flags, pages=tuple(PAGES)):
        if pages not in folders:
            data, masks = root / f"data{len(folders)}", root / f"masks{len(folders)}"
            data.mkdir()
            masks.mkdir()
            for name in pages:
                image, mask, _ = drawn_page(**PAGES[name])
                write_png(image, str(data / f"{name}.png"))
                write_png(mask, str(masks / f"{name}_mask.png"))
            folders[pages] = (data, masks)
        if (pages, flags) not in done:
            data, masks = folders[pages]
            out = root / f"out{len(done)}"
            argv = ["-d", str(data), "-o", str(out), "--mask_folder", str(masks)]
            printed = io.StringIO()
            with contextlib.redirect_stdout(printed):
                digitize.run(digitize.get_parser().parse_args(argv + [*FLAGS, *flags]))
            done[pages, flags] = (out, printed.getvalue())
        return done[pages, flags]

    return run


def signals(folder, name):
    with open(folder / f"{name}.dat", "rb") as dat, open(
        folder / f"{name}.hea", "rb"
    ) as hea:
        return dat.read(), hea.read()


def qc_rows(folder):
    with open(folder / "qc.csv", newline="") as f:
        return {row["record"]: row for row in csv.DictReader(f)}


def lines_of(printed, name, start=""):
    """The lines run() printed that name the record, optionally those starting so."""
    names_it = re.compile(rf"record {re.escape(name)}(?!\w)")
    return [
        line
        for line in printed.splitlines()
        if names_it.search(line) and line.startswith(start)
    ]


def drawn_snr(folder, name):
    """SNR in dB of every lead of a digitised page against the signal it was drawn from
    (sample n is 1/20 mm at 25 mm/s and 500 Hz; means removed)."""
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        record = wfdb.rdrecord(str(folder / name))
    snr = {}
    for index, lead in enumerate(record.sig_name):
        read = record.p_signal[:, index]
        drawn = GAIN[lead] * beats(np.arange(read.size) / FREQUENCY)
        valid = np.isfinite(read)
        drawn = drawn[valid] - drawn[valid].mean()
        error = (read[valid] - read[valid].mean()) - drawn
        snr[lead] = float(10 * np.log10(np.sum(drawn ** 2) / np.sum(error ** 2)))
    return snr


def test_run_reads_every_drawn_page_straight(runs):
    out, _ = runs()
    assert all(float(row["rotation_angle"]) == 0.0 for row in qc_rows(out).values())


def test_run_without_the_flag_writes_off_and_falls_back_as_before(runs):
    out, printed = runs()
    rows = qc_rows(out)
    assert sorted(rows) == sorted(PAGES)
    assert all(row["grid_rescue"] == "off" for row in rows.values())
    assert "Grid rescue" not in printed
    for name in ("fit", "fitphase", "fitgap"):
        assert rows[name]["column_mapping"] == "uniform: no column grid"
        assert lines_of(printed, name, "WARNING: no column grid") == [
            f"WARNING: no column grid for record {name} ({NO_GRID}), "
            "falling back to --time_mapping bbox."
        ]
    for name in ("phase", "phasebad"):
        assert rows[name]["column_mapping"] == "uniform: grid lines not used"
        assert (
            len(lines_of(printed, name, "WARNING: grid lines not used for record")) == 1
        )
    # --grid_rescue off is the run without the flag.
    out, printed = runs(pages=ONE)
    off, printed_off = runs("--grid_rescue", "off", pages=ONE)
    assert printed_off == printed
    assert all(signals(off, name) == signals(out, name) for name in ONE)
    assert qc_rows(off) == qc_rows(out)


def test_run_rescues_the_pages_whose_map_stands(runs):
    plain, _ = runs()
    out, printed = runs("--grid_rescue", "map")
    rows, rows_plain = qc_rows(out), qc_rows(plain)
    for name, how in RESCUED.items():
        assert rows[name]["grid_rescue"] == how, name
        assert rows[name]["column_mapping"] == "lines", name
        assert signals(out, name) != signals(plain, name), name
        # One plain line, and none of the WARNING lines of the check it failed.
        rescue_lines = lines_of(printed, name, "Grid rescue for record")
        assert len(rescue_lines) == 1, name
        assert rescue_lines[0].startswith(
            f"Grid rescue for record {name}: {how}, read on"
        )
        assert (
            f"shift {float(rows[name]['column_mapping_shift_px']):.2f} px"
            in rescue_lines[0]
        )
        assert lines_of(printed, name, "WARNING: no column grid") == []
        assert lines_of(printed, name, "WARNING: grid lines not used") == []
        assert lines_of(printed, name, "WARNING: column mapping not used") == []
        # The contrast and the shift of the lines are in the QC row as measured.
        assert np.isfinite(float(rows[name]["grid_line_contrast"]))
        assert np.isfinite(float(rows[name]["grid_line_shift_px"]))
    assert (
        rows_plain["phase"]["grid_line_shift_px"] == rows["phase"]["grid_line_shift_px"]
    )


def test_run_reads_a_rescued_page_as_it_was_drawn(runs):
    plain, _ = runs()
    out, _ = runs("--grid_rescue", "map")
    even = drawn_snr(plain, "even")
    for name in RESCUED:
        before, after = drawn_snr(plain, name), drawn_snr(out, name)
        # On the map every lead is read as on the even page (32.7 dB in the median),
        # within 5 dB of it; aVL is drawn 0.1 mV high and reads 14 dB on any page.
        assert np.median(list(after.values())) > 30.0, (name, after)
        assert all(after[lead] > even[lead] - 5.0 for lead in after), (name, after)
        # Without it the page is read on bbox or on the grid of the masks: the rhythm
        # strip is lost (0 dB) and the median lead is 7 dB or more behind.
        assert before[RHYTHM_LEAD] < 5.0 and after[RHYTHM_LEAD] > 30.0, (name, before)
        gain = np.median(list(after.values())) - np.median(list(before.values()))
        assert gain > 5.0, (name, gain)


def test_run_leaves_a_refused_page_exactly_as_without_the_flag(runs):
    plain, printed_plain = runs()
    out, printed = runs("--grid_rescue", "map")
    rows, rows_plain = qc_rows(out), qc_rows(plain)
    for name, pattern in REFUSED.items():
        assert signals(out, name) == signals(plain, name), name
        assert re.fullmatch(pattern, rows[name]["grid_rescue"]), rows[name]["grid_rescue"]
        # The QC row but for that column, and every WARNING line, are those without it.
        ours = {key: value for key, value in rows[name].items() if key != "grid_rescue"}
        theirs = {k: v for k, v in rows_plain[name].items() if k != "grid_rescue"}
        assert ours == theirs, name
        assert lines_of(printed, name, "WARNING") == lines_of(
            printed_plain, name, "WARNING"
        )
        assert len(lines_of(printed, name, "WARNING")) >= 1, name
        # --verbose says why the second try was refused, after the WARNING.
        refused_lines = lines_of(printed, name, "Grid rescue for record")
        assert len(refused_lines) == 1 and ": refused (" in refused_lines[0], name
        assert rows[name]["grid_rescue"][len("refused: ") :] in refused_lines[0]


def test_run_leaves_a_page_without_a_failed_check_alone(runs):
    plain, printed_plain = runs()
    out, printed = runs("--grid_rescue", "map")
    rows, rows_plain = qc_rows(out), qc_rows(plain)
    for name in UNTOUCHED:
        assert signals(out, name) == signals(plain, name), name
        assert rows[name]["grid_rescue"] == "", name
        assert {**rows[name], "grid_rescue": "off"} == rows_plain[name], name
        assert lines_of(printed, name) == lines_of(printed_plain, name), name
    assert rows["squeezed"]["column_mapping"] == "lines"
    assert rows["even"]["column_mapping"] == "uniform: dead band"


def test_run_says_a_rescue_without_verbose_and_a_refusal_only_with_it(runs):
    pages = ("fit", "phase", "fitgap", "phasebad")
    loud, printed_loud = runs("--grid_rescue", "map")
    out, printed = runs("--grid_rescue", "map", "--no-verbose", pages=pages)
    assert all(signals(out, name) == signals(loud, name) for name in pages)
    rows_loud = qc_rows(loud)
    assert qc_rows(out) == {name: rows_loud[name] for name in pages}
    said = [line for line in printed.splitlines() if line.startswith("Grid rescue")]
    assert sorted(said) == sorted(
        lines_of(printed_loud, name, "Grid rescue for record")[0]
        for name in ("fit", "phase")
    )
    # The refused pages keep their WARNING, and nothing else is said of them.
    for name in ("fitgap", "phasebad"):
        assert lines_of(printed, name, "WARNING") == lines_of(
            printed_loud, name, "WARNING"
        )
        assert [line for line in lines_of(printed, name) if "WARNING" not in line] == []


def test_run_measures_the_map_of_a_rescued_page_once(tmp_path, monkeypatch):
    # The map the rescue was decided on is the map the page is read on: neither the
    # grid lines nor the map are measured a second time. And the second try is that of
    # the run: its page, its snap offset and its --column_mapping_median.
    data, masks = tmp_path / "data", tmp_path / "masks"
    data.mkdir()
    masks.mkdir()
    for name in ("fit", "phase", "squeezed"):
        image, mask, _ = drawn_page(**PAGES[name])
        write_png(image, str(data / f"{name}.png"))
        write_png(mask, str(masks / f"{name}_mask.png"))
    calls = {
        "fit_column_grid": [],
        "refine_grid_from_lines": [],
        "measure_column_mapping": [],
    }
    for function in calls:
        original = getattr(digitize, function)

        def recorded(*args, _name=function, _original=original, **kwargs):
            given = inspect.signature(_original).bind(*args, **kwargs).arguments
            calls[_name].append(dict(given))
            return _original(*args, **kwargs)

        monkeypatch.setattr(digitize, function, recorded)
    argv = ["-d", str(data), "-o", str(tmp_path / "out"), "--mask_folder", str(masks)]
    argv += [*FLAGS, "--grid_rescue", "map", "--no-verbose"]
    argv += ["--column_mapping_median", "27"]
    with contextlib.redirect_stdout(io.StringIO()):
        digitize.run(digitize.get_parser().parse_args(argv))
    assert [
        row["grid_rescue"] for _, row in sorted(qc_rows(tmp_path / "out").items())
    ] == ["fit", "phase", ""]
    assert len(calls["measure_column_mapping"]) == 3
    assert len(calls["refine_grid_from_lines"]) == 3
    # Every map, those of the two rescues among them, with the flags of the run.
    assert all(
        (given["snap_offset"], given["median_mm"]) == (0.0, 27.0)
        for given in calls["measure_column_mapping"]
    )
    assert all(given["snap_offset"] == 0.0 for given in calls["refine_grid_from_lines"])
    # One fit per page, and a second one for the page that failed it: of the same
    # page height, with the second tolerance, and silent.
    assert len(calls["fit_column_grid"]) == 4
    assert all(given["image_height"] == HEIGHT for given in calls["fit_column_grid"])
    second = [given for given in calls["fit_column_grid"] if "tolerance" in given]
    assert len(second) == 1
    assert (second[0]["record"], second[0]["pitch"]) == ("fit", "page")
    assert second[0]["tolerance"] == digitize.GRID_RESCUE_RESIDUAL_TOLERANCE
    assert second[0]["quiet"] is True


@pytest.mark.parametrize(
    "others, pages",
    [
        # Without the column map both checks are left as they are; without the origin
        # from the grid lines only the fit is there to fail, and without the grid none.
        (("--column_mapping", "uniform"), FEW),
        (("--grid_origin", "masks"), ONE),
        (("--time_mapping", "bbox"), ONE),
    ],
)
def test_run_needs_the_flags_of_the_column_map(runs, others, pages):
    plain, printed_plain = runs(*others, pages=pages)
    out, printed = runs(*others, "--grid_rescue", "map", pages=pages)
    assert all(signals(out, name) == signals(plain, name) for name in pages)
    assert qc_rows(out) == qc_rows(plain)
    assert all(row["grid_rescue"] == "off" for row in qc_rows(out).values())
    note = [line for line in printed.splitlines() if line.startswith("Grid rescue")]
    assert note == [
        "Grid rescue is off for this run: --grid_rescue map needs --time_mapping "
        "grid, --grid_origin lines and --column_mapping lines."
    ]
    assert printed.replace(note[0] + "\n", "") == printed_plain


# ----------------------------------------------------------------------------- QC
def test_grid_rescue_qc_is_the_column_of_a_run_with_the_flag():
    assert digitize.grid_rescue_qc(None) == ""
    assert digitize.grid_rescue_qc({"how": "fit+phase", "refused": ""}) == "fit+phase"
    refused = {"how": "phase", "refused": "dead band"}
    assert digitize.grid_rescue_qc(refused) == "refused: dead band"


def test_append_qc_row_writes_the_two_columns_after_all_others(tmp_path):
    qc = {
        "column_mapping": "lines",
        "grid_rescue": "fit+phase",
        "row_mapping": "lines 3/4",
    }
    digitize.append_qc_row(str(tmp_path), "rec", "column", qc, 0.0)
    with open(tmp_path / "qc.csv", newline="") as f:
        reader = csv.DictReader(f)
        row = next(reader)
        # Appended at the end: no column that was there moves.
        assert reader.fieldnames[-4:] == [
            "column_mapping",
            "column_mapping_shift_px",
            "grid_rescue",
            "row_mapping",
        ]
        assert len(reader.fieldnames) == 34
        assert reader.fieldnames.index("paper_normalisation") == 12
        assert reader.fieldnames.index("column_mapping_shift_px") == 31
    assert row["grid_rescue"] == "fit+phase"
    assert row["row_mapping"] == "lines 3/4"
