"""Unit tests for --sheet_curl in src/run/digitize.py and for src/run/sheet_curl.py.

The stage sits between the perspective stage and the segmentation: it measures a
smooth map from the pixels of a page to the coordinates of its grid lines and warps
the page, and a mask that was predicted on it, into the frame in which the lines are
straight. It is off unless --sheet_curl lines asks for it.
No model and no data files: the pages are drawn here, a quarter of a page with a
1 mm / 5 mm grid at 200 dpi and no trace, flat or as a sheet shows it that is bent by
a known map, half a sine across each side, which no homography describes. The label
masks are drawn through the same map, so that a trace that is straight on the sheet
is bent on the page as the sheet is.
"""
import argparse
import ast
import contextlib
import csv
import inspect
import io
import json
import math
import os
import re
import shutil
from functools import lru_cache
from types import SimpleNamespace

import cv2
import numpy as np
import pytest
import torch
import wfdb
from torchvision.io.image import read_image, write_png

from config import LEAD_LABEL_MAPPING, SHORT_SIGNAL_LENGTH_SEC
from src.run import digitize, paper_normalisation, sheet_curl

# ------------------------------------------------------------------ the drawn page
# A quarter of a page with the grid of a whole one: at the 200 dpi scale, and cheap.
WIDTH, HEIGHT = 1100, 850
PERIOD = 200 / 25.4  # px per printed millimetre at 200 dpi
FOLDERS = ["-d", "in", "-o", "out"]
# name -> (px the sheet is bent by, whether it has grid lines). What the stage makes
# of each is pinned in the tests of the estimator.
PAGES = {
    # Flat: the map moves it by less than a hundredth of a pixel.
    "flat": (0.0, True),
    # Bent by 12 px: the map moves the page by 7.6 px, inside a dead band of 10 px.
    "slight": (12.0, True),
    # Bent by 20 px: the map moves the page by 12.6 px, above a dead band of 10 px and
    # inside the default one of 15 px.
    "bent": (20.0, True),
    # No grid at all: nothing to measure.
    "blank": (0.0, False),
}


def bend(x, y, amplitude):
    """Where on the flat sheet the point (x, y) of the page lies: the known map.

    Half a sine across each side, as a sheet that bulges: not a polynomial, and far
    from any homography. Its largest displacement is amplitude px, at the centre.
    """
    return (
        x + 0.6 * amplitude * np.sin(np.pi * y / HEIGHT),
        y + 0.8 * amplitude * np.sin(np.pi * x / WIDTH),
    )


@lru_cache(maxsize=None)
def drawn_page(amplitude=0.0, lines=True):
    """A drawn page [3, H, W] uint8: a red 1 mm grid, every fifth line bold, no trace.

    The page shows at (x, y) what the flat sheet has at bend(x, y, amplitude), drawn
    there and not resampled. Without lines the page is blank. Cached, so no caller may
    write into a page.
    """
    y, x = np.mgrid[0:HEIGHT, 0:WIDTH].astype(np.float64)
    on_sheet_x, on_sheet_y = bend(x + 0.5, y + 0.5, amplitude)

    def family(lines):
        thin = np.abs(lines - np.round(lines)) * PERIOD
        bold = np.abs(lines / 5 - np.round(lines / 5)) * 5 * PERIOD
        return 45 * np.exp(-0.5 * (thin / 0.6) ** 2) + 55 * np.exp(
            -0.5 * (bold / 0.8) ** 2
        )

    darkness = family(on_sheet_x / PERIOD) + family(on_sheet_y / PERIOD)
    darkness = np.clip(darkness, 0, 200) * bool(lines)
    image = np.full((3, HEIGHT, WIDTH), 255, dtype=np.uint8)
    # Red lines, as on a printed page: darkest in the two lower channels.
    image[1] = image[2] = (255 - darkness).astype(np.uint8)
    return torch.from_numpy(image)


@lru_cache(maxsize=None)
def label_mask(amplitude=0.0):
    """A flat trace per lead in the column of the standard layout, [1, H, W] uint8.

    Three pixels thick and straight ON THE SHEET: on a page that is bent by amplitude
    the trace is bent with it, as a segmentation of that page would find it.
    """
    label = np.zeros((1, HEIGHT, WIDTH), dtype=np.uint8)
    pitch = WIDTH / 5
    y, x = np.mgrid[0:HEIGHT, 0:WIDTH].astype(np.float64)
    on_sheet_x, on_sheet_y = bend(x + 0.5, y + 0.5, amplitude)
    for lead, value in LEAD_LABEL_MAPPING.items():
        column = int(digitize.STANDARD_LEAD_OFFSETS_SEC[lead] / SHORT_SIGNAL_LENGTH_SEC)
        row = HEIGHT // 6 + HEIGHT // 17 * (value % 12)
        start = pitch / 2 + column * pitch
        trace = np.abs(on_sheet_y - (row + 0.5)) <= 1.5
        label[0][trace & (on_sheet_x >= start) & (on_sheet_x < start + pitch)] = value
    return torch.from_numpy(label)


def page(name):
    return drawn_page(*PAGES[name])


def mask_of(name):
    return label_mask(PAGES[name][0])


def helpers():
    return digitize.sheet_curl_helpers()


def measured(name, **options):
    """measure_field() on a drawn page with these options: (field, info)."""
    return sheet_curl.measure_field(page(name), options, helpers())


def rows_spanned(mask):
    """How many rows the trace of each lead spans in a label mask [1, H, W]."""
    label = mask[0].numpy()
    spans = []
    for value in LEAD_LABEL_MAPPING.values():
        rows = np.flatnonzero((label == value).any(axis=1))
        spans.append(int(rows[-1] - rows[0] + 1))
    return spans


def map_error(field, amplitude):
    """(RMS, max) in px of a field against the known map, in the gauge of the field.

    The grid lines give the map up to a shift and one scale, which the estimator fixes
    by the smallest displacement: the truth gets the shift and the scale that put it
    nearest to the field, and what is left is the error.
    """
    x, y = np.meshgrid(np.arange(10.0, WIDTH, 20), np.arange(10.0, HEIGHT, 20))
    X, Y = field.straighten(x, y)
    true_x, true_y = bend(x, y, amplitude)
    scale, shift_x, shift_y = sheet_curl._least_squares_gauge(true_x, true_y, X, Y)
    error = np.hypot(shift_x + scale * true_x - X, shift_y + scale * true_y - Y)
    return float(np.sqrt(np.mean(error**2))), float(error.max())


def same_number(a, b):
    """Two numbers agree to nine digits, NaN being NaN: a fit is not bit for bit the
    same in two calls on every machine."""
    return (a != a and b != b) or a == pytest.approx(b, rel=1e-9, abs=1e-9)


def same_info(a, b):
    """Two info dicts hold the same keys and values, but for the time they took."""
    return a.keys() == b.keys() and all(
        same_number(a[key], b[key]) if isinstance(a[key], float) else a[key] == b[key]
        for key in a
        if key != "seconds"
    )


def lattice_field(dx=0.0, dy=0.0):
    """A field that moves every point of a page by (dx, dy), every node measured."""
    step = sheet_curl.LATTICE
    shape = (math.ceil(HEIGHT / step) + 1, math.ceil(WIDTH / step) + 1)
    measured_everywhere = np.ones(shape, bool)
    return sheet_curl.Field(
        WIDTH, HEIGHT, step, np.full(shape, dx), np.full(shape, dy), measured_everywhere
    )


def recorded(monkeypatch, module, name):
    """Record every call of module.<name> and pass it on: [(args, kwargs), ...]."""
    original, calls = getattr(module, name), []

    def recorder(*args, **kwargs):
        calls.append((args, kwargs))
        return original(*args, **kwargs)

    monkeypatch.setattr(module, name, recorder)
    return calls


# ------------------------------------------------------------ flags and constants
def test_the_stage_is_off_unless_it_is_asked_for():
    parse = digitize.get_parser().parse_args
    default = parse(FOLDERS)
    assert default.sheet_curl == "off"
    # The defaults of the four options are the constants of the module.
    assert default.sheet_curl_tolerance == sheet_curl.TOLERANCE == 0.25
    assert default.sheet_curl_passes == sheet_curl.PASSES == 3
    assert default.sheet_curl_min_shift == sheet_curl.MIN_SHIFT_PX == 15.0
    assert default.sheet_curl_scope == sheet_curl.SCOPE == "all"
    # The dead band of the stage is not the threshold of the estimator.
    assert sheet_curl.ESTIMATOR_MIN_SHIFT_PX == 1.0 < sheet_curl.MIN_SHIFT_PX
    assert isinstance(default.sheet_curl_tolerance, float)
    assert isinstance(default.sheet_curl_passes, int)
    assert isinstance(default.sheet_curl_min_shift, float)
    written_out = [
        "--sheet_curl", "off", "--sheet_curl_tolerance", "0.25", "--sheet_curl_passes",
        "3", "--sheet_curl_min_shift", "15", "--sheet_curl_scope", "all",
    ]
    assert parse(FOLDERS + written_out) == default
    # lines is another run, in that argument and in no other.
    lines = vars(parse(FOLDERS + ["--sheet_curl", "lines"]))
    assert {key for key in lines if lines[key] != vars(default)[key]} == {"sheet_curl"}
    asked = parse(
        FOLDERS
        + ["--sheet_curl", "lines", "--sheet_curl_tolerance", "0.1"]
        + ["--sheet_curl_passes", "2", "--sheet_curl_min_shift", "1.5"]
        + ["--sheet_curl_scope", "normalised"]
    )
    assert (asked.sheet_curl, asked.sheet_curl_scope) == ("lines", "normalised")
    assert asked.sheet_curl_tolerance == 0.1 and asked.sheet_curl_passes == 2
    assert asked.sheet_curl_min_shift == 1.5


def test_both_ends_of_the_ranges_are_in_them():
    parse = digitize.get_parser().parse_args
    assert sheet_curl.MAX_TOLERANCE == 0.5 and sheet_curl.MAX_PASSES == 5
    half = parse(FOLDERS + ["--sheet_curl_tolerance", "0.5"]).sheet_curl_tolerance
    assert half == 0.5
    small = parse(FOLDERS + ["--sheet_curl_tolerance", "1e-6"]).sheet_curl_tolerance
    assert small == 1e-6
    for passes in (1, 5):
        given = parse(FOLDERS + ["--sheet_curl_passes", str(passes)])
        assert given.sheet_curl_passes == passes
    assert parse(FOLDERS + ["--sheet_curl_min_shift", "0"]).sheet_curl_min_shift == 0.0
    large = parse(FOLDERS + ["--sheet_curl_min_shift", "1e6"]).sheet_curl_min_shift
    assert large == 1e6


@pytest.mark.parametrize(
    "text", ["0", "-0.1", "0.5001", "1", "nan", "inf", "-inf", "wide", ""]
)
def test_a_tolerance_outside_half_a_period_is_refused(text, capsys):
    # Refused when the arguments are read, before any page is.
    flag = "--sheet_curl_tolerance"
    with pytest.raises(SystemExit):
        digitize.get_parser().parse_args(FOLDERS + [f"{flag}={text}"])
    error = capsys.readouterr().err.rstrip()
    assert error.endswith(
        f"argument {flag}: {text!r} is not a share of the grid line period above 0 "
        f"and up to 0.5"
    )
    with pytest.raises(argparse.ArgumentTypeError, match="above 0 and up to 0.5"):
        digitize.parse_sheet_curl_tolerance(text)


@pytest.mark.parametrize("text", ["0", "6", "-1", "2.5", "nan", "two", ""])
def test_passes_are_a_whole_number_from_1_to_5(text, capsys):
    flag = "--sheet_curl_passes"
    with pytest.raises(SystemExit):
        digitize.get_parser().parse_args(FOLDERS + [f"{flag}={text}"])
    error = capsys.readouterr().err.rstrip()
    assert error.endswith(
        f"argument {flag}: {text!r} is not a whole number from 1 to 5"
    )
    with pytest.raises(argparse.ArgumentTypeError, match="whole number from 1 to 5"):
        digitize.parse_sheet_curl_passes(text)


@pytest.mark.parametrize("text", ["-1", "-0.001", "nan", "inf", "far", ""])
def test_the_dead_band_is_a_number_of_pixels_of_0_or_more(text, capsys):
    flag = "--sheet_curl_min_shift"
    with pytest.raises(SystemExit):
        digitize.get_parser().parse_args(FOLDERS + [f"{flag}={text}"])
    error = capsys.readouterr().err.rstrip()
    assert error.endswith(
        f"argument {flag}: {text!r} is not a number of pixels of 0 or more"
    )
    with pytest.raises(argparse.ArgumentTypeError, match="pixels of 0 or more"):
        digitize.parse_sheet_curl_min_shift(text)


@pytest.mark.parametrize(
    "flag, text", [("--sheet_curl", "on"), ("--sheet_curl_scope", "photo")]
)
def test_the_stage_and_its_scope_take_their_two_words_only(flag, text, capsys):
    with pytest.raises(SystemExit):
        digitize.get_parser().parse_args(FOLDERS + [flag, text])
    assert f"argument {flag}: invalid choice: {text!r}" in capsys.readouterr().err
    assert sheet_curl.SCOPES == ("normalised", "all")


def test_the_other_options_of_the_estimator_are_constants_of_the_module():
    # What the driver the stage was developed with had as options, at its defaults.
    assert sheet_curl.DEFAULTS == {
        "degree": 4,
        "tol": 0.25,
        # The threshold of the estimator, the driver's: not the dead band of the stage.
        "min_shift": 1.0,
        "max_shift": None,
        "lines": 16,
        "band": 50,
        "overlap": 4,
        "blocks": 4,
        "passes": 3,
        "cover": 0.5,
    }
    assert sheet_curl.DEGREE == 4 and sheet_curl.WINDOW_LINES == 16
    assert sheet_curl.BAND_ROWS == 50 and sheet_curl.WINDOW_OVERLAP == 4
    assert sheet_curl.START_BLOCKS == 4 and sheet_curl.MIN_COVER == 0.5
    assert sheet_curl.MAX_SHIFT_PX is None
    assert sheet_curl.LATTICE == 8 and sheet_curl.MIN_CONTRAST == 2.0
    assert sheet_curl.JACOBIAN_FLOOR == 0.5
    assert (sheet_curl.AFTER_SHARE, sheet_curl.AFTER_MARGIN) == (0.25, 1.15)
    assert sheet_curl.AFTER_SLACK == 0.002
    # Every default unless another value is given, and no key that is no option.
    assert sheet_curl._resolved(None) == sheet_curl.DEFAULTS
    assert sheet_curl._resolved({"tol": 0.1})["tol"] == 0.1
    assert sheet_curl._resolved({"tol": 0.1})["passes"] == 3
    with pytest.raises(ValueError, match="mask_weight: no option of the sheet curl"):
        sheet_curl._resolved({"mask_weight": 0.15})


def test_the_module_does_not_import_the_digitiser():
    tree = ast.parse(inspect.getsource(sheet_curl))
    imported = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            imported.update(alias.name.split(".")[0] for alias in node.names)
        elif isinstance(node, ast.ImportFrom):
            imported.add(node.module.split(".")[0])
    assert imported == {"cv2", "math", "numpy", "scipy", "time", "torch"}
    # What it takes from the digitiser it names, and every name it uses is named.
    used = {
        node.attr
        for node in ast.walk(tree)
        if isinstance(node, ast.Attribute)
        and isinstance(node.value, ast.Name)
        and node.value.id == "D"
    }
    assert used == set(sheet_curl.HELPERS) and len(sheet_curl.HELPERS) == 17


def test_the_helpers_are_those_of_the_digitiser_at_the_time_of_the_call(monkeypatch):
    given = helpers()
    assert sorted(vars(given)) == sorted(sheet_curl.HELPERS)
    for name in sheet_curl.HELPERS:
        assert getattr(given, name) is getattr(digitize, name), name
    assert given.PERSPECTIVE_MAX_SHIFT == 0.05 and given.PERSPECTIVE_MIN_WINDOWS == 24

    # A function that is replaced in the digitiser for the time of a run is the one
    # the stage calls, as a driver or a test replaces it.
    def stand_in(image):
        return digitize._ink(image)

    monkeypatch.setattr(digitize, "_darkness", stand_in)
    assert helpers()._darkness is stand_in and given._darkness is not stand_in


# ------------------------------------------------------------------ the estimator
def test_the_windows_are_those_of_the_perspective_stage_at_its_own_length():
    # The windows are derived from _grid_phase_field(), which has their length, step
    # and band fixed: at those the amplitudes and the centres are the same numbers.
    darkness = digitize._darkness(page("flat"))
    period = digitize._grid_line_period(digitize._band_profiles(darkness)[0])
    for axis in (0, 1):
        amplitudes, x, y = digitize._grid_phase_field(darkness, period, axis)
        profiles, centres = digitize._band_profiles(
            darkness.T if axis == 1 else darkness, digitize.GRID_LINE_BAND_HEIGHT
        )
        starts, length, comb = sheet_curl._window_combs(
            profiles, period, digitize.PERSPECTIVE_WINDOW_LINES, 2, False
        )
        along, across = np.meshgrid(starts + length / 2, centres)
        mine = (along, across) if axis == 0 else (across, along)
        assert np.array_equal(comb(period), amplitudes)
        assert np.array_equal(mine[0], x) and np.array_equal(mine[1], y)
    # With an edge window the last one ends on the last pixel.
    starts, length, _ = sheet_curl._window_combs(profiles, period, 16, 4, True)
    assert starts[-1] + length == profiles.shape[1] and length == round(16 * period)
    assert sheet_curl._window_combs(profiles[:, :50], period, 16, 4, True) is None


def test_a_flat_page_is_inside_the_dead_band():
    field, info = measured("flat")
    assert info["decision"] == "dead band" and info["reason"] == ""
    assert info["shift_max"] < 0.05 and info["shift_all"] < 0.05
    assert info["carrier"] == "page"
    assert info["period"] == pytest.approx(PERIOD, rel=1e-3)
    assert info["res_field"] < 0.005 and info["passes"] == 1
    assert info["windows_u"] > 400 and info["windows_v"] > 400
    # The map is told, and nothing was warped.
    assert field is not None and field.page is None
    assert np.isnan(info["res_after"]) and np.isnan(info["shift_after"])
    # For the stage it is a page to leave alone: no field.
    kept, same = sheet_curl.straighten(page("flat"), None, helpers())
    assert kept is None and same["decision"] == "dead band"
    # The dead band of the first measurement: no map was accepted, nothing was warped.
    assert "accepted" not in same and np.isnan(same["res_after"])


def test_a_bent_page_is_straightened_to_its_known_map():
    before = page("bent").clone()
    field, info = measured("bent")
    assert info["decision"] == "applied" and info["reason"] == ""
    # The map is the known one to a twentieth of a pixel, in the gauge of the field:
    # 0.011 px RMS and 0.021 px at the worst point when this was written.
    rms, worst = map_error(field, 20.0)
    assert rms < 0.05 and worst < 0.15
    # Doing nothing, the map that moves no point, is 6 px RMS off it.
    assert map_error(lattice_field(), 20.0)[0] > 5.0
    # The numbers of the fit: no projective map describes the sheet, the field does.
    assert info["res_proj"] > 0.25 and info["res_field"] < 0.01
    assert info["res_field"] < info["res_all"] + 1e-9 < 0.01
    assert 12.0 < info["shift_max"] < 13.5 and info["shift_rms"] < info["shift_max"]
    assert info["shift_all"] >= info["shift_max"]
    assert info["carrier"] == "page" and info["passes"] == 1 and info["rounds"] >= 3
    assert info["cover"] > 0.95 and info["share"] > 0.95 and info["jac_min"] > 0.9
    assert info["scale"] == pytest.approx(1.0, abs=0.01)
    assert info["inverse"] <= sheet_curl.INVERSE_TOLERANCE
    # The check on the straightened page: its lines are straight.
    assert info["res_after"] < 0.01 and info["shift_after"] < 0.1
    # The straightened page is a new one of the same kind; the page was not written.
    assert torch.equal(page("bent"), before)
    assert field.page.dtype == np.uint8 and field.page.shape == (3, HEIGHT, WIDTH)
    assert not np.shares_memory(field.page, page("bent").numpy())
    assert not np.array_equal(field.page, page("bent").numpy())
    # Measured again, it is a flat page.
    _, again = sheet_curl.measure_field(field.page, None, helpers())
    assert again["decision"] == "dead band" and again["shift_max"] < 0.1
    # An array is read as a tensor is.
    as_array, info_array = sheet_curl.measure_field(page("bent").numpy(), {}, helpers())
    assert np.allclose(as_array.dx, field.dx, rtol=0, atol=1e-9)
    assert np.allclose(as_array.dy, field.dy, rtol=0, atol=1e-9)
    assert np.array_equal(as_array.page, field.page)
    assert same_info(info_array, info)


def test_the_dead_band_is_the_smallest_shift_that_is_applied():
    # The threshold of the estimator, min_shift of measure_field(). The same page, the
    # same map: only what is made of it differs.
    kept, info_kept = measured("slight", min_shift=10.0)
    assert info_kept["decision"] == "dead band" and 7.0 < info_kept["shift_max"] < 8.5
    assert kept.page is None
    moved, info_moved = measured("slight", min_shift=2.0)
    assert info_moved["decision"] == "applied"
    assert same_number(info_moved["shift_max"], info_kept["shift_max"])
    assert moved.page is not None and map_error(moved, 12.0)[0] < 0.05
    # The bent page is above that threshold and inside a wider one.
    assert measured("bent", min_shift=10.0)[1]["decision"] == "applied"
    _, wide = measured("bent", min_shift=13.5)
    assert wide["decision"] == "dead band"


def test_a_page_whose_lines_do_not_carry_the_map_is_refused():
    # A fit above the tolerance, here a tolerance far below any fit.
    field, info = measured("bent", tol=0.0005)
    assert field is None and info["decision"] == "refused"
    assert re.fullmatch(
        r"grid line phase is 0\.00\d periods off the field", info["reason"]
    )
    assert info["res_field"] > 0.0005 and np.isnan(info["shift_max"])
    # A page without grid lines.
    field, info = measured("blank")
    assert field is None and info["decision"] == "refused"
    assert info["reason"] == "no grid line period found" and info["carrier"] == "none"
    # A page of noise has a period and no comb.
    noise = torch.from_numpy(
        np.random.default_rng(0).integers(0, 256, (3, HEIGHT, WIDTH)).astype(np.uint8)
    )
    field, info = sheet_curl.measure_field(noise, None, helpers())
    assert field is None and info["decision"] == "refused"
    assert info["reason"].startswith("no grid lines found (contrast ")
    assert info["contrast"] < sheet_curl.MIN_CONTRAST
    # Something that is no page.
    field, info = sheet_curl.measure_field(
        page("bent").numpy().astype(float), None, helpers()
    )
    assert field is None and info["decision"] == "refused"
    assert info["reason"] == "page is float64 [3, 850, 1100], not uint8 [C, H, W]"
    small = torch.full((3, 60, 80), 255, dtype=torch.uint8)
    field, info = sheet_curl.measure_field(small, None, helpers())
    assert field is None and info["reason"].startswith("image too small for the grid")


@pytest.mark.parametrize("passes", [1, 2, 3, 5])
def test_the_passes_are_the_measurements_that_may_be_composed(passes):
    # With no dead band the straightened page always asks for something more, and
    # what it asks for is composed into the map until the passes are used up.
    field, info = measured("bent", min_shift=0.0, passes=passes)
    assert info["decision"] == "applied" and info["passes"] == passes
    assert map_error(field, 20.0)[0] < 0.05
    # At the threshold of the estimator the first map leaves nothing to ask for.
    _, once = measured("bent", passes=passes)
    assert once["decision"] == "applied" and once["passes"] == 1


def test_a_map_that_folds_or_asks_for_too_much_is_refused():
    opt = sheet_curl._resolved(None)

    def judged(field):
        info = sheet_curl._blank_info(opt)
        return sheet_curl._judge(helpers(), field, info, opt), info

    reason, info = judged(lattice_field())
    assert reason == "" and info["jac_min"] == 1.0 and info["shift_all"] == 0.0
    assert info["scale"] == 1.0 and info["gauge_dx"] == info["gauge_dy"] == 0.0
    # One node moved by 6 px leaves a quarter of the cell next to it.
    dented = lattice_field()
    dented.dx[50, 60] = 6.0
    reason, info = judged(dented)
    assert info["jac_min"] == pytest.approx(0.25)
    assert reason.startswith("the map stretches or squeezes the page by more than a")
    folded = lattice_field()
    folded.dx[50, 60] = 9.0
    reason, info = judged(folded)
    assert info["jac_min"] < 0 and reason.startswith("the map folds (Jacobian")
    # More than PERSPECTIVE_MAX_SHIFT of the page width is a misread grid.
    limit = digitize.PERSPECTIVE_MAX_SHIFT * WIDTH
    reason, info = judged(lattice_field(limit - 1.0))
    assert reason == ""
    reason, info = judged(lattice_field(limit + 1.0))
    assert reason == (
        "grid lines ask for a 56 px correction somewhere on the page (56 px where "
        "they were measured)"
    )


def test_an_exception_inside_the_measurement_is_a_refusal_for_the_stage():
    broken = helpers()

    def no_darkness(image):
        raise RuntimeError("no\n   darkness")

    broken._darkness = no_darkness
    with pytest.raises(RuntimeError):
        sheet_curl.measure_field(page("bent"), None, broken)
    field, info = sheet_curl.straighten(page("bent"), None, broken)
    assert field is None and info["decision"] == "refused"
    # One line, whatever the message was.
    assert info["reason"] == "error: RuntimeError: no darkness"
    assert np.isnan(info["shift_max"]) and info["carrier"] == "none"
    # An option that is none is no page that broke: it is raised.
    with pytest.raises(ValueError, match="no option of the sheet curl stage"):
        sheet_curl.measure_field(page("bent"), {"speed": 3}, helpers())


def test_the_stage_hands_on_a_field_only_for_a_page_it_straightens():
    field, info = sheet_curl.straighten(page("bent"), None, helpers(), 10.0)
    assert info["decision"] == "applied" and field.page is not None
    assert "accepted" not in info
    for name, options, decision in (
        ("slight", None, "dead band"),
        ("flat", None, "dead band"),
        ("blank", None, "refused"),
        ("bent", {"tol": 0.0005}, "refused"),
    ):
        field, info = sheet_curl.straighten(page(name), options, helpers(), 10.0)
        assert field is None and info["decision"] == decision, name
    # With the default dead band, 15 px, the bent page is left alone as well.
    field, info = sheet_curl.straighten(page("bent"), None, helpers())
    assert field is None and info["decision"] == "dead band" and info["accepted"]
    outside = sheet_curl.out_of_scope({"tol": 0.1})
    assert outside["decision"] == "out of scope" and outside["reason"] == ""
    assert outside["carrier"] == "none" and np.isnan(outside["shift_max"])


# ------------------------------------------------------------------------ the map
def test_a_field_moves_a_page_and_its_labels_where_it_says():
    strip = torch.zeros((1, HEIGHT, WIDTH), dtype=torch.uint8)
    strip[0, 400:406, 300:900], strip[0, 600:605, 100:400] = 3, 7
    # The identity gives the page and the mask back.
    same = lattice_field()
    assert same.invert() == 0.0
    assert np.array_equal(same.warp_labels(strip.numpy()), strip.numpy())
    assert np.array_equal(same.warp_array(page("bent").numpy()), page("bent").numpy())
    # A field that moves every point by (3, -2): the labels land 3 px right and 2 up.
    moved = lattice_field(3.0, -2.0)
    X, Y = moved.straighten(np.array([10.0, 500.0]), np.array([20.0, 700.0]))
    assert np.allclose(X, [13.0, 503.0]) and np.allclose(Y, [18.0, 698.0])
    assert moved.invert() <= sheet_curl.INVERSE_TOLERANCE
    labels = moved.warp_labels(strip.numpy())
    assert labels.dtype == np.uint8 and labels.shape == (1, HEIGHT, WIDTH)
    assert np.array_equal(labels[0, 398:404, 303:903], strip[0, 400:406, 300:900])
    assert np.count_nonzero(labels) == np.count_nonzero(strip.numpy())
    # What comes in from beyond the page is no label, and black on the page.
    assert not labels[0, -2:, :].any() and not labels[0, :, :3].any()
    warped = moved.warp_array(page("flat").numpy())
    assert not warped[:, -2:, :].any() and not warped[:, :, :3].any()
    flat = page("flat").numpy()
    assert np.array_equal(warped[:, 100:200, 103:203], flat[:, 102:202, 100:200])
    # The map of a bent page holds no label that was not in the mask.
    field, _ = measured("bent")
    labels = field.warp_labels(mask_of("bent").numpy())
    assert set(np.unique(labels)) == set(np.unique(mask_of("bent").numpy()))
    # One map after another is the second map of the first.
    both = lattice_field(3.0, -2.0).compose(lattice_field(0.5, 4.0))
    X, Y = both.straighten(np.array([100.0]), np.array([200.0]))
    assert np.allclose([X[0], Y[0]], [103.5, 202.0])
    assert same.jacobian().shape == (same.dx.shape[0] - 1, same.dx.shape[1] - 1)
    assert np.all(same.jacobian() == 1.0)


def test_the_mask_of_a_bent_page_goes_where_its_page_goes():
    # Traces that are straight on the sheet are bent on the page, and straight again
    # in the frame of the map: three rows thick, and one more for the rounding.
    field, _ = measured("bent")
    given = mask_of("bent")
    assert min(rows_spanned(given)) >= 6 and max(rows_spanned(given)) >= 11
    straight = torch.from_numpy(field.warp_labels(given.numpy()))
    assert max(rows_spanned(straight)) <= 4
    assert max(rows_spanned(mask_of("flat"))) == 3


# ------------------------------------------- the lines, the QC values, the record
def a_fit(**values):
    info = sheet_curl._blank_info(sheet_curl.DEFAULTS)
    info.update(
        decision="applied", carrier="page", period=7.87412, windows_u=561,
        windows_v=550, res_proj=0.54912, res_field=0.00151, res_after=0.00149,
        res_all=0.00163, shift_max=12.6114, shift_rms=6.2, shift_all=13.0,
        scale=0.999771, jac_min=0.99812, passes=1, cover=0.98431, share=1.0,
    )
    info.update(values)
    return info


def test_the_verbose_line_names_the_decision_and_the_numbers_of_the_fit():
    numbers = (
        "carrier page 7.874 px, windows 561 + 550, residual 0.5491 periods off the "
        "projective part, 0.0015 off the field, 0.0015 after the warp, 0.0016 over "
        "all windows, shift max 12.61 px, rms 6.20 px, whole page 13.00 px, scale "
        "0.99977, jac_min 0.998, passes 1, cover 0.984, share 1.000"
    )
    line = sheet_curl.verbose_line("rec", a_fit())
    assert line == f"Sheet curl for record rec: applied, {numbers}"
    line = sheet_curl.verbose_line("rec", a_fit(decision="dead band"))
    assert line == f"Sheet curl for record rec: dead band, {numbers}"
    # A map that was accepted and is below the dead band of the stage says so.
    line = sheet_curl.verbose_line("rec", a_fit(decision="dead band", accepted=True))
    assert line == f"Sheet curl for record rec: dead band of the accepted map, {numbers}"
    # A refused page shows the numbers it got as far as, and nan for the rest.
    refused = sheet_curl._blank_info(sheet_curl.DEFAULTS)
    refused["reason"] = "no grid line period found"
    line = sheet_curl.verbose_line("rec", refused)
    assert line.startswith("Sheet curl for record rec: refused, carrier none nan px, ")
    assert line.endswith("passes 0, cover nan, share nan")
    # A page out of scope was not measured.
    outside = sheet_curl.out_of_scope(None)
    assert sheet_curl.verbose_line("rec", outside) == (
        "Sheet curl for record rec: out of scope, not a page the paper normalisation "
        "warped onto the page frame"
    )
    for info in (a_fit(), refused, outside):
        line = sheet_curl.verbose_line("rec", info)
        assert "WARNING" not in line and "\n" not in line


def test_the_warning_of_a_refused_page_says_why_and_that_the_page_is_kept():
    refused = sheet_curl._blank_info(sheet_curl.DEFAULTS)
    refused["reason"] = "grid line phase is 0.312 periods off the field"
    assert sheet_curl.warning_line("rec", refused) == (
        "WARNING: sheet curl not straightened for record rec (grid line phase is "
        "0.312 periods off the field), keeping the page as it is."
    )


def test_the_qc_values_are_the_decision_and_the_shift_of_a_page_with_a_map():
    assert sheet_curl.qc_values(a_fit()) == ("applied", 12.6114)
    assert sheet_curl.qc_values(a_fit(decision="dead band")) == ("dead band", 12.6114)
    text, shift = sheet_curl.qc_values(a_fit(decision="refused", reason="no lines"))
    assert text == "refused: no lines" and np.isnan(shift)
    text, shift = sheet_curl.qc_values(sheet_curl.out_of_scope(None))
    assert text == "out of scope" and np.isnan(shift)


def test_the_record_holds_what_defines_the_map_and_the_map_of_an_applied_page():
    options = {"tol": 0.25, "passes": 3}
    field, info = sheet_curl.straighten(page("bent"), options, helpers(), 10.0)
    block = sheet_curl.mask_record(info, field, options, "all", 10.0)
    assert list(block) == [
        "version", "decision", "reason", "scope", "dead_band", "options", "constants",
        "shift_max", "page_size", "points",
    ]
    # The dead band of the stage next to the options the page was measured with,
    # where min_shift is the threshold of the estimator.
    assert block["dead_band"] == 10.0 and block["options"]["min_shift"] == 1.0
    assert block["version"] == sheet_curl.VERSION == 1
    assert block["decision"] == "applied" and block["reason"] == ""
    assert block["scope"] == "all"
    assert block["options"] == sheet_curl.DEFAULTS
    assert block["constants"]["lattice"] == 8
    assert block["constants"]["jacobian_floor"] == 0.5
    assert block["constants"]["degree_schedule"] == [2, 3]
    assert block["shift_max"] == info["shift_max"]
    assert block["page_size"] == [WIDTH, HEIGHT]
    # The map at 9 x 7 points, the corners of the page among them: [x, y, X, Y].
    points = np.array(block["points"])
    assert sheet_curl.CHECK_POINTS == (9, 7) and points.shape == (63, 4)
    assert points[0, :2].tolist() == [0.0, 0.0]
    assert points[-1, :2].tolist() == [float(WIDTH), float(HEIGHT)]
    X, Y = field.straighten(points[:, 0], points[:, 1])
    assert np.array_equal(points[:, 2], X) and np.array_equal(points[:, 3], Y)
    moved = np.hypot(points[:, 2] - points[:, 0], points[:, 3] - points[:, 1])
    assert moved.max() > 5
    # Plain values: it goes into a JSON as it is, and comes back as it was.
    assert json.loads(json.dumps(block, allow_nan=False)) == block

    # A page that was not straightened has a record without a map.
    kept, info_kept = sheet_curl.straighten(page("slight"), options, helpers(), 10.0)
    block = sheet_curl.mask_record(info_kept, kept, options, "normalised", 10.0)
    assert block["decision"] == "dead band" and block["scope"] == "normalised"
    # A dead band below the threshold of the estimator takes its place, and the
    # default dead band is the constant of the module.
    assert sheet_curl.mask_record(info_kept, None, options, "all", 0.5)["options"][
        "min_shift"
    ] == 0.5
    assert sheet_curl.mask_record(info_kept, None, options, "all")["dead_band"] == 15.0
    assert block["shift_max"] == info_kept["shift_max"]
    assert "points" not in block and "page_size" not in block
    unmeasured = sheet_curl.out_of_scope(options)
    outside = sheet_curl.mask_record(unmeasured, None, options, "normalised")
    assert outside["decision"] == "out of scope" and outside["shift_max"] is None
    refused, info_refused = sheet_curl.straighten(page("blank"), options, helpers())
    block = sheet_curl.mask_record(info_refused, refused, options, "all")
    assert block["decision"] == "refused" and block["shift_max"] is None
    assert block["reason"] == "no grid line period found"
    assert json.loads(json.dumps(block, allow_nan=False)) == block


def test_a_mask_is_in_the_straightened_frame_only_when_its_record_says_applied():
    field = lattice_field(3.0, -2.0)
    applied = sheet_curl.mask_record(a_fit(), field, None, "all")
    # No record, or the record of a page that was kept: a mask of the page as it was.
    for block in (None, {}, {"decision": "dead band"}, {"decision": None}):
        assert sheet_curl.mask_frame(block, field) == (False, "")
        assert sheet_curl.mask_frame(block, None) == (False, "")
    # A record that is no object cannot be read: a problem, and the mask is not warped
    # on a guess. null in the JSON is no record (None above).
    for block in ("applied", ["applied"], 1, True):
        for now in (field, None):
            assert sheet_curl.mask_frame(block, now) == (
                True,
                "has a sheet curl record that is no object",
            ), block
    for decision in ("dead band", "refused", "out of scope"):
        kept = sheet_curl.mask_record(a_fit(decision=decision), None, None, "all")
        assert sheet_curl.mask_frame(kept, field) == (False, "")
    # The same map: the mask fits.
    assert sheet_curl.mask_frame(applied, field) == (True, "")
    assert sheet_curl.MASK_SHIFT_TOLERANCE == digitize.PERSPECTIVE_MASK_SHIFT_TOLERANCE
    assert sheet_curl.mask_frame(applied, lattice_field(3.05, -2.05)) == (True, "")
    # Another map, at the points of the record.
    assert sheet_curl.mask_frame(applied, lattice_field(3.5, -2.0)) == (
        True,
        "was predicted on a page straightened by a sheet curl map that is 0.50 px off "
        "the one used now",
    )
    # The distance is printed to the hundredth: a map 0.12 px off is not '0.1 px off'.
    assert sheet_curl.mask_frame(applied, lattice_field(3.12, -2.0))[1].endswith(
        "that is 0.12 px off the one used now"
    )
    # A page that is kept now.
    kept_now = "was predicted on a page whose sheet curl was straightened, the page "
    assert sheet_curl.mask_frame(applied, None) == (
        True,
        kept_now + "is now kept as it is",
    )
    assert sheet_curl.mask_frame(applied, None, "dead band") == (
        True,
        kept_now + "is now kept as it is (dead band)",
    )
    # A record that cannot be compared is one that does not fit, and is not warped.
    for points in (None, [], [[1.0, 2.0]], [[0.0, 0.0, float("nan"), 0.0]], "points"):
        broken = {**applied, "points": points}
        assert sheet_curl.mask_frame(broken, field) == (
            True,
            "has a sheet curl record without the points of its map",
        ), points
    without = {key: value for key, value in applied.items() if key != "points"}
    assert sheet_curl.mask_frame(without, field)[1].endswith("the points of its map")
    # An applied record needs ALL its check points and the size of the page they are
    # on: one point of the 63, 62 of them, or another page, is no check of the map.
    for points in (applied["points"][:1], applied["points"][:-1]):
        assert sheet_curl.mask_frame({**applied, "points": points}, field) == (
            True,
            "has a sheet curl record without the points of its map",
        ), len(points)
    sizes = (None, [WIDTH, HEIGHT - 1], [HEIGHT, WIDTH], [WIDTH], "page", [None, None])
    for size in sizes:
        assert sheet_curl.mask_frame({**applied, "page_size": size}, field) == (
            True,
            "has a sheet curl record without the points of its map",
        ), size
    no_size = {key: value for key, value in applied.items() if key != "page_size"}
    assert sheet_curl.mask_frame(no_size, field)[1].endswith("the points of its map")
    assert sheet_curl.mask_frame({**applied, "page_size": [WIDTH, HEIGHT]}, field) == (
        True,
        "",
    )
    assert sheet_curl.mask_frame({**applied, "version": 2}, field) == (
        True,
        "has a sheet curl record of version 2, this code reads 1",
    )


def test_check_mask_sheet_curl_warns_in_the_words_of_the_frame_check(capsys):
    field = lattice_field(3.0, -2.0)
    applied = sheet_curl.mask_record(a_fit(), field, None, "all")
    assert digitize.check_mask_sheet_curl(None, "rec", field) is False
    assert digitize.check_mask_sheet_curl(applied, "rec", field) is True
    assert capsys.readouterr().out == ""
    assert digitize.check_mask_sheet_curl(applied, "rec", None, "refused") is True
    assert capsys.readouterr().out == (
        "WARNING: mask of record rec was predicted on a page whose sheet curl was "
        "straightened, the page is now kept as it is (refused); the mask does not "
        "fit.\n"
    )
    assert digitize.check_mask_sheet_curl(applied, "rec", lattice_field(4.2)) is True
    assert capsys.readouterr().out == (
        "WARNING: mask of record rec was predicted on a page straightened by a sheet "
        "curl map that is 2.33 px off the one used now; the mask does not fit.\n"
    )
    # A record that is no object is said as well, and the mask is used as it is.
    assert digitize.check_mask_sheet_curl("applied", "rec", field) is True
    assert capsys.readouterr().out == (
        "WARNING: mask of record rec has a sheet curl record that is no object; the "
        "mask does not fit.\n"
    )


def test_save_mask_files_writes_the_record_of_the_stage_after_all_others(tmp_path):
    mask = torch.zeros((1, 6, 8), dtype=torch.uint8)
    plain = '{"rot_angle": 0.5, "height": 6, "width": 8}'
    # Without a record the JSON is the one it always was.
    digitize.save_mask_files(mask, "none", str(tmp_path), 0.5, curl=None)
    assert (tmp_path / "none_mask.json").read_text() == plain
    block = {"version": 1, "decision": "dead band"}
    digitize.save_mask_files(mask, "rec", str(tmp_path), 0.5, curl=block)
    assert (tmp_path / "rec_mask.json").read_text() == (
        plain[:-1] + ', "sheet_curl": {"version": 1, "decision": "dead band"}}'
    )
    paper = {"version": "v"}
    digitize.save_mask_files(mask, "both", str(tmp_path), 0.5, paper=paper, curl=block)
    assert list(json.loads((tmp_path / "both_mask.json").read_text())) == [
        "rot_angle", "height", "width", "paper_normalisation", "sheet_curl",
    ]
    # The parameters drivers call it with keep their places; the new one is last.
    parameters = inspect.signature(digitize.save_mask_files).parameters
    assert list(parameters) == [
        "mask", "record", "output_folder", "rot_angle", "homography", "scale",
        "paper", "curl",
    ]
    assert parameters["curl"].default is None


# ------------------------------------------------------------------- end to end
# Every run saves its mask, so its output folder is the mask folder of a later run.
BASE = ("--time_mapping", "bbox", "--save_mask")
# The stage on every page, which is the default scope, with a dead band of 10 px: the
# page bent by 20 px is straightened (12.6 px) and the one bent by 12 px is not (7.6).
LINES = ("--sheet_curl", "lines", "--sheet_curl_min_shift", "10")
# The same with the scope that only takes a page the paper normalisation warped.
NORMALISED = (*LINES, "--sheet_curl_scope", "normalised")
FITS_NOT = "the mask does not fit."
SAID = "Sheet curl for record"
NOT_STRAIGHTENED = "WARNING: sheet curl not straightened"
# The columns of qc.csv before the stage, in their order.
OLD_COLUMNS = [
    "record", "placement", "sharpen", "einthoven_rms", "einthoven_rms_demedian",
    "einthoven_ratio", "einthoven_n", "goldberger_rms", "goldberger_rms_demedian",
    "goldberger_ratio", "goldberger_n", "max_offset_deviation", "paper_normalisation",
    "baseline_scale", "baseline_shift_px", "baseline_disagreement_px",
    "grid_line_contrast", "grid_line_shift_px", "rotation_angle", "rotation_coarse",
    "rotation_residual_px", "perspective_shift_px", "perspective_residual_px",
    "grid_period_px", "resolution_scale", "trace_estimator", "ink_leads", "ink_p95",
    "trace_shift_px", "trace_shift_rows", "column_mapping", "column_mapping_shift_px",
    "grid_rescue", "row_mapping", "perspective_residual_rel",
]


def run_digitiser(data, out, masks, *flags):
    """run() with a mask folder; returns what it printed."""
    argv = ["-d", str(data), "-o", str(out), "--mask_folder", str(masks)]
    printed = io.StringIO()
    with contextlib.redirect_stdout(printed):
        digitize.run(digitize.get_parser().parse_args(argv + [*BASE, *flags]))
    return printed.getvalue()


@pytest.fixture(scope="module")
def runs(tmp_path_factory):
    """run() on one drawn page with a saved mask, each page, mask and flags once.

    Returns a function: (name, *flags, masks=None) -> (output folder, what run()
    printed). masks is the folder the mask is read from: the label mask of the page
    alone, without a JSON, unless given.
    """
    root = tmp_path_factory.mktemp("sheet_curl")
    folders, done = {}, {}

    def run(name, *flags, masks=None):
        if name not in folders:
            data, bare = root / f"data_{name}", root / f"masks_{name}"
            data.mkdir()
            bare.mkdir()
            write_png(page(name), str(data / f"{name}.png"))
            write_png(mask_of(name), str(bare / f"{name}_mask.png"))
            folders[name] = (data, bare)
        data, bare = folders[name]
        masks = bare if masks is None else masks
        if (name, flags, masks) not in done:
            out = root / f"out{len(done)}"
            done[name, flags, masks] = (out, run_digitiser(data, out, masks, *flags))
        return done[name, flags, masks]

    return run


def qc_row(folder):
    """(the one row of qc.csv, its header)."""
    with open(folder / "qc.csv", newline="") as f:
        reader = csv.DictReader(f)
        rows = list(reader)
    assert len(rows) == 1
    return rows[0], reader.fieldnames


def same_row(a, b, but=()):
    """Two QC rows agree, in every column or in all but some: the texts as they are,
    the numbers as same_number() has it."""

    def same(x, y):
        try:
            return same_number(float(x), float(y))
        except ValueError:
            return x == y

    return a.keys() == b.keys() and all(same(a[k], b[k]) for k in a if k not in but)


def mask_meta(folder, name):
    with open(folder / f"{name}_mask.json") as f:
        return json.load(f)


def signal_files(folder, name):
    """The bytes of the signals a run wrote for a page, header and samples."""
    return [(folder / f"{name}{end}").read_bytes() for end in (".hea", ".dat")]


def lines_of(printed, name, start=""):
    """The lines run() printed that name the record, optionally those starting so."""
    names_it = re.compile(rf"record {re.escape(name)}(?!\w)")
    return [
        line
        for line in printed.splitlines()
        if names_it.search(line) and line.startswith(start)
    ]


def other_lines(printed):
    """What run() printed but for the lines of the stage."""
    return [line for line in printed.splitlines() if not line.startswith(SAID)]


def signal_range(folder, name):
    """Largest minus smallest sample of every lead of a written record: what a flat
    trace leaves of it is the bend of its mask."""
    signals = wfdb.rdrecord(str(folder / name)).p_signal
    return np.nanmax(signals, axis=0) - np.nanmin(signals, axis=0)


CURL_COLUMNS = ("sheet_curl", "sheet_curl_shift_px")


@pytest.mark.parametrize("name", ["flat", "bent", "blank"])
def test_a_run_without_the_stage_is_the_run_it_was(runs, name):
    out, printed = runs(name)
    row, header = qc_row(out)
    # No line of the stage, no record next to the mask, and the columns that were
    # there where they were: the two of the stage come after them and say off.
    assert "heet curl" not in printed
    assert list(mask_meta(out, name)) == ["rot_angle", "height", "width"]
    assert header[:35] == OLD_COLUMNS and header[35:] == list(CURL_COLUMNS)
    assert row["sheet_curl"] == "off" and np.isnan(float(row["sheet_curl_shift_px"]))
    # The mask is saved as it was given.
    assert torch.equal(read_image(str(out / f"{name}_mask.png")), mask_of(name))
    # off written out is that run, and so is any option of the stage without it.
    options = (
        "--sheet_curl_tolerance", "0.1", "--sheet_curl_passes", "1",
        "--sheet_curl_min_shift", "0", "--sheet_curl_scope", "normalised",
    )
    for flags in (("--sheet_curl", "off"), options, ("--sheet_curl", "off", *options)):
        named, printed_named = runs(name, *flags)
        assert printed_named == printed, flags
        assert same_row(qc_row(named)[0], row), flags
        assert signal_files(named, name) == signal_files(out, name), flags
        assert (named / f"{name}_mask.json").read_bytes() == (
            out / f"{name}_mask.json"
        ).read_bytes(), flags
        assert (named / f"{name}_mask.png").read_bytes() == (
            out / f"{name}_mask.png"
        ).read_bytes(), flags


def test_a_run_without_the_stage_calls_no_function_of_it(tmp_path, monkeypatch):
    def never(name):
        def called(*args, **kwargs):
            raise AssertionError(f"{name} was called with the stage off")

        return called

    for name in ("sheet_curl_helpers", "check_mask_sheet_curl"):
        monkeypatch.setattr(digitize, name, never(name))
    public = [
        name
        for name, value in vars(sheet_curl).items()
        if inspect.isfunction(value) or inspect.isclass(value)
    ]
    assert {"measure_field", "straighten", "mask_frame", "mask_record", "Field"} <= set(
        public
    )
    for name in public:
        monkeypatch.setattr(sheet_curl, name, never(name))
    data, masks = tmp_path / "data", tmp_path / "masks"
    data.mkdir()
    masks.mkdir()
    write_png(page("bent"), str(data / "bent.png"))
    write_png(mask_of("bent"), str(masks / "bent_mask.png"))
    # With a mask without a JSON, and with the mask and the JSON a run saves.
    printed = run_digitiser(data, tmp_path / "first", masks)
    again = run_digitiser(data, tmp_path / "second", tmp_path / "first")
    off = run_digitiser(data, tmp_path / "third", masks, "--sheet_curl", "off")
    assert "heet curl" not in printed + again + off
    assert signal_files(tmp_path / "second", "bent") == signal_files(
        tmp_path / "first", "bent"
    )
    # The stand-ins do stand in the way of a run with the stage.
    with pytest.raises(AssertionError, match="called with the stage off"):
        run_digitiser(data, tmp_path / "fourth", masks, *LINES)


def test_run_straightens_a_bent_page_and_the_mask_that_comes_with_it(runs):
    off, printed_off = runs("bent")
    out, printed = runs("bent", *LINES)
    row, header = qc_row(out)
    # One line of the stage, after the line of the perspective stage, and no WARNING.
    said = lines_of(printed, "bent", SAID)
    assert len(said) == 1 and NOT_STRAIGHTENED not in printed
    assert re.fullmatch(
        r"Sheet curl for record bent: applied, carrier page 7\.87\d px, windows \d+ "
        r"\+ \d+, residual 0\.\d{4} periods off the projective part, 0\.\d{4} off "
        r"the field, 0\.\d{4} after the warp, 0\.\d{4} over all windows, shift max "
        r"12\.\d\d px, rms \d\.\d\d px, whole page 1\d\.\d\d px, scale [01]\.\d{5}, "
        r"jac_min 0\.\d{3}, passes 1, cover 0\.\d{3}, share [01]\.\d{3}",
        said[0],
    )
    every = printed.splitlines()
    before = lines_of(printed, "bent", "Perspective for record")
    assert len(before) == 1 and every.index(said[0]) == every.index(before[0]) + 1
    # The stages before it print what they printed.
    start = every.index(said[0])
    assert every[:start] == printed_off.splitlines()[:start]
    # The two columns of the stage, after all others.
    assert header[-2:] == list(CURL_COLUMNS) and header[:35] == OLD_COLUMNS
    assert row["sheet_curl"] == "applied"
    assert 12.0 < float(row["sheet_curl_shift_px"]) < 13.5
    row_off = qc_row(off)[0]
    for column in OLD_COLUMNS[12:25]:
        assert same_row({column: row[column]}, {column: row_off[column]}), column
    # The mask that came with the page was warped with it: the mask this run saves
    # lives on the straightened page, where the traces are straight.
    assert min(rows_spanned(mask_of("bent"))) >= 6
    saved = read_image(str(out / "bent_mask.png"))
    assert max(rows_spanned(saved)) <= 4
    assert torch.equal(read_image(str(off / "bent_mask.png")), mask_of("bent"))
    # So are the signals: with the stage off the bend of the sheet is read as signal.
    assert signal_files(out, "bent") != signal_files(off, "bent")
    assert signal_range(out, "bent").sum() < 0.5 * signal_range(off, "bent").sum()
    # And the record next to the mask says so.
    meta = mask_meta(out, "bent")
    assert list(meta) == ["rot_angle", "height", "width", "sheet_curl"]
    block = meta["sheet_curl"]
    assert block["decision"] == "applied" and block["scope"] == "all"
    assert block["options"] == sheet_curl.DEFAULTS and block["dead_band"] == 10.0
    assert block["shift_max"] == pytest.approx(float(row["sheet_curl_shift_px"]))
    assert block["page_size"] == [WIDTH, HEIGHT] and len(block["points"]) == 63


def test_the_mask_is_warped_by_the_map_of_the_page_and_the_page_is_a_new_one(
    tmp_path, monkeypatch
):
    data, masks = tmp_path / "data", tmp_path / "masks"
    data.mkdir()
    masks.mkdir()
    write_png(page("bent"), str(data / "bent.png"))
    write_png(mask_of("bent"), str(masks / "bent_mask.png"))
    asked = recorded(monkeypatch, sheet_curl, "straighten")
    cut = recorded(monkeypatch, digitize, "cut_binary")
    run_digitiser(data, tmp_path / "out", masks, *LINES)
    # The stage is asked once, with the options of the command line and the helpers.
    assert len(asked) == 1 and len(cut) == 1
    (given, options, handed, band), kwargs = asked[0]
    assert kwargs == {}
    # The dead band of the command line is the stage's and no option of the estimator.
    assert options == {"tol": 0.25, "passes": 3} and band == 10.0
    assert sorted(vars(handed)) == sorted(sheet_curl.HELPERS)
    # On the page as the stages before it left it, here the page as it was drawn, and
    # that page is not written into.
    assert torch.equal(given, page("bent"))
    field, info = sheet_curl.measure_field(page("bent"), options, helpers())
    assert info["decision"] == "applied"
    # What goes on is the straightened page, a new tensor, and the mask through the
    # same map: nearest, as Field.warp_labels() does it.
    (mask_used, page_used), _ = cut[0]
    assert page_used is not given and not np.shares_memory(
        page_used.numpy(), given.numpy()
    )
    assert page_used.dtype == torch.uint8
    assert tuple(page_used.shape) == (3, HEIGHT, WIDTH)
    assert np.array_equal(page_used.numpy(), field.page)
    assert mask_used.dtype == torch.uint8
    assert tuple(mask_used.shape) == (1, HEIGHT, WIDTH)
    assert np.array_equal(mask_used.numpy(), field.warp_labels(mask_of("bent").numpy()))
    # The saved mask is the one that was used.
    assert torch.equal(read_image(str(tmp_path / "out" / "bent_mask.png")), mask_used)

    # Other options of the command line reach the stage as they are given.
    flags = (
        "--sheet_curl_tolerance", "0.1", "--sheet_curl_passes", "2",
        "--sheet_curl_min_shift", "1.5",
    )
    run_digitiser(data, tmp_path / "other", masks, *LINES, *flags)
    assert asked[1][0][1] == {"tol": 0.1, "passes": 2} and asked[1][0][3] == 1.5


def test_a_mask_saved_for_the_straightened_page_is_used_as_it_is(runs):
    first, _ = runs("bent", *LINES)
    out, printed = runs("bent", *LINES, masks=first)
    # The record says applied and the map of this run is the one of the record: no
    # WARNING, the mask is not warped a second time, and the signals are the same.
    assert FITS_NOT not in printed
    assert len(lines_of(printed, "bent", SAID)) == 1
    assert signal_files(out, "bent") == signal_files(first, "bent")
    saved, given = out / "bent_mask.png", first / "bent_mask.png"
    assert saved.read_bytes() == given.read_bytes()
    assert same_row(qc_row(out)[0], qc_row(first)[0])
    meta, meta_first = mask_meta(out, "bent"), mask_meta(first, "bent")
    assert meta["sheet_curl"]["decision"] == "applied"
    assert np.allclose(
        meta["sheet_curl"]["points"], meta_first["sheet_curl"]["points"], atol=1e-6
    )
    # Warped again, its traces would be bent the other way.
    field, _ = measured("bent")
    twice = field.warp_labels(read_image(str(first / "bent_mask.png")).numpy())
    assert max(rows_spanned(torch.from_numpy(twice))) >= 6


@pytest.mark.parametrize(
    "flags, kept",
    [
        ((), "--sheet_curl off"),
        (NORMALISED, "out of scope"),
        ((*LINES, "--sheet_curl_min_shift", "50"), "dead band"),
        ((*LINES, "--sheet_curl_tolerance", "0.0005"), "refused"),
    ],
)
def test_a_mask_of_a_straightened_page_does_not_fit_a_page_that_is_kept(
    runs, flags, kept
):
    first, _ = runs("bent", *LINES)
    out, printed = runs("bent", *flags, masks=first)
    # The WARNING of the frame, once, and the run goes on with the mask as it is.
    said = lines_of(printed, "bent", "WARNING: mask of record")
    assert said == [
        "WARNING: mask of record bent was predicted on a page whose sheet curl was "
        f"straightened, the page is now kept as it is ({kept}); {FITS_NOT}"
    ]
    assert printed.count(FITS_NOT) == 1
    assert (out / "bent.dat").exists() and (out / "bent.hea").exists()
    saved, given = out / "bent_mask.png", first / "bent_mask.png"
    assert saved.read_bytes() == given.read_bytes()
    decision = "off" if kept == "--sheet_curl off" else kept
    assert qc_row(out)[0]["sheet_curl"].split(":")[0] == decision
    # It comes with the other warnings of the frame, before the page is read.
    every = printed.splitlines()
    read = next(i for i, line in enumerate(every) if line.startswith("QC bent:"))
    assert every.index(said[0]) < read


def test_a_mask_straightened_by_another_map_does_not_fit(runs, tmp_path):
    first, _ = runs("bent", *LINES)

    def moved_by(shift):
        folder = tmp_path / f"moved_{shift}"
        folder.mkdir()
        shutil.copy(first / "bent_mask.png", folder / "bent_mask.png")
        meta = mask_meta(first, "bent")
        for point in meta["sheet_curl"]["points"]:
            point[2] += shift
        (folder / "bent_mask.json").write_text(json.dumps(meta))
        return folder

    # Half a pixel off at the points of the record: said, and the mask is used as it
    # is, not warped by the map of this run on top of its own.
    out, printed = runs("bent", *LINES, masks=moved_by(0.5))
    assert lines_of(printed, "bent", "WARNING: mask of record") == [
        "WARNING: mask of record bent was predicted on a page straightened by a sheet "
        f"curl map that is 0.50 px off the one used now; {FITS_NOT}"
    ]
    assert signal_files(out, "bent") == signal_files(first, "bent")
    # Saved again, the mask keeps the record it came with, not the map of this run.
    assert mask_meta(out, "bent")["sheet_curl"] == mask_meta(
        tmp_path / "moved_0.5", "bent"
    )["sheet_curl"]
    # Within a tenth of a pixel it is the same map.
    out, printed = runs("bent", *LINES, masks=moved_by(0.05))
    assert FITS_NOT not in printed
    assert signal_files(out, "bent") == signal_files(first, "bent")


def test_a_mask_saved_with_the_stage_on_for_a_page_that_was_kept_is_warped(runs):
    # The record of a page inside the dead band says that its mask lives on the page
    # as it was: a later run that straightens the page warps the mask with it, as it
    # warps a mask without a record.
    kept, printed_kept = runs("slight", *LINES)
    row = qc_row(kept)[0]
    assert row["sheet_curl"] == "dead band"
    assert 7.0 < float(row["sheet_curl_shift_px"]) < 8.5
    assert NOT_STRAIGHTENED not in printed_kept
    # The map was measured, checked and accepted, and is below the dead band.
    said = lines_of(printed_kept, "slight", SAID)[0]
    assert said.startswith(
        "Sheet curl for record slight: dead band of the accepted map, carrier page "
    )
    assert re.search(r", 0\.\d{4} after the warp, .*, passes 1, ", said)
    block = mask_meta(kept, "slight")["sheet_curl"]
    assert block["decision"] == "dead band" and "points" not in block
    assert block["shift_max"] == pytest.approx(float(row["sheet_curl_shift_px"]))
    # Kept: the signals, the mask and every other column are those of the run without.
    off, printed_off = runs("slight")
    assert signal_files(kept, "slight") == signal_files(off, "slight")
    assert torch.equal(read_image(str(kept / "slight_mask.png")), mask_of("slight"))
    assert same_row(row, qc_row(off)[0], but=CURL_COLUMNS)
    assert other_lines(printed_kept) == printed_off.splitlines()

    narrow = (*LINES, "--sheet_curl_min_shift", "2")
    bare, _ = runs("slight", *narrow)
    assert qc_row(bare)[0]["sheet_curl"] == "applied"
    again, printed = runs("slight", *narrow, masks=kept)
    assert FITS_NOT not in printed
    assert signal_files(again, "slight") == signal_files(bare, "slight")
    assert max(rows_spanned(read_image(str(again / "slight_mask.png")))) <= 4
    assert signal_files(again, "slight") != signal_files(off, "slight")


def test_a_flat_page_is_left_exactly_as_it_is(runs):
    off, printed_off = runs("flat")
    out, printed = runs("flat", *LINES)
    row = qc_row(out)[0]
    assert row["sheet_curl"] == "dead band" and float(row["sheet_curl_shift_px"]) < 0.05
    assert NOT_STRAIGHTENED not in printed and FITS_NOT not in printed
    assert len(lines_of(printed, "flat", SAID)) == 1
    assert other_lines(printed) == printed_off.splitlines()
    assert signal_files(out, "flat") == signal_files(off, "flat")
    assert (out / "flat_mask.png").read_bytes() == (off / "flat_mask.png").read_bytes()
    assert same_row(row, qc_row(off)[0], but=CURL_COLUMNS)
    assert mask_meta(out, "flat")["sheet_curl"]["decision"] == "dead band"
    # The dead band of the first measurement: nothing was warped to check a map.
    said = lines_of(printed, "flat", SAID)[0]
    assert said.startswith("Sheet curl for record flat: dead band, carrier page ")
    assert ", nan after the warp, " in said


@pytest.mark.parametrize(
    "name, flags, reason",
    [
        ("blank", (), r"no grid line period found"),
        (
            "bent",
            ("--sheet_curl_tolerance", "0.0005"),
            r"grid line phase is 0\.00\d periods off the field",
        ),
    ],
)
def test_a_refused_page_is_kept_with_a_warning(runs, name, flags, reason):
    off, printed_off = runs(name)
    out, printed = runs(name, *LINES, *flags)
    warning = lines_of(printed, name, NOT_STRAIGHTENED)
    assert len(warning) == 1 and printed.count(NOT_STRAIGHTENED) == 1
    found = re.fullmatch(
        rf"WARNING: sheet curl not straightened for record {name} \(({reason})\), "
        r"keeping the page as it is\.",
        warning[0],
    )
    assert found
    # The line of the stage comes right after it.
    said = lines_of(printed, name, SAID)
    assert len(said) == 1 and said[0].startswith(f"{SAID} {name}: refused, carrier ")
    every = printed.splitlines()
    assert every.index(said[0]) == every.index(warning[0]) + 1
    # The reason is in the QC row and in the record, and the page is the one it was.
    row = qc_row(out)[0]
    assert row["sheet_curl"] == f"refused: {found.group(1)}"
    assert np.isnan(float(row["sheet_curl_shift_px"]))
    block = mask_meta(out, name)["sheet_curl"]
    assert block["decision"] == "refused" and block["reason"] == found.group(1)
    assert block["shift_max"] is None and "points" not in block
    assert signal_files(out, name) == signal_files(off, name)
    assert (out / f"{name}_mask.png").read_bytes() == (
        off / f"{name}_mask.png"
    ).read_bytes()
    assert same_row(row, qc_row(off)[0], but=CURL_COLUMNS)
    assert [
        line for line in other_lines(printed) if not line.startswith(NOT_STRAIGHTENED)
    ] == printed_off.splitlines()


def test_the_stage_only_warns_without_verbose(runs):
    loud, _ = runs("bent", *LINES)
    out, printed = runs("bent", *LINES, "--no-verbose")
    assert SAID not in printed and NOT_STRAIGHTENED not in printed
    # The page is straightened all the same, and the QC columns are what say so.
    assert same_row(qc_row(out)[0], qc_row(loud)[0])
    assert signal_files(out, "bent") == signal_files(loud, "bent")
    # A page inside the dead band says nothing at all, a refused one its WARNING.
    _, printed = runs("slight", *LINES, "--no-verbose")
    assert "heet curl" not in printed
    narrow = ("--sheet_curl_tolerance", "0.0005")
    _, printed = runs("bent", *LINES, *narrow, "--no-verbose")
    assert printed.count(NOT_STRAIGHTENED) == 1 and SAID not in printed


def test_a_page_that_breaks_the_measurement_is_kept_and_the_run_goes_on(
    tmp_path, monkeypatch
):
    data, masks = tmp_path / "data", tmp_path / "masks"
    data.mkdir()
    masks.mkdir()
    write_png(page("bent"), str(data / "bent.png"))
    write_png(mask_of("bent"), str(masks / "bent_mask.png"))
    plain = run_digitiser(data, tmp_path / "plain", masks)
    real = digitize.sheet_curl_helpers

    def broken():
        handed = real()

        def no_darkness(image):
            raise RuntimeError("no darkness")

        handed._darkness = no_darkness
        return handed

    monkeypatch.setattr(digitize, "sheet_curl_helpers", broken)
    printed = run_digitiser(data, tmp_path / "out", masks, *LINES)
    assert lines_of(printed, "bent", NOT_STRAIGHTENED) == [
        "WARNING: sheet curl not straightened for record bent (error: RuntimeError: "
        "no darkness), keeping the page as it is."
    ]
    assert qc_row(tmp_path / "out")[0]["sheet_curl"] == (
        "refused: error: RuntimeError: no darkness"
    )
    assert signal_files(tmp_path / "out", "bent") == signal_files(
        tmp_path / "plain", "bent"
    )
    assert plain.splitlines()[-1] == printed.splitlines()[-1] == "Done."


# -------------------------------------------------------------------- the scope
def paper_record(image, decision):
    """The paper normalisation record of a page that is the view its mask was
    predicted on: what --save_mask writes for it, with this decision."""
    digest = digitize.page_sha1(image)
    block = {
        "version": paper_normalisation.VERSION,
        "decision": decision,
        "reason": "",
        "input_size": [WIDTH, HEIGHT],
        "input_sha1": digest,
        "output_size": [WIDTH, HEIGHT],
        "view_sha1": digest,
    }
    if decision == "normalised":
        block["pre_size"] = [WIDTH, HEIGHT]
        block["warp"] = np.eye(3).tolist()
    return block


@pytest.fixture(scope="module")
def paper_masks(tmp_path_factory):
    """decision -> a folder with the mask of the bent page and a JSON whose paper
    normalisation record says that decision of the page."""
    root = tmp_path_factory.mktemp("sheet_curl_paper")
    folders = {}

    def folder(decision):
        if decision not in folders:
            folders[decision] = root / decision
            folders[decision].mkdir()
            write_png(mask_of("bent"), str(folders[decision] / "bent_mask.png"))
            meta = {"rot_angle": 0.0, "height": HEIGHT, "width": WIDTH}
            meta["paper_normalisation"] = paper_record(page("bent"), decision)
            (folders[decision] / "bent_mask.json").write_text(json.dumps(meta))
        return folders[decision]

    return folder


def test_a_page_that_was_not_normalised_is_out_of_scope(runs):
    # A mask without a paper record: the page is used as given, and is no page the
    # paper normalisation warped. The scope normalised leaves it alone, unmeasured.
    off, printed_off = runs("bent")
    out, printed = runs("bent", *NORMALISED)
    assert lines_of(printed, "bent", SAID) == [
        "Sheet curl for record bent: out of scope, not a page the paper normalisation "
        "warped onto the page frame"
    ]
    assert NOT_STRAIGHTENED not in printed
    row = qc_row(out)[0]
    assert row["paper_normalisation"] == "as given"
    assert row["sheet_curl"] == "out of scope"
    assert np.isnan(float(row["sheet_curl_shift_px"]))
    assert same_row(row, qc_row(off)[0], but=CURL_COLUMNS)
    assert other_lines(printed) == printed_off.splitlines()
    assert signal_files(out, "bent") == signal_files(off, "bent")
    block = mask_meta(out, "bent")["sheet_curl"]
    assert block["decision"] == "out of scope" and block["scope"] == "normalised"
    assert block["shift_max"] is None and "points" not in block
    # The default scope is every page, and all written out is that run.
    everywhere, printed_everywhere = runs("bent", *LINES)
    assert qc_row(everywhere)[0]["sheet_curl"] == "applied"
    named, printed_named = runs("bent", *LINES, "--sheet_curl_scope", "all")
    assert printed_named == printed_everywhere
    assert same_row(qc_row(named)[0], qc_row(everywhere)[0])
    assert signal_files(named, "bent") == signal_files(everywhere, "bent")
    # Nor is it measured with --paper_normalisation off.
    out, _ = runs("bent", *NORMALISED, "--paper_normalisation", "off")
    assert qc_row(out)[0]["paper_normalisation"] == "off"
    assert qc_row(out)[0]["sheet_curl"] == "out of scope"


def test_a_page_the_record_of_its_mask_calls_normalised_is_in_scope(runs, paper_masks):
    everywhere, _ = runs("bent", *LINES)
    # The decision is the one of the mask's record, replayed and not made again.
    masks = paper_masks("normalised")
    out, printed = runs("bent", *NORMALISED, masks=masks)
    row = qc_row(out)[0]
    assert row["paper_normalisation"] == "normalised" and row["sheet_curl"] == "applied"
    assert "Paper normalisation for record bent: normalised from the mask's record" in (
        printed
    )
    assert signal_files(out, "bent") == signal_files(everywhere, "bent")
    assert mask_meta(out, "bent")["sheet_curl"]["scope"] == "normalised"
    # The record of the paper is saved again next to the one of the stage.
    assert list(mask_meta(out, "bent")) == [
        "rot_angle", "height", "width", "paper_normalisation", "sheet_curl",
    ]
    # With --paper_normalisation off the record is still what says which page it is.
    off_flag, _ = runs("bent", *NORMALISED, "--paper_normalisation", "off", masks=masks)
    assert qc_row(off_flag)[0]["sheet_curl"] == "applied"


@pytest.mark.parametrize("decision", ["pass_chain", "pass_fullframe", "failed"])
def test_a_page_the_paper_normalisation_left_alone_is_out_of_scope(
    runs, paper_masks, decision
):
    out, printed = runs("bent", *NORMALISED, masks=paper_masks(decision))
    row = qc_row(out)[0]
    assert row["paper_normalisation"] == decision
    assert row["sheet_curl"] == "out of scope"
    assert lines_of(printed, "bent", SAID)[0].startswith(f"{SAID} bent: out of scope, ")
    # Unless the scope is every page.
    out, _ = runs("bent", *LINES, masks=paper_masks(decision))
    assert qc_row(out)[0]["sheet_curl"] == "applied"


def model_run(folder, name, decision, *flags):
    """run() on a drawn page without a mask: stand-ins for the paper normalisation,
    which says this decision of the page and leaves it as it is, and for the model,
    which returns the label mask of a flat page.

    Returns what was printed (log), the output folder (out) and the pages the model
    was asked about (seen).
    """
    data, out = folder / f"data_{name}", folder / "_".join(("out", name, decision))
    if not data.exists():
        data.mkdir(parents=True)
        write_png(page(name), str(data / f"{name}.png"))
    seen = []

    def normalise_paper(image):
        block = paper_record(image, decision)
        return image, block, {"decision": decision, "reason": ""}, None

    def model(
        image, dataset_name, model_folder, device="auto", disable_tta=False, fold="all"
    ):
        seen.append(image)
        # A copy: the label is a cached one, and run() owns the mask it is given.
        return label_mask(0.0).clone()

    log = io.StringIO()
    with pytest.MonkeyPatch.context() as patch:
        patch.setattr(digitize, "normalise_paper", normalise_paper)
        patch.setattr(digitize, "predict_mask_nnunet", model)
        argv = ["-d", str(data), "-o", str(out), *BASE, *flags]
        with contextlib.redirect_stdout(log):
            digitize.run(digitize.get_parser().parse_args(argv))
    return SimpleNamespace(log=log.getvalue(), out=out, seen=seen)


def test_the_model_is_asked_about_the_straightened_page(tmp_path):
    # Without a mask folder the paper normalisation decides now: a page it calls
    # normalised is in scope, and the segmentation sees the page after the stage.
    run = model_run(tmp_path, "bent", "normalised", *NORMALISED)
    row = qc_row(run.out)[0]
    assert row["paper_normalisation"] == "normalised" and row["sheet_curl"] == "applied"
    field, _ = measured("bent")
    assert len(run.seen) == 1 and np.array_equal(run.seen[0].numpy(), field.page)
    # The mask it returns lives on that page: it is saved as it is, with a record
    # that says so, and is not warped.
    assert torch.equal(read_image(str(run.out / "bent_mask.png")), label_mask(0.0))
    block = mask_meta(run.out, "bent")["sheet_curl"]
    assert block["decision"] == "applied" and len(block["points"]) == 63
    assert FITS_NOT not in run.log
    # A run on that mask and its two records gives the same signals.
    data = tmp_path / "data_bent"
    printed = run_digitiser(data, tmp_path / "again", run.out, *NORMALISED)
    assert FITS_NOT not in printed
    assert signal_files(tmp_path / "again", "bent") == signal_files(run.out, "bent")
    assert same_row(qc_row(tmp_path / "again")[0], row)

    # With the stage off the model sees the page as it is, as it always did.
    plain = model_run(tmp_path / "plain", "bent", "normalised")
    assert len(plain.seen) == 1 and torch.equal(plain.seen[0], page("bent"))
    assert "sheet_curl" not in mask_meta(plain.out, "bent")
    # And so it does for a page the paper normalisation left alone.
    alone = model_run(tmp_path / "alone", "bent", "pass_fullframe", *NORMALISED)
    assert torch.equal(alone.seen[0], page("bent"))
    assert qc_row(alone.out)[0]["sheet_curl"] == "out of scope"


# ----------------------------------------------------------------------------- QC
def test_append_qc_row_writes_the_two_columns_after_all_others(tmp_path):
    qc = {"perspective_residual_rel": 0.12, "sheet_curl": "refused: no lines, none"}
    qc["sheet_curl_shift_px"] = float("nan")
    digitize.append_qc_row(str(tmp_path), "rec", "column", qc, 0.0)
    moved = {"sheet_curl": "applied", "sheet_curl_shift_px": 12.5}
    digitize.append_qc_row(str(tmp_path), "moved", "column", moved, 0.0)
    # A row without them, as a caller of before the columns would write.
    digitize.append_qc_row(str(tmp_path), "old", "column", {"row_mapping": "off"}, 0.0)
    with open(tmp_path / "qc.csv", newline="") as f:
        reader = csv.DictReader(f)
        row, second, old = list(reader)
        # Appended at the end: no column that was there moves.
        assert reader.fieldnames[-3:] == [
            "perspective_residual_rel",
            "sheet_curl",
            "sheet_curl_shift_px",
        ]
        assert len(reader.fieldnames) == 37
        assert reader.fieldnames[:35] == OLD_COLUMNS
        assert reader.fieldnames.index("paper_normalisation") == 12
        assert reader.fieldnames.index("row_mapping") == 33
        assert reader.fieldnames.index("perspective_residual_rel") == 34
    # A reason with a comma is one cell.
    assert row["sheet_curl"] == "refused: no lines, none"
    assert np.isnan(float(row["sheet_curl_shift_px"]))
    assert second["sheet_curl"] == "applied"
    assert float(second["sheet_curl_shift_px"]) == 12.5
    assert old["sheet_curl"] == "" and old["sheet_curl_shift_px"] == ""


# ------------------------------------------------- 1. every page of a folder
# mutation: state_kept_between_pages
def test_every_page_of_a_folder_gets_its_own_decision(tmp_path, monkeypatch):
    # Two pages in one folder, read in this order. The first is a page the paper
    # normalisation warped, as the record of its mask says, and is straightened; the
    # second comes with a mask without a record and is out of scope: nothing of the
    # first page, neither its map nor its straightened pixels, reaches the second.
    data, masks = tmp_path / "data", tmp_path / "masks"
    data.mkdir()
    masks.mkdir()
    for name in ("first", "second"):
        write_png(page("bent"), str(data / f"{name}.png"))
        write_png(mask_of("bent"), str(masks / f"{name}_mask.png"))
    meta = {"rot_angle": 0.0, "height": HEIGHT, "width": WIDTH}
    meta["paper_normalisation"] = paper_record(page("bent"), "normalised")
    (masks / "first_mask.json").write_text(json.dumps(meta))
    listed = os.listdir
    monkeypatch.setattr(digitize.os, "listdir", lambda folder: sorted(listed(folder)))
    printed = run_digitiser(data, tmp_path / "out", masks, *NORMALISED)
    run_digitiser(data, tmp_path / "off", masks)
    with open(tmp_path / "out" / "qc.csv", newline="") as f:
        rows = list(csv.DictReader(f))
    assert [row["record"] for row in rows] == ["first", "second"]
    assert [row["sheet_curl"] for row in rows] == ["applied", "out of scope"]
    assert 12.0 < float(rows[0]["sheet_curl_shift_px"]) < 13.5
    assert np.isnan(float(rows[1]["sheet_curl_shift_px"]))
    assert FITS_NOT not in printed
    # The first page is straightened, the second is the page it was.
    out, off = tmp_path / "out", tmp_path / "off"
    assert signal_files(out, "first") != signal_files(off, "first")
    assert max(rows_spanned(read_image(str(out / "first_mask.png")))) <= 4
    assert signal_files(out, "second") == signal_files(off, "second")
    assert torch.equal(read_image(str(out / "second_mask.png")), mask_of("bent"))
    assert mask_meta(out, "first")["sheet_curl"]["decision"] == "applied"
    block = mask_meta(out, "second")["sheet_curl"]
    assert block["decision"] == "out of scope" and "points" not in block


# ------------------------------- 2. after a page the perspective stage warped
SHEAR = 0.02
# What takes the shear out of the page: the homography of the perspective stage.
UNSHEAR = np.array([[1.0, SHEAR, -SHEAR * HEIGHT / 2], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0]])


def sheared_page(amplitude):
    """The page of drawn_page(amplitude), photographed with a shear: it shows at (x, y)
    what the bent page shows at (x + SHEAR (y - H / 2), y)."""
    y, x = np.mgrid[0:HEIGHT, 0:WIDTH].astype(np.float64)
    on_sheet_x, on_sheet_y = bend(
        x + 0.5 + SHEAR * (y + 0.5 - HEIGHT / 2), y + 0.5, amplitude
    )

    def family(lines):
        thin = np.abs(lines - np.round(lines)) * PERIOD
        bold = np.abs(lines / 5 - np.round(lines / 5)) * 5 * PERIOD
        return 45 * np.exp(-0.5 * (thin / 0.6) ** 2) + 55 * np.exp(
            -0.5 * (bold / 0.8) ** 2
        )

    darkness = np.clip(family(on_sheet_x / PERIOD) + family(on_sheet_y / PERIOD), 0, 200)
    image = np.full((3, HEIGHT, WIDTH), 255, dtype=np.uint8)
    image[1] = image[2] = (255 - darkness).astype(np.uint8)
    return torch.from_numpy(image)


def recorded_results(monkeypatch, module, name):
    """Record every call of module.<name> with what it returned."""
    original, calls = getattr(module, name), []

    def recorder(*args, **kwargs):
        result = original(*args, **kwargs)
        calls.append((args, kwargs, result))
        return result

    monkeypatch.setattr(module, name, recorder)
    return calls


# mutation: measured_before_the_perspective
def test_the_stage_measures_the_page_the_perspective_stage_left(tmp_path, monkeypatch):
    # A sheet bent by 12 px, photographed with a shear. The stages before are stood in
    # for, so that what they leave is known: no rotation, and a perspective stage that
    # takes the shear out. The mask lives in the frame of that stage, as every mask.
    drawn = sheared_page(12.0)
    data, masks = tmp_path / "data", tmp_path / "masks"
    data.mkdir()
    masks.mkdir()
    write_png(drawn, str(data / "page.png"))
    write_png(label_mask(12.0), str(masks / "page_mask.png"))

    def no_rotation(image, method="hough"):
        info = {"reason": "", "deltas": [], "coarse": 0.0, "period": PERIOD}
        return 0.0, {**info, "contrast": 100.0, "residual": 0.0}

    def unshear(image, tolerance=None):
        info = {"shift": 8.5, "residual": 0.1, "residual_rel": 0.0127}
        return UNSHEAR.copy(), {**info, "period": PERIOD, "windows": 100, "reason": ""}

    monkeypatch.setattr(digitize, "estimate_rotation", no_rotation)
    monkeypatch.setattr(digitize, "estimate_perspective", unshear)
    asked = recorded_results(monkeypatch, sheet_curl, "straighten")
    flags = (*LINES, "--sheet_curl_min_shift", "2")
    printed = run_digitiser(data, tmp_path / "out", masks, *flags)
    # The page the stage is given is the warp of the perspective stage, not the page
    # that was read, and the map it finds there is the bend alone.
    (given, *_), _, (field, info) = asked[0]
    rectified = digitize.warp_page(drawn, UNSHEAR)
    assert torch.equal(given, rectified) and not torch.equal(given, drawn)
    assert info["decision"] == "applied" and 7.0 < info["shift_max"] < 8.5
    assert map_error(field, 12.0)[0] < 0.1
    # Measured on the page as it was read, the map holds the shear as well.
    whole, _ = sheet_curl.measure_field(drawn, {"min_shift": 2.0}, helpers())
    assert map_error(whole, 12.0)[0] > 1.0
    # The mask is saved straightened, with the frame of both stages next to it.
    out = tmp_path / "out"
    assert max(rows_spanned(read_image(str(out / "page_mask.png")))) <= 4
    meta = mask_meta(out, "page")
    assert list(meta) == ["rot_angle", "height", "width", "homography", "sheet_curl"]
    assert np.allclose(meta["homography"], UNSHEAR)
    assert meta["sheet_curl"]["decision"] == "applied"
    assert FITS_NOT not in printed
    # A second run on that mask fits both frames and gives the same signals.
    again = run_digitiser(data, tmp_path / "again", out, *flags)
    assert FITS_NOT not in again
    assert signal_files(tmp_path / "again", "page") == signal_files(out, "page")


# ----------------------------------------------------- 3. one map after another
# mutation: compose_at_the_wrong_point
def test_a_composed_map_is_the_second_map_at_the_points_of_the_first():
    # Two maps that do not commute, both linear, so that the lattice holds them
    # exactly: a shear along x that grows with y, then a stretch along x with a slope.
    step = sheet_curl.LATTICE
    shape = (math.ceil(HEIGHT / step) + 1, math.ceil(WIDTH / step) + 1)
    gx, gy = np.meshgrid(np.arange(shape[1]) * float(step), np.arange(shape[0]) * float(step))
    everywhere = np.ones(shape, bool)
    first = sheet_curl.Field(WIDTH, HEIGHT, step, 0.03 * gy, 0.0 * gy, everywhere)
    second = sheet_curl.Field(WIDTH, HEIGHT, step, 0.02 * gx, 0.01 * gx, everywhere)
    x, y = np.array([100.0, 500.0, 900.0]), np.array([200.0, 400.0, 100.0])
    one_x, one_y = first.straighten(x, y)
    two_x, two_y = second.straighten(one_x, one_y)
    X, Y = first.compose(second).straighten(x, y)
    assert np.allclose(X, two_x, atol=1e-9) and np.allclose(Y, two_y, atol=1e-9)
    # The other order is another map, by a tenth of a pixel and more here.
    other_x, other_y = second.compose(first).straighten(x, y)
    assert np.min(np.hypot(other_x - X, other_y - Y)) > 0.1
    # And so is the second map taken at the points of the page instead of at those
    # the first map made of them.
    assert np.max(np.abs(X - (x + 0.03 * y + 0.02 * x))) > 0.05


# -------------------------------- 4. the check on the straightened page, step 8
def measurements(monkeypatch, *answers):
    """Stand in for the measurements of a page, one answer per call:
    (shift_max, res_proj) of the map that measurement finds."""
    left = list(answers)

    def stand_in(D, array, opt):
        shift, res_proj = left.pop(0)
        info = sheet_curl._blank_info(opt)
        info.update(
            carrier="page", period=PERIOD, windows_u=500, windows_v=500, share=1.0,
            res_proj=res_proj, res_field=0.002, res_all=0.002, cover=1.0,
            proj_shift=0.0, shift_max=shift, shift_rms=shift, shift_all=shift,
            scale=1.0, jac_min=1.0, rounds=3,
        )
        return lattice_field(shift), info

    monkeypatch.setattr(sheet_curl, "_measure_once", stand_in)
    return left


# mutations: no_after_shift_check, no_after_residual_check, dead_band_le
def test_a_straightened_page_that_is_not_straight_is_refused(monkeypatch):
    def decided(*answers, **options):
        left = measurements(monkeypatch, *answers)
        field, info = sheet_curl.measure_field(page("flat"), options, helpers())
        assert left == []
        return field, info

    # The page asks for 20 px and the straightened page for nothing more: applied.
    field, info = decided((20.0, 0.3), (0.0, 0.002))
    assert info["decision"] == "applied" and field.page is not None
    assert (info["passes"], info["res_after"], info["shift_after"]) == (1, 0.002, 0.0)
    # The straightened page still asks for more than the dead band and more than a
    # quarter of what the page asked for, with no pass left: the map did not do it.
    field, info = decided((60.0, 0.3), (16.0, 0.002), passes=1)
    assert field is None and info["decision"] == "refused"
    assert info["reason"] == (
        "the straightened page asks for 16.0 px more, the page as it was for 60.0 px"
    )
    # A quarter of it or less, above the dead band, is let through.
    field, info = decided((60.0, 0.3), (14.0, 0.002), passes=1)
    assert info["decision"] == "applied" and info["shift_after"] == 14.0
    # Lines that are not straight after the warp: above the tolerance ...
    field, info = decided((20.0, 0.3), (0.0, 0.26))
    assert field is None and info["reason"] == (
        "lines are not straight after the warp (0.2600 periods off a projective fit, "
        "0.3000 before, 0.2500 allowed)"
    )
    # ... or less straight than the measurement before could tell.
    field, info = decided((20.0, 0.01), (0.0, 0.02))
    assert field is None and info["reason"] == (
        "lines are not straight after the warp (0.0200 periods off a projective fit, "
        "0.0100 before, 0.0135 allowed)"
    )
    # The threshold of the estimator: a first map that moves a point by exactly that
    # much is applied, a hair below it is the dead band.
    field, info = decided((10.0, 0.3), (0.0, 0.002), min_shift=10.0)
    assert info["decision"] == "applied"
    field, info = decided((10.0, 0.3), min_shift=float(np.nextafter(10.0, 11.0)))
    assert info["decision"] == "dead band" and field.page is None


# ------------------------------------------- 5. a grid on a part of the page
def part_of_the_grid(kept):
    """The bent page with its grid lines only where kept(x / W, y / H) says so."""
    y, x = np.mgrid[0:HEIGHT, 0:WIDTH]
    image = page("bent").clone()
    image[:, torch.from_numpy(~kept(x / WIDTH, y / HEIGHT))] = 255
    return image


# mutations: no_span_check_b, no_cover_check, no_inverse_check
def test_a_grid_on_a_part_of_the_page_carries_no_map_of_the_page():
    # Lines on the left 40 % of the page: neither family spans half of its width.
    left = part_of_the_grid(lambda x, y: x < 0.40)
    field, info = sheet_curl.measure_field(left, None, helpers())
    assert field is None and info["decision"] == "refused"
    assert info["reason"] == "vertical grid lines only span 434 x 800 px"
    # Lines on a band along the diagonal: both families span the page, and what
    # their windows cover is less than half of it (MIN_COVER).
    band = part_of_the_grid(lambda x, y: np.abs(y - x) < 0.15)
    field, info = sheet_curl.measure_field(band, None, helpers())
    assert field is None and info["decision"] == "refused"
    assert info["reason"] == "grid lines cover 42 % of the page"
    assert 0.40 < info["cover"] < sheet_curl.MIN_COVER
    # A wider band is enough for a map.
    wide = part_of_the_grid(lambda x, y: np.abs(y - x) < 0.30)
    field, info = sheet_curl.measure_field(wide, None, helpers())
    assert info["decision"] == "applied" and info["cover"] > sheet_curl.MIN_COVER
    # Lines everywhere but in one corner: the smooth part is held there from the
    # nearest node with lines, the two sides of the corner disagree, and the map has
    # no inverse the warp could use.
    corner = part_of_the_grid(lambda x, y: (x < 0.7) | (y < 0.7))
    field, info = sheet_curl.measure_field(corner, None, helpers())
    assert field is None and info["decision"] == "refused"
    assert info["reason"].startswith(
        "the map stretches the page too much for the inverse to converge ("
    )
    assert info["inverse"] > sheet_curl.INVERSE_TOLERANCE and info["cover"] > 0.85


# ----------------------------------------------------------- further, small ones
# mutation: scope_scaled_too
def test_a_page_the_paper_normalisation_only_rescaled_is_out_of_scope(tmp_path):
    run = model_run(tmp_path, "bent", "scaled", *NORMALISED)
    row = qc_row(run.out)[0]
    assert row["paper_normalisation"] == "scaled"
    assert row["sheet_curl"] == "out of scope"
    assert torch.equal(run.seen[0], page("bent"))


# ------------------------------ 0. a fit between the two tolerances, a known map
def blurred_page(amplitude, sigma):
    """The drawn page bent by amplitude px, out of focus by a Gaussian of sigma px."""
    channels = drawn_page(amplitude, True).numpy()
    return np.stack([cv2.GaussianBlur(channel, (0, 0), sigma) for channel in channels])


def test_a_fit_between_the_two_tolerances_is_not_the_map_of_the_page():
    # A sheet bent by 38 px on this quarter of a page, out of focus: the windows are
    # put a line off somewhere, and the fit that comes of it is 0.17 periods off its
    # lines (0.29 is no relation at all). The tolerance of the grid line stages (0.1)
    # refuses it.
    image = blurred_page(38.0, 2.5)
    field, info = sheet_curl.measure_field(image, {"tol": 0.1}, helpers())
    assert field is None and info["decision"] == "refused"
    assert info["reason"] == "grid line phase is 0.169 periods off the field"
    # KNOWN LIMIT, pinned so that a change of it shows: the default tolerance (0.25)
    # lets that fit through, the check on the straightened page passes as well, and
    # the map that is applied is more than 5 px RMS and 15 px at the worst point off
    # the known one (a right map is a fraction of a pixel off, see the bend of 36 px
    # below).
    field, info = sheet_curl.measure_field(image, None, helpers())
    assert info["decision"] == "applied"
    assert 0.15 < info["res_field"] < 0.19 and 0.15 < info["res_after"] < 0.20
    rms, worst = map_error(field, 38.0)
    assert rms > 5.0 and worst > 15.0
    # A bend of 36 px with the same blur is read right, with a fit ten times closer:
    # the residual does tell the two apart, the default tolerance does not.
    field, info = sheet_curl.measure_field(blurred_page(36.0, 2.5), None, helpers())
    assert info["decision"] == "applied" and info["res_field"] < 0.02
    assert map_error(field, 36.0)[0] < 0.3


# ----------------------------------- 6. the two other clauses of the dead band
# mutations: dead_band_no_second_clause, dead_band_shift_all
def test_the_dead_band_reads_the_measured_nodes_and_what_the_smooth_part_explains(
    monkeypatch,
):
    def decided(first, *later):
        """The decision when the measurements of a page say this, one dict per
        measurement, and how many of them were not asked for."""
        left = [first, *later]

        def stand_in(D, array, opt):
            info = sheet_curl._blank_info(opt)
            info.update(
                carrier="page", period=PERIOD, windows_u=500, windows_v=500, share=1.0,
                res_proj=0.3, res_field=0.002, res_all=0.002, cover=0.9,
                proj_shift=0.0, shift_max=0.0, shift_rms=0.0, shift_all=0.0, scale=1.0,
                jac_min=1.0, rounds=3,
            )
            info.update(left.pop(0))
            return lattice_field(info["shift_max"]), info

        monkeypatch.setattr(sheet_curl, "_measure_once", stand_in)
        # The threshold of the estimator, here 10 px: these are its clauses.
        options = {"min_shift": 10.0}
        field, info = sheet_curl.measure_field(page("flat"), options, helpers())
        return info["decision"], len(left)

    # What the straightened page says when nothing is left to ask for.
    straight = {"res_proj": 0.002}
    # The dead band is about the nodes the lines were measured at: a map that moves
    # the page by 20 px only where it is held from the nearest measured node, and by
    # 5 px where the lines are, is inside it, and nothing is measured a second time.
    assert decided({"shift_max": 5.0, "shift_all": 20.0}, straight) == ("dead band", 1)
    # A smooth part that explains the lines no better than the projective part is
    # nothing to apply, whatever it moves, while that part moves less than the band.
    no_better = {"shift_max": 20.0, "shift_all": 20.0, "res_proj": 0.002}
    assert decided({**no_better, "proj_shift": 3.0}, straight) == ("dead band", 1)
    # With a projective part that moves the page by more than the band it is applied,
    # and so is a smooth part that does explain the lines.
    assert decided({**no_better, "proj_shift": 12.0}, straight) == ("applied", 0)
    better = {"shift_max": 20.0, "shift_all": 20.0}
    assert decided(better, straight) == ("applied", 0)


# ------------------------------------------- 7. the second candidate of the carrier
# mutation: carrier_page_only
def test_a_blurred_bent_page_is_read_on_the_comb_of_its_windows():
    # Out of focus and bent, the 1 mm lines are not coherent over the page: the
    # strongest comb of the page-wide profile is a harmonic of the bold lines, a third
    # of 5 mm, where the windows have no comb. The windows carry the 1 mm lines.
    image = blurred_page(20.0, 2.5)
    darkness = digitize._darkness(torch.from_numpy(image))
    page_wide = digitize._grid_line_period(digitize._band_profiles(darkness)[0])
    assert page_wide == pytest.approx(5 * PERIOD / 3, rel=0.01)
    field, info = sheet_curl.measure_field(image, None, helpers())
    assert info["carrier"] == "windows"
    assert info["period"] == pytest.approx(PERIOD, rel=1e-3)
    assert info["decision"] == "applied" and info["res_field"] < 0.01
    assert map_error(field, 20.0)[0] < 0.1
    # The same page in focus is read on the page-wide period.
    _, sharp = sheet_curl.measure_field(page("bent"), None, helpers())
    assert sharp["carrier"] == "page"


# ------------------------------------------------ 8. the judge of a composed map
# mutation: no_judge_after_compose
def test_a_composed_map_is_judged_as_the_first_one_is(monkeypatch):
    # The page asks for 40 px, the straightened page for 20 px more: composed, the map
    # asks for 60 px, more than PERSPECTIVE_MAX_SHIFT of the page width (55 px).
    measurements(monkeypatch, (40.0, 0.3), (20.0, 0.002), (0.0, 0.002))
    field, info = sheet_curl.measure_field(page("flat"), None, helpers())
    assert field is None and info["decision"] == "refused" and info["passes"] == 2
    assert info["reason"] == (
        "grid lines ask for a 60 px correction somewhere on the page (60 px where "
        "they were measured)"
    )


# ------------------------------------------ 9. the judge of the first map, a page
# mutation: judge_reason_ignored
def test_a_page_whose_first_map_tears_is_refused_before_anything_is_warped():
    # Lines everywhere but in two opposite corners: the smooth part is held there
    # from the nearest node with lines, and the map squeezes a cell of the lattice to
    # a quarter where the two sides of a corner meet.
    torn = part_of_the_grid(
        lambda x, y: ((x < 0.7) | (y < 0.7)) & ((x > 0.3) | (y > 0.3))
    )
    field, info = sheet_curl.measure_field(torn, None, helpers())
    assert field is None and info["decision"] == "refused" and info["passes"] == 0
    assert info["reason"].startswith(
        "the map stretches or squeezes the page by more than a factor of two "
        "somewhere (Jacobian determinant 0.2"
    )
    assert info["jac_min"] < sheet_curl.JACOBIAN_FLOOR and info["cover"] > 0.75
    assert np.isnan(info["inverse"]) and np.isnan(info["res_after"])


# ------------------------------------- the record of a mask that is saved again
@pytest.mark.parametrize(
    "flags",
    [
        (),
        NORMALISED,
        (*LINES, "--sheet_curl_min_shift", "50"),
        (*LINES, "--sheet_curl_tolerance", "0.0005"),
    ],
    ids=["off", "out of scope", "dead band", "refused"],
)
def test_a_mask_that_does_not_fit_keeps_its_record_when_it_is_saved_again(runs, flags):
    # A run that keeps the page, on a mask of the straightened page: it says that the
    # mask does not fit, and saves it again.
    first, _ = runs("bent", *LINES)
    kept, printed = runs("bent", *flags, masks=first)
    assert printed.count(FITS_NOT) == 1
    # The mask that is saved is the one that was given, and it lives on a straightened
    # page: what is saved next to it has to say so still, and not what this run did
    # with the page (nothing, with the stage off).
    assert (kept / "bent_mask.png").read_bytes() == (first / "bent_mask.png").read_bytes()
    assert mask_meta(kept, "bent")["sheet_curl"] == mask_meta(first, "bent")["sheet_curl"]
    assert mask_meta(kept, "bent")["sheet_curl"]["decision"] == "applied"
    # So that a later run that straightens the page uses the mask as it is, and does
    # not warp it a second time without a word.
    again, printed = runs("bent", *LINES, masks=kept)
    assert FITS_NOT not in printed
    assert signal_files(again, "bent") == signal_files(first, "bent")


def test_a_record_that_cannot_be_read_is_said_and_the_mask_is_not_warped(
    runs, tmp_path
):
    first, _ = runs("bent", *LINES)
    bare, _ = runs("bent", *LINES)

    def with_record(name, change, mask=first):
        folder = tmp_path / name
        folder.mkdir()
        shutil.copy(mask / "bent_mask.png", folder / "bent_mask.png")
        meta = mask_meta(first, "bent")
        change(meta)
        (folder / "bent_mask.json").write_text(json.dumps(meta))
        return folder

    def a_string(meta):
        meta["sheet_curl"] = "applied"

    def one_point(meta):
        meta["sheet_curl"]["points"] = meta["sheet_curl"]["points"][:1]

    def no_size(meta):
        del meta["sheet_curl"]["page_size"]

    # A record that is no object, and an applied record without its whole map: the
    # WARNING of the frame, and the mask of the straightened page is used as it is.
    for name, change, problem in (
        ("string", a_string, "has a sheet curl record that is no object"),
        ("point", one_point, "has a sheet curl record without the points of its map"),
        ("size", no_size, "has a sheet curl record without the points of its map"),
    ):
        out, printed = runs("bent", *LINES, masks=with_record(name, change))
        assert lines_of(printed, "bent", "WARNING: mask of record") == [
            f"WARNING: mask of record bent {problem}; {FITS_NOT}"
        ], name
        assert signal_files(out, "bent") == signal_files(first, "bent"), name
        # Saved again, it keeps what it came with: the next run says it again.
        saved = mask_meta(out, "bent")["sheet_curl"]
        assert saved == mask_meta(tmp_path / name, "bent")["sheet_curl"], name

    # null is no record: the mask of the page as it was is warped with the page, as a
    # mask without a JSON is, and nothing is said.
    def null(meta):
        meta["sheet_curl"] = None

    given = tmp_path / "given"
    given.mkdir()
    write_png(mask_of("bent"), str(given / "bent_mask.png"))
    out, printed = runs("bent", *LINES, masks=with_record("null", null, mask=given))
    assert FITS_NOT not in printed
    assert signal_files(out, "bent") == signal_files(bare, "bent")
    assert mask_meta(out, "bent")["sheet_curl"]["decision"] == "applied"


# ------------------------------------------------- the dead band of the stage
def test_the_dead_band_of_the_stage_is_taken_on_the_map_it_would_apply(monkeypatch):
    asked = recorded(monkeypatch, sheet_curl, "measure_field")

    def staged(band, *answers, **options):
        left = measurements(monkeypatch, *answers)
        field, info = sheet_curl.straighten(page("flat"), options, helpers(), band)
        assert left == []
        return field, info

    # A map that moves a measured point by exactly the dead band is applied ...
    field, info = staged(15.0, (15.0, 0.3), (0.0, 0.002))
    assert info["decision"] == "applied" and field.page is not None
    assert "accepted" not in info
    # ... and the estimator was asked with its own threshold, not with the dead band.
    assert asked[-1][0][1]["min_shift"] == sheet_curl.ESTIMATOR_MIN_SHIFT_PX == 1.0
    # A hair below it the page is left as it is: the map was measured, warped by,
    # checked and accepted, and the stage hands on no field.
    below = float(np.nextafter(15.0, 0.0))
    field, info = staged(15.0, (below, 0.3), (0.0, 0.002))
    assert field is None and info["decision"] == "dead band" and info["accepted"] is True
    assert info["shift_max"] == below and info["passes"] == 1
    assert (info["res_after"], info["shift_after"]) == (0.002, 0.0)
    assert sheet_curl.qc_values(info) == ("dead band", below)
    block = sheet_curl.mask_record(info, field, None, "all", 15.0)
    assert block["decision"] == "dead band" and block["shift_max"] == below
    assert block["dead_band"] == 15.0 and "points" not in block
    assert sheet_curl.verbose_line("rec", info).startswith(
        "Sheet curl for record rec: dead band of the accepted map, carrier page "
    )
    # The dead band is taken on the FINAL map. A first map of 8 px, below the dead
    # band, whose straightened page asks for 5 px and then for 4 px more: composed
    # over three passes the map moves the page by 17 px, and it is applied.
    answers = ((8.0, 0.3), (5.0, 0.002), (4.0, 0.002), (0.0, 0.002))
    field, info = staged(15.0, *answers)
    assert info["decision"] == "applied" and info["passes"] == 3
    assert info["shift_max"] == pytest.approx(17.0) and field.page is not None
    # With two passes the same page ends at 13 px with 4 px left to ask for, more than
    # the check lets pass (a quarter of 13 px): refused, and a refusal is a refusal
    # whatever the dead band is, also for a map that is inside it.
    field, info = staged(15.0, *answers[:3], passes=2)
    assert field is None and info["decision"] == "refused"
    assert info["reason"] == (
        "the straightened page asks for 4.0 px more, the page as it was for 13.0 px"
    )
    # The threshold of the estimator ends the passes, not the dead band: a
    # straightened page that asks for 5 px more gets them, below a dead band of 15 px.
    field, info = staged(15.0, (20.0, 0.3), (5.0, 0.002), (0.0, 0.002))
    assert info["decision"] == "applied" and info["passes"] == 2
    assert info["shift_max"] == pytest.approx(25.0)
    # The first measurement inside the threshold of the estimator is its dead band.
    field, info = staged(15.0, (0.6, 0.3))
    assert field is None and info["decision"] == "dead band" and "accepted" not in info
    # A dead band at or below that threshold IS the threshold: the stage is
    # measure_field() with it and nothing else.
    field, info = staged(0.5, (0.7, 0.3), (0.0, 0.002))
    assert info["decision"] == "applied" and asked[-1][0][1]["min_shift"] == 0.5
    field, info = staged(0.5, (0.4, 0.3))
    assert field is None and info["decision"] == "dead band" and "accepted" not in info
    field, info = staged(1.0, (1.0, 0.3), (0.0, 0.002))
    assert info["decision"] == "applied" and asked[-1][0][1]["min_shift"] == 1.0
    # A threshold given with the options is kept, below the dead band.
    field, info = staged(15.0, (16.0, 0.3), (2.0, 0.002), min_shift=3.0)
    assert info["decision"] == "applied" and info["passes"] == 1
    assert asked[-1][0][1]["min_shift"] == 3.0
    assert sheet_curl.estimator_options(None)["min_shift"] == 1.0
    assert sheet_curl.estimator_options({"min_shift": 3.0}, 2.0)["min_shift"] == 2.0


def test_a_page_inside_the_default_dead_band_is_the_page_it_was(runs):
    # The stage with nothing but its defaults: every page in scope, a dead band of
    # 15 px. The page bent by 20 px has a good map of 12.6 px and is left alone.
    off, printed_off = runs("bent")
    out, printed = runs("bent", "--sheet_curl", "lines")
    row = qc_row(out)[0]
    assert row["sheet_curl"] == "dead band"
    assert 12.0 < float(row["sheet_curl_shift_px"]) < 13.5
    said = lines_of(printed, "bent", SAID)
    assert len(said) == 1 and said[0].startswith(
        "Sheet curl for record bent: dead band of the accepted map, carrier page "
    )
    assert NOT_STRAIGHTENED not in printed and FITS_NOT not in printed
    assert other_lines(printed) == printed_off.splitlines()
    assert signal_files(out, "bent") == signal_files(off, "bent")
    assert torch.equal(read_image(str(out / "bent_mask.png")), mask_of("bent"))
    assert same_row(row, qc_row(off)[0], but=CURL_COLUMNS)
    block = mask_meta(out, "bent")["sheet_curl"]
    assert block["decision"] == "dead band" and block["scope"] == "all"
    assert block["dead_band"] == 15.0 and "points" not in block
    # Exactly its own shift as the dead band, the page is applied; a hair above, not.
    shift = block["shift_max"]
    at, _ = runs("bent", "--sheet_curl", "lines", "--sheet_curl_min_shift", repr(shift))
    assert qc_row(at)[0]["sheet_curl"] == "applied"
    assert signal_files(at, "bent") == signal_files(runs("bent", *LINES)[0], "bent")
    above = repr(float(np.nextafter(shift, 20.0)))
    over, _ = runs("bent", "--sheet_curl", "lines", "--sheet_curl_min_shift", above)
    assert qc_row(over)[0]["sheet_curl"] == "dead band"
    # A mask saved for the page straightened at 10 px does not fit the default run.
    _, printed = runs("bent", "--sheet_curl", "lines", masks=runs("bent", *LINES)[0])
    assert lines_of(printed, "bent", "WARNING: mask of record") == [
        "WARNING: mask of record bent was predicted on a page whose sheet curl was "
        f"straightened, the page is now kept as it is (dead band); {FITS_NOT}"
    ]


# ------------------------------------------------ the model path, a page that is kept
def test_the_model_is_asked_about_the_page_as_it_is_when_the_stage_keeps_it(tmp_path):
    # Without a mask folder, a page the stage measures and leaves alone: the model
    # sees the page as it is, its mask is saved as it is, and the record says so.
    kept = model_run(tmp_path, "slight", "normalised", *NORMALISED)
    assert qc_row(kept.out)[0]["sheet_curl"] == "dead band"
    assert len(kept.seen) == 1 and torch.equal(kept.seen[0], page("slight"))
    assert torch.equal(read_image(str(kept.out / "slight_mask.png")), label_mask(0.0))
    block = mask_meta(kept.out, "slight")["sheet_curl"]
    assert block["decision"] == "dead band" and "points" not in block
    # And a page it refuses.
    narrow = (*NORMALISED, "--sheet_curl_tolerance", "0.0005")
    refused = model_run(tmp_path / "refused", "bent", "normalised", *narrow)
    assert qc_row(refused.out)[0]["sheet_curl"].startswith("refused: grid line phase")
    assert len(refused.seen) == 1 and torch.equal(refused.seen[0], page("bent"))
    assert refused.log.count(NOT_STRAIGHTENED) == 1
    block = mask_meta(refused.out, "bent")["sheet_curl"]
    assert block["decision"] == "refused" and "points" not in block
    # A run on that mask and its record, with the same flags, gives the same signals.
    data = tmp_path / "refused" / "data_bent"
    printed = run_digitiser(data, tmp_path / "again", refused.out, *narrow)
    assert FITS_NOT not in printed
    assert signal_files(tmp_path / "again", "bent") == signal_files(refused.out, "bent")


# -------------------------------------------------------- the default reading
def shown_bent(image, mask, amplitude):
    """A page and its label mask as a sheet shows them that is bent by amplitude px:
    half a sine across each side. Resampled, the page bicubic and the mask nearest."""
    height, width = image.shape[1:]
    y, x = np.mgrid[0:height, 0:width].astype(np.float32)
    source_x = x + 0.6 * amplitude * np.sin(np.pi * (y + 0.5) / height)
    source_y = y + 0.8 * amplitude * np.sin(np.pi * (x + 0.5) / width)
    channels = np.ascontiguousarray(np.moveaxis(image.numpy(), 0, 2))
    shown = cv2.remap(
        channels, source_x, source_y, cv2.INTER_CUBIC, borderMode=cv2.BORDER_REPLICATE
    )
    label = cv2.remap(mask.numpy()[0], source_x, source_y, cv2.INTER_NEAREST)
    shown = np.ascontiguousarray(np.moveaxis(shown, 2, 0))
    return torch.from_numpy(shown), torch.from_numpy(label[None])


def read_by_default(folder, image, mask, *flags):
    """run() with every flag at its default but these: (signals, QC row, printed)."""
    data, masks, out = folder / "data", folder / "masks", folder / "out"
    data.mkdir(parents=True)
    masks.mkdir()
    write_png(image, str(data / "page.png"))
    write_png(mask, str(masks / "page_mask.png"))
    argv = ["-d", str(data), "-o", str(out), "--mask_folder", str(masks), *flags]
    printed = io.StringIO()
    with contextlib.redirect_stdout(printed):
        digitize.run(digitize.get_parser().parse_args(argv))
    with open(out / "qc.csv", newline="") as f:
        row = next(csv.DictReader(f))
    return wfdb.rdrecord(str(out / "page")).p_signal, row, printed.getvalue()


def snr_against(reference, signal):
    """dB of every lead of a reading against another one, sample by sample."""
    found = []
    for lead in range(reference.shape[1]):
        both = np.isfinite(reference[:, lead]) & np.isfinite(signal[:, lead])
        noise = signal[both, lead] - reference[both, lead]
        found.append(
            10 * np.log10(np.sum(reference[both, lead] ** 2) / np.sum(noise**2))
        )
    return np.array(found)


def test_a_bent_page_is_read_as_the_flat_one_under_the_default_reading(tmp_path):
    # Every other end to end test reads with --time_mapping bbox, on a quarter page
    # without a column grid. Here: the full-size drawn page of the row mapping tests
    # with its traces, shown through a bend of 40 px, read with every flag at its
    # default (grid time mapping, column map, row maps) and --sheet_curl lines alone.
    import test_row_mapping as rows

    image, mask, _ = rows.drawn_page(**{**rows.PAGES["even"], "snap": 0.5})
    flat, flat_row, _ = read_by_default(tmp_path / "flat", image, mask)
    bent, label = shown_bent(image, mask, 40.0)
    # With the stage off the bend is read as signal and as time: the page is lost.
    off, off_row, printed_off = read_by_default(tmp_path / "off", bent, label)
    assert off_row["sheet_curl"] == "off" and np.nanmedian(snr_against(flat, off)) < 5.0
    # Straightened, every lead is the lead of the flat page to 15 dB and more, on the
    # same column grid. The map moves the page by about 25.5 px, far from the dead
    # band of 15 px on either side.
    on, row, printed = read_by_default(tmp_path / "on", bent, label, "--sheet_curl", "lines")
    assert row["sheet_curl"] == "applied"
    assert 23.0 < float(row["sheet_curl_shift_px"]) < 28.0
    gained = snr_against(flat, on)
    assert np.nanmin(gained) > 15.0 and np.nanmedian(gained) > 20.0
    assert row["column_mapping"] == flat_row["column_mapping"]
    assert "Einthoven check failed" not in printed
    assert "column mapping not used" not in printed
