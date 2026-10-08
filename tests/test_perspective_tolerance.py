"""Unit tests for --perspective_tolerance in src/run/digitize.py.

estimate_perspective() refuses the homography of the grid lines when the residual of
a fit is above GRID_LINE_MAX_SLOPE_RESIDUAL, a tenth, of the carrier period. The flag
puts another tolerance into that test, for the perspective stage of run() alone: the
rotation stage reads the constant, and so does the chain of the paper normalisation.
The default of the flag is PERSPECTIVE_TOLERANCE, 0.15, so a run that names no flag
widens the stage, and --perspective_tolerance 0.1 is the stage at its own tolerance,
the run before 0.15 became the default.
No model and no data files: the pages are drawn here, a quarter of a page with a
1 mm / 5 mm grid at 200 dpi and no trace, whose line pitch breathes or whose lines
wave as on a printed and scanned sheet, which no homography describes and which the
fit is therefore left with as its residual.
"""
import argparse
import contextlib
import csv
import inspect
import io
import json
import re
from functools import lru_cache
from types import SimpleNamespace

import numpy as np
import pytest
import torch
from torchvision.io.image import write_png

from config import LEAD_LABEL_MAPPING, SHORT_SIGNAL_LENGTH_SEC
from src.run import digitize

# ------------------------------------------------------------------ the drawn page
# A quarter of a page with the grid of a whole one: at the 200 dpi scale, and cheap.
WIDTH, HEIGHT = 1100, 850
PERIOD = 200 / 25.4  # px per printed millimetre at 200 dpi
SWINGS = 1.5  # how often a displacement swings over the page
STAGE = digitize.GRID_LINE_MAX_SLOPE_RESIDUAL  # the tolerance of the stage itself
FLAG = "--perspective_tolerance"
FOLDERS = ["-d", "in", "-o", "out"]


@lru_cache(maxsize=None)
def drawn_page(along=0.0, across=0.0, shear=0.0, lines=True):
    """A drawn page [3, H, W] uint8: a red 1 mm grid, every fifth line bold, no trace.

    along moves the lines of both families along their normal by that many pixels, in
    a sine that swings 1.5 times over the page: a pitch that breathes. The lines stay
    straight, so the rotation stage reads the page, while the windows of the
    perspective fit are off every projective map. across moves the lines by a sine of
    the position along them: lines that wave, which the rotation stage is left with as
    well. shear leans the vertical lines by that many pixels per pixel of height.
    Without lines the page is blank. Cached, so no caller may write into a page.
    """
    y, x = np.mgrid[0:HEIGHT, 0:WIDTH].astype(np.float64)
    x, y = x + 0.5, y + 0.5
    swing_x = np.sin(2 * np.pi * SWINGS * x / WIDTH)
    swing_y = np.sin(2 * np.pi * SWINGS * y / HEIGHT)
    u = (x + along * swing_x + across * swing_y + shear * (y - HEIGHT / 2)) / PERIOD
    v = (y + along * swing_y + across * swing_x) / PERIOD

    def family(lines):
        thin = np.abs(lines - np.round(lines)) * PERIOD
        bold = np.abs(lines / 5 - np.round(lines / 5)) * 5 * PERIOD
        return 45 * np.exp(-0.5 * (thin / 0.6) ** 2) + 55 * np.exp(
            -0.5 * (bold / 0.8) ** 2
        )

    darkness = np.clip(family(u) + family(v), 0, 200) * bool(lines)
    image = np.full((3, HEIGHT, WIDTH), 255, dtype=np.uint8)
    # Red lines, as on a printed page: darkest in the two lower channels.
    image[1] = image[2] = (255 - darkness).astype(np.uint8)
    return torch.from_numpy(image)


# name -> the arguments of drawn_page(). What the stages make of each is pinned in
# test_the_pages_are_what_they_are_drawn_for.
PAGES = {
    # Straight and even: the dead band of the stage.
    "even": {},
    # A pitch that breathes by 1.2 px leaves the fit 0.083 of the period: inside the
    # tolerance of the stage, which rectifies the page by 4 px.
    "below": {"along": 1.2},
    # By 1.6 px: 0.110 of the period in the first round, 0.111 in the second.
    "above": {"along": 1.6},
    # The same on a sheet that leans by a quarter of a degree: the rotation stage
    # turns the page, and the perspective is measured on the turned one.
    "leaning": {"along": 1.6, "shear": 0.004},
    # Lines that wave by 1.5 px: 0.127 of the period off a line for the rotation
    # stage, 0.126 off a fit that asks for no correction (0.01 px).
    "waved": {"across": 1.5},
    # No grid at all: no stage finds a period, at any tolerance.
    "blank": {"lines": False},
}
OFF_THE_FIT = r"grid line phase is \d\.\d\d px off the fit"
OFF_A_LINE = r"grid line phase is \d\.\d\d px off a line"
# A tolerance above the residual of every page here, and one between the tolerance of
# the stage and the residual of the pages that are above it.
WIDE, NARROW = 0.15, 0.105


def page(name):
    return drawn_page(**PAGES[name])


@lru_cache(maxsize=None)
def label_mask():
    """A flat trace per lead in the column of the standard layout, [1, H, W] uint8."""
    label = np.zeros((1, HEIGHT, WIDTH), dtype=np.uint8)
    pitch = WIDTH / 5
    for lead, value in LEAD_LABEL_MAPPING.items():
        column = int(digitize.STANDARD_LEAD_OFFSETS_SEC[lead] / SHORT_SIGNAL_LENGTH_SEC)
        row = HEIGHT // 6 + HEIGHT // 17 * (value % 12)
        start = int(pitch / 2 + column * pitch)
        label[0, row, start : start + int(pitch)] = value
    return torch.from_numpy(label)


def same_number(a, b):
    """Two numbers agree to nine digits, NaN being NaN: the fit is not bit for bit the
    same in two calls on every machine."""
    return (a != a and b != b) or a == pytest.approx(b, rel=1e-9, abs=1e-9)


def same_info(a, b):
    """Two info dicts hold the same keys and values."""
    return a.keys() == b.keys() and all(
        same_number(a[key], b[key]) if isinstance(a[key], float) else a[key] == b[key]
        for key in a
    )


def recorded(monkeypatch, name):
    """Record every call of digitize.<name> and pass it on: [(args, kwargs), ...]."""
    original, calls = getattr(digitize, name), []

    def recorder(*args, **kwargs):
        calls.append((args, kwargs))
        return original(*args, **kwargs)

    monkeypatch.setattr(digitize, name, recorder)
    return calls


# ------------------------------------------------------------ flag and constants
def test_the_flag_is_0_15_unless_another_tolerance_is_given():
    parse = digitize.get_parser().parse_args
    # The stage keeps its tenth: the default of the flag is above it, and in the range.
    assert STAGE == 0.1
    assert digitize.PERSPECTIVE_TOLERANCE == 0.15
    assert digitize.PERSPECTIVE_MAX_TOLERANCE == 0.5
    default = parse(FOLDERS).perspective_tolerance
    assert default == 0.15 and isinstance(default, float)
    assert default == digitize.PERSPECTIVE_TOLERANCE
    assert parse(FOLDERS + [FLAG, "0.15"]).perspective_tolerance == 0.15
    # Both ends of the range are in it.
    assert parse(FOLDERS + [FLAG, "0.1"]).perspective_tolerance == STAGE
    assert parse(FOLDERS + [FLAG, "0.5"]).perspective_tolerance == 0.5


def test_a_run_that_does_not_name_the_flag_has_the_arguments_of_one_with_0_15():
    parse = digitize.get_parser().parse_args
    assert parse(FOLDERS) == parse(FOLDERS + [FLAG, "0.15"])
    assert vars(parse(FOLDERS)) == vars(parse(FOLDERS + [FLAG, "1.5e-1"]))
    # The tenth written out is another run, in that argument and in no other.
    tenth = parse(FOLDERS + [FLAG, "0.1"])
    assert parse(FOLDERS) != tenth
    default = vars(parse(FOLDERS))
    assert {key for key in default if default[key] != vars(tenth)[key]} == {
        "perspective_tolerance"
    }


def test_the_default_of_the_flag_is_no_default_of_the_fit():
    # 0.15 is what the command line gives a run that names no flag, and nothing else:
    # the constant of the stage, which the rotation stage and the chain of the paper
    # normalisation read, stays a tenth, and the fit that is called without a
    # tolerance refuses a page between the two, as a caller of the function had it.
    assert digitize.GRID_LINE_MAX_SLOPE_RESIDUAL == 0.1 < digitize.PERSPECTIVE_TOLERANCE
    refused, info = digitize.estimate_perspective(page("above"))
    assert refused is None and re.fullmatch(OFF_THE_FIT, info["reason"])
    assert STAGE < info["residual_rel"] < digitize.PERSPECTIVE_TOLERANCE
    matrix, info_default = digitize.estimate_perspective(
        page("above"), digitize.PERSPECTIVE_TOLERANCE
    )
    assert matrix is not None and info_default["reason"] == ""


@pytest.mark.parametrize(
    "text",
    ["0.05", "0.0999", "0", "-0.15", "0.5001", "1", "nan", "inf", "-inf", "wide", ""],
)
def test_the_flag_only_widens_the_stage(text, capsys):
    # Below the tolerance of the stage, above half a period, or no number at all:
    # refused when the arguments are read, before any page is.
    with pytest.raises(SystemExit):
        digitize.get_parser().parse_args(FOLDERS + [f"{FLAG}={text}"])
    error = capsys.readouterr().err.rstrip()
    assert f"argument {FLAG}: {text!r} is not a share of the grid line period" in error
    assert error.endswith(
        "from 0.1, the tolerance of the stage itself, to 0.5: the flag only widens "
        "the stage"
    )
    with pytest.raises(argparse.ArgumentTypeError, match="only widens the stage"):
        digitize.parse_perspective_tolerance(text)


def test_the_tolerance_is_an_argument_of_the_perspective_fit_alone():
    def names(function):
        return list(inspect.signature(function).parameters)

    assert names(digitize.estimate_perspective) == ["image", "tolerance"]
    assert names(digitize._perspective_grid_map) == [
        "darkness",
        "width",
        "height",
        "tolerance",
    ]
    for function in (digitize.estimate_perspective, digitize._perspective_grid_map):
        assert inspect.signature(function).parameters["tolerance"].default is None
    # Neither the rotation stage nor the chain of the paper normalisation has one.
    assert names(digitize.grid_line_slope) == ["image", "axis"]
    assert names(digitize.estimate_rotation) == ["image", "method"]
    assert names(digitize.paper_chain) == ["image"]


# ------------------------------------------------------------------------ the fit
def test_the_pages_are_what_they_are_drawn_for():
    share = {}
    for name in ("even", "below", "above"):
        angle, rotation = digitize.estimate_rotation(page(name), "lines")
        assert angle == 0.0 and rotation["reason"] == "", name
        matrix, info = digitize.estimate_perspective(page(name))
        share[name] = info["residual_rel"]
        assert (matrix is not None) == (name == "below"), name
        assert bool(info["reason"]) == (name == "above"), name
    assert share["even"] < 0.01
    assert 0.07 < share["below"] < 0.095
    assert 0.105 < share["above"] < 0.12

    # The leaning sheet is turned first, and is then the page above.
    angle, rotation = digitize.estimate_rotation(page("leaning"), "lines")
    assert rotation["reason"] == "" and 0.2 < angle < 0.3
    turned = digitize.warp_page(
        page("leaning"), digitize.rotation_homography(angle, WIDTH, HEIGHT)
    )
    matrix, info = digitize.estimate_perspective(turned)
    assert matrix is None and re.fullmatch(OFF_THE_FIT, info["reason"])
    assert 0.105 < info["residual_rel"] < 0.12

    # The waved page is above the tenth for both stages.
    angle, rotation = digitize.estimate_rotation(page("waved"), "lines")
    assert np.isnan(angle) and re.fullmatch(OFF_A_LINE, rotation["reason"])
    matrix, info = digitize.estimate_perspective(page("waved"))
    assert matrix is None and re.fullmatch(OFF_THE_FIT, info["reason"])
    assert 0.12 < info["residual_rel"] < 0.135

    # The blank page has no period to measure a residual on.
    matrix, info = digitize.estimate_perspective(page("blank"))
    assert matrix is None and info["reason"] == "no grid line period found"
    assert np.isnan(info["residual_rel"])


def test_a_fit_above_a_tenth_is_refused_unless_the_tolerance_is_above_it(monkeypatch):
    refused, info = digitize.estimate_perspective(page("above"))
    assert refused is None and re.fullmatch(OFF_THE_FIT, info["reason"])
    assert np.isnan(info["shift"])
    assert info["residual"] > STAGE * info["period"]

    # A tolerance between the tenth and the residual refuses the page as well, with
    # the reason and the numbers of the stage.
    still, info_narrow = digitize.estimate_perspective(page("above"), NARROW)
    assert still is None and same_info(info_narrow, info)
    assert info["residual_rel"] > NARROW

    matrix, info_wide = digitize.estimate_perspective(page("above"), WIDE)
    assert matrix is not None and info_wide["reason"] == ""
    assert info_wide["shift"] > digitize.PERSPECTIVE_MIN_SHIFT_PX
    assert info_wide["residual_rel"] < WIDE
    # The second round, on the page the first one warped, is above the tenth too.
    assert info_wide["residual"] > STAGE * info_wide["period"]

    # It is the homography of the fit as it is without the check: nothing but the
    # test of the residual takes the tolerance.
    widest, info_widest = digitize.estimate_perspective(
        page("above"), digitize.PERSPECTIVE_MAX_TOLERANCE
    )
    assert np.allclose(widest, matrix, rtol=0, atol=1e-9)
    assert same_info(info_widest, info_wide)
    monkeypatch.setattr(digitize, "GRID_LINE_MAX_SLOPE_RESIDUAL", float("inf"))
    unchecked, info_unchecked = digitize.estimate_perspective(page("above"))
    assert np.allclose(unchecked, matrix, rtol=0, atol=1e-9)
    assert same_info(info_unchecked, info_wide)


def test_a_fit_inside_a_tenth_is_the_same_at_any_tolerance():
    for name in ("even", "below"):
        plain, info = digitize.estimate_perspective(page(name))
        assert info["reason"] == ""
        for tolerance in (STAGE, WIDE, digitize.PERSPECTIVE_MAX_TOLERANCE):
            matrix, info_given = digitize.estimate_perspective(page(name), tolerance)
            assert same_info(info_given, info), (name, tolerance)
            if plain is None:
                assert matrix is None
            else:
                assert np.allclose(matrix, plain, rtol=0, atol=1e-9)


def test_the_tolerance_is_the_module_constant_read_at_the_call_unless_given(
    monkeypatch,
):
    # Not when the function is defined: a driver that sets the constant for the time
    # of a call still reaches the fit, and a tolerance that is given wins over it.
    matrix, info = digitize.estimate_perspective(page("above"), WIDE)
    monkeypatch.setattr(digitize, "GRID_LINE_MAX_SLOPE_RESIDUAL", WIDE)
    by_constant, info_constant = digitize.estimate_perspective(page("above"))
    assert np.allclose(by_constant, matrix, rtol=0, atol=1e-9)
    assert same_info(info_constant, info)
    refused, info_refused = digitize.estimate_perspective(page("above"), NARROW)
    assert refused is None and re.fullmatch(OFF_THE_FIT, info_refused["reason"])


def test_estimate_perspective_without_the_argument_is_what_it_was(monkeypatch):
    # The fit is called with the page and its size and nothing else, as it always
    # was: a stand-in that takes no tolerance, as the ones of the drivers, is enough.
    # What comes back is the stage with its own tolerance written out, on a page of
    # every kind.
    fit, rounds = digitize._perspective_grid_map, []

    def three_arguments(darkness, width, height):
        rounds.append((width, height))
        return fit(darkness, width, height)

    results = {
        name: digitize.estimate_perspective(page(name), STAGE)
        for name in ("even", "below", "above", "waved")
    }
    with monkeypatch.context() as patch:
        patch.setattr(digitize, "_perspective_grid_map", three_arguments)
        for name, (expected, expected_info) in results.items():
            rounds.clear()
            matrix, info = digitize.estimate_perspective(page(name))
            assert rounds == [(WIDTH, HEIGHT)] * (2 if name == "below" else 1), name
            assert same_info(info, expected_info), name
            if expected is None:
                assert matrix is None, name
            else:
                assert np.allclose(matrix, expected, rtol=0, atol=1e-9), name
    # One key more than it had, and none less.
    assert sorted(results["even"][1]) == [
        "period",
        "reason",
        "residual",
        "residual_rel",
        "shift",
        "windows",
    ]
    # A tolerance that is given goes to the fit of every round, by its name.
    calls = recorded(monkeypatch, "_perspective_grid_map")
    digitize.estimate_perspective(page("above"), WIDE)
    assert [len(args) for args, _ in calls] == [3, 3]
    assert [kwargs for _, kwargs in calls] == [{"tolerance": WIDE}] * 2


def test_the_fit_tests_its_residual_against_the_tolerance_it_is_given():
    darkness = digitize._darkness(page("above"))
    refused, info = digitize._perspective_grid_map(darkness, WIDTH, HEIGHT)
    assert refused is None and re.fullmatch(OFF_THE_FIT, info["reason"])
    still, info_narrow = digitize._perspective_grid_map(
        darkness, WIDTH, HEIGHT, tolerance=NARROW
    )
    assert still is None and same_info(info_narrow, info)
    matrix, info_wide = digitize._perspective_grid_map(
        darkness, WIDTH, HEIGHT, tolerance=WIDE
    )
    # The same fit of the same windows: only what is made of its residual differs.
    assert matrix is not None and info_wide["reason"] == ""
    assert same_info({**info_wide, "reason": info["reason"]}, info)
    assert NARROW * info["period"] < info["residual"] < WIDE * info["period"]


# ------------------------------------------------------------------- residual_rel
def two_rounds(first, second):
    """A stand-in for the fit: a residual of first, then second, of a period of 8 px.

    Tested against the tolerance as the fit tests it, and with a map that leans the
    page enough for a second round. Returns (fit, the tolerance of every call).
    """
    seen = []

    def fit(darkness, width, height, tolerance=None):
        if tolerance is None:
            tolerance = digitize.GRID_LINE_MAX_SLOPE_RESIDUAL
        seen.append(tolerance)
        share = first if len(seen) == 1 else second
        info = {"period": 8.0, "residual": share * 8.0, "windows": 300, "reason": ""}
        if info["residual"] > tolerance * 8.0:
            info["reason"] = f"grid line phase is {info['residual']:.2f} px off the fit"
            return None, info
        lean = 0.02 if len(seen) == 1 else 0.004
        return np.array([[1.0, lean, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0]]), info

    return fit, seen


def test_residual_rel_is_the_residual_of_one_round_over_its_period(monkeypatch):
    rounds = recorded(monkeypatch, "_perspective_grid_map")
    # One round, refused: the number the tolerance was compared with.
    _, info = digitize.estimate_perspective(page("above"))
    assert len(rounds) == 1 and info["reason"]
    assert info["residual_rel"] == info["residual"] / info["period"]
    assert info["period"] == pytest.approx(PERIOD, rel=0.01)
    # One round, accepted and left alone inside the dead band.
    rounds.clear()
    matrix, info = digitize.estimate_perspective(page("waved"), WIDE)
    assert len(rounds) == 1 and matrix is None and info["reason"] == ""
    assert info["shift"] < digitize.PERSPECTIVE_MIN_SHIFT_PX
    assert info["residual_rel"] == info["residual"] / info["period"]
    assert STAGE < info["residual_rel"] < WIDE


def test_residual_rel_is_the_larger_one_of_two_rounds(monkeypatch):
    # On a drawn page: the second round, on the warped page, is the larger one.
    fit, fits = digitize._perspective_grid_map, []

    def keeping(*args, **kwargs):
        result = fit(*args, **kwargs)
        fits.append(result[1])
        return result

    with monkeypatch.context() as patch:
        patch.setattr(digitize, "_perspective_grid_map", keeping)
        matrix, info = digitize.estimate_perspective(page("above"), WIDE)
    shares = [round_["residual"] / round_["period"] for round_ in fits]
    assert matrix is not None and len(shares) == 2
    assert info["residual_rel"] == pytest.approx(max(shares), rel=1e-9)
    assert all(STAGE < share < WIDE for share in shares)

    # Whichever round it is: info["residual"] is the one of the last round, in pixels.
    blank = torch.zeros((3, 400, 600), dtype=torch.uint8)
    for first, second in ((0.12, 0.08), (0.08, 0.12)):
        stand_in, seen = two_rounds(first, second)
        monkeypatch.setattr(digitize, "_perspective_grid_map", stand_in)
        matrix, info = digitize.estimate_perspective(blank, WIDE)
        # The tolerance is the one of the test of every round.
        assert seen == [WIDE, WIDE]
        assert matrix is not None and info["reason"] == ""
        assert info["shift"] > digitize.PERSPECTIVE_MIN_SHIFT_PX
        assert info["residual_rel"] == pytest.approx(0.12)
        assert info["residual"] == pytest.approx(second * 8.0)

    # Without a tolerance each round is tested against the tenth: a page inside it in
    # the first round and above it in the second is refused in the second.
    stand_in, seen = two_rounds(0.08, 0.12)
    monkeypatch.setattr(digitize, "_perspective_grid_map", stand_in)
    matrix, info = digitize.estimate_perspective(blank)
    assert seen == [STAGE, STAGE]
    assert matrix is None and info["reason"] == "grid line phase is 0.96 px off the fit"
    assert info["residual_rel"] == pytest.approx(0.12)


def test_residual_rel_leaves_out_a_round_without_a_residual(monkeypatch):
    # No round got as far as a residual: NaN, as the residual itself.
    blank = torch.full((3, HEIGHT, WIDTH), 255, dtype=torch.uint8)
    matrix, info = digitize.estimate_perspective(blank, WIDE)
    assert matrix is None and info["reason"] == "no grid line period found"
    assert np.isnan(info["residual"]) and np.isnan(info["residual_rel"])

    # A second round that finds no grid lines leaves the first one standing.
    calls = []

    def fit(darkness, width, height, tolerance=None):
        calls.append(tolerance)
        if len(calls) == 1:
            info = {"period": 8.0, "residual": 0.96, "windows": 300, "reason": ""}
            return np.array([[1.0, 0.02, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0]]), info
        nan = float("nan")
        return None, {"period": nan, "residual": nan, "windows": 0, "reason": "none"}

    monkeypatch.setattr(digitize, "_perspective_grid_map", fit)
    matrix, info = digitize.estimate_perspective(
        torch.zeros((3, 400, 600), dtype=torch.uint8), WIDE
    )
    assert calls == [WIDE, WIDE]
    assert matrix is None and info["reason"] == "none"
    assert np.isnan(info["residual"])
    assert info["residual_rel"] == pytest.approx(0.12)


# ----------------------------------------------------------------------- the line
def test_perspective_tolerance_line_says_what_was_done_with_the_page():
    info = {"residual_rel": 0.12345, "residual": 0.97, "period": 7.87, "shift": 5.4}
    start = (
        "Perspective for record rec: accepted at --perspective_tolerance 0.15, "
        "residual 0.1235 of the period (the stage's own tolerance is 0.1), "
    )
    line = digitize.perspective_tolerance_line("rec", 0.15, info, True)
    assert line == start + "homography applied"
    line = digitize.perspective_tolerance_line("rec", 0.15, info, False)
    assert line == start + "inside the dead band, page kept"
    assert "WARNING" not in line and "\n" not in line


# ---------------------------------------------- the rotation stage and the chain
def test_the_rotation_stage_keeps_its_tenth():
    # The lines of the waved page are 0.127 of the period off a line, which a
    # perspective stage at 0.15 would take: the rotation stage refuses them.
    info = digitize.grid_line_slope(page("waved"))
    assert re.fullmatch(OFF_A_LINE, info["reason"]) and np.isnan(info["angle"])
    assert STAGE * info["period"] < info["residual"] < WIDE * info["period"]
    assert digitize.GRID_LINE_MAX_SLOPE_RESIDUAL == 0.1


def test_the_chain_of_the_paper_normalisation_measures_at_the_tenth(monkeypatch):
    calls = recorded(monkeypatch, "estimate_perspective")
    info, stages = digitize.paper_chain(page("above"))
    # One call, with the page it measured on and no tolerance.
    assert len(calls) == 1
    (measured,), kwargs = calls[0]
    assert measured is stages["measured"] and kwargs == {}
    assert info["ok"] is False and re.fullmatch(OFF_THE_FIT, info["persp_reason"])
    assert info["rot_reason"] == info["res_reason"] == ""
    assert stages["H_rect"] is None and info["persp_applied"] == 0
    assert stages["perspective_info"]["residual_rel"] > STAGE


# ------------------------------------------------------------------- end to end
# Every run saves its mask, so its output folder is the mask folder of a later run.
BASE = ("--time_mapping", "bbox", "--save_mask")
# 0.15 is the default, so TOLERANCE is the default written out, and the run of the
# stage at its own tolerance, which the run without the flag was before, asks for 0.1.
TOLERANCE = (FLAG, "0.15")
TENTH = (FLAG, "0.1")
FITS_NOT = "the mask does not fit."
ACCEPTED = "accepted at --perspective_tolerance"


@pytest.fixture(scope="module")
def runs(tmp_path_factory):
    """run() on one drawn page with a saved mask, each page, mask and flags once.

    Returns a function: (name, *flags, masks=None) -> (output folder, what run()
    printed). masks is the folder the mask is read from: the label mask alone,
    without a JSON, unless given.
    """
    root = tmp_path_factory.mktemp("perspective_tolerance")
    folders, done = {}, {}

    def run(name, *flags, masks=None):
        if name not in folders:
            data, bare = root / f"data_{name}", root / f"masks_{name}"
            data.mkdir()
            bare.mkdir()
            write_png(page(name), str(data / f"{name}.png"))
            write_png(label_mask(), str(bare / f"{name}_mask.png"))
            folders[name] = (data, bare)
        data, bare = folders[name]
        masks = bare if masks is None else masks
        if (name, flags, masks) not in done:
            out = root / f"out{len(done)}"
            argv = ["-d", str(data), "-o", str(out), "--mask_folder", str(masks)]
            printed = io.StringIO()
            with contextlib.redirect_stdout(printed):
                digitize.run(digitize.get_parser().parse_args(argv + [*BASE, *flags]))
            done[name, flags, masks] = (out, printed.getvalue())
        return done[name, flags, masks]

    return run


def qc_row(folder):
    """(the one row of qc.csv, its header)."""
    with open(folder / "qc.csv", newline="") as f:
        reader = csv.DictReader(f)
        rows = list(reader)
    assert len(rows) == 1
    return rows[0], reader.fieldnames


def same_row(a, b):
    """Two QC rows agree: the texts as they are, the numbers as same_number() has it."""

    def same(x, y):
        try:
            return same_number(float(x), float(y))
        except ValueError:
            return x == y

    return a.keys() == b.keys() and all(same(a[key], b[key]) for key in a)


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


def perspective_warning(printed, name):
    return lines_of(printed, name, "WARNING: grid lines not used for the perspective")


def test_run_with_a_tenth_refuses_the_page_as_it_always_did(runs):
    out, printed = runs("above", *TENTH)
    row, header = qc_row(out)
    warning = perspective_warning(printed, "above")
    assert len(warning) == 1
    assert re.fullmatch(
        r"WARNING: grid lines not used for the perspective of record above "
        rf"\({OFF_THE_FIT}\), keeping the rotated page\.",
        warning[0],
    )
    assert ACCEPTED not in printed
    assert float(row["rotation_angle"]) == 0.0
    assert np.isnan(float(row["perspective_shift_px"]))
    # The residual in pixels, and the same as a share of the period: the last column
    # before the two of the sheet curl stage.
    share = float(row["perspective_residual_rel"])
    assert header[-5] == "perspective_residual_rel"
    assert 0.105 < share < 0.12
    in_pixels = float(row["perspective_residual_px"])
    assert share == pytest.approx(in_pixels / PERIOD, rel=0.01)
    # The page was only rotated, by no angle: its mask is in the frame of the page.
    assert "homography" not in mask_meta(out, "above")


@pytest.mark.parametrize("name", ["even", "below", "above", "waved"])
def test_a_run_that_does_not_name_the_flag_is_the_run_with_0_15(runs, name):
    # 0.15 is the default: --perspective_tolerance 0.15 is the run without the flag, to
    # the lines it prints, its QC row and the frame of its mask, on a page of every
    # kind. The signals of these pages, which have no trace, are the same at a tenth
    # as well, so they are compared as they are and do not tell the default from it.
    out, printed = runs(name)
    named, printed_named = runs(name, *TOLERANCE)
    assert printed_named == printed
    assert same_row(qc_row(named)[0], qc_row(out)[0])
    assert signal_files(named, name) == signal_files(out, name)
    meta, meta_named = mask_meta(out, name), mask_meta(named, name)
    assert sorted(meta) == sorted(meta_named)
    assert ("homography" in meta) == (name in ("below", "above"))
    if "homography" in meta:
        assert np.allclose(
            meta_named["homography"], meta["homography"], rtol=0, atol=1e-9
        )
    # It lets the pages above the tenth through, and says so of each.
    above = name in ("above", "waved")
    assert printed.count(ACCEPTED) == (1 if above else 0)
    assert perspective_warning(printed, name) == []
    # The tenth written out is the stage at its own tolerance, the run without the flag
    # before 0.15 became the default: it lets no page through, rectifies only the page
    # the stage itself reads, and is the default run on a page inside the tenth alone.
    tenth, printed_tenth = runs(name, *TENTH)
    assert ACCEPTED not in printed_tenth
    assert ("homography" in mask_meta(tenth, name)) == (name == "below")
    assert len(perspective_warning(printed_tenth, name)) == (1 if above else 0)
    assert (printed_tenth == printed) == (not above)
    assert same_row(qc_row(tenth)[0], qc_row(out)[0]) == (not above)


def test_run_at_0_15_rectifies_the_page_and_says_so(runs):
    out, printed = runs("above", *TOLERANCE)
    row, header = qc_row(out)
    assert perspective_warning(printed, "above") == []
    share = float(row["perspective_residual_rel"])
    assert header[-5] == "perspective_residual_rel"
    assert 0.105 < share < 0.12
    # One line more, right after the line of the stage.
    said = lines_of(printed, "above", "Perspective for record")
    assert len(said) == 2 and said[0].startswith("Perspective for record above: shift ")
    assert said[1] == (
        "Perspective for record above: accepted at --perspective_tolerance 0.15, "
        f"residual {share:.4f} of the period (the stage's own tolerance is 0.1), "
        "homography applied"
    )
    every = printed.splitlines()
    assert every[every.index(said[0]) + 1] == said[1]
    assert printed.count(ACCEPTED) == 1
    shift = float(row["perspective_shift_px"])
    assert digitize.PERSPECTIVE_MIN_SHIFT_PX < shift < 10.0
    # The mask of this run is in the frame of that homography.
    homography = np.array(mask_meta(out, "above")["homography"])
    assert digitize._homography_shift(homography, WIDTH, HEIGHT) == pytest.approx(shift)
    # But for the perspective, the row is the one of the run with a tenth.
    plain = qc_row(runs("above", *TENTH)[0])[0]
    differ = {key for key in row if not same_row({key: row[key]}, {key: plain[key]})}
    assert differ == {
        "perspective_shift_px",
        "perspective_residual_px",
        "perspective_residual_rel",
    }


def test_a_tolerance_below_the_residual_of_a_page_leaves_it_refused(runs):
    plain, printed_plain = runs("above", *TENTH)
    out, printed = runs("above", FLAG, str(NARROW))
    assert printed == printed_plain
    assert len(perspective_warning(printed, "above")) == 1 and ACCEPTED not in printed
    assert same_row(qc_row(out)[0], qc_row(plain)[0])
    assert "homography" not in mask_meta(out, "above")


@pytest.mark.parametrize("name", ["even", "below"])
def test_run_reads_a_page_inside_the_tenth_the_same_at_a_wider_tolerance(runs, name):
    plain, printed_plain = runs(name, *TENTH)
    row = qc_row(plain)[0]
    assert float(row["perspective_residual_rel"]) < STAGE
    assert perspective_warning(printed_plain, name) == []
    rectified = float(row["perspective_shift_px"]) > digitize.PERSPECTIVE_MIN_SHIFT_PX
    assert rectified == (name == "below")
    for tolerance in ("0.15", "0.5"):
        out, printed = runs(name, FLAG, tolerance)
        # To the lines it prints: no page here is one a wider tolerance lets through.
        assert printed == printed_plain and ACCEPTED not in printed, tolerance
        assert same_row(qc_row(out)[0], qc_row(plain)[0]), tolerance
        meta, meta_plain = mask_meta(out, name), mask_meta(plain, name)
        assert ("homography" in meta) == ("homography" in meta_plain) == rectified
        if rectified:
            assert np.allclose(
                meta["homography"], meta_plain["homography"], rtol=0, atol=1e-9
            )


def test_a_mask_fits_the_frame_of_the_tolerance_it_was_saved_at(runs):
    plain, _ = runs("above", *TENTH)
    wide, _ = runs("above", *TOLERANCE)
    # Saved at a tenth, the mask lives on the page as it was rotated: it fits a run at
    # a tenth, and not the page a wider tolerance rectifies.
    _, printed = runs("above", *TENTH, masks=plain)
    assert FITS_NOT not in printed
    _, printed = runs("above", *TOLERANCE, masks=plain)
    said = lines_of(printed, "above", "WARNING: mask of record")
    assert len(said) == 1 and printed.count(FITS_NOT) == 1
    found = re.fullmatch(
        r"WARNING: mask of record above was predicted in a frame whose corners are "
        r"(\d+\.\d) px off the one used now; the mask does not fit\.",
        said[0],
    )
    assert found and float(found.group(1)) > 1.0
    assert printed.count(ACCEPTED) == 1
    # Saved at 0.15, it fits a run at the same tolerance, and no other.
    _, printed = runs("above", *TOLERANCE, masks=wide)
    assert FITS_NOT not in printed and printed.count(ACCEPTED) == 1
    _, printed = runs("above", *TENTH, masks=wide)
    assert printed.count(FITS_NOT) == 1
    assert len(perspective_warning(printed, "above")) == 1


def test_a_mask_saved_before_0_15_was_the_default_needs_the_tenth_written_out(runs):
    # A run that named no flag saved its masks at a tenth then. Such a mask of a page
    # between the two tolerances does not fit the run that names no flag now, which
    # rectifies the page, and the run says so; with --perspective_tolerance 0.1 it
    # fits as it did.
    before, _ = runs("above", *TENTH)
    _, printed = runs("above", masks=before)
    said = lines_of(printed, "above", "WARNING: mask of record")
    assert len(said) == 1 and said[0].endswith(FITS_NOT)
    assert printed.count(FITS_NOT) == 1 and printed.count(ACCEPTED) == 1
    _, printed = runs("above", *TENTH, masks=before)
    assert FITS_NOT not in printed and ACCEPTED not in printed
    # The mask the run of now saves fits that run, and the one with 0.15 written out.
    now, _ = runs("above")
    for flags in ((), TOLERANCE):
        _, printed = runs("above", *flags, masks=now)
        assert FITS_NOT not in printed and printed.count(ACCEPTED) == 1, flags
    # A page inside the tenth, and one that is accepted above it and kept, are in the
    # frame they were in: their masks of then fit the run of now.
    for name in ("even", "below", "waved"):
        before, _ = runs(name, *TENTH)
        _, printed = runs(name, masks=before)
        assert FITS_NOT not in printed, name


def test_run_names_a_page_that_is_kept_inside_the_dead_band(runs):
    plain, printed_plain = runs("waved", *TENTH)
    out, printed = runs("waved", *TOLERANCE)
    row = qc_row(out)[0]
    assert len(perspective_warning(printed_plain, "waved")) == 1
    assert perspective_warning(printed, "waved") == []
    share = float(row["perspective_residual_rel"])
    assert 0.12 < share < 0.135
    assert lines_of(printed, "waved", "Perspective for record")[1] == (
        "Perspective for record waved: accepted at --perspective_tolerance 0.15, "
        f"residual {share:.4f} of the period (the stage's own tolerance is 0.1), "
        "inside the dead band, page kept"
    )
    assert float(row["perspective_shift_px"]) < digitize.PERSPECTIVE_MIN_SHIFT_PX
    # No homography: the page is in the frame it has at a tenth, and a mask that was
    # saved at a tenth fits.
    assert "homography" not in mask_meta(out, "waved")
    _, printed_again = runs("waved", *TOLERANCE, masks=plain)
    assert FITS_NOT not in printed_again and printed_again.count(ACCEPTED) == 1


@pytest.mark.parametrize("flags", [(), (FLAG, "0.15"), (FLAG, "0.5")])
def test_run_leaves_the_rotation_stage_at_its_tenth(runs, flags):
    # At the default of a run that names no flag as at a tolerance that is given.
    plain, printed_plain = runs("waved", *TENTH)
    out, printed = runs("waved", *flags)
    start = "WARNING: grid lines not used for the rotation"
    warning = lines_of(printed, "waved", start)
    assert len(warning) == 1 and warning == lines_of(printed_plain, "waved", start)
    assert re.fullmatch(
        r"WARNING: grid lines not used for the rotation of record waved "
        rf"\({OFF_A_LINE}\), keeping the whole degree Hough angle\.",
        warning[0],
    )
    row, row_plain = qc_row(out)[0], qc_row(plain)[0]
    for column in ("rotation_angle", "rotation_coarse", "rotation_residual_px"):
        assert row[column] == row_plain[column], column
    # A residual the perspective stage of this run accepts, and the rotation does not.
    assert float(row["rotation_residual_px"]) / PERIOD > STAGE
    assert float(row["rotation_residual_px"]) / PERIOD < 0.15
    assert printed.count(ACCEPTED) == 1


def test_run_says_nothing_of_the_tolerance_without_verbose(runs):
    loud, _ = runs("above", *TOLERANCE)
    out, printed = runs("above", *TOLERANCE, "--no-verbose")
    assert "Perspective for record" not in printed
    assert perspective_warning(printed, "above") == []
    # Nor does the run that names no flag, which is that run.
    default, printed_default = runs("above", "--no-verbose")
    assert printed_default == printed
    assert same_row(qc_row(default)[0], qc_row(out)[0])
    # The page is rectified all the same, and the QC column is what says so.
    assert same_row(qc_row(out)[0], qc_row(loud)[0])
    assert float(qc_row(out)[0]["perspective_residual_rel"]) > STAGE
    assert np.allclose(
        mask_meta(out, "above")["homography"],
        mask_meta(loud, "above")["homography"],
        rtol=0,
        atol=1e-9,
    )


def test_the_flag_is_one_of_the_perspective_stage_only(runs):
    plain, printed_plain = runs("above", "--perspective", "off", *TENTH)
    out, printed = runs("above", "--perspective", "off", *TOLERANCE)
    assert printed == printed_plain and "Perspective for record" not in printed
    assert qc_row(out)[0] == qc_row(plain)[0]
    # Nor does the default of the flag reach a run without the stage.
    default, printed_default = runs("above", "--perspective", "off")
    assert printed_default == printed_plain
    assert qc_row(default)[0] == qc_row(plain)[0]
    # The stage did not measure: no residual of either kind.
    row = qc_row(out)[0]
    assert np.isnan(float(row["perspective_residual_px"]))
    assert np.isnan(float(row["perspective_residual_rel"]))
    assert "homography" not in mask_meta(out, "above")


def test_run_hands_the_tolerance_to_the_stage_only_when_it_widens_the_stage(
    tmp_path, monkeypatch
):
    # With the tenth written out the stage is called with the page and nothing else,
    # as it always was: a stand-in that takes no tolerance, as the ones of the drivers
    # and of the other tests, still stands in. A tolerance above the tenth goes along
    # by its name, and so does the default of a run that names no flag.
    data, masks = tmp_path / "data", tmp_path / "masks"
    data.mkdir()
    masks.mkdir()
    write_png(page("above"), str(data / "above.png"))
    write_png(label_mask(), str(masks / "above_mask.png"))
    calls = recorded(monkeypatch, "estimate_perspective")
    for index, flags in enumerate(((), (FLAG, "0.1"), TOLERANCE, (FLAG, "0.5"))):
        argv = ["-d", str(data), "-o", str(tmp_path / f"out{index}")]
        argv += ["--mask_folder", str(masks), "--time_mapping", "bbox", *flags]
        with contextlib.redirect_stdout(io.StringIO()):
            digitize.run(digitize.get_parser().parse_args(argv))
    assert [len(args) for args, _ in calls] == [1, 1, 1, 1]
    assert [kwargs for _, kwargs in calls] == [
        {"tolerance": 0.15},
        {},
        {"tolerance": 0.15},
        {"tolerance": 0.5},
    ]


def mask_run(folder, name, *flags):
    """run() on one drawn page with a saved mask, in a folder of its own.

    For a test that puts a stand-in into digitize: the runs fixture keeps what it ran
    for the tests after it, and must not keep that. Returns (output folder, what run()
    printed).
    """
    data, masks, out = folder / "data", folder / "masks", folder / "out"
    data.mkdir()
    masks.mkdir()
    write_png(page(name), str(data / f"{name}.png"))
    write_png(label_mask(), str(masks / f"{name}_mask.png"))
    argv = ["-d", str(data), "-o", str(out), "--mask_folder", str(masks)]
    printed = io.StringIO()
    with contextlib.redirect_stdout(printed):
        digitize.run(digitize.get_parser().parse_args(argv + [*BASE, *flags]))
    return out, printed.getvalue()


def test_run_writes_and_names_the_larger_round_when_it_is_not_the_last(
    tmp_path, monkeypatch
):
    # 0.12 of the period in the first round and 0.08 in the second: the last round
    # alone looks like a page the stage itself accepts, and the page is still one
    # that only a tolerance above the tenth let through. The QC column and the line
    # hold the first.
    stand_in, seen = two_rounds(0.12, 0.08)
    monkeypatch.setattr(digitize, "_perspective_grid_map", stand_in)
    out, printed = mask_run(tmp_path, "even", *TOLERANCE)
    assert seen == [0.15, 0.15]
    assert perspective_warning(printed, "even") == []
    row = qc_row(out)[0]
    assert float(row["perspective_residual_rel"]) == pytest.approx(0.12)
    # In pixels it is the last round, as it always was.
    assert float(row["perspective_residual_px"]) == pytest.approx(0.08 * 8.0)
    said = lines_of(printed, "even", "Perspective for record")
    assert len(said) == 2 and said[1] == (
        "Perspective for record even: accepted at --perspective_tolerance 0.15, "
        "residual 0.1200 of the period (the stage's own tolerance is 0.1), "
        "homography applied"
    )
    assert printed.count(ACCEPTED) == 1
    assert "homography" in mask_meta(out, "even")


def test_run_under_a_driver_that_widens_the_constant_prints_no_line_of_the_flag(
    tmp_path, monkeypatch
):
    # A driver from before the flag widens the stage by the module constant, which it
    # sets for the time of the call in a stand-in of one argument, and ran without
    # the flag: the run that --perspective_tolerance 0.1 is now. The page is
    # rectified, as under that driver it always was, and the line would say 'accepted
    # at --perspective_tolerance 0.1' of a page that a tolerance of 0.1 refuses: it is
    # for a run whose tolerance widens the stage, and no other.
    stage = digitize.estimate_perspective

    def driver(image):
        with pytest.MonkeyPatch.context() as patch:
            patch.setattr(digitize, "GRID_LINE_MAX_SLOPE_RESIDUAL", WIDE)
            return stage(image)

    monkeypatch.setattr(digitize, "estimate_perspective", driver)
    out, printed = mask_run(tmp_path, "above", *TENTH)
    assert digitize.GRID_LINE_MAX_SLOPE_RESIDUAL == STAGE
    assert perspective_warning(printed, "above") == []
    said = lines_of(printed, "above", "Perspective for record")
    assert len(said) == 1 and said[0].startswith("Perspective for record above: shift ")
    assert ACCEPTED not in printed
    row = qc_row(out)[0]
    assert float(row["perspective_shift_px"]) > digitize.PERSPECTIVE_MIN_SHIFT_PX
    assert "homography" in mask_meta(out, "above")
    # The QC column holds the residual of such a page all the same.
    assert STAGE < float(row["perspective_residual_rel"]) < WIDE
    # Such a driver has to name the tenth. The run that names no flag hands its 0.15
    # along by its name, which a stand-in of one argument does not take: the run
    # stops, and reads no page at another tolerance than the driver set.
    unpinned = tmp_path / "unpinned"
    unpinned.mkdir()
    with pytest.raises(TypeError, match="unexpected keyword argument 'tolerance'"):
        mask_run(unpinned, "above")


# ------------------------------------------- with the paper normalisation (auto)
def auto_run(folder, name, *flags):
    """run() on a drawn page without a mask, the label in place of the model.

    --paper_normalisation auto, the default, decides the page, and with the defaults
    of the three stages run() takes what the chain measured.
    Returns what was printed (log), the output folder (out), every call of
    estimate_perspective() as (args, kwargs) (calls) and what normalise_paper()
    returned for the page (decided).
    """
    data, out = folder / f"data_{name}", folder / "_".join(("out", name, *flags))
    if not data.exists():
        data.mkdir()
        write_png(page(name), str(data / f"{name}.png"))
    decided = []
    paper = digitize.normalise_paper

    def normalise_paper(image):
        decided.append(paper(image))
        return decided[-1]

    def model(
        image, dataset_name, model_folder, device="auto", disable_tta=False, fold="all"
    ):
        assert tuple(image.shape) == (3, HEIGHT, WIDTH)
        # A copy: the label is a cached one, and run() owns the mask it is given.
        return label_mask().clone()

    log = io.StringIO()
    with pytest.MonkeyPatch.context() as patch:
        calls = recorded(patch, "estimate_perspective")
        patch.setattr(digitize, "normalise_paper", normalise_paper)
        patch.setattr(digitize, "predict_mask_nnunet", model)
        argv = ["-d", str(data), "-o", str(out), *BASE, *flags]
        with contextlib.redirect_stdout(log):
            digitize.run(digitize.get_parser().parse_args(argv))
    assert len(decided) == 1
    return SimpleNamespace(log=log.getvalue(), out=out, calls=calls, decided=decided[0])


@pytest.mark.parametrize("name", ["above", "leaning"])
def test_a_page_the_chain_refused_is_measured_again_at_the_tolerance_of_the_run(
    tmp_path, name
):
    plain = auto_run(tmp_path, name, *TENTH)
    wide = auto_run(tmp_path, name, *TOLERANCE)
    default = auto_run(tmp_path, name)
    # The chain refuses the page for its residual at any tolerance of the run, the
    # paper stage leaves the page as it is, and its decision and what it was made on
    # are the same in all three runs.
    for run in (plain, wide, default):
        _, block, meta, stages = run.decided
        assert block["decision"] == "pass_fullframe" and stages is not None
        assert meta["chain"]["ok"] is False
        assert re.fullmatch(OFF_THE_FIT, meta["chain"]["persp_reason"])
        assert stages["H_rect"] is None
        assert same_info(run.decided[2]["chain"], plain.decided[2]["chain"])
        assert run.decided[1] == plain.decided[1]
        stored = mask_meta(run.out, name)["paper_normalisation"]
        assert stored == plain.decided[1]

    # At a tenth the refusal of the chain is the one of the run: one measurement.
    stages = plain.decided[3]
    assert len(plain.calls) == 1
    assert plain.calls[0][0][0] is stages["measured"] and plain.calls[0][1] == {}
    assert len(perspective_warning(plain.log, name)) == 1 and ACCEPTED not in plain.log
    assert "homography" not in mask_meta(plain.out, name)

    # Above it the run measures again: the page the chain measured, in the frame the
    # chain measured it in, at the tolerance of the run. The run that names no flag
    # does so at its 0.15.
    row_plain = qc_row(plain.out)[0]
    for run in (wide, default):
        stages = run.decided[3]
        assert len(run.calls) == 2
        (first,), first_kwargs = run.calls[0]
        (second,), second_kwargs = run.calls[1]
        assert first is stages["measured"] and first_kwargs == {}
        assert second is stages["measured"] and second_kwargs == {"tolerance": 0.15}
        assert (stages["measured"] is stages["image"]) == (name == "above")
        assert perspective_warning(run.log, name) == []
        assert run.log.count(ACCEPTED) == 1 and "homography applied" in run.log
        row = qc_row(run.out)[0]
        assert row["paper_normalisation"] == row_plain["paper_normalisation"]
        assert row["rotation_angle"] == row_plain["rotation_angle"]
        assert (float(row["rotation_angle"]) == 0.0) == (name == "above")
        assert STAGE < float(row["perspective_residual_rel"]) < 0.15
        assert float(row["perspective_shift_px"]) > digitize.PERSPECTIVE_MIN_SHIFT_PX
        assert "homography" in mask_meta(run.out, name)
    assert default.log == wide.log
    assert same_row(qc_row(default.out)[0], qc_row(wide.out)[0])


def test_a_page_the_chain_accepted_is_not_measured_again(tmp_path):
    plain = auto_run(tmp_path, "below", *TENTH)
    wide = auto_run(tmp_path, "below", *TOLERANCE)
    default = auto_run(tmp_path, "below")
    for run in (plain, wide, default):
        _, block, meta, stages = run.decided
        assert block["decision"] == "pass_chain" and meta["chain"]["ok"] is True
        # Taken from the chain: one measurement, at the tolerance of the stage.
        assert len(run.calls) == 1
        assert run.calls[0][0][0] is stages["measured"] and run.calls[0][1] == {}
        assert stages["H_rect"] is not None
        assert np.allclose(
            mask_meta(run.out, "below")["homography"],
            stages["H_rect"],
            rtol=0,
            atol=1e-9,
        )
    assert wide.log == default.log == plain.log and ACCEPTED not in wide.log
    assert same_row(qc_row(wide.out)[0], qc_row(plain.out)[0])
    assert same_row(qc_row(default.out)[0], qc_row(plain.out)[0])
    assert float(qc_row(wide.out)[0]["perspective_residual_rel"]) < STAGE


def test_a_page_the_chain_refused_for_another_reason_is_refused_again(tmp_path):
    # The chain refuses a page without grid lines for want of a period, which no
    # tolerance mends. A run above the tenth, the one that names no flag included,
    # measures a page the chain refused again whatever the reason: one measurement
    # more, to the same refusal, and but for that the run with a tenth.
    plain = auto_run(tmp_path, "blank", *TENTH)
    wide = auto_run(tmp_path, "blank", *TOLERANCE)
    default = auto_run(tmp_path, "blank")
    for run in (plain, wide, default):
        _, block, meta, stages = run.decided
        assert block["decision"] == "pass_fullframe" and stages is not None
        assert meta["chain"]["persp_reason"] == "no grid line period found"
        assert run.calls[0][0][0] is stages["measured"] and run.calls[0][1] == {}
        assert len(perspective_warning(run.log, "blank")) == 1
        assert ACCEPTED not in run.log
        assert "homography" not in mask_meta(run.out, "blank")
        assert np.isnan(float(qc_row(run.out)[0]["perspective_residual_rel"]))
        assert same_row(qc_row(run.out)[0], qc_row(plain.out)[0])
        assert run.log == plain.log
    assert len(plain.calls) == 1
    for run in (wide, default):
        assert len(run.calls) == 2
        (second,), second_kwargs = run.calls[1]
        assert second is run.decided[3]["measured"]
        assert second_kwargs == {"tolerance": 0.15}


# ----------------------------------------------------------------------------- QC
def test_append_qc_row_writes_the_column_after_all_others(tmp_path):
    qc = {"row_mapping": "lines 3/4", "perspective_residual_rel": 0.1234}
    digitize.append_qc_row(str(tmp_path), "rec", "column", qc, 0.0)
    # A row without it, as a caller of before the column would write.
    digitize.append_qc_row(str(tmp_path), "old", "column", {"row_mapping": "off"}, 0.0)
    with open(tmp_path / "qc.csv", newline="") as f:
        reader = csv.DictReader(f)
        row, old = list(reader)
        # Appended at the end: no column that was there moves.
        assert reader.fieldnames[-7:-4] == [
            "grid_rescue",
            "row_mapping",
            "perspective_residual_rel",
        ]
        assert len(reader.fieldnames) == 39
        assert reader.fieldnames.index("paper_normalisation") == 12
        assert reader.fieldnames.index("perspective_residual_px") == 22
        assert reader.fieldnames.index("row_mapping") == 33
    assert float(row["perspective_residual_rel"]) == pytest.approx(0.1234)
    assert old["perspective_residual_rel"] == "" and old["row_mapping"] == "off"
