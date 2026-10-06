"""Unit tests for the rotation estimate of src/run/digitize.py.

No model, no data files: every page is synthesised with numpy/cv2/torch.
"""
import csv
import json
import time
from functools import lru_cache

import cv2
import numpy as np
import pytest
import torch
from torchvision.io.image import write_png
from torchvision.transforms.functional import rotate

from src.run import digitize

HEIGHT, WIDTH = 1700, 2200
# 1 mm at 200 dpi, the pitch of the printed grid lines of a generator page.
PERIOD = 200 / 25.4


@lru_cache(maxsize=None)
def _page(angle_deg=0.0, period=PERIOD):
    """A page with a 1 mm grid, a few traces and a text blob, rotated by angle_deg.

    cv2 rotates the *content*, so the page comes back tilted by +angle_deg and the
    angle that straightens it again is the negative one. Drawing a page takes a
    moment, so they are cached and no caller may write into one.
    """
    image = np.full((HEIGHT, WIDTH, 3), 255, dtype=np.uint8)
    # Light red grid lines, every fifth one darker and thicker as on a printed page.
    for k in range(int(WIDTH / period) + 1):
        x = int(round(k * period))
        if x < WIDTH:
            heavy = k % 5 == 0
            cv2.line(image, (x, 0), (x, HEIGHT - 1),
                     (120, 120, 255) if heavy else (190, 190, 255), 2 if heavy else 1)
    for k in range(int(HEIGHT / period) + 1):
        y = int(round(k * period))
        if y < HEIGHT:
            heavy = k % 5 == 0
            cv2.line(image, (0, y), (WIDTH - 1, y),
                     (120, 120, 255) if heavy else (190, 190, 255), 2 if heavy else 1)
    # Traces and a text-like blob, so that the band median has something to reject.
    rng = np.random.default_rng(0)
    for row in (300, 700, 1100, 1500):
        x = np.arange(0, WIDTH, 5)
        y = row + 60 * np.sin(2 * np.pi * x / 300) + rng.normal(0, 1, x.size)
        cv2.polylines(
            image, [np.stack([x, y], axis=1).astype(np.int32)], False, (0, 0, 0), 2
        )
    cv2.rectangle(image, (100, 60), (500, 110), (0, 0, 0), -1)
    if angle_deg:
        image = _tilt(image, angle_deg)
    return image


def _tilt(image, angle_deg):
    """Turn the content of a page by angle_deg, about its centre and onto white."""
    matrix = cv2.getRotationMatrix2D((WIDTH / 2, HEIGHT / 2), angle_deg, 1.0)
    return cv2.warpAffine(
        image,
        matrix,
        (WIDTH, HEIGHT),
        flags=cv2.INTER_LINEAR,
        borderMode=cv2.BORDER_CONSTANT,
        borderValue=(255, 255, 255),
    )


def _tensor(np_image):
    """The page as run() reads it: [3, H, W] uint8."""
    return torch.from_numpy(np_image.transpose(2, 0, 1).copy())


def _blank():
    return np.full((HEIGHT, WIDTH, 3), 255, dtype=np.uint8)


@lru_cache(maxsize=None)
def _vertical_lines_page(angle_deg=0.0, period=PERIOD):
    """The vertical family of _page() alone: no horizontal lines, traces or text.

    The bare page of tests/test_digitize_grid.py, and the one the fine Hough
    transform stumbles over: with no other line family to outvote them, the
    collinear alignments across the edge pixels of the 1 mm lines gather enough
    votes to pass as a line family of their own.
    """
    image = np.full((HEIGHT, WIDTH, 3), 255, dtype=np.uint8)
    for k in range(int(WIDTH / period) + 1):
        x = int(round(k * period))
        if x < WIDTH:
            heavy = k % 5 == 0
            cv2.line(image, (x, 0), (x, HEIGHT - 1),
                     (120, 120, 255) if heavy else (190, 190, 255), 2 if heavy else 1)
    if angle_deg:
        image = _tilt(image, angle_deg)
    return image


def _no_grid_page():
    """A page with lines the Hough transform sees, but no printed 1 mm grid.

    The horizontal rules give the Hough transform its angle, the few irregular
    vertical bars leave a profile that is no comb, so the contrast stays low.
    """
    image = _blank()
    for y in range(100, HEIGHT - 100, 50):
        cv2.line(image, (0, y), (WIDTH - 1, y), (0, 0, 0), 2)
    for x in (137, 402, 913, 1290, 1755, 2050):
        cv2.line(image, (x, 0), (x, HEIGHT - 1), (0, 0, 0), 3)
    return image


# ------------------------------------------------------ get_rotation_angle_fine
@pytest.mark.parametrize("tilt", [0.3, 1.5, -0.7])
def test_get_rotation_angle_fine_recovers_a_fraction_of_a_degree(tilt):
    assert digitize.get_rotation_angle_fine(_page(tilt)) == pytest.approx(
        -tilt, abs=0.1
    )


def test_get_rotation_angle_only_finds_whole_degrees():
    # What the fine Hough transform is for: the whole degree one cannot see 0.3.
    coarse = digitize.get_rotation_angle(_page(0.3))
    assert np.isnan(coarse) or abs(coarse - round(coarse)) < 1e-9


def test_get_rotation_angle_fine_is_nan_without_lines():
    assert np.isnan(digitize.get_rotation_angle_fine(_blank()))


def test_get_lines_keeps_its_default_behaviour():
    # A quarter page is enough here and keeps the three Hough transforms quick.
    page = np.ascontiguousarray(_page(2.0)[: HEIGHT // 2, : WIDTH // 2])
    default = digitize.get_lines(page, threshold_HoughLines=600)
    # The extra keywords only matter when they are given.
    same = digitize.get_lines(
        page, threshold_HoughLines=600, theta_resolution=np.pi / 180
    )
    assert np.array_equal(default, same)
    fine = digitize.get_lines(
        page,
        threshold_HoughLines=600,
        theta_resolution=np.deg2rad(0.1),
        theta_window=np.deg2rad(31),
    )

    def near_horizontal(lines):
        return np.sum(np.abs(lines[:, 0, 1] - np.pi / 2) < np.deg2rad(30))

    # The window only drops what filter_lines drops, and a fine theta step splits
    # every line into several near duplicate hits.
    assert np.all(np.abs(fine[:, 0, 1] - np.pi / 2) <= np.deg2rad(31))
    assert near_horizontal(fine) > near_horizontal(default)


# -------------------------------------------------------------------- filter_lines
def _filter_lines_pair_by_pair(
    lines, degree_window=20, parallelism_count=0, parallelism_window=2
):
    """The reference: filter_lines as it counted the pairs one by one."""
    parallelism_radian = np.deg2rad(parallelism_window)
    filtered_lines = []

    if lines is not None:
        for line in lines:
            for rho, theta in line:
                if digitize.is_within_x_degrees_of_horizontal(theta, degree_window):
                    filtered_lines.append((rho, theta))

    parallel_lines = []
    if len(filtered_lines) > 0:
        for rho, theta in filtered_lines:
            count = 0
            for comp_rho, comp_theta in filtered_lines:
                if (
                    abs(theta - comp_theta) < parallelism_radian
                    or abs((theta - comp_theta) - np.pi) < parallelism_radian
                ):
                    count += 1
            if count >= parallelism_count:
                parallel_lines.append((rho, theta))

    if len(parallel_lines) == 0:
        parallel_lines = None
    else:
        parallel_lines = np.array(parallel_lines)[:, np.newaxis, :]

    return parallel_lines


def _hough_lines(thetas, rng):
    """The angles as cv2 hands them over: a float32 [n, 1, 2] array, rho and theta."""
    rho = rng.uniform(-2000, 2000, len(thetas))
    return np.stack([rho, thetas], axis=1).astype(np.float32)[:, np.newaxis, :]


def _thetas(rng, n, kind, parallelism_window):
    """n angles of the given kind, in radians."""
    if kind == "horizontal":
        return np.pi / 2 + rng.normal(0, np.deg2rad(3), n)
    if kind == "spread":
        return rng.uniform(0, np.pi, n)
    if kind == "wrap":
        # Half of them just above zero, half just below pi, which is the only way the
        # second branch of the count sees anything. They sit on a grid of a thousandth
        # of a radian: that branch rounds its difference to float32 around pi, where a
        # step is 2e-7 wide, and no difference of this grid comes that close to a
        # window edge, so the rounding cannot decide a pair either way.
        grid = rng.integers(0, 60, n) / 1000.0
        return np.where(rng.random(n) < 0.5, grid, np.pi - grid)
    if kind == "window_apart":
        # Whole multiples of the window apart, where the open edge has to decide.
        return np.pi / 2 + np.deg2rad(parallelism_window) * rng.integers(-3, 4, n)
    if kind == "duplicates":
        return np.repeat(rng.uniform(0, np.pi, max(n // 4, 1)), 4)[:n]
    raise ValueError(kind)


# The two combinations the code base uses, one of them the defaults, and wider ones
# on top: a degree window beyond 90 is what leaves angles a whole pi apart in at all.
_FILTER_CASES = [
    (30, 3, 2),
    (20, 0, 2),
    (95, 4, 2),
    (20, 5, 0.5),
    (5, 2, 10),
    (30, 3, 0),
]


@pytest.mark.parametrize(
    "degree_window,parallelism_count,parallelism_window", _FILTER_CASES
)
def test_filter_lines_keeps_what_the_pair_by_pair_count_kept(
    degree_window, parallelism_count, parallelism_window
):
    """The sorted count is the old double loop, line for line and bit for bit."""
    rng = np.random.default_rng(20250919)
    for kind in ("horizontal", "spread", "wrap", "window_apart", "duplicates"):
        for n in (0, 1, 2, 5, 40, 200):
            lines = _hough_lines(_thetas(rng, n, kind, parallelism_window), rng)
            expected = _filter_lines_pair_by_pair(
                lines, degree_window, parallelism_count, parallelism_window
            )
            kept = digitize.filter_lines(
                lines, degree_window, parallelism_count, parallelism_window
            )
            if expected is None:
                assert kept is None, (kind, n)
            else:
                assert kept is not None, (kind, n)
                assert (kept.dtype, kept.shape) == (expected.dtype, expected.shape)
                assert np.array_equal(kept, expected), (kind, n)


def test_filter_lines_keeps_the_empty_answers():
    rng = np.random.default_rng(1)
    assert digitize.filter_lines(None) is None
    assert digitize.filter_lines(_hough_lines(np.array([]), rng)) is None
    # Nothing inside the degree window, and nothing with enough parallel lines.
    vertical = _hough_lines(np.array([0.0, 0.05]), rng)
    assert digitize.filter_lines(vertical, degree_window=2) is None
    assert (
        digitize.filter_lines(vertical, degree_window=95, parallelism_count=3) is None
    )


def test_filter_lines_counts_the_wrap_around_in_one_direction_only():
    """Two lines a whole pi apart are the same line, but the count is one-sided.

    It only ever tested theta - comp_theta - pi, so the upper of the two sees the
    lower one below it while the lower one sees nothing, and that stays that way.
    """
    rng = np.random.default_rng(2)
    lines = _hough_lines(np.array([0.01, np.pi - 0.01]), rng)
    kept = digitize.filter_lines(lines, degree_window=95, parallelism_count=2)
    assert kept.shape == (1, 1, 2)
    assert kept[0, 0, 1] == pytest.approx(np.pi - 0.01, abs=1e-6)
    # And the lower line alone counts itself only, so nothing is left.
    assert (
        digitize.filter_lines(lines[:1], degree_window=95, parallelism_count=2) is None
    )


def test_filter_lines_takes_the_lines_of_a_vector_page_in_a_moment():
    """What the sorted count is for: a noise-free page floods the fine accumulator.

    A vector rendered page leaves tens of thousands of near duplicate lines where a
    scan leaves a few hundred, and counting those pair by pair is quadratic: 4000
    lines already took 15 s, the 18000 of such a page took over seven minutes.
    """
    rng = np.random.default_rng(3)
    lines = _hough_lines(np.pi / 2 + rng.normal(0, np.deg2rad(2), 20000), rng)
    start = time.perf_counter()
    kept = digitize.filter_lines(
        lines, degree_window=30, parallelism_count=3, parallelism_window=2
    )
    elapsed = time.perf_counter() - start
    assert kept.shape == (20000, 1, 2)
    # Measured at 0.12 s, with room for a machine that has other work to do.
    assert elapsed < 2.0


# ---------------------------------------------------------------- grid_line_slope
@pytest.mark.parametrize("tilt", [0.3, -0.45])
def test_grid_line_slope_recovers_the_tilt_on_both_axes(tilt):
    page = _tensor(_page(tilt))
    vertical = digitize.grid_line_slope(page, axis=0)
    horizontal = digitize.grid_line_slope(page, axis=1)
    assert vertical["reason"] == "" and horizontal["reason"] == ""
    # Both axes give the angle that straightens the page, with the same sign.
    assert vertical["angle"] == pytest.approx(-tilt, abs=0.01)
    assert horizontal["angle"] == pytest.approx(-tilt, abs=0.01)
    assert vertical["period"] == pytest.approx(PERIOD, rel=1e-3)
    assert vertical["contrast"] > digitize.GRID_LINE_MIN_CONTRAST
    assert vertical["residual"] < 0.1
    assert vertical["bands"] >= digitize.GRID_LINE_MIN_SLOPE_BANDS


def test_grid_line_slope_sign_is_the_one_rotate_undoes():
    """The empirical sign check: rotate() by the angle must straighten the page."""
    page = _tensor(_page(0.8))
    angle = digitize.grid_line_slope(page, axis=0)["angle"]
    straightened = rotate(page, angle)
    assert digitize.grid_line_slope(straightened, axis=0)["angle"] == pytest.approx(
        0.0, abs=0.01
    )
    assert digitize.grid_line_slope(straightened, axis=1)["angle"] == pytest.approx(
        0.0, abs=0.01
    )
    # And it is the same sign the Hough transform reports for the same page.
    assert np.sign(digitize.get_rotation_angle_fine(_page(0.8))) == np.sign(angle)


def test_grid_line_slope_is_nan_on_a_blank_page():
    info = digitize.grid_line_slope(_tensor(_blank()))
    assert np.isnan(info["angle"])
    assert info["reason"] != ""


def test_grid_line_slope_needs_grid_lines():
    info = digitize.grid_line_slope(_tensor(_no_grid_page()))
    assert np.isnan(info["angle"])
    assert "no grid lines found" in info["reason"]


def test_grid_line_slope_needs_enough_bands():
    small = _tensor(_page()[: 3 * digitize.GRID_LINE_BAND_HEIGHT])
    info = digitize.grid_line_slope(small)
    assert np.isnan(info["angle"])
    assert info["reason"] == "image too small for the grid line profile"


def test_grid_line_slope_takes_a_numpy_page_as_well():
    page = _page(0.3)
    from_numpy = digitize.grid_line_slope(page.transpose(2, 0, 1))
    from_tensor = digitize.grid_line_slope(_tensor(page))
    assert from_numpy["angle"] == pytest.approx(from_tensor["angle"], abs=1e-9)


# --------------------------------------------------------------- estimate_rotation
def test_estimate_rotation_hough_is_the_old_angle():
    page = _page(2.0)
    angle, info = digitize.estimate_rotation(_tensor(page), "hough")
    expected = digitize.get_rotation_angle(page)
    assert angle == expected
    assert info["coarse"] == expected
    assert info["reason"] == "" and info["deltas"] == []


@pytest.mark.parametrize("tilt", [0.3, 1.5, -0.7, 2.0])
def test_estimate_rotation_lines_recovers_the_tilt(tilt):
    angle, info = digitize.estimate_rotation(_tensor(_page(tilt)), "lines")
    assert info["reason"] == ""
    assert angle == pytest.approx(-tilt, abs=0.01)


def test_estimate_rotation_lines_refines_between_two_hough_steps():
    # 0.37 degrees is not a multiple of the 0.1 degree theta step.
    angle, info = digitize.estimate_rotation(_tensor(_page(0.37)), "lines")
    assert info["coarse"] == pytest.approx(-0.4, abs=1e-9)
    assert angle == pytest.approx(-0.37, abs=0.01)
    assert angle != info["coarse"]


def test_estimate_rotation_lines_keeps_the_hough_angle_of_a_straight_page():
    page = _tensor(_page())
    angle, info = digitize.estimate_rotation(page, "lines")
    # The refinement is within the noise, so the page keeps its Hough angle exactly.
    assert angle == info["coarse"]
    assert abs(info["deltas"][0]) < digitize.ROTATION_REFINE_MIN_DEG


def test_estimate_rotation_lines_measures_twice_without_a_hough_angle(monkeypatch):
    monkeypatch.setattr(digitize, "get_rotation_angle_fine", lambda image: np.nan)
    angle, info = digitize.estimate_rotation(_tensor(_page(1.5)), "lines")
    assert np.isnan(info["coarse"])
    assert len(info["deltas"]) == 2
    assert angle == pytest.approx(-1.5, abs=0.01)


def test_estimate_rotation_lines_is_nan_on_a_blank_page():
    angle, info = digitize.estimate_rotation(_tensor(_blank()), "lines")
    assert np.isnan(angle)
    assert info["reason"] != ""


def test_estimate_rotation_lines_drops_a_coarse_angle_the_grid_lines_reject():
    """The degenerate page: a straight one the fine Hough transform reads as tilted.

    Turning the page by that angle smears its comb, so the grid lines cannot veto it
    where they are asked first; measured on the page as it came they can, and they
    read the page for what it is.
    """
    page = _vertical_lines_page()
    fine = digitize.get_rotation_angle_fine(page)
    # Measured at 16.7 degrees, while the whole degree transform leaves the page be.
    assert abs(fine) > 1.0
    assert np.isnan(digitize.get_rotation_angle(page))

    angle, info = digitize.estimate_rotation(_tensor(page), "lines")
    assert angle == pytest.approx(0.0, abs=0.05)
    assert info["reason"] == ""
    # The rejected angle is still what the QC column has to show.
    assert info["coarse"] == fine


def test_estimate_rotation_lines_finds_the_tilt_under_a_spurious_coarse_angle():
    page = _vertical_lines_page(0.3)
    # Measured at 21.8 degrees, nowhere near the tilt the page really has.
    assert abs(digitize.get_rotation_angle_fine(page)) > 1.0
    angle, info = digitize.estimate_rotation(_tensor(page), "lines")
    assert info["reason"] == ""
    # Not the spurious angle but the tilt the grid lines measure on the page itself.
    assert angle == pytest.approx(-0.3, abs=0.01)


def test_estimate_rotation_lines_falls_back_to_the_whole_degree_angle(monkeypatch):
    """With no grid lines to ask, "lines" answers what "hough" would have answered."""
    page = _tilt(_no_grid_page(), 2.0)
    fine = digitize.get_rotation_angle_fine(page)
    calls = _spy(monkeypatch, "get_rotation_angle")
    angle, info = digitize.estimate_rotation(_tensor(page), "lines")
    # Both transforms read this page, and the answer is the whole degree one, asked
    # once and only because the grid lines were unreadable on either page.
    assert len(calls) == 1
    assert angle == digitize.get_rotation_angle(page)
    assert not np.isnan(angle)
    assert "no grid lines found" in info["reason"]
    assert info["coarse"] == fine


def test_estimate_rotation_lines_is_nan_when_neither_transform_is_trusted(monkeypatch):
    """The fine angle is spurious and the whole degree one finds nothing: no angle."""
    monkeypatch.setattr(digitize, "get_rotation_angle_fine", lambda image: 12.0)
    monkeypatch.setattr(digitize, "get_rotation_angle", lambda image: np.nan)
    angle, info = digitize.estimate_rotation(_tensor(_no_grid_page()), "lines")
    assert np.isnan(angle)
    assert info["coarse"] == 12.0
    assert "no grid lines found" in info["reason"]


def test_estimate_rotation_lines_does_not_ask_the_whole_degree_transform(monkeypatch):
    """The common path pays for the grid lines only, not for a second transform."""
    calls = _spy(monkeypatch, "get_rotation_angle")
    angle, info = digitize.estimate_rotation(_tensor(_page(0.37)), "lines")
    assert info["reason"] == "" and angle == pytest.approx(-0.37, abs=0.01)
    assert calls == []


# --------------------------------------------------------------- check_mask_rotation
def _write_mask_meta(folder, record, rot_angle):
    folder.mkdir(exist_ok=True)
    with open(folder / f"{record}_mask.json", "w") as f:
        json.dump({"rot_angle": rot_angle, "height": 10, "width": 10}, f)


def test_check_mask_rotation_warns_on_a_different_angle(tmp_path, capsys):
    _write_mask_meta(tmp_path, "rec", 0.0)
    assert digitize.check_mask_rotation(str(tmp_path), "rec", -0.37) == 0.0
    out = capsys.readouterr().out
    assert "mask of record rec was predicted at rot_angle 0.0000" in out
    assert "the image is now rotated by -0.3700" in out
    assert "the mask does not fit." in out


def test_check_mask_rotation_is_quiet_within_the_tolerance(tmp_path, capsys):
    _write_mask_meta(tmp_path, "rec", -0.37)
    digitize.check_mask_rotation(str(tmp_path), "rec", -0.3701)
    assert capsys.readouterr().out == ""


def test_check_mask_rotation_is_quiet_without_a_json(tmp_path, capsys):
    assert np.isnan(digitize.check_mask_rotation(str(tmp_path), "missing", 0.0))
    assert capsys.readouterr().out == ""


# --------------------------------------------------------------------- parser and qc
def test_parser_rotation_defaults_to_lines():
    parser = digitize.get_parser()
    args = parser.parse_args(["-d", "data", "-o", "out"])
    assert args.rotation == "lines"
    args = parser.parse_args(["-d", "data", "-o", "out", "--rotation", "hough"])
    assert args.rotation == "hough"


def test_parser_interpolation_defaults_to_nearest():
    parser = digitize.get_parser()
    args = parser.parse_args(["-d", "data", "-o", "out"])
    assert args.interpolation == "nearest"
    args = parser.parse_args(["-d", "data", "-o", "out", "--interpolation", "bicubic"])
    assert args.interpolation == "bicubic"


def test_append_qc_row_writes_the_rotation_columns(tmp_path):
    qc = {"rotation_angle": -0.37, "rotation_coarse": -0.4, "rotation_residual_px": 0.1}
    digitize.append_qc_row(str(tmp_path), "rec", "column", qc, 0.0)
    with open(tmp_path / "qc.csv", newline="") as f:
        reader = csv.DictReader(f)
        row = next(reader)
        # The new columns are appended, the old ones keep their order.
        assert reader.fieldnames[:2] == ["record", "placement"]
        assert reader.fieldnames[-17:-14] == [
            "rotation_angle",
            "rotation_coarse",
            "rotation_residual_px",
        ]
    assert float(row["rotation_angle"]) == pytest.approx(-0.37)
    assert float(row["rotation_coarse"]) == pytest.approx(-0.4)
    assert float(row["rotation_residual_px"]) == pytest.approx(0.1)


# --------------------------------------------------------------------- end to end
def _label_mask(rows=None):
    """A flat trace per lead, in the column of the standard layout."""
    label = np.zeros((HEIGHT, WIDTH), dtype=np.uint8)
    pitch = WIDTH / 5
    for lead, value in digitize.LEAD_LABEL_MAPPING.items():
        column = int(digitize.STANDARD_LEAD_OFFSETS_SEC[lead] / 2.5)
        row = 300 + 100 * (value % 12)
        start = int(pitch / 2 + column * pitch)
        label[row, start : start + int(pitch)] = value
    return label


def test_run_with_rotation_lines_reports_the_angle(tmp_path, capsys):
    data_folder, mask_folder = tmp_path / "data", tmp_path / "masks"
    data_folder.mkdir()
    mask_folder.mkdir()
    write_png(_tensor(_page(0.37)), str(data_folder / "rec.png"))
    write_png(torch.from_numpy(_label_mask())[None], str(mask_folder / "rec_mask.png"))
    _write_mask_meta(mask_folder, "rec", 0.0)

    args = digitize.get_parser().parse_args(
        [
            "-d", str(data_folder),
            "-o", str(tmp_path / "out"),
            "--mask_folder", str(mask_folder),
            "--time_mapping", "bbox",
            "--rotation", "lines",
        ]
    )
    digitize.run(args)

    out = capsys.readouterr().out
    assert "Rotation for record rec:" in out
    assert "the mask does not fit." in out
    with open(tmp_path / "out" / "qc.csv", newline="") as f:
        row = next(csv.DictReader(f))
    assert float(row["rotation_angle"]) == pytest.approx(-0.37, abs=0.01)
    assert float(row["rotation_coarse"]) == pytest.approx(-0.4, abs=1e-9)
    assert float(row["rotation_residual_px"]) < 1.0


def test_run_with_rotation_lines_leaves_a_page_alone_without_an_angle(
    tmp_path, monkeypatch, capsys
):
    """What the fallback means downstream: the page is left alone, not turned by 12."""
    monkeypatch.setattr(digitize, "get_rotation_angle_fine", lambda image: 12.0)
    monkeypatch.setattr(digitize, "get_rotation_angle", lambda image: np.nan)
    _run_page(tmp_path, _no_grid_page(), "--rotation", "lines")
    out = capsys.readouterr().out
    assert "keeping the whole degree Hough angle" in out
    assert "No rotation angle found for record rec" in out
    assert _qc_angle(tmp_path) == 0.0


# ------------------------------------------------------ interpolation of the page
def test_bicubic_leaves_the_grid_lines_straighter_than_rotate():
    """What the flag is for, measured without a model.

    rotate() interpolates nearest, so the vertical lines of a turned page step from
    one column to the next instead of leaning. grid_line_slope fits a straight phase
    drift through the bands and its residual is what those steps leave behind, while
    the bicubic warp moves the same lines smoothly.
    """
    page = _tensor(_page(0.3))
    rotation = digitize.rotation_homography(-0.3, WIDTH, HEIGHT)
    nearest = digitize.grid_line_slope(rotate(page, -0.3))["residual"]
    bicubic = digitize.grid_line_slope(digitize.warp_page(page, rotation))["residual"]
    # Measured: 0.169 px of steps against 0.015 px, an order of magnitude apart.
    assert nearest > 0.1
    assert bicubic < nearest / 5


def _spy(monkeypatch, name):
    """Count the calls of digitize.<name> and keep their arguments, behaviour as is."""
    calls = []
    real = getattr(digitize, name)

    def spy(*args, **kwargs):
        calls.append(args)
        return real(*args, **kwargs)

    monkeypatch.setattr(digitize, name, spy)
    return calls


def _run_page(tmp_path, page, *flags):
    """run() over one synthetic page with the flat mask, no model and no data files."""
    data_folder, mask_folder = tmp_path / "data", tmp_path / "masks"
    data_folder.mkdir()
    mask_folder.mkdir()
    write_png(_tensor(page), str(data_folder / "rec.png"))
    write_png(torch.from_numpy(_label_mask())[None], str(mask_folder / "rec_mask.png"))
    digitize.run(
        digitize.get_parser().parse_args(
            [
                "-d", str(data_folder),
                "-o", str(tmp_path / "out"),
                "--mask_folder", str(mask_folder),
                "--time_mapping", "bbox",
                *flags,
            ]
        )
    )


def _qc_angle(tmp_path):
    with open(tmp_path / "out" / "qc.csv", newline="") as f:
        return float(next(csv.DictReader(f))["rotation_angle"])


def test_run_with_nearest_turns_the_page_with_rotate(tmp_path, monkeypatch):
    # All three stages resample the page themselves, which the spies would count, so
    # the interpolation tests keep them as they were: --rotation hough rotates nothing
    # of its own, and the whole degree Hough transform reads this page as -2.0.
    rotates, warps = _spy(monkeypatch, "rotate"), _spy(monkeypatch, "warp_page")
    _run_page(
        tmp_path, _page(2.0), "--rotation", "hough", "--perspective", "off",
        "--resolution", "keep",
    )
    assert _qc_angle(tmp_path) == pytest.approx(-2.0)
    assert len(rotates) == 1 and rotates[0][1] == pytest.approx(-2.0)
    assert warps == []


def test_run_with_bicubic_warps_the_page_once(tmp_path, monkeypatch):
    rotates, warps = _spy(monkeypatch, "rotate"), _spy(monkeypatch, "warp_page")
    _run_page(
        tmp_path, _page(2.0), "--interpolation", "bicubic", "--rotation", "hough",
        "--perspective", "off", "--resolution", "keep",
    )
    angle = _qc_angle(tmp_path)
    assert len(warps) == 1
    assert np.allclose(warps[0][1], digitize.rotation_homography(angle, WIDTH, HEIGHT))
    assert rotates == []


def test_run_with_bicubic_leaves_a_straight_page_alone(tmp_path, monkeypatch):
    rotates, warps = _spy(monkeypatch, "rotate"), _spy(monkeypatch, "warp_page")
    _run_page(
        tmp_path, _page(), "--interpolation", "bicubic", "--rotation", "hough",
        "--perspective", "off", "--resolution", "keep",
    )
    # An angle of zero keeps the old path, so the page is not resampled at all.
    assert _qc_angle(tmp_path) == 0.0
    assert warps == []
    assert len(rotates) == 1 and rotates[0][1] == 0.0


def test_run_with_perspective_reuses_the_bicubic_page(tmp_path, monkeypatch):
    warps = _spy(monkeypatch, "warp_page")
    _run_page(
        tmp_path, _page(2.0), "--interpolation", "bicubic", "--perspective", "lines"
    )
    # One warp for both purposes: the perspective is measured on the turned page, and
    # a page whose grid is straight again needs no rectifying warp on top.
    assert len(warps) == 1


def test_run_with_perspective_does_not_warp_a_straight_page(tmp_path, monkeypatch):
    warps = _spy(monkeypatch, "warp_page")
    _run_page(tmp_path, _page(), "--perspective", "lines")
    # The measurement would only warp by the identity, so it takes the image itself.
    assert warps == []


# ---------------------------------------------------------- the stages by default
def test_run_uses_all_three_stages_by_default(tmp_path):
    """What the three defaults are for, with no stage flag given at all.

    A page tilted by a fraction of a degree is what the whole degree Hough transform
    of --rotation hough cannot read, and the perspective and the resolution leave
    their QC columns at NaN as long as they do not run.
    """
    _run_page(tmp_path, _page(0.37))
    with open(tmp_path / "out" / "qc.csv", newline="") as f:
        row = next(csv.DictReader(f))

    assert float(row["rotation_angle"]) == pytest.approx(-0.37, abs=0.01)
    assert float(row["rotation_coarse"]) == pytest.approx(-0.4, abs=1e-9)
    assert float(row["rotation_residual_px"]) < 1.0
    assert np.isfinite(float(row["perspective_shift_px"]))
    assert float(row["perspective_residual_px"]) < 0.5
    assert float(row["grid_period_px"]) == pytest.approx(PERIOD, rel=1e-3)
    # The page is drawn at the 200 dpi scale, so the stage ran and kept its pixels.
    assert float(row["resolution_scale"]) == 1.0
