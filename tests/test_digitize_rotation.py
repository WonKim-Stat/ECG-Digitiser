"""Unit tests for the rotation estimate of src/run/digitize.py.

No model, no data files: every page is synthesised with numpy/cv2/torch.
"""
import csv
import json
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
        matrix = cv2.getRotationMatrix2D((WIDTH / 2, HEIGHT / 2), angle_deg, 1.0)
        image = cv2.warpAffine(
            image,
            matrix,
            (WIDTH, HEIGHT),
            flags=cv2.INTER_LINEAR,
            borderMode=cv2.BORDER_CONSTANT,
            borderValue=(255, 255, 255),
        )
    return image


def _tensor(np_image):
    """The page as run() reads it: [3, H, W] uint8."""
    return torch.from_numpy(np_image.transpose(2, 0, 1).copy())


def _blank():
    return np.full((HEIGHT, WIDTH, 3), 255, dtype=np.uint8)


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


def test_estimate_rotation_lines_keeps_the_coarse_angle_without_grid_lines():
    page = _no_grid_page()
    angle, info = digitize.estimate_rotation(_tensor(page), "lines")
    assert angle == digitize.get_rotation_angle_fine(page)
    assert "no grid lines found" in info["reason"]


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
def test_parser_rotation_defaults_to_hough():
    parser = digitize.get_parser()
    args = parser.parse_args(["-d", "data", "-o", "out"])
    assert args.rotation == "hough"
    args = parser.parse_args(["-d", "data", "-o", "out", "--rotation", "lines"])
    assert args.rotation == "lines"


def test_append_qc_row_writes_the_rotation_columns(tmp_path):
    qc = {"rotation_angle": -0.37, "rotation_coarse": -0.4, "rotation_residual_px": 0.1}
    digitize.append_qc_row(str(tmp_path), "rec", "column", qc, 0.0)
    with open(tmp_path / "qc.csv", newline="") as f:
        reader = csv.DictReader(f)
        row = next(reader)
        # The new columns are appended, the old ones keep their order.
        assert reader.fieldnames[:2] == ["record", "placement"]
        assert reader.fieldnames[-5:-2] == [
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
