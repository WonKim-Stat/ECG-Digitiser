"""Unit tests for the perspective estimate of src/run/digitize.py.

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
# Pixel centres sit at i + 0.5 in the pipeline and at i in cv2.
HALF = np.array([[1, 0, -0.5], [0, 1, -0.5], [0, 0, 1]], float)


@lru_cache(maxsize=None)
def _page(grid="both"):
    """A page with a 1 mm grid, a few traces and a text blob.

    grid="vertical" leaves the horizontal lines out, so that only one line family is
    left. Drawing a page takes a moment, so they are cached and no caller may write
    into one.
    """
    image = np.full((HEIGHT, WIDTH, 3), 255, dtype=np.uint8)
    # Light red grid lines, every fifth one darker and thicker as on a printed page.
    for k in range(int(WIDTH / PERIOD) + 1):
        x = int(round(k * PERIOD))
        if x < WIDTH:
            heavy = k % 5 == 0
            cv2.line(image, (x, 0), (x, HEIGHT - 1),
                     (120, 120, 255) if heavy else (190, 190, 255), 2 if heavy else 1)
    if grid == "both":
        for k in range(int(HEIGHT / PERIOD) + 1):
            y = int(round(k * PERIOD))
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
    return image


def _tensor(np_image):
    """The page as run() reads it: [3, H, W] uint8."""
    return torch.from_numpy(np_image.transpose(2, 0, 1).copy())


def _blank():
    return np.full((HEIGHT, WIDTH, 3), 255, dtype=np.uint8)


def _no_grid_page():
    """A page with lines the Hough transform sees, but no printed 1 mm grid."""
    image = _blank()
    for y in range(100, HEIGHT - 100, 50):
        cv2.line(image, (0, y), (WIDTH - 1, y), (0, 0, 0), 2)
    for x in (137, 402, 913, 1290, 1755, 2050):
        cv2.line(image, (x, 0), (x, HEIGHT - 1), (0, 0, 0), 3)
    return image


def _corner_homography(fraction, seed):
    """A page whose four corners are pulled inwards by up to fraction of the page."""
    rng = np.random.default_rng(seed)
    source = np.float32([[0, 0], [WIDTH, 0], [WIDTH, HEIGHT], [0, HEIGHT]])
    sign = np.float32([[1, 1], [-1, 1], [-1, -1], [1, -1]])
    move = rng.uniform(0.2, 1.0, (4, 2)).astype(np.float32)
    move *= fraction * np.float32([WIDTH, HEIGHT])
    return cv2.getPerspectiveTransform(source, source + sign * move).astype(float)


@lru_cache(maxsize=None)
def _warped(name):
    """The page under a known homography, plus that homography (clean -> page)."""
    if name == "shear":
        # A badly fed page: the vertical lines lean by a quarter of a degree.
        homography = np.array([[1.0, 0.0044, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0]])
    elif name == "tilted":
        # A photographed page: a perspective on top of a tilt of 1.5 degrees.
        homography = _corner_homography(0.01, 3) @ digitize.rotation_homography(
            -1.5, WIDTH, HEIGHT
        )
    else:
        homography = _corner_homography(0.005 if name == "persp05" else 0.015, 1)
    warped = cv2.warpPerspective(
        _page(),
        HALF @ homography @ np.linalg.inv(HALF),
        (WIDTH, HEIGHT),
        flags=cv2.INTER_CUBIC,
        borderMode=cv2.BORDER_CONSTANT,
        borderValue=(255, 255, 255),
    )
    return warped, homography


def _leftover(total, truth):
    """Largest thing left of clean page -> final frame after a scale and a shift.

    The pipeline may rescale and move the page, and nothing else: whatever a lattice
    over the page keeps after those three parameters are fitted is the error.
    """
    x, y = np.meshgrid(
        np.linspace(200, WIDTH - 200, 30), np.linspace(200, HEIGHT - 200, 20)
    )
    points = np.stack([x.ravel(), y.ravel()], axis=1)
    mapped = digitize._apply_homography(total @ truth, points)
    n = len(points)
    design = np.zeros((2 * n, 3))
    design[:n, 0], design[:n, 1] = points[:, 0], 1.0
    design[n:, 0], design[n:, 2] = points[:, 1], 1.0
    target = np.concatenate([mapped[:, 0], mapped[:, 1]])
    fit = np.linalg.lstsq(design, target, rcond=None)[0]
    return float(np.abs(target - design @ fit).max())


def _psi(amplitudes):
    """Where the lines of every window sit, in cycles, as the fit reads them."""
    return -np.angle(amplitudes) / (2 * np.pi)


# ------------------------------------------------------------- _grid_phase_field
def test_grid_phase_field_lays_a_window_lattice_over_the_page():
    amplitudes, x, y = digitize._grid_phase_field(
        digitize._darkness(_tensor(_page())), PERIOD, 0
    )
    assert amplitudes.shape == x.shape == y.shape
    assert amplitudes.shape[0] == HEIGHT // digitize.GRID_LINE_BAND_HEIGHT
    assert amplitudes.shape[1] > 10
    # x runs along the windows of a band, y along the bands, both inside the page.
    assert np.all(np.diff(x, axis=1) > 0) and np.all(np.diff(x, axis=0) == 0)
    assert np.all(np.diff(y, axis=0) > 0) and np.all(np.diff(y, axis=1) == 0)
    assert 0 < x.min() and x.max() < WIDTH and 0 < y.min() and y.max() < HEIGHT
    # Every window is about 32 lines long and they overlap by half.
    assert np.diff(x, axis=1)[0, 0] == pytest.approx(
        digitize.PERSPECTIVE_WINDOW_LINES * PERIOD / 2, abs=1.0
    )


def test_grid_phase_field_swaps_the_axes_for_the_horizontal_lines():
    amplitudes, x, y = digitize._grid_phase_field(
        digitize._darkness(_tensor(_page())), PERIOD, 1
    )
    assert amplitudes.shape == x.shape == y.shape
    assert amplitudes.shape[0] == WIDTH // digitize.GRID_LINE_BAND_HEIGHT
    # The windows now run down the page, the bands across it.
    assert np.all(np.diff(y, axis=1) > 0) and np.all(np.diff(y, axis=0) == 0)
    assert np.all(np.diff(x, axis=0) > 0) and np.all(np.diff(x, axis=1) == 0)


def test_grid_phase_field_follows_a_page_moved_to_the_right():
    darkness = digitize._darkness(_tensor(_page()))
    here, _, _ = digitize._grid_phase_field(darkness, PERIOD, 0)
    there, _, _ = digitize._grid_phase_field(np.roll(darkness, 2, axis=1), PERIOD, 0)
    # The lines of a window sit at (psi + k) * period, so two pixels to the right
    # move psi by 2 / period; the opposite sign would be off by twice that.
    moved = (_psi(there) - _psi(here) + 0.5) % 1.0 - 0.5
    assert np.median(moved) == pytest.approx(2 / PERIOD, abs=0.01)
    # Only the window the roll wrapped around disagrees.
    assert np.abs(moved[:, 1:] - 2 / PERIOD).max() < 0.02


def test_grid_phase_field_has_no_amplitude_on_a_blank_page():
    amplitudes, _, _ = digitize._grid_phase_field(
        digitize._darkness(_tensor(_blank())), PERIOD, 0
    )
    assert amplitudes.size > 0
    assert np.abs(amplitudes).max() == pytest.approx(0.0, abs=1e-9)


def test_grid_phase_field_is_empty_on_a_page_without_room_for_a_window():
    narrow = _tensor(_page())[:, :, : int(10 * PERIOD)]
    amplitudes, x, y = digitize._grid_phase_field(digitize._darkness(narrow), PERIOD, 0)
    assert amplitudes.size == 0 and x.size == 0 and y.size == 0


# ----------------------------------------------------------- estimate_perspective
@pytest.mark.parametrize("name", ["persp05", "persp15", "shear"])
def test_estimate_perspective_straightens_a_warped_page(name):
    page, truth = _warped(name)
    matrix, info = digitize.estimate_perspective(_tensor(page))
    assert info["reason"] == ""
    assert matrix is not None
    assert info["shift"] > digitize.PERSPECTIVE_MIN_SHIFT_PX
    assert info["residual"] < digitize.GRID_LINE_MAX_SLOPE_RESIDUAL * info["period"]
    assert info["period"] == pytest.approx(PERIOD, rel=0.02)
    assert _leftover(matrix, truth) <= 0.15


def test_estimate_perspective_composes_with_the_rotation_stage():
    page, truth = _warped("tilted")
    angle, rotation_info = digitize.estimate_rotation(_tensor(page), "lines")
    assert rotation_info["reason"] == ""
    # The perspective turns the page a little as well, so the tilt the grid lines
    # show is not the 1.5 degrees the page was built with, only most of it.
    assert 1.0 < angle < 1.5
    # As run() does it: the perspective is measured in the frame of the final warp.
    rotation = digitize.rotation_homography(angle, WIDTH, HEIGHT)
    matrix, info = digitize.estimate_perspective(
        digitize.warp_page(_tensor(page), rotation)
    )
    assert info["reason"] == ""
    assert _leftover(matrix @ rotation, truth) <= 0.15


def test_estimate_perspective_keeps_the_centre_and_the_scale():
    page, _ = _warped("persp15")
    matrix, _ = digitize.estimate_perspective(_tensor(page))
    centre = np.array([[WIDTH / 2, HEIGHT / 2]])
    assert digitize._apply_homography(matrix, centre)[0] == pytest.approx(
        centre[0], abs=1e-6
    )
    around = centre + np.array([[1.0, 0], [0, 1.0], [-1.0, 0], [0, -1.0]])
    mapped = digitize._apply_homography(matrix, around)
    jacobian = np.stack([mapped[0] - mapped[2], mapped[1] - mapped[3]], axis=1) / 2
    assert np.linalg.det(jacobian) == pytest.approx(1.0, abs=1e-6)


def test_estimate_perspective_leaves_a_straight_page_alone():
    matrix, info = digitize.estimate_perspective(_tensor(_page()))
    # The dead band: no homography, and nothing to warn about either.
    assert matrix is None
    assert info["reason"] == ""
    assert info["shift"] < digitize.PERSPECTIVE_MIN_SHIFT_PX
    assert info["windows"] > digitize.PERSPECTIVE_MIN_WINDOWS


def test_estimate_perspective_takes_a_numpy_page_as_well():
    page, _ = _warped("persp05")
    from_numpy, _ = digitize.estimate_perspective(page.transpose(2, 0, 1))
    from_tensor, _ = digitize.estimate_perspective(_tensor(page))
    assert np.allclose(from_numpy, from_tensor, atol=1e-12)


def test_estimate_perspective_gives_up_on_a_blank_page():
    matrix, info = digitize.estimate_perspective(_tensor(_blank()))
    assert matrix is None
    assert info["reason"] == "no grid line period found"


def test_estimate_perspective_needs_grid_lines():
    matrix, info = digitize.estimate_perspective(_tensor(_no_grid_page()))
    assert matrix is None
    assert "no grid lines found" in info["reason"]


def test_estimate_perspective_needs_both_line_families():
    matrix, info = digitize.estimate_perspective(_tensor(_page("vertical")))
    # The vertical lines alone fix u, nothing says where the rows of the grid are.
    assert matrix is None
    assert info["reason"] != ""


# ---------------------------------------------------- rotation_homography, warp_page
def _line_positions(image):
    """Where the vertical lines sit in every band, relative to the middle band."""
    profiles, _ = digitize._band_profiles(digitize._darkness(image))
    comb = digitize._grid_line_comb(profiles, PERIOD)
    positions = -np.unwrap(np.angle(comb)) / (2 * np.pi) * PERIOD
    return positions - positions[len(positions) // 2]


def test_rotation_homography_is_the_map_rotate_applies():
    page = _tensor(_page())
    turned = _line_positions(rotate(page, 1.5))
    right = digitize.warp_page(page, digitize.rotation_homography(1.5, WIDTH, HEIGHT))
    wrong = digitize.warp_page(page, digitize.rotation_homography(-1.5, WIDTH, HEIGHT))
    # The lines of the two agree band by band, the other sign turns the page the
    # other way and drifts by tens of pixels over the page.
    assert np.abs(_line_positions(right) - turned).max() < 0.3
    assert np.abs(_line_positions(wrong) - turned).max() > 10.0


def test_warp_page_with_the_identity_keeps_every_pixel():
    page = _tensor(_page())
    assert torch.equal(digitize.warp_page(page, np.eye(3)), page)


# -------------------------------------------------------------- mask frame and QC
def _write_mask_meta(folder, record, rot_angle, homography=None):
    folder.mkdir(exist_ok=True)
    meta = {"rot_angle": rot_angle, "height": 10, "width": 10}
    if homography is not None:
        meta["homography"] = np.asarray(homography).tolist()
    with open(folder / f"{record}_mask.json", "w") as f:
        json.dump(meta, f)


def _tilt(amount):
    """A homography that pulls the top right corner of a 10 px frame down."""
    return np.array([[1.0, 0.0, 0.0], [amount, 1.0, 0.0], [0.0, 0.0, 1.0]])


def test_save_mask_files_stores_the_homography_only_when_there_is_one(tmp_path):
    mask = torch.zeros((1, 6, 8), dtype=torch.uint8)
    digitize.save_mask_files(mask, "plain", str(tmp_path), 0.5)
    digitize.save_mask_files(mask, "warped", str(tmp_path), 0.5, _tilt(0.01))
    with open(tmp_path / "plain_mask.json") as f:
        assert "homography" not in json.load(f)
    with open(tmp_path / "warped_mask.json") as f:
        meta = json.load(f)
    assert meta["height"] == 6 and meta["width"] == 8
    assert np.allclose(meta["homography"], _tilt(0.01))


def test_check_mask_rotation_warns_when_the_frames_differ(tmp_path, capsys):
    _write_mask_meta(tmp_path, "rec", 0.0, _tilt(0.05))
    assert digitize.check_mask_rotation(str(tmp_path), "rec", 0.0, _tilt(0.02)) == 0.0
    out = capsys.readouterr().out
    assert "was predicted in a frame whose corners are 0.3 px off" in out
    assert "the mask does not fit." in out


def test_check_mask_rotation_is_quiet_on_the_same_frame(tmp_path, capsys):
    _write_mask_meta(tmp_path, "rec", 0.0, _tilt(0.05))
    digitize.check_mask_rotation(str(tmp_path), "rec", 0.0, _tilt(0.05))
    assert capsys.readouterr().out == ""


def test_check_mask_rotation_warns_when_only_one_frame_is_warped(tmp_path, capsys):
    # An older JSON without a homography means the plain rotation of its angle.
    _write_mask_meta(tmp_path, "rec", 0.0)
    digitize.check_mask_rotation(str(tmp_path), "rec", 0.0, _tilt(0.05))
    assert "the mask does not fit." in capsys.readouterr().out


def test_check_mask_rotation_keeps_its_old_behaviour(tmp_path, capsys):
    _write_mask_meta(tmp_path, "rec", -0.37)
    assert digitize.check_mask_rotation(str(tmp_path), "rec", -0.3701) == -0.37
    assert capsys.readouterr().out == ""


def test_parser_perspective_defaults_to_lines():
    parser = digitize.get_parser()
    args = parser.parse_args(["-d", "data", "-o", "out"])
    assert args.perspective == "lines"
    args = parser.parse_args(["-d", "data", "-o", "out", "--perspective", "off"])
    assert args.perspective == "off"


def test_append_qc_row_writes_the_perspective_columns(tmp_path):
    qc = {"perspective_shift_px": 3.25, "perspective_residual_px": 0.06}
    digitize.append_qc_row(str(tmp_path), "rec", "column", qc, 0.0)
    with open(tmp_path / "qc.csv", newline="") as f:
        reader = csv.DictReader(f)
        row = next(reader)
        # The new columns are appended, the old ones keep their order.
        assert reader.fieldnames[:2] == ["record", "placement"]
        assert reader.fieldnames[-9:-7] == [
            "perspective_shift_px",
            "perspective_residual_px",
        ]
    assert float(row["perspective_shift_px"]) == pytest.approx(3.25)
    assert float(row["perspective_residual_px"]) == pytest.approx(0.06)


# --------------------------------------------------------------------- end to end
def _label_mask():
    """A flat trace per lead, in the column of the standard layout."""
    label = np.zeros((HEIGHT, WIDTH), dtype=np.uint8)
    pitch = WIDTH / 5
    for lead, value in digitize.LEAD_LABEL_MAPPING.items():
        column = int(digitize.STANDARD_LEAD_OFFSETS_SEC[lead] / 2.5)
        row = 300 + 100 * (value % 12)
        start = int(pitch / 2 + column * pitch)
        label[row, start : start + int(pitch)] = value
    return label


def test_run_with_perspective_lines_reports_the_shift(tmp_path, capsys):
    data_folder, mask_folder = tmp_path / "data", tmp_path / "masks"
    data_folder.mkdir()
    mask_folder.mkdir()
    page, _ = _warped("persp15")
    write_png(_tensor(page), str(data_folder / "rec.png"))
    write_png(torch.from_numpy(_label_mask())[None], str(mask_folder / "rec_mask.png"))
    with open(mask_folder / "rec_mask.json", "w") as f:
        json.dump({"rot_angle": 0.0, "height": HEIGHT, "width": WIDTH}, f)

    args = digitize.get_parser().parse_args(
        [
            "-d", str(data_folder),
            "-o", str(tmp_path / "out"),
            "--mask_folder", str(mask_folder),
            "--time_mapping", "bbox",
            "--perspective", "lines",
        ]
    )
    digitize.run(args)

    out = capsys.readouterr().out
    assert "Perspective for record rec:" in out
    # The mask was predicted on the page as it came, which is not this frame.
    assert "the mask does not fit." in out
    with open(tmp_path / "out" / "qc.csv", newline="") as f:
        row = next(csv.DictReader(f))
    assert float(row["perspective_shift_px"]) > digitize.PERSPECTIVE_MIN_SHIFT_PX
    assert float(row["perspective_residual_px"]) < 0.5


def test_run_without_the_stage_leaves_the_qc_columns_empty(tmp_path, capsys):
    data_folder, mask_folder = tmp_path / "data", tmp_path / "masks"
    data_folder.mkdir()
    mask_folder.mkdir()
    write_png(_tensor(_page()), str(data_folder / "rec.png"))
    write_png(torch.from_numpy(_label_mask())[None], str(mask_folder / "rec_mask.png"))

    args = digitize.get_parser().parse_args(
        [
            "-d", str(data_folder),
            "-o", str(tmp_path / "out"),
            "--mask_folder", str(mask_folder),
            "--time_mapping", "bbox",
            "--perspective", "off",
        ]
    )
    digitize.run(args)

    assert "Perspective for record rec:" not in capsys.readouterr().out
    with open(tmp_path / "out" / "qc.csv", newline="") as f:
        row = next(csv.DictReader(f))
    assert np.isnan(float(row["perspective_shift_px"]))
    assert np.isnan(float(row["perspective_residual_px"]))
