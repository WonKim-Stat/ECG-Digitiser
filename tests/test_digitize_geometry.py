"""Unit tests for the pure geometry helpers in src/run/digitize.py.

No model, no data files: everything is synthesised with numpy/cv2/torch.
"""
import cv2
import numpy as np
import pytest
import torch

from src.run import digitize


def _grid_image(angle_deg, size=1600, spacing=50, thickness=2):
    """White canvas with near-horizontal lines tilted by `angle_deg`.

    Sign convention (verified against get_rotation_angle): a line drawn going
    *down* to the right in image coordinates (y increasing with x, i.e. a
    clockwise tilt on screen) yields a POSITIVE recovered angle.
    """
    image = np.full((size, size, 3), 255, dtype=np.uint8)
    slope = np.tan(np.deg2rad(angle_deg))
    for y in range(100, size - 100, spacing):
        cv2.line(
            image,
            (0, int(y)),
            (size - 1, int(y + (size - 1) * slope)),
            (0, 0, 0),
            thickness,
        )
    return image


# --------------------------------------------------------- get_median_degrees
def test_get_median_degrees_horizontal_lines_are_zero():
    # theta = pi/2 is a horizontal line in the Hough parameterisation.
    lines = np.array([[[10.0, np.pi / 2]], [[60.0, np.pi / 2]]])
    assert digitize.get_median_degrees(lines) == pytest.approx(0.0)


def test_get_median_degrees_matches_the_theta_convention():
    theta = np.deg2rad(95.0)
    lines = np.array([[[10.0, theta]], [[20.0, theta]], [[30.0, theta]]])
    # angle = -(90 - theta_degrees) = +5 degrees
    assert digitize.get_median_degrees(lines) == pytest.approx(5.0, abs=1e-3)


def test_get_median_degrees_uses_the_median_not_the_mean():
    thetas = np.deg2rad([92.0, 92.0, 92.0, 130.0])
    lines = np.array([[[float(i), t]] for i, t in enumerate(thetas)])
    assert digitize.get_median_degrees(lines) == pytest.approx(2.0, abs=1e-3)


# -------------------------------------------------------------- filter_lines
def test_filter_lines_returns_none_for_no_lines():
    assert digitize.filter_lines(None) is None
    assert digitize.filter_lines(np.empty((0, 1, 2))) is None


def test_filter_lines_drops_lines_outside_the_degree_window():
    lines = np.array(
        [
            [[10.0, np.deg2rad(90.0)]],
            [[20.0, np.deg2rad(91.0)]],
            [[30.0, np.deg2rad(0.0)]],  # vertical -> outside a 20 degree window
        ]
    )
    kept = digitize.filter_lines(lines, degree_window=20, parallelism_count=0)
    assert kept.shape == (2, 1, 2)
    assert np.allclose(kept[:, 0, 1], np.deg2rad([90.0, 91.0]))


def test_filter_lines_enforces_parallelism_count():
    lines = np.array(
        [
            [[10.0, np.deg2rad(90.0)]],
            [[20.0, np.deg2rad(90.5)]],
            [[30.0, np.deg2rad(91.0)]],
            [[40.0, np.deg2rad(105.0)]],  # within 30 deg of horizontal, but alone
        ]
    )
    kept = digitize.filter_lines(
        lines, degree_window=30, parallelism_count=3, parallelism_window=2
    )
    assert kept.shape == (3, 1, 2)
    assert np.deg2rad(105.0) not in kept[:, 0, 1]
    # With a stricter requirement than any cluster can satisfy, nothing is kept.
    assert (
        digitize.filter_lines(
            lines, degree_window=30, parallelism_count=10, parallelism_window=2
        )
        is None
    )


# ----------------------------------------------------------- get_rotation_angle
@pytest.mark.parametrize("angle", [0.0, 3.0, -3.0, 7.0, -7.0])
def test_get_rotation_angle_recovers_a_known_tilt(angle):
    image = _grid_image(angle)
    recovered = digitize.get_rotation_angle(image)
    assert not np.isnan(recovered)
    assert recovered == pytest.approx(angle, abs=1.0)


def test_get_rotation_angle_is_nan_without_lines():
    blank = np.full((1600, 1600, 3), 255, dtype=np.uint8)
    assert np.isnan(digitize.get_rotation_angle(blank))


def test_get_lines_returns_none_on_a_blank_image():
    blank = np.full((1600, 1600, 3), 255, dtype=np.uint8)
    assert digitize.get_lines(blank) is None


# ----------------------------------------------------------------- cut_to_mask
def test_cut_to_mask_crops_to_the_mask_bounding_box():
    img = torch.arange(3 * 10 * 12, dtype=torch.float32).reshape(3, 10, 12)
    mask = torch.zeros(1, 10, 12)
    mask[0, 3:6, 4:9] = 1

    cropped = digitize.cut_to_mask(img, mask)
    assert cropped.shape == (3, 3, 5)
    assert torch.equal(cropped, img[:, 3:6, 4:9])

    cropped, y1, x1 = digitize.cut_to_mask(img, mask, return_y1=True)
    assert (y1, x1) == (3, 4)
    assert cropped.shape == (3, 3, 5)


def test_cut_to_mask_single_pixel_mask():
    img = torch.arange(1 * 5 * 5, dtype=torch.float32).reshape(1, 5, 5)
    mask = torch.zeros(1, 5, 5)
    mask[0, 2, 3] = 1
    cropped, y1, x1 = digitize.cut_to_mask(img, mask, return_y1=True)
    assert cropped.shape == (1, 1, 1)
    assert (y1, x1) == (2, 3)
    assert cropped.item() == img[0, 2, 3].item()


# ------------------------------------------------------------------ cut_binary
def test_cut_binary_splits_leads_and_records_offsets():
    h, w = 40, 60
    label_mask = torch.zeros(1, h, w, dtype=torch.int64)
    label_mask[0, 5:10, 2:20] = digitize.LEAD_LABEL_MAPPING["I"]
    label_mask[0, 20:24, 30:50] = digitize.LEAD_LABEL_MAPPING["V3"]
    image = torch.arange(3 * h * w, dtype=torch.float32).reshape(3, h, w)

    masks, positions, images = digitize.cut_binary(label_mask, image)

    assert set(masks) == set(digitize.LEAD_LABEL_MAPPING)
    assert masks["I"].shape == (1, 5, 18)
    assert positions["I"] == {"y1": 5, "x1": 2}
    assert images["I"].shape == (3, 5, 18)
    assert torch.equal(images["I"], image[:, 5:10, 2:20])
    assert int(masks["I"].sum()) == 5 * 18

    assert masks["V3"].shape == (1, 4, 20)
    assert positions["V3"] == {"y1": 20, "x1": 30}

    # Leads with no pixels are reported as None in all three dicts.
    for lead in ("II", "aVR", "V6"):
        assert masks[lead] is None
        assert positions[lead] is None
        assert images[lead] is None


# --------------------------------------------------------------- resolve_device
@pytest.mark.parametrize("device", ["cpu", "cuda", "mps"])
def test_resolve_device_passes_explicit_choices_through(device):
    assert digitize.resolve_device(device) == device


def test_resolve_device_auto_returns_a_known_device():
    assert digitize.resolve_device("auto") in {"cuda", "mps", "cpu"}


# -------------------------------------------------------------------- vectorise
def test_vectorise_smoke_shape_and_finiteness():
    """Smoke test only.

    vectorise() mixes the Y_SHIFT_RATIO baseline convention (which assumes a
    full page image) with the cropped mask, so the absolute amplitudes are not
    meaningful for a synthetic canvas. We only assert the output shape and
    finiteness.
    """
    h, w = 200, 250
    mask = torch.zeros(1, h, w)
    x = np.arange(w)
    y = (100 + 20 * np.sin(2 * np.pi * x / 50)).astype(int)
    mask[0, y, x] = 1

    image_rotated = torch.zeros(3, 1700, 2200)
    sec_per_pixel = 2.5 / w  # short lead: 2.5 s across the crop
    out = digitize.vectorise(
        image_rotated,
        mask,
        signal_cropped=50,
        sec_per_pixel=sec_per_pixel,
        mV_per_pixel=25 * sec_per_pixel / 10,
        y_shift_ratio=digitize.Y_SHIFT_RATIO,
        lead="I",
    )
    assert out.shape == (int(2.5 * digitize.FREQUENCY),)
    assert torch.isfinite(out).all()
