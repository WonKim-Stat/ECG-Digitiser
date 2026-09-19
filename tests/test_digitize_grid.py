"""Unit tests for the shared column grid in src/run/digitize.py.

Covers fit_column_grid(), vectorise_grid() and the --time_mapping / --grid_pitch
flags. No model and no data files: a full 3x4 label mask with rhythm strip is
synthesised with numpy from a known grid (g0, P) and a known analytic signal,
then cut with cut_binary() exactly like run() does.
"""
import os
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


# --------------------------------------------------------------- the fake page
HEIGHT = 1700
WIDTH = 2200
# A 2.5 s column is 62.5 mm of the 215.9 mm page height.
P_TRUE = HEIGHT * 6.25 / 21.59
G0_TRUE = 118.0
MV_PER_PIXEL = 6.25 / P_TRUE
SHORT_SAMPLES = int(SHORT_SIGNAL_LENGTH_SEC * FREQUENCY)
LONG_SAMPLES = int(LONG_SIGNAL_LENGTH_SEC * FREQUENCY)

RHYTHM_LEAD = "II"
# The model never predicts a short lead II when a rhythm strip is present.
SHORT_COLUMNS = {
    "I": 0, "III": 0,
    "aVR": 1, "aVL": 1, "aVF": 1,
    "V1": 2, "V2": 2, "V3": 2,
    "V4": 3, "V5": 3, "V6": 3,
}
LEAD_ORDER = list(SHORT_COLUMNS) + [RHYTHM_LEAD]


def analytic_signal(t, lead):
    """A smooth, deterministic trace: a slow sine plus one narrow QRS bump."""
    k = LEAD_ORDER.index(lead)
    bump_centre = 0.55 + 0.13 * (k % 5)
    return (
        0.35 * np.sin(2 * np.pi * (0.7 * t + k / 13.0))
        + 0.55 * np.exp(-0.5 * ((t - bump_centre) / 0.035) ** 2)
    )


def _baseline_row(lead, is_long):
    ratio = Y_SHIFT_RATIO["full" if is_long else lead]
    return (1 - ratio) * HEIGHT


def _draw_lead(label, lead, g0=G0_TRUE, pitch=P_TRUE, x_shift=0):
    """Draw one 1 pixel thick trace into the label image."""
    is_long = lead == RHYTHM_LEAD
    column = 0 if is_long else SHORT_COLUMNS[lead]
    n_columns = 4 if is_long else 1
    start = g0 + column * pitch
    end = start + n_columns * pitch

    # Pixel c covers [c, c + 1), so it belongs to the column if its centre does.
    columns = np.arange(int(np.ceil(start - 0.5)), int(np.ceil(end - 0.5)))
    t = (columns + 0.5 - start) * SHORT_SIGNAL_LENGTH_SEC / pitch
    rows = np.rint(
        _baseline_row(lead, is_long) - analytic_signal(t, lead) / (6.25 / pitch)
    ).astype(int)
    label[rows, columns + x_shift] = LEAD_LABEL_MAPPING[lead]


def make_label_mask(leads=None, x_shifts=None, g0=G0_TRUE, pitch=P_TRUE):
    """A full standard 3x4 label mask with a 10 s rhythm strip, as [H, W] uint8."""
    leads = LEAD_ORDER if leads is None else leads
    x_shifts = x_shifts or {}
    label = np.zeros((HEIGHT, WIDTH), dtype=np.uint8)
    for lead in leads:
        _draw_lead(label, lead, g0=g0, pitch=pitch, x_shift=x_shifts.get(lead, 0))
    return label


def shave_start(label, lead, n_pixels):
    """Remove the n leftmost pixel columns of one lead from the label mask."""
    label = label.copy()
    is_lead = label == LEAD_LABEL_MAPPING[lead]
    filled = np.flatnonzero(is_lead.any(axis=0))
    first = filled[0]
    label[:, first : first + n_pixels][is_lead[:, first : first + n_pixels]] = 0
    return label


def cut(label):
    """Split the label mask the way run() does."""
    mask = torch.from_numpy(label)[None]
    image = torch.zeros(3, HEIGHT, WIDTH, dtype=torch.uint8)
    masks, positions, _ = digitize.cut_binary(mask, image)
    return masks, positions, image


def grid_signal(label, lead, pitch="page", image_height=HEIGHT):
    """fit_column_grid + vectorise_grid for one lead of a label mask."""
    masks, positions, image = cut(label)
    g0, P, long_leads, reason = digitize.fit_column_grid(
        masks, positions, image_height, pitch
    )
    assert g0 is not None, reason
    is_long = lead in long_leads
    out = digitize.vectorise_grid(
        image,
        masks[lead],
        positions[lead],
        g0,
        P,
        0 if is_long else SHORT_COLUMNS[lead],
        is_long,
        Y_SHIFT_RATIO,
        lead,
    )
    return out.numpy(), g0, P


def legacy_signal(label, lead):
    """digitize.vectorise() for one lead, with sec_per_pixel as run() computes it."""
    masks, positions, image = cut(label)
    widths = [m.shape[2] for m in masks.values() if m is not None]
    median = np.median(widths)
    sec_per_pixel = SHORT_SIGNAL_LENGTH_SEC / np.mean(
        [w for w in widths if w < 2 * median]
    )
    return digitize.vectorise(
        image,
        masks[lead],
        positions[lead]["y1"],
        sec_per_pixel,
        25 * sec_per_pixel / 10,
        Y_SHIFT_RATIO,
        lead,
    ).numpy()


# -------------------------------------------------------------- fit_column_grid
@pytest.mark.parametrize("pitch", ["page", "fit"])
def test_fit_column_grid_recovers_the_known_grid(pitch):
    masks, positions, _ = cut(make_label_mask())
    g0, P, long_leads, reason = digitize.fit_column_grid(
        masks, positions, HEIGHT, pitch
    )
    assert reason == ""
    assert long_leads == [RHYTHM_LEAD]
    assert g0 == pytest.approx(G0_TRUE, abs=0.5)
    assert P == pytest.approx(P_TRUE, abs=0.2)


def test_fit_column_grid_without_a_rhythm_strip():
    label = make_label_mask(leads=list(SHORT_COLUMNS))
    masks, positions, _ = cut(label)
    g0, P, long_leads, reason = digitize.fit_column_grid(masks, positions, HEIGHT)
    assert (g0, P) == (None, None)
    assert long_leads == []
    assert "rhythm" in reason


def test_fit_column_grid_needs_a_short_lead_in_every_column():
    leads = [lead for lead in LEAD_ORDER if lead not in ("V4", "V5", "V6")]
    masks, positions, _ = cut(make_label_mask(leads=leads))
    g0, P, _, reason = digitize.fit_column_grid(masks, positions, HEIGHT)
    assert (g0, P) == (None, None)
    assert "column" in reason


def test_fit_column_grid_falls_back_to_the_fitted_pitch_on_a_page_mismatch(capsys):
    masks, positions, _ = cut(make_label_mask())
    # A page height 10 % off makes the page pitch disagree with the drawn one.
    g0, P, _, reason = digitize.fit_column_grid(
        masks, positions, int(HEIGHT * 1.1), "page"
    )
    assert reason == ""
    assert P == pytest.approx(P_TRUE, abs=0.2)
    out = capsys.readouterr().out
    assert "does not match the fitted pitch" in out


def test_fit_column_grid_rejects_a_misplaced_column():
    shifts = {lead: 40 for lead in ("aVR", "aVL", "aVF")}
    masks, positions, _ = cut(make_label_mask(x_shifts=shifts))
    g0, P, _, reason = digitize.fit_column_grid(masks, positions, HEIGHT, "page")
    assert (g0, P) == (None, None)
    assert "off the grid" in reason


def test_shaving_a_whole_column_barely_moves_the_page_pitch_origin():
    label = make_label_mask()
    _, g0_intact, _ = grid_signal(label, "I", "page")
    for lead in ("aVR", "aVL", "aVF"):
        label = shave_start(label, lead, 1)
    _, g0_shaved, _ = grid_signal(label, "I", "page")
    # With a fixed pitch one moved edge of eight moves the origin by 1/8 px.
    assert abs(g0_shaved - g0_intact) <= 0.13


# --------------------------------------------------------------- vectorise_grid
def test_vectorise_grid_matches_the_analytic_short_lead():
    masks, positions, image = cut(make_label_mask())
    g0, P, long_leads, _ = digitize.fit_column_grid(masks, positions, HEIGHT, "page")
    out = digitize.vectorise_grid(
        image, masks["V5"], positions["V5"], g0, P,
        SHORT_COLUMNS["V5"], False, Y_SHIFT_RATIO, "V5",
    )
    assert isinstance(out, torch.Tensor)
    assert out.dtype == torch.float32
    assert out.shape == (SHORT_SAMPLES,)

    expected = analytic_signal(np.arange(SHORT_SAMPLES) / FREQUENCY, "V5")
    error = out.numpy() - expected
    snr = 10 * np.log10(np.mean(expected**2) / np.mean(error**2))
    assert snr > 20, f"SNR {snr:.1f} dB"


def test_vectorise_grid_rhythm_strip_is_ten_seconds():
    masks, positions, image = cut(make_label_mask())
    g0, P, long_leads, _ = digitize.fit_column_grid(masks, positions, HEIGHT, "page")
    out = digitize.vectorise_grid(
        image, masks[RHYTHM_LEAD], positions[RHYTHM_LEAD], g0, P,
        0, True, Y_SHIFT_RATIO, RHYTHM_LEAD,
    )
    assert out.shape == (LONG_SAMPLES,)
    expected = analytic_signal(np.arange(LONG_SAMPLES) / FREQUENCY, RHYTHM_LEAD)
    error = out.numpy() - expected
    snr = 10 * np.log10(np.mean(expected**2) / np.mean(error**2))
    assert snr > 20, f"SNR {snr:.1f} dB"


@pytest.mark.parametrize("lead", ["aVF", "V5"])
def test_vectorise_grid_pixel_centre_convention_has_no_lag(lead):
    """Would fail if the -0.5 pixel centre correction were dropped.

    The exact drawing grid is handed to vectorise_grid, so any lag comes from
    the sampling convention alone. Half a pixel is 1.25 samples here, and the
    narrow bump makes the best-fitting lag sharply defined.
    """
    masks, positions, image = cut(make_label_mask())
    out = digitize.vectorise_grid(
        image, masks[lead], positions[lead], G0_TRUE, P_TRUE,
        SHORT_COLUMNS[lead], False, Y_SHIFT_RATIO, lead,
    ).numpy()

    samples = np.arange(SHORT_SAMPLES)
    lags = np.arange(-4.0, 4.0001, 0.05)
    errors = [
        np.mean((out - analytic_signal((samples + lag) / FREQUENCY, lead)) ** 2)
        for lag in lags
    ]
    best = lags[int(np.argmin(errors))]
    assert abs(best) < 0.25, f"best lag {best:.2f} samples"


# ------------------------------------------------- robustness to a shaved start
@pytest.mark.parametrize("lead", ["aVF", "V5"])
@pytest.mark.parametrize("n_pixels", [1, 2, 3])
def test_shaved_start_leaves_the_grid_output_almost_unchanged(lead, n_pixels):
    label = make_label_mask()
    shaved = shave_start(label, lead, n_pixels)

    grid_intact, _, _ = grid_signal(label, lead, "page")
    grid_shaved, _, _ = grid_signal(shaved, lead, "page")
    diff = grid_shaved - grid_intact
    # Only the first few samples change: np.interp holds the missing left edge.
    assert np.max(np.abs(diff[10:])) < 0.02
    assert np.corrcoef(grid_shaved, grid_intact)[0, 1] > 0.999

    # The legacy bounding box mapping stretches the whole lead instead.
    legacy_intact = legacy_signal(label, lead)
    legacy_shaved = legacy_signal(shaved, lead)
    grid_rms = float(np.sqrt(np.mean(diff**2)))
    legacy_rms = float(np.sqrt(np.mean((legacy_shaved - legacy_intact) ** 2)))
    assert legacy_rms > grid_rms, f"legacy {legacy_rms:.4f} vs grid {grid_rms:.4f}"


# ----------------------------------------------------------------- parser flags
def test_parser_time_mapping_and_grid_pitch():
    parser = digitize.get_parser()
    args = parser.parse_args(["-d", "data", "-o", "out"])
    assert args.time_mapping == "bbox"
    assert args.grid_pitch == "page"

    args = parser.parse_args(
        ["-d", "data", "-o", "out", "--time_mapping", "grid", "--grid_pitch", "fit"]
    )
    assert args.time_mapping == "grid"
    assert args.grid_pitch == "fit"

    with pytest.raises(SystemExit):
        parser.parse_args(["-d", "data", "-o", "out", "--time_mapping", "nope"])


# ------------------------------------------------------------------- end to end
def _write_case(tmp_path, name, label):
    data_folder = tmp_path / f"{name}_data"
    mask_folder = tmp_path / f"{name}_masks"
    data_folder.mkdir(exist_ok=True)
    mask_folder.mkdir(exist_ok=True)
    blank = torch.full((3, HEIGHT, WIDTH), 255, dtype=torch.uint8)
    write_png(blank, str(data_folder / f"{name}.png"))
    write_png(torch.from_numpy(label)[None], str(mask_folder / f"{name}_mask.png"))
    return data_folder, mask_folder


def _run(tmp_path, name, label, time_mapping):
    data_folder, mask_folder = _write_case(tmp_path, name, label)
    output_folder = tmp_path / f"{name}_{time_mapping}_out"
    args = digitize.get_parser().parse_args(
        [
            "-d", str(data_folder),
            "-o", str(output_folder),
            "--mask_folder", str(mask_folder),
            "--time_mapping", time_mapping,
            "--no-verbose",
        ]
    )
    digitize.run(args)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        record = wfdb.rdrecord(str(output_folder / name))
    return record


def test_run_with_time_mapping_grid_places_every_lead_in_its_column(tmp_path):
    record = _run(tmp_path, "rec", make_label_mask(), "grid")
    signals = record.p_signal
    assert signals.shape == (LONG_SAMPLES, len(LEAD_ORDER))

    for lead, column in SHORT_COLUMNS.items():
        channel = signals[:, record.sig_name.index(lead)]
        start = column * SHORT_SAMPLES
        window = channel[start : start + SHORT_SAMPLES]
        assert np.all(np.isfinite(window))
        assert np.all(np.isnan(np.delete(channel, np.s_[start : start + SHORT_SAMPLES])))
        expected = analytic_signal(np.arange(SHORT_SAMPLES) / FREQUENCY, lead)
        assert np.max(np.abs(window - expected)) < 0.05

    rhythm = signals[:, record.sig_name.index(RHYTHM_LEAD)]
    assert np.all(np.isfinite(rhythm))
    expected = analytic_signal(np.arange(LONG_SAMPLES) / FREQUENCY, RHYTHM_LEAD)
    assert np.max(np.abs(rhythm - expected)) < 0.05


def test_run_falls_back_to_bbox_without_a_rhythm_strip(tmp_path, capsys):
    label = make_label_mask(leads=list(SHORT_COLUMNS))
    grid = _run(tmp_path, "norhythm", label, "grid")
    out = capsys.readouterr().out
    assert "falling back to --time_mapping bbox" in out
    assert "rhythm" in out

    bbox = _run(tmp_path, "norhythm", label, "bbox")
    assert grid.sig_name == bbox.sig_name
    assert np.array_equal(grid.p_signal, bbox.p_signal, equal_nan=True)
