"""Unit tests for the shared column grid in src/run/digitize.py.

Covers fit_column_grid(), vectorise_grid() and the --time_mapping / --grid_pitch
flags. No model and no data files: a full 3x4 label mask with rhythm strip is
synthesised with numpy from a known grid (g0, P) and a known analytic signal,
then cut with cut_binary() exactly like run() does.
"""
import csv
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
    """A smooth, deterministic trace: a slow sine plus one narrow QRS bump.

    III and aVF follow from their neighbours so that the Einthoven and the
    Goldberger sum vanish on the fake page, as they do on a real one.
    """
    if lead == "III":
        return analytic_signal(t, "II") - analytic_signal(t, "I")
    if lead == "aVF":
        return -(analytic_signal(t, "aVR") + analytic_signal(t, "aVL"))
    k = LEAD_ORDER.index(lead)
    bump_centre = 0.55 + 0.13 * (k % 5)
    return (
        0.35 * np.sin(2 * np.pi * (0.7 * t + k / 13.0))
        + 0.55 * np.exp(-0.5 * ((t - bump_centre) / 0.035) ** 2)
    )


def _baseline_row(lead, is_long, scale=1.0):
    ratio = Y_SHIFT_RATIO["full" if is_long else lead]
    return digitize.baseline_row(ratio, HEIGHT, scale)


def _draw_lead(label, lead, g0=G0_TRUE, pitch=P_TRUE, x_shift=0, scale=1.0):
    """Draw one 1 pixel thick trace into the label image."""
    is_long = lead == RHYTHM_LEAD
    column = 0 if is_long else SHORT_COLUMNS[lead]
    n_columns = 4 if is_long else 1
    start = g0 + column * pitch
    end = start + n_columns * pitch

    # Pixel c covers [c, c + 1), so it belongs to the column if its centre does.
    columns = np.arange(int(np.ceil(start - 0.5)), int(np.ceil(end - 0.5)))
    t = (columns + 0.5 - start) * SHORT_SIGNAL_LENGTH_SEC / pitch
    # Row r covers [r, r + 1), so the trace darkens the pixel row containing it.
    rows = np.floor(
        _baseline_row(lead, is_long, scale) - analytic_signal(t, lead) / (6.25 / pitch)
    ).astype(int)
    label[rows, columns + x_shift] = LEAD_LABEL_MAPPING[lead]


def make_label_mask(leads=None, x_shifts=None, g0=G0_TRUE, pitch=P_TRUE, scale=1.0):
    """A full standard 3x4 label mask with a 10 s rhythm strip, as [H, W] uint8."""
    leads = LEAD_ORDER if leads is None else leads
    x_shifts = x_shifts or {}
    label = np.zeros((HEIGHT, WIDTH), dtype=np.uint8)
    for lead in leads:
        _draw_lead(
            label, lead, g0=g0, pitch=pitch, x_shift=x_shifts.get(lead, 0), scale=scale
        )
    return label


def make_rescaled_label(scale):
    """A page rescaled about its centre, as a crop that keeps the image size does."""
    return make_label_mask(
        g0=WIDTH / 2 + scale * (G0_TRUE - WIDTH / 2),
        pitch=P_TRUE * scale,
        scale=scale,
    )


def shave_start(label, lead, n_pixels):
    """Remove the n leftmost pixel columns of one lead from the label mask."""
    label = label.copy()
    is_lead = label == LEAD_LABEL_MAPPING[lead]
    filled = np.flatnonzero(is_lead.any(axis=0))
    first = filled[0]
    label[:, first : first + n_pixels][is_lead[:, first : first + n_pixels]] = 0
    return label


def extend_end(label, lead, x_end):
    """Continue a lead's trace to the right, up to pixel column x_end exclusive."""
    label = label.copy()
    value = LEAD_LABEL_MAPPING[lead]
    last = np.flatnonzero((label == value).any(axis=0))[-1]
    label[np.flatnonzero(label[:, last] == value)[-1], last + 1 : x_end] = value
    return label


def extend_start(label, lead, x_start):
    """Continue a lead's trace to the left, back to pixel column x_start."""
    label = label.copy()
    value = LEAD_LABEL_MAPPING[lead]
    first = np.flatnonzero((label == value).any(axis=0))[0]
    label[np.flatnonzero(label[:, first] == value)[0], x_start:first] = value
    return label


def _blank_page():
    """The white page image, as run() reads it: [3, H, W] uint8."""
    return torch.full((3, HEIGHT, WIDTH), 255, dtype=torch.uint8)


def _draw_grid_lines(image, g0=G0_TRUE, pitch=P_TRUE):
    """Draw the printed 1 mm grid lines of one page into the image, in place.

    The generator draws with matplotlib, whose Agg backend puts a thin vertical
    line at x on the pixel column round(x), whose centre is round(x) + 0.5: that
    is the half pixel GRID_LINE_SNAP_OFFSET takes back off.
    """
    step = pitch / digitize.GRID_LINES_PER_COLUMN
    k = np.arange(np.floor(-g0 / step), np.ceil((WIDTH - g0) / step))
    columns = np.round(g0 + k * step).astype(int)
    # Light lines on the two lower channels, as the red grid of a printed page.
    image[1:, :, columns[(columns >= 0) & (columns < WIDTH)]] = 200
    return image


def cut(label):
    """Split the label mask the way run() does."""
    mask = torch.from_numpy(label)[None]
    image = torch.zeros(3, HEIGHT, WIDTH, dtype=torch.uint8)
    masks, positions, _ = digitize.cut_binary(mask, image)
    return masks, positions, image


def grid_signal(label, lead, pitch="page", image_height=HEIGHT, scale=None):
    """fit_column_grid + vectorise_grid for one lead of a label mask."""
    masks, positions, image = cut(label)
    g0, P, long_leads, reason = digitize.fit_column_grid(
        masks, positions, image_height, pitch
    )
    assert g0 is not None, reason
    # Without a scale given, take the one run() derives from the fitted pitch.
    if scale is None:
        scale = P / (image_height * digitize.PAGE_PITCH_RATIO)
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
        scale,
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


def test_fit_column_grid_does_not_force_the_page_pitch_on_a_rescaled_page(capsys):
    # A 0.2 % crop that keeps the image size rescales the page by about 0.4 %.
    pitch = P_TRUE * 1.004
    masks, positions, _ = cut(make_label_mask(pitch=pitch))
    g0, P, _, reason = digitize.fit_column_grid(masks, positions, HEIGHT, "page")
    assert reason == ""
    assert P == pytest.approx(pitch, abs=0.5)
    assert "does not match the fitted pitch" in capsys.readouterr().out


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


def _overrun_to_the_edge(label):
    """V4 and V5 running on past their column, to the right edge of the image."""
    return extend_end(extend_end(label, "V4", WIDTH), "V5", WIDTH)


@pytest.mark.parametrize("pitch", ["page", "fit"])
def test_fit_column_grid_ignores_leads_running_to_the_image_edge(pitch):
    label = make_label_mask()
    masks, positions, _ = cut(_overrun_to_the_edge(label))
    g0, P, _, reason = digitize.fit_column_grid(masks, positions, HEIGHT, pitch)
    assert reason == ""
    assert g0 == pytest.approx(G0_TRUE, abs=0.5)
    assert P == pytest.approx(P_TRUE, abs=0.2)

    masks, positions, _ = cut(label)
    g0_intact, P_intact, _, _ = digitize.fit_column_grid(
        masks, positions, HEIGHT, pitch
    )
    assert (g0, P) == pytest.approx((g0_intact, P_intact), abs=1e-6)


def test_a_lead_running_to_the_image_edge_keeps_its_grid_signal():
    label = make_label_mask()
    intact, _, _ = grid_signal(label, "V4", "page")
    out, _, _ = grid_signal(_overrun_to_the_edge(label), "V4", "page")
    assert out.shape == (SHORT_SAMPLES,)
    assert np.allclose(out, intact, atol=1e-5)


def test_fit_column_grid_ignores_a_lead_starting_too_early():
    label = make_label_mask()
    masks, positions, _ = cut(label)
    g0_intact, P_intact, _, _ = digitize.fit_column_grid(
        masks, positions, HEIGHT, "page"
    )
    early = extend_start(label, "aVL", int(G0_TRUE + P_TRUE) - 60)
    masks, positions, _ = cut(early)
    g0, P, _, reason = digitize.fit_column_grid(masks, positions, HEIGHT, "page")
    assert reason == ""
    assert (g0, P) == pytest.approx((g0_intact, P_intact), abs=1e-6)


def test_grid_inliers_flags_only_the_edge_off_the_grid():
    origin, pitch = 100.0, 500.0
    lead_edges = [(c, origin + c * pitch, True) for c in range(4)]
    lead_edges += [(c, origin + c * pitch, False) for c in range(1, 5)]
    lead_edges.append((3, origin + 3 * pitch + 61, False))
    assert list(digitize._grid_inliers(lead_edges)) == [True] * 8 + [False]


# ---------------------------------------------------------- refine_grid_from_lines
# A page whose columns start between two pixels, as a real one does.
G0_LINES = G0_TRUE + 0.3


@pytest.mark.parametrize("as_numpy", [False, True])
def test_refine_grid_from_lines_recovers_a_sub_pixel_origin(as_numpy):
    image = _draw_grid_lines(_blank_page(), G0_LINES)
    if as_numpy:
        image = image.numpy()
    # The mask grid is a whole pixel off, the lines must put it back between two.
    g0, P, info = digitize.refine_grid_from_lines(image, G0_LINES - 0.4, P_TRUE)
    assert info["reason"] == ""
    assert info["contrast"] > digitize.GRID_LINE_MIN_CONTRAST
    assert g0 == pytest.approx(G0_LINES, abs=0.05)
    assert P == pytest.approx(P_TRUE, abs=0.02)
    assert info["shift"] == pytest.approx(0.4, abs=0.05)


def test_refine_grid_from_lines_follows_a_rescaled_page():
    # The same 0.4 % rescale as the fitted pitch test, with a 0.3 px pitch error.
    pitch = P_TRUE * 1.004
    image = _draw_grid_lines(_blank_page(), G0_LINES, pitch)
    g0, P, info = digitize.refine_grid_from_lines(image, G0_LINES, pitch - 0.3)
    assert info["reason"] == ""
    assert P == pytest.approx(pitch, abs=0.03)
    assert g0 == pytest.approx(G0_LINES, abs=0.1)


def test_refine_grid_from_lines_ignores_traces_and_text():
    image = _draw_grid_lines(_blank_page(), G0_LINES)
    # Black strokes over whole rows and one label sized block: the median of a
    # 100 row band must not see them.
    image[:, 300:305, :] = 0
    image[:, 1000:1004, 100:2100] = 0
    image[:, 700:760, 200:900] = 0
    g0, P, info = digitize.refine_grid_from_lines(image, G0_LINES - 0.4, P_TRUE)
    assert info["reason"] == ""
    assert g0 == pytest.approx(G0_LINES, abs=0.05)
    assert P == pytest.approx(P_TRUE, abs=0.02)


def test_refine_grid_from_lines_keeps_the_mask_grid_on_a_blank_page():
    g0, P, info = digitize.refine_grid_from_lines(_blank_page(), G0_TRUE, P_TRUE)
    assert (g0, P) == (G0_TRUE, P_TRUE)
    assert info["contrast"] < digitize.GRID_LINE_MIN_CONTRAST
    assert "no grid lines" in info["reason"]


def test_refine_grid_from_lines_keeps_the_mask_grid_off_the_lines():
    image = _draw_grid_lines(_blank_page(), G0_LINES)
    # 3.5 px is nearly half of the 7.87 px line period: the columns of the mask
    # do not start on a grid line, so the lines say nothing about their origin.
    g0, P, info = digitize.refine_grid_from_lines(image, G0_LINES + 3.5, P_TRUE)
    assert (g0, P) == (G0_LINES + 3.5, P_TRUE)
    assert info["contrast"] > digitize.GRID_LINE_MIN_CONTRAST
    assert info["shift"] == pytest.approx(-3.5, abs=0.05)
    assert "off the mask origin" in info["reason"]


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


# --------------------------------------------------------------- page baseline
def test_y_shift_ratios_match_the_generator_row_geometry():
    # The generator puts 4 rows on a page of rows + 2 = 6 row heights, the short
    # lead rows at 3.5, 2.5 and 1.5 row heights above the bottom edge.
    page_height = 21.59
    row_height = page_height / 6
    rows = {
        "I": 3.5, "aVR": 3.5, "V1": 3.5, "V4": 3.5,
        "II": 2.5, "aVL": 2.5, "V2": 2.5, "V5": 2.5,
        "III": 1.5, "aVF": 1.5, "V3": 1.5, "V6": 1.5,
    }
    for lead, height in rows.items():
        assert Y_SHIFT_RATIO[lead] == pytest.approx(
            height * row_height / page_height, rel=1e-12
        )
    # The rhythm strip sits at row_height / 2 - lead_name_offset + 0.8, offset 0.5.
    assert Y_SHIFT_RATIO["full"] == pytest.approx(
        (row_height / 2 - 0.5 + 0.8) / page_height, rel=1e-12
    )


def test_baseline_row_rescales_the_page_about_its_centre():
    ratio = Y_SHIFT_RATIO["I"]
    row = (1 - ratio) * HEIGHT
    assert digitize.baseline_row(ratio, HEIGHT) == pytest.approx(row)

    # The page centre is a fixed point, every other row moves with its distance.
    assert digitize.baseline_row(0.5, HEIGHT, 1.03) == pytest.approx(HEIGHT / 2)
    for scale in (0.98, 1.012):
        moved = digitize.baseline_row(ratio, HEIGHT, scale)
        assert moved - row == pytest.approx((scale - 1) * (row - HEIGHT / 2))


@pytest.mark.parametrize("lead", ["I", "V5", "aVF", RHYTHM_LEAD])
def test_vectorise_grid_follows_a_rescaled_page(lead):
    # A 1.2 % rescale: the page pitch is rejected, so the fitted pitch gives the scale.
    label = make_rescaled_label(1.012)
    out, _, _ = grid_signal(label, lead, "page")
    samples = LONG_SAMPLES if lead == RHYTHM_LEAD else SHORT_SAMPLES
    expected = analytic_signal(np.arange(samples) / FREQUENCY, lead)
    assert abs(np.median(out - expected)) < 0.005


def test_a_rescaled_page_needs_the_scaled_baseline():
    # The rhythm strip is farthest from the page centre, so it drifts the most.
    label = make_rescaled_label(1.012)
    out, _, _ = grid_signal(label, RHYTHM_LEAD, "page", scale=1.0)
    expected = analytic_signal(np.arange(LONG_SAMPLES) / FREQUENCY, RHYTHM_LEAD)
    assert abs(np.median(out - expected)) > 0.05


# ------------------------------------------------------ estimate_baseline_shift
def _limb_leads(lowered_px=0.0, leads_lowered=None, rhythm=True):
    """The limb leads of the fake page, as run() hands them to the estimator."""
    short_time = np.arange(SHORT_SAMPLES) / FREQUENCY
    long_time = np.arange(LONG_SAMPLES) / FREQUENCY
    signals = {}
    for lead in ("I", "II", "III", "aVR", "aVL", "aVF"):
        is_long = rhythm and lead == RHYTHM_LEAD
        signal = analytic_signal(long_time if is_long else short_time, lead)
        if leads_lowered is None or lead in leads_lowered:
            signal = signal - lowered_px * MV_PER_PIXEL
        signals[lead] = torch.from_numpy(signal.astype(np.float32))
    return signals, [RHYTHM_LEAD] if rhythm else []


@pytest.mark.parametrize("rhythm", [True, False])
def test_estimate_baseline_shift_recovers_a_common_offset(rhythm):
    signals, long_leads = _limb_leads(lowered_px=3.7, rhythm=rhythm)
    shift, disagreement = digitize.estimate_baseline_shift(
        signals, long_leads, MV_PER_PIXEL, P_TRUE
    )
    assert shift == pytest.approx(3.7, abs=1e-3)
    assert disagreement == pytest.approx(0.0, abs=1e-3)


def test_estimate_baseline_shift_keeps_the_page_baseline_when_the_rules_disagree(capsys):
    goldberger = ("aVR", "aVL", "aVF")
    signals, long_leads = _limb_leads(lowered_px=5.0, leads_lowered=goldberger)
    shift, disagreement = digitize.estimate_baseline_shift(
        signals, long_leads, MV_PER_PIXEL, P_TRUE, "rec"
    )
    assert shift == 0.0
    assert disagreement == pytest.approx(5.0, abs=1e-3)
    out = capsys.readouterr().out
    assert "baseline estimates disagree for record rec" in out
    assert "keeping the page baseline" in out


@pytest.mark.parametrize("missing", ["III", "aVL"])
def test_estimate_baseline_shift_needs_both_rules(missing):
    signals, long_leads = _limb_leads(lowered_px=2.0)
    signals[missing] = None
    shift, disagreement = digitize.estimate_baseline_shift(
        signals, long_leads, MV_PER_PIXEL, P_TRUE
    )
    assert shift == 0.0
    assert np.isnan(disagreement)


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
    assert args.time_mapping == "grid"
    assert args.grid_pitch == "page"
    assert args.enable_tta is False
    assert parser.parse_args(["-d", "data", "-o", "out", "--enable_tta"]).enable_tta
    # The old flag is still accepted.
    parser.parse_args(["-d", "data", "-o", "out", "--disable_tta"])
    args = parser.parse_args(["-d", "data", "-o", "out", "--time_mapping", "bbox"])
    assert args.time_mapping == "bbox"

    args = parser.parse_args(
        ["-d", "data", "-o", "out", "--time_mapping", "grid", "--grid_pitch", "fit"]
    )
    assert args.time_mapping == "grid"
    assert args.grid_pitch == "fit"

    with pytest.raises(SystemExit):
        parser.parse_args(["-d", "data", "-o", "out", "--time_mapping", "nope"])


def test_parser_baseline():
    parser = digitize.get_parser()
    assert parser.parse_args(["-d", "data", "-o", "out"]).baseline == "leads"
    args = parser.parse_args(["-d", "data", "-o", "out", "--baseline", "page"])
    assert args.baseline == "page"

    with pytest.raises(SystemExit):
        parser.parse_args(["-d", "data", "-o", "out", "--baseline", "nope"])


def test_parser_grid_origin():
    parser = digitize.get_parser()
    assert parser.parse_args(["-d", "data", "-o", "out"]).grid_origin == "lines"
    args = parser.parse_args(["-d", "data", "-o", "out", "--grid_origin", "masks"])
    assert args.grid_origin == "masks"

    with pytest.raises(SystemExit):
        parser.parse_args(["-d", "data", "-o", "out", "--grid_origin", "nope"])


# ------------------------------------------------------- offsets and mask gaps
def test_compute_grid_lead_offsets_uses_the_grid_columns():
    # All leads of column 1 start 3 px late: the placement must stay at 2.5 s.
    label = make_label_mask()
    for lead in ["aVR", "aVL", "aVF"]:
        label = shave_start(label, lead, 3)
    masks, positions, _ = cut(label)
    g0, P, long_leads, _ = digitize.fit_column_grid(masks, positions, HEIGHT)
    lengths = {
        lead: LONG_SAMPLES if lead in long_leads else SHORT_SAMPLES
        for lead, mask in masks.items()
        if mask is not None
    }
    offsets = digitize.compute_grid_lead_offsets(positions, lengths, long_leads, g0, P)

    assert offsets[RHYTHM_LEAD] == {"raw": 0.0, "snapped": 0.0}
    for lead, column in SHORT_COLUMNS.items():
        assert offsets[lead]["snapped"] == pytest.approx(column * 2.5)
        assert offsets[lead]["raw"] == pytest.approx(column * 2.5, abs=0.05)


def test_max_mask_gap_counts_the_empty_columns():
    label = make_label_mask()
    masks, _, _ = cut(label)
    assert digitize.max_mask_gap(masks["V5"]) == 0

    is_lead = label == LEAD_LABEL_MAPPING["V5"]
    first = np.flatnonzero(is_lead.any(axis=0))[0]
    label[:, first + 100 : first + 112][is_lead[:, first + 100 : first + 112]] = 0
    masks, _, _ = cut(label)
    assert digitize.max_mask_gap(masks["V5"]) == 12


# ------------------------------------------------------------------- end to end
def _write_case(tmp_path, name, label, image=None):
    data_folder = tmp_path / f"{name}_data"
    mask_folder = tmp_path / f"{name}_masks"
    data_folder.mkdir(exist_ok=True)
    mask_folder.mkdir(exist_ok=True)
    write_png(_blank_page() if image is None else image, str(data_folder / f"{name}.png"))
    write_png(torch.from_numpy(label)[None], str(mask_folder / f"{name}_mask.png"))
    return data_folder, mask_folder


def _output_folder(tmp_path, name, time_mapping, baseline=None, grid_origin=None):
    suffix = time_mapping if baseline is None else f"{time_mapping}_{baseline}"
    if grid_origin is not None:
        suffix = f"{suffix}_{grid_origin}"
    return tmp_path / f"{name}_{suffix}_out"


def _run(tmp_path, name, label, time_mapping, baseline=None, image=None, grid_origin=None):
    data_folder, mask_folder = _write_case(tmp_path, name, label, image)
    output_folder = _output_folder(tmp_path, name, time_mapping, baseline, grid_origin)
    argv = [
        "-d", str(data_folder),
        "-o", str(output_folder),
        "--mask_folder", str(mask_folder),
        "--time_mapping", time_mapping,
        "--no-verbose",
    ]
    if baseline is not None:
        argv += ["--baseline", baseline]
    if grid_origin is not None:
        argv += ["--grid_origin", grid_origin]
    args = digitize.get_parser().parse_args(argv)
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


def test_run_reports_a_gap_in_a_lead(tmp_path, capsys):
    label = make_label_mask()
    is_lead = label == LEAD_LABEL_MAPPING["V5"]
    first = np.flatnonzero(is_lead.any(axis=0))[0]
    label[:, first + 100 : first + 112][is_lead[:, first + 100 : first + 112]] = 0
    _run(tmp_path, "gap", label, "grid")
    out = capsys.readouterr().out
    assert "lead V5 of record gap has a gap of 12 px" in out


def _lead_median(record, lead):
    channel = record.p_signal[:, record.sig_name.index(lead)]
    return float(np.nanmedian(channel))


def test_run_with_baseline_leads_absorbs_a_shifted_page(tmp_path):
    label = make_label_mask()
    shifted = np.roll(label, 10, axis=0)
    reference = _run(tmp_path, "unshifted", label, "grid", "leads")
    leads = _run(tmp_path, "shifted", shifted, "grid", "leads")
    page = _run(tmp_path, "shifted", shifted, "grid", "page")

    for lead in LEAD_ORDER:
        reference_median = _lead_median(reference, lead)
        assert _lead_median(leads, lead) == pytest.approx(reference_median, abs=0.01)
        # Without the lead baseline the whole record follows the shifted page.
        assert _lead_median(page, lead) - reference_median == pytest.approx(
            -10 * MV_PER_PIXEL, abs=0.01
        )

    qc_path = _output_folder(tmp_path, "shifted", "grid", "leads") / "qc.csv"
    with open(qc_path, newline="") as f:
        row = next(csv.DictReader(f))
    assert float(row["baseline_scale"]) == pytest.approx(1.0, abs=1e-6)
    assert float(row["baseline_shift_px"]) == pytest.approx(10.0, abs=0.5)
    assert np.isfinite(float(row["baseline_disagreement_px"]))


@pytest.mark.parametrize("g0", [G0_TRUE, G0_LINES])
def test_run_with_grid_origin_lines_uses_the_printed_grid(tmp_path, g0):
    # Mask and grid lines from the same page origin, whole pixel and between two.
    name = f"lines{g0:.1f}".replace(".", "_")
    image = _draw_grid_lines(_blank_page(), g0)
    record = _run(
        tmp_path, name, make_label_mask(g0=g0), "grid", image=image,
        grid_origin="lines",
    )
    assert record.p_signal.shape == (LONG_SAMPLES, len(LEAD_ORDER))
    rhythm = record.p_signal[:, record.sig_name.index(RHYTHM_LEAD)]
    expected = analytic_signal(np.arange(LONG_SAMPLES) / FREQUENCY, RHYTHM_LEAD)
    assert np.max(np.abs(rhythm - expected)) < 0.05

    qc_path = _output_folder(tmp_path, name, "grid", grid_origin="lines") / "qc.csv"
    with open(qc_path, newline="") as f:
        row = next(csv.DictReader(f))
    assert float(row["grid_line_contrast"]) > digitize.GRID_LINE_MIN_CONTRAST
    assert abs(float(row["grid_line_shift_px"])) < 1


def test_run_with_grid_origin_lines_keeps_the_masks_without_grid_lines(tmp_path, capsys):
    label = make_label_mask()
    lines = _run(tmp_path, "nolines", label, "grid", grid_origin="lines")
    out = capsys.readouterr().out
    assert "grid lines not used for record nolines" in out
    assert "keeping the grid of the masks" in out

    masks = _run(tmp_path, "nolines", label, "grid", grid_origin="masks")
    assert lines.sig_name == masks.sig_name
    assert np.allclose(lines.p_signal, masks.p_signal, equal_nan=True)
