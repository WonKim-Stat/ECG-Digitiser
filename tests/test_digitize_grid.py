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


def punch_gap(label, lead, x, width=12):
    """Erase width pixel columns of a lead from the label mask, from column x on."""
    label = label.copy()
    is_lead = label == LEAD_LABEL_MAPPING[lead]
    label[:, x : x + width][is_lead[:, x : x + width]] = 0
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


def add_stray_blob(label, lead, x):
    """Label a single pixel of a lead in column x, as the model does at a page edge."""
    label = label.copy()
    value = LEAD_LABEL_MAPPING[lead]
    last = np.flatnonzero((label == value).any(axis=0))[-1]
    label[np.flatnonzero(label[:, last] == value)[-1], x] = value
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


def test_refine_grid_from_lines_snap_offset_moves_the_origin():
    # A scanned page draws its lines where they are: without the half pixel snap of
    # the generator the same lines put the origin half a pixel further right.
    image = _draw_grid_lines(_blank_page(), G0_LINES)
    g0_snapped, P_snapped, _ = digitize.refine_grid_from_lines(image, G0_LINES, P_TRUE)
    g0, P, info = digitize.refine_grid_from_lines(
        image, G0_LINES, P_TRUE, snap_offset=0
    )
    assert info["reason"] == ""
    assert g0 - g0_snapped == pytest.approx(digitize.GRID_LINE_SNAP_OFFSET, abs=1e-6)
    assert P == pytest.approx(P_snapped, abs=1e-9)


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


# -------------------------------------------------------------- trace estimator
# A flat lead for the ink weights, far enough from the page edges to draw around.
FLAT_ROW = 900


def _inked_page(label, value=0):
    """The page of a label mask: every labelled pixel drawn in grey, the rest white."""
    page = np.full((3, HEIGHT, WIDTH), 255, dtype=np.uint8)
    page[:, label > 0] = value
    return torch.from_numpy(page)


def _flat_lead(band, ink_rows, ink_value=0, red_row=None, white_column=None):
    """One flat lead of V5: a mask band of rows and the ink drawn under it.

    band and ink_rows are row offsets from FLAT_ROW, ink_value the grey level the
    ink is drawn in, red_row an offset drawn as a coloured grid line instead and
    white_column a column of the lead left without any ink at all.
    Returns (mask, position, image), as run() hands them to vectorise_grid.
    """
    column = SHORT_COLUMNS["V5"]
    start = int(np.ceil(G0_TRUE + column * P_TRUE - 0.5))
    end = int(np.ceil(G0_TRUE + (column + 1) * P_TRUE - 0.5))
    label = np.zeros((HEIGHT, WIDTH), dtype=np.uint8)
    label[[FLAT_ROW + offset for offset in band], start:end] = LEAD_LABEL_MAPPING["V5"]
    image = torch.full((3, HEIGHT, WIDTH), 255, dtype=torch.uint8)
    for offset in ink_rows:
        image[:, FLAT_ROW + offset, start:end] = ink_value
    if red_row is not None:
        image[0, FLAT_ROW + red_row, start:end] = 255
        image[1:, FLAT_ROW + red_row, start:end] = 0
    if white_column is not None:
        image[:, :, start + white_column] = 255
    masks, positions, _ = digitize.cut_binary(torch.from_numpy(label)[None], image)
    return masks["V5"], positions["V5"], image


def _flat_signal(mask, position, image, ink_map=None, info=None):
    """The one value of a flat lead, in mV, as vectorise_grid reads it."""
    out = digitize.vectorise_grid(
        image, mask, position, G0_TRUE, P_TRUE,
        SHORT_COLUMNS["V5"], False, Y_SHIFT_RATIO, "V5", 1.0,
        ink_map=ink_map, info=info,
    )
    return float(np.median(out.numpy()))


def _row_in_mV(row):
    """The value vectorise_grid gives a short lead of V5 read on this row."""
    # Pixel row r covers [r, r + 1), so the trace on it sits at row + 0.5.
    return (_baseline_row("V5", False) - (row + 0.5)) * MV_PER_PIXEL


def _mask_mean_profile(mask, position):
    """The mask mean profile of one lead, as vectorise_grid computes it."""
    binary = mask[0].numpy() > 0
    count = binary.sum(axis=0)
    filled = np.flatnonzero(count > 0)
    rows = np.arange(binary.shape[0])[:, None]
    mean = position["y1"] + (binary * rows).sum(axis=0)[filled] / count[filled]
    return binary, filled, mean


def test_vectorise_grid_without_an_ink_map_never_looks_at_the_page():
    """The default estimator reads the mask alone, to the bit."""
    label = make_label_mask()
    masks, positions, blank = cut(label)
    arguments = (
        positions["V5"], G0_TRUE, P_TRUE,
        SHORT_COLUMNS["V5"], False, Y_SHIFT_RATIO, "V5",
    )
    default = digitize.vectorise_grid(blank, masks["V5"], *arguments).numpy()
    explicit = digitize.vectorise_grid(
        blank, masks["V5"], *arguments, 1.0, ink_map=None
    ).numpy()
    inked = digitize.vectorise_grid(
        _inked_page(label), masks["V5"], *arguments
    ).numpy()
    assert np.array_equal(default, explicit)
    assert np.array_equal(default, inked)


def test_vectorise_grid_ink_keeps_the_time_of_a_mask_that_is_too_wide():
    """A mask a pixel wider on the left reads a sloped trace too early.

    The ink is where it always was, so the weights hold the trace in place while
    the mask mean of every column is pulled halfway to its right hand neighbour.
    """
    label = make_label_mask(leads=["V5"])
    page = _inked_page(label)
    wide = np.maximum(label, np.roll(label, -1, axis=1))
    masks, positions, _ = digitize.cut_binary(torch.from_numpy(wide)[None], page)

    def best_lag(ink_map):
        out = digitize.vectorise_grid(
            page, masks["V5"], positions["V5"], G0_TRUE, P_TRUE,
            SHORT_COLUMNS["V5"], False, Y_SHIFT_RATIO, "V5", 1.0, ink_map=ink_map,
        ).numpy()
        samples = np.arange(SHORT_SAMPLES)
        lags = np.arange(-4.0, 4.0001, 0.05)
        errors = [
            np.mean((out - analytic_signal((samples + lag) / FREQUENCY, "V5")) ** 2)
            for lag in lags
        ]
        return lags[int(np.argmin(errors))]

    ink = best_lag(digitize._ink(page))
    mask = best_lag(None)
    assert abs(ink) < abs(mask), f"ink {ink:.2f}, mask {mask:.2f} samples"
    # Half a pixel of the extra width, which is 1.02 samples wide here.
    assert abs(mask) > 0.4
    assert abs(ink) < 0.25


def test_ink_ignores_a_coloured_grid_line_that_darkness_would_follow():
    # A symmetric band, so the mask mean is the row of the trace as well.
    mask, position, image = _flat_lead(band=(-2, -1, 0, 1, 2), ink_rows=(0,), red_row=2)
    assert _flat_signal(mask, position, image) == pytest.approx(_row_in_mV(FLAT_ROW))

    ink = _flat_signal(mask, position, image, digitize._ink(image))
    assert ink == pytest.approx(_row_in_mV(FLAT_ROW))
    # _darkness counts the red line as ink, which drags the row halfway onto it.
    darkness = _flat_signal(mask, position, image, digitize._darkness(image))
    assert darkness == pytest.approx(_row_in_mV(FLAT_ROW + 1))


def test_ink_keeps_the_mask_mean_of_a_lead_whose_stroke_is_too_faint():
    # An off centre band, so a weighted row cannot come out as the mask mean.
    band, faint = (-1, 0, 1, 2, 3), int(0.5 * 255)
    mask, position, image = _flat_lead(band, ink_rows=(0,), ink_value=faint)
    info = {}
    ink = _flat_signal(mask, position, image, digitize._ink(image), info)
    assert info["ink_used"] is False
    assert info["ink_p95"] == pytest.approx(255 - faint, abs=1)
    assert ink == _flat_signal(mask, position, image)
    assert ink == pytest.approx(_row_in_mV(FLAT_ROW + 1))


def test_ink_falls_back_to_the_mask_mean_in_a_column_without_ink():
    band, white_column = (-1, 0, 1, 2, 3), 100
    mask, position, image = _flat_lead(band, ink_rows=(0,), white_column=white_column)
    binary, filled, mean = _mask_mean_profile(mask, position)
    profile = digitize._ink_profile(
        digitize._ink(image), binary, position, mean, filled
    )
    assert np.all(np.isfinite(profile))
    # Only the white column keeps the mask mean, every other one follows the ink.
    assert list(np.flatnonzero(profile != FLAT_ROW)) == [white_column]
    assert profile[white_column] == FLAT_ROW + 1


# ------------------------------------------------------------------ trace shift
# Rows a clean bar of the page below contributes: 8 tall, minus the two corner rows
# at each end.
BAR_ROWS = 8 - 2 * digitize.TRACE_SHIFT_CORNER_ROWS


def _bar_page(
    bars=10, spacing=12, first=6, offset=1, ink_width=3, faint=None, neighbour=None,
    snug=False, far_right=False,
):
    """A page of steep three pixel strokes whose label mask sits offset px left.

    Every bar is 8 rows tall, so its columns are steep and BAR_ROWS of its rows lie
    between the corners; the bars are far enough apart for their ink windows not to
    meet. faint draws one bar in pale grey, neighbour labels another lead one pixel
    beyond one bar's window edge (True = beside every bar), snug ends the page on the
    last bar's ink so that its window reaches past the right border, and far_right
    labels another lead in the last column of the page.
    Returns (labelled, ink_map), as run() hands them to measure_trace_shift.
    """
    last = first + (bars - 1) * spacing
    height = 30
    width = last + 3 if snug else first + bars * spacing
    labelled = np.zeros((height, width), dtype=np.uint8)
    page = np.full((3, height, width), 255, dtype=np.uint8)
    for bar in range(bars):
        column, top = first + bar * spacing, 8 + bar % 3
        page[:, top : top + 8, column : column + ink_width] = (
            200 if bar == faint else 0
        )
        labelled[top : top + 8, column - offset : column - offset + 3] = 1
        if neighbour is True or neighbour == bar:
            labelled[top : top + 8, column - offset + 4] = 2
        if far_right:
            labelled[top : top + 8, width - 1] = 2
    return labelled, digitize._ink(torch.from_numpy(page))


def test_measure_trace_shift_finds_a_mask_that_sits_left_of_the_ink():
    for offset in (0, 1):
        dx, info = digitize.measure_trace_shift(*_bar_page(offset=offset))
        assert info["rows"] == 10 * BAR_ROWS
        assert dx == pytest.approx(offset)
    # An ink two pixels wide under a three pixel mask is half a pixel off.
    dx, info = digitize.measure_trace_shift(*_bar_page(ink_width=2))
    assert dx == pytest.approx(0.5)


def test_measure_trace_shift_ignores_a_lead_without_a_steep_stroke():
    labelled, ink_map = _bar_page()
    # Three rows per column is flatter than TRACE_SHIFT_MIN_RUN asks for.
    labelled = labelled * (np.cumsum(labelled > 0, axis=0) <= 3)
    dx, info = digitize.measure_trace_shift(labelled, ink_map)
    assert (dx, info["rows"]) == (0.0, 0)


def test_measure_trace_shift_needs_enough_rows():
    dx, info = digitize.measure_trace_shift(*_bar_page(bars=5))
    assert info["rows"] == 5 * BAR_ROWS < digitize.TRACE_SHIFT_MIN_ROWS
    # The median is still reported, it is only not trusted.
    assert info["median"] == pytest.approx(1.0)
    assert dx == 0.0


def test_measure_trace_shift_drops_a_window_another_lead_reaches_into():
    _, info = digitize.measure_trace_shift(*_bar_page(neighbour=3))
    assert info["rows"] == 9 * BAR_ROWS
    _, info = digitize.measure_trace_shift(*_bar_page(neighbour=True))
    assert info["rows"] == 0


def test_measure_trace_shift_drops_a_window_over_the_page_border():
    # The first bar's window reaches one pixel past the left edge of the page, and
    # the last column of every row holds mask pixels, which is what a window looked
    # up with a negative column would read instead of the paper left of the page.
    for far_right in (False, True):
        dx, info = digitize.measure_trace_shift(*_bar_page(first=2, far_right=far_right))
        assert info["rows"] == 9 * BAR_ROWS
        assert dx == pytest.approx(1.0)

    # The same one pixel over the right edge, where there is no column to read at all.
    dx, info = digitize.measure_trace_shift(*_bar_page(snug=True))
    assert info["rows"] == 9 * BAR_ROWS
    assert dx == pytest.approx(1.0)


def test_measure_trace_shift_drops_a_window_without_ink_of_its_own():
    dx, info = digitize.measure_trace_shift(*_bar_page(faint=4))
    assert info["rows"] == 9 * BAR_ROWS
    assert dx == pytest.approx(1.0)


def test_trace_shift_px_shares_the_measurement_with_the_estimator():
    assert digitize.trace_shift_px(0.3, "mask") == pytest.approx(0.3)
    assert digitize.trace_shift_px(0.3, "ink") == pytest.approx(
        0.3 * digitize.TRACE_SHIFT_INK_SHARE
    )
    # Inside the dead band nothing is applied, on either side of zero.
    for measured in (0.04, -0.04):
        assert digitize.trace_shift_px(measured, "mask") == 0.0
    assert digitize.trace_shift_px(0.06, "mask") == pytest.approx(0.06)
    # The share is taken first, so it decides what falls into the band.
    assert digitize.trace_shift_px(0.07, "ink") == 0.0


def test_vectorise_grid_without_a_shift_samples_where_it_always_did():
    masks, positions, image = cut(make_label_mask())
    arguments = (
        positions["V5"], G0_TRUE, P_TRUE,
        SHORT_COLUMNS["V5"], False, Y_SHIFT_RATIO, "V5",
    )
    default = digitize.vectorise_grid(image, masks["V5"], *arguments).numpy()
    explicit = digitize.vectorise_grid(
        image, masks["V5"], *arguments, 1.0, x_shift=0.0
    ).numpy()
    assert np.array_equal(default, explicit)


@pytest.mark.parametrize("x_shift", [-1.0, 1.0])
def test_vectorise_grid_x_shift_moves_the_trace_in_time(x_shift):
    """A shift to the right reads the trace later, by its own pixels."""
    masks, positions, image = cut(make_label_mask())
    out = digitize.vectorise_grid(
        image, masks["V5"], positions["V5"], G0_TRUE, P_TRUE,
        SHORT_COLUMNS["V5"], False, Y_SHIFT_RATIO, "V5", 1.0, x_shift=x_shift,
    ).numpy()

    samples = np.arange(SHORT_SAMPLES)
    lags = np.arange(-4.0, 4.0001, 0.05)
    errors = [
        np.mean((out - analytic_signal((samples + lag) / FREQUENCY, "V5")) ** 2)
        for lag in lags
    ]
    best = lags[int(np.argmin(errors))]
    samples_per_pixel = SHORT_SAMPLES / P_TRUE
    assert best == pytest.approx(-x_shift * samples_per_pixel, abs=0.15)


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


def test_parser_grid_line_offset():
    parser = digitize.get_parser()
    args = parser.parse_args(["-d", "data", "-o", "out"])
    assert args.grid_line_offset == digitize.GRID_LINE_SNAP_OFFSET
    args = parser.parse_args(["-d", "data", "-o", "out", "--grid_line_offset", "0"])
    assert args.grid_line_offset == 0.0


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


def _column_window(column, n_columns=1, g0=G0_TRUE, pitch=P_TRUE):
    """The image x range the grid sampling of a lead of that column reads."""
    return g0 + column * pitch, g0 + (column + n_columns) * pitch


def test_max_mask_gap_ignores_a_blob_outside_the_window():
    masks, positions, _ = cut(add_stray_blob(make_label_mask(), "V5", WIDTH - 1))
    window = _column_window(SHORT_COLUMNS["V5"])
    x1 = positions["V5"]["x1"]

    # Without a window the empty stretch up to the blob passes as a gap in the lead.
    assert digitize.max_mask_gap(masks["V5"]) > 100
    assert digitize.max_mask_gap(masks["V5"], x1, window) == 0


def test_max_mask_gap_still_counts_a_gap_inside_the_window():
    label = make_label_mask()
    first = np.flatnonzero((label == LEAD_LABEL_MAPPING["V5"]).any(axis=0))[0]
    label = add_stray_blob(punch_gap(label, "V5", first + 100), "V5", WIDTH - 1)
    masks, positions, _ = cut(label)
    window = _column_window(SHORT_COLUMNS["V5"])

    assert digitize.max_mask_gap(masks["V5"], positions["V5"]["x1"], window) == 12


def test_max_mask_gap_counts_an_early_end_bridged_towards_a_blob():
    # V5 stops at pixel column 2000 and is interpolated on towards the blob, so the
    # empty columns up to the end of its window at 2086.5 are read after all.
    label = punch_gap(make_label_mask(), "V5", 2001, width=86)
    masks, positions, _ = cut(add_stray_blob(label, "V5", WIDTH - 1))
    window = _column_window(SHORT_COLUMNS["V5"])

    assert digitize.max_mask_gap(masks["V5"]) == 198
    assert digitize.max_mask_gap(masks["V5"], positions["V5"]["x1"], window) == 85


def test_max_mask_gap_with_a_window_beside_the_lead():
    label = make_label_mask()
    first = np.flatnonzero((label == LEAD_LABEL_MAPPING["V5"]).any(axis=0))[0]
    masks, positions, _ = cut(punch_gap(label, "V5", first + 100))
    # The gap of V5 is in the last column of the page, none of it in the first.
    window = _column_window(0)

    assert digitize.max_mask_gap(masks["V5"], positions["V5"]["x1"], window) == 0


def test_max_mask_gap_spans_every_column_of_the_long_lead():
    label = make_label_mask()
    first = np.flatnonzero((label == LEAD_LABEL_MAPPING[RHYTHM_LEAD]).any(axis=0))[0]
    label = punch_gap(label, RHYTHM_LEAD, first + int(3 * P_TRUE) + 100)
    masks, positions, _ = cut(label)
    x1 = positions[RHYTHM_LEAD]["x1"]

    # The gap is in the last column, which only the four column window covers.
    assert digitize.max_mask_gap(masks[RHYTHM_LEAD], x1, _column_window(0, 4)) == 12
    assert digitize.max_mask_gap(masks[RHYTHM_LEAD], x1, _column_window(0)) == 0


def test_mask_overhang_measures_the_pixels_outside_the_window():
    label = make_label_mask()
    window = _column_window(SHORT_COLUMNS["V5"])
    masks, positions, _ = cut(label)
    assert digitize.mask_overhang(masks["V5"], positions["V5"]["x1"], window) == 0.0

    # Pixel c covers [c, c + 1), so a blob in column c reaches out to c + 0.5.
    masks, positions, _ = cut(add_stray_blob(label, "V5", WIDTH - 1))
    overhang = digitize.mask_overhang(masks["V5"], positions["V5"]["x1"], window)
    assert overhang == pytest.approx(WIDTH - 0.5 - window[1])

    masks, positions, _ = cut(add_stray_blob(label, "V5", 5))
    overhang = digitize.mask_overhang(masks["V5"], positions["V5"]["x1"], window)
    assert overhang == pytest.approx(window[0] - 5.5)


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


def _run(
    tmp_path, name, label, time_mapping, baseline=None, image=None, grid_origin=None,
    verbose=False, trace_estimator="mask", trace_shift="off", column_mapping=None,
):
    data_folder, mask_folder = _write_case(tmp_path, name, label, image)
    output_folder = _output_folder(tmp_path, name, time_mapping, baseline, grid_origin)
    # The stages under test are the column grid and the baseline: the pages here show
    # the vertical lines of that grid and nothing else, which leaves the rotation, the
    # perspective and the resolution nothing to correct, only seconds and warnings.
    # The trace estimator and the page shift are the plain mask ones here for the same
    # reason; None passes no flag at all, which is how the defaults are tested. The
    # sharpening is left at its default throughout, and so is the column mapping: no
    # page here has the bold 5 mm lines it needs, most have no grid lines at all, so
    # it never moves a sample and every lead is read on the uniform columns.
    argv = [
        "-d", str(data_folder),
        "-o", str(output_folder),
        "--mask_folder", str(mask_folder),
        "--time_mapping", time_mapping,
        "--rotation", "hough",
        "--perspective", "off",
        "--resolution", "keep",
        "--verbose" if verbose else "--no-verbose",
    ]
    if baseline is not None:
        argv += ["--baseline", baseline]
    if grid_origin is not None:
        argv += ["--grid_origin", grid_origin]
    if trace_estimator is not None:
        argv += ["--trace_estimator", trace_estimator]
    if trace_shift is not None:
        argv += ["--trace_shift", trace_shift]
    if column_mapping is not None:
        argv += ["--column_mapping", column_mapping]
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


def test_run_does_not_report_a_blob_outside_the_column_as_a_gap(tmp_path, capsys):
    label = make_label_mask()
    blob = _run(
        tmp_path, "blob", add_stray_blob(label, "V4", WIDTH - 1), "grid", verbose=True
    )
    out = capsys.readouterr().out
    assert "has a gap of" not in out
    assert "Lead V4 of record blob has mask pixels up to 113 px outside" in out

    # The blob is never sampled, so the lead comes out as it does without it.
    clean = _run(tmp_path, "noblob", label, "grid")
    channel = blob.p_signal[:, blob.sig_name.index("V4")]
    expected = clean.p_signal[:, clean.sig_name.index("V4")]
    assert np.array_equal(channel, expected, equal_nan=True)


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


def test_parser_trace_estimator_defaults_to_ink():
    parser = digitize.get_parser()
    args = parser.parse_args(["-d", "data", "-o", "out"])
    assert args.trace_estimator == "ink"
    args = parser.parse_args(["-d", "data", "-o", "out", "--trace_estimator", "mask"])
    assert args.trace_estimator == "mask"


def test_append_qc_row_writes_the_trace_estimator_columns(tmp_path):
    qc = {"trace_estimator": "ink", "ink_leads": 13, "ink_p95": 232.0}
    digitize.append_qc_row(str(tmp_path), "rec", "column", qc, 0.0)
    with open(tmp_path / "qc.csv", newline="") as f:
        reader = csv.DictReader(f)
        row = next(reader)
        # The new columns are appended, the old ones keep their order.
        assert reader.fieldnames[:2] == ["record", "placement"]
        assert reader.fieldnames[-9:-6] == ["trace_estimator", "ink_leads", "ink_p95"]
    assert row["trace_estimator"] == "ink"
    assert int(row["ink_leads"]) == 13
    assert float(row["ink_p95"]) == pytest.approx(232.0)


def test_run_with_trace_estimator_ink_reads_every_lead(tmp_path, capsys):
    label = make_label_mask()
    page = _inked_page(label)
    ink = _run(
        tmp_path, "ink", label, "grid", image=page, verbose=True,
        trace_estimator="ink",
    )
    out = capsys.readouterr().out
    leads = len(LEAD_ORDER)
    assert f"Trace estimator for record ink: ink on {leads}/{leads} leads" in out
    assert "ink p95 255" in out

    # The mask is drawn from the ink here, so the weights have nothing to move.
    mask = _run(tmp_path, "inkmask", label, "grid", image=page, verbose=True)
    assert "Trace estimator for record" not in capsys.readouterr().out
    assert ink.sig_name == mask.sig_name
    assert np.array_equal(ink.p_signal, mask.p_signal, equal_nan=True)

    with open(_output_folder(tmp_path, "ink", "grid") / "qc.csv", newline="") as f:
        row = next(csv.DictReader(f))
    assert row["trace_estimator"] == "ink"
    assert int(row["ink_leads"]) == leads
    assert float(row["ink_p95"]) == pytest.approx(255.0)


def test_parser_trace_shift_defaults_to_page():
    parser = digitize.get_parser()
    args = parser.parse_args(["-d", "data", "-o", "out"])
    assert args.trace_shift == "page"
    args = parser.parse_args(["-d", "data", "-o", "out", "--trace_shift", "off"])
    assert args.trace_shift == "off"


def test_append_qc_row_writes_the_trace_shift_columns(tmp_path):
    qc = {"trace_shift_px": 0.18, "trace_shift_rows": 4752}
    digitize.append_qc_row(str(tmp_path), "rec", "column", qc, 0.0)
    with open(tmp_path / "qc.csv", newline="") as f:
        reader = csv.DictReader(f)
        row = next(reader)
        # The new columns are appended, the old ones keep their order.
        assert reader.fieldnames[:2] == ["record", "placement"]
        assert reader.fieldnames[-6:-4] == ["trace_shift_px", "trace_shift_rows"]
    assert float(row["trace_shift_px"]) == pytest.approx(0.18)
    assert int(row["trace_shift_rows"]) == 4752


def _connected_label(label):
    """Close the vertical gaps between neighbouring columns, as a drawn trace does.

    The one pixel per column of _draw_lead is no stroke a shift can be measured on:
    a column needs TRACE_SHIFT_MIN_RUN pixels before it counts as steep.
    """
    out = label.copy()
    for value in LEAD_LABEL_MAPPING.values():
        is_lead = label == value
        columns = np.flatnonzero(is_lead.any(axis=0))
        rows = np.argmax(is_lead, axis=0)
        for column in columns[1:]:
            low, high = sorted((rows[column - 1], rows[column]))
            out[low : high + 1, column] = value
    return out


def test_run_with_trace_shift_page_leaves_a_page_on_its_own_ink_alone(tmp_path, capsys):
    label = _connected_label(make_label_mask())
    page = _inked_page(label)
    shifted = _run(
        tmp_path, "shift", label, "grid", image=page, verbose=True,
        trace_estimator="ink", trace_shift="page",
    )
    out = capsys.readouterr().out
    assert "Trace shift for record shift: +0.00 px from" in out

    # The mask is the ink, so there is nothing to move and nothing moves.
    off = _run(
        tmp_path, "noshift", label, "grid", image=page, verbose=True,
        trace_estimator="ink",
    )
    assert "Trace shift for record" not in capsys.readouterr().out
    assert shifted.sig_name == off.sig_name
    assert np.array_equal(shifted.p_signal, off.p_signal, equal_nan=True)

    with open(_output_folder(tmp_path, "shift", "grid") / "qc.csv", newline="") as f:
        row = next(csv.DictReader(f))
    assert float(row["trace_shift_px"]) == 0.0
    # Enough steep rows on the QRS flanks for the median to be used at all.
    assert int(row["trace_shift_rows"]) >= digitize.TRACE_SHIFT_MIN_ROWS


# --------------------------------------------------- the trace stages by default
def test_run_uses_the_ink_estimator_and_the_page_shift_by_default(tmp_path, capsys):
    """What the two defaults are for, with no trace flag given at all.

    The page here is drawn from its own masks, so neither stage has anything to move;
    what shows that they ran is the QC row and the lines they print. The page has no
    grid lines either, so the default column mapping keeps the uniform columns.
    """
    label = _connected_label(make_label_mask())
    page = _inked_page(label)
    record = _run(
        tmp_path, "default", label, "grid", image=page, verbose=True,
        trace_estimator=None, trace_shift=None,
    )
    out = capsys.readouterr().out
    assert "Trace estimator for record default: ink on" in out
    assert "Trace shift for record default: +0.00 px from" in out
    assert "Column mapping for record default: uniform: grid lines not used" in out

    with open(_output_folder(tmp_path, "default", "grid") / "qc.csv", newline="") as f:
        row = next(csv.DictReader(f))
    assert row["trace_estimator"] == "ink"
    assert row["sharpen"] == "bandlimited"
    assert int(row["ink_leads"]) == len(LEAD_ORDER)
    assert float(row["trace_shift_px"]) == 0.0
    # Enough steep rows on the QRS flanks for the median to be used at all.
    assert int(row["trace_shift_rows"]) >= digitize.TRACE_SHIFT_MIN_ROWS
    # --column_mapping lines is the default, and without grid lines it measures no map.
    assert row["column_mapping"] == "uniform: grid lines not used"
    assert row["column_mapping_shift_px"] == "nan"

    # Nothing to move, so the defaults read this page where the plain masks do.
    masks = _run(tmp_path, "defaultmask", label, "grid", image=page)
    assert record.sig_name == masks.sig_name
    assert np.array_equal(record.p_signal, masks.p_signal, equal_nan=True)

    # And where the uniform columns, the time axis before the default, read it.
    uniform = _run(
        tmp_path, "defaultuniform", label, "grid", image=page,
        trace_estimator=None, trace_shift=None, column_mapping="uniform",
    )
    qc_path = _output_folder(tmp_path, "defaultuniform", "grid") / "qc.csv"
    with open(qc_path, newline="") as f:
        assert next(csv.DictReader(f))["column_mapping"] == "uniform"
    assert record.sig_name == uniform.sig_name
    assert np.array_equal(record.p_signal, uniform.p_signal, equal_nan=True)
