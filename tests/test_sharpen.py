"""Unit tests for --sharpen bandlimited in src/run/digitize.py.

The band limited sharpening is cur + LP35(R2a8 - cur) on the 500 Hz row trace of
vectorise_grid(): cur is np.interp of the column profile, R2a8 the aperture corrected
(a = 1/8) profile through a cubic spline per run of columns, LP35 a zero phase
Butterworth low-pass. No model and no data files: the profiles, the lead masks and
the one page of the run() test are synthesised with numpy.
"""
import csv
import warnings

import numpy as np
import pytest
import torch
from scipy.interpolate import CubicSpline
from scipy.signal import butter, sosfiltfilt
import wfdb
from torchvision.io.image import write_png

from config import (
    FREQUENCY,
    LEAD_LABEL_MAPPING,
    SHORT_SIGNAL_LENGTH_SEC,
    Y_SHIFT_RATIO,
)
from src.run import digitize


HEIGHT = 1700
WIDTH = 2200
# A 2.5 s column is 62.5 mm of the 215.9 mm page height, as on the generator pages.
P_PAGE = HEIGHT * 6.25 / 21.59
G0 = 118.0
SHORT_SAMPLES = int(SHORT_SIGNAL_LENGTH_SEC * FREQUENCY)
SHORT_COLUMNS = {
    "I": 0, "III": 0,
    "aVR": 1, "aVL": 1, "aVF": 1,
    "V1": 2, "V2": 2, "V3": 2,
    "V4": 3, "V5": 3, "V6": 3,
}
RHYTHM_LEAD = "II"


# ----------------------------------------------------------------- helpers
def _x(g0, pitch, n_samples=SHORT_SAMPLES, column=0):
    """The sampling positions of vectorise_grid, in column index units."""
    samples_per_column = SHORT_SIGNAL_LENGTH_SEC * FREQUENCY
    return g0 + column * pitch + np.arange(n_samples) * pitch / samples_per_column - 0.5


def _sharpen(filled, profile, x, columns=None):
    """(linear trace, sharpened trace) of a profile on the integer columns filled."""
    columns = filled if columns is None else columns
    sampled = np.interp(x, columns, profile)
    out = digitize._bandlimited_sharpen(columns, profile, filled, x, sampled, FREQUENCY)
    return sampled, out


def _lead_mask(trace_rows, x1=600, y1=500, thickness=3):
    """A lead mask as run() hands it to vectorise_grid: ([1, h, w] tensor, position).

    trace_rows holds the top row of the stroke in every column of the crop, or -1 for
    a column without any mask pixel.
    """
    trace_rows = np.asarray(trace_rows)
    height = int(trace_rows.max()) + thickness + 1
    binary = np.zeros((height, trace_rows.size), dtype=np.uint8)
    for c, r in enumerate(trace_rows):
        if r >= 0:
            binary[r : r + thickness, c] = 1
    return torch.from_numpy(binary)[None], {"y1": y1, "x1": x1}


def _wavy_rows(width=520):
    """A slow wave with a narrow spike, and an island of 2 columns between two gaps."""
    c = np.arange(width)
    rows = 80 + np.round(20 * np.sin(2 * np.pi * c / 90.0)).astype(int)
    rows = rows - np.round(45 * np.exp(-0.5 * ((c - 300) / 3.0) ** 2)).astype(int)
    rows[200:206] = -1
    rows[208:214] = -1
    return rows


def _expected(mask, position, pitch, column, lead, x_shift=0.0, sharpen="none"):
    """vectorise_grid's short lead written out: mask mean, np.interp, sharpening."""
    binary = mask[0].numpy() > 0
    count = binary.sum(axis=0)
    filled = np.flatnonzero(count > 0)
    rows = np.arange(binary.shape[0])[:, None]
    profile = position["y1"] + (binary * rows).sum(axis=0)[filled] / count[filled]
    x = _x(G0, pitch, column=column)
    columns = position["x1"] + filled
    if x_shift:
        columns = columns + x_shift
    sampled = np.interp(x, columns, profile)
    if sharpen == "bandlimited":
        sampled = digitize._bandlimited_sharpen(
            columns, profile, filled, x, sampled, FREQUENCY
        )
    baseline = digitize.baseline_row(Y_SHIFT_RATIO[lead], HEIGHT, 1.0)
    mV_per_pixel = 25 * SHORT_SIGNAL_LENGTH_SEC / pitch / 10
    return ((baseline - (sampled + 0.5)) * mV_per_pixel).astype(np.float32)


def _vectorise(mask, position, lead="aVR", **kwargs):
    image = torch.zeros((3, HEIGHT, WIDTH), dtype=torch.uint8)
    out = digitize.vectorise_grid(
        image, mask, position, G0, P_PAGE, SHORT_COLUMNS[lead], False,
        Y_SHIFT_RATIO, lead, 1.0, **kwargs,
    )
    return out.numpy()


def _amplitude(signal, f_hz):
    """Amplitude of the f_hz component of a 500 Hz trace, least squares sine fit."""
    t = np.arange(signal.size) / FREQUENCY
    design = np.c_[
        np.sin(2 * np.pi * f_hz * t), np.cos(2 * np.pi * f_hz * t), np.ones_like(t)
    ]
    coef, *_ = np.linalg.lstsq(design, signal, rcond=None)
    return float(np.hypot(coef[0], coef[1]))


# ------------- 1. the flag: keyword default none, CLI default bandlimited
def test_sharpen_none_is_the_linear_interpolation_bitwise():
    mask, position = _lead_mask(_wavy_rows())
    expected = _expected(mask, position, P_PAGE, 1, "aVR")
    assert np.array_equal(_vectorise(mask, position, sharpen="none"), expected)
    # Without the keyword it is the same code path.
    assert np.array_equal(_vectorise(mask, position), expected)

    # With the page shift the columns move, the runs stay those of filled.
    sharpened = _vectorise(mask, position, x_shift=0.3, sharpen="bandlimited")
    assert np.array_equal(
        sharpened,
        _expected(mask, position, P_PAGE, 1, "aVR", 0.3, sharpen="bandlimited"),
    )
    assert not np.array_equal(
        sharpened, _vectorise(mask, position, x_shift=0.3, sharpen="none")
    )
    with pytest.raises(ValueError, match="sharpen"):
        _vectorise(mask, position, sharpen="nope")


def test_parser_sharpen_defaults_to_bandlimited():
    parser = digitize.get_parser()
    assert parser.parse_args(["-d", "data", "-o", "out"]).sharpen == "bandlimited"
    args = parser.parse_args(["-d", "data", "-o", "out", "--sharpen", "bandlimited"])
    assert args.sharpen == "bandlimited"
    args = parser.parse_args(["-d", "data", "-o", "out", "--sharpen", "none"])
    assert args.sharpen == "none"
    with pytest.raises(SystemExit):
        parser.parse_args(["-d", "data", "-o", "out", "--sharpen", "nope"])


# ------------------------------------------------------ 2. constant profile
def test_constant_profile_stays_bitwise_constant():
    x = _x(1000.3, 500.0)
    # Runs of every kind: long ones, a gap, a run of 2 the spline leaves alone.
    filled = np.r_[np.arange(980, 1200), np.arange(1210, 1212), np.arange(1230, 1560)]
    profile = np.full(filled.size, 812.3)
    sampled, out = _sharpen(filled, profile, x)
    assert np.array_equal(out, sampled)

    # Through vectorise_grid: a flat stroke reads the same with and without it.
    mask, position = _lead_mask(np.full(520, 40))
    assert np.array_equal(
        _vectorise(mask, position, sharpen="bandlimited"),
        _vectorise(mask, position, sharpen="none"),
    )


# --------------------------------------------------------- 3. linear ramp
def test_linear_ramp_is_unchanged():
    x = _x(1000.3, 500.0)
    # The ends of a run repeat their own value, which moves the two end columns of
    # a ramp by a times its slope; the run reaches 40 columns past the window on both
    # sides, where that and the spline around it have died out.
    filled = np.arange(int(np.floor(x[0])) - 40, int(np.ceil(x[-1])) + 40)
    profile = 700.0 + 0.7 * filled
    sampled, out = _sharpen(filled, profile, x)
    assert np.max(np.abs(out - sampled)) <= 1e-9


# --------------------------------------------------- 4. band restoration
@pytest.mark.parametrize("f_hz", [20.0, 80.0])
def test_sharpen_restores_the_band_and_only_the_band(f_hz):
    # 500 px per 2.5 s column window: 200 columns per second, 1250 samples per window.
    pitch, g0, amplitude = 500.0, 1000.3, 10.0
    x = _x(g0, pitch)
    filled = np.arange(int(np.floor(x[0])) - 40, int(np.ceil(x[-1])) + 40)

    def trace(X):
        # Physical position X; sample n sits at X = x + 0.5, time n / FREQUENCY.
        t = (X - g0) * SHORT_SIGNAL_LENGTH_SEC / pitch
        return 700.0 + amplitude * np.sin(2 * np.pi * f_hz * t)

    # Pixel column c covers [c, c + 1); its value is the column average of the
    # trace, the mean of its two edge values, which is the midrange a thin mask has
    # in a column the trace crosses monotonically: the aperture the sharpening undoes.
    c = filled.astype(float)
    profile = 0.5 * (trace(c) + trace(c + 1.0))
    sampled, out = _sharpen(filled, profile, x)

    linear_ratio = _amplitude(sampled, f_hz) / amplitude
    sharp_ratio = _amplitude(out, f_hz) / amplitude
    if f_hz == 20.0:
        assert abs(sharp_ratio - 1) < 0.03, sharp_ratio
        assert abs(sharp_ratio - 1) < abs(linear_ratio - 1), (sharp_ratio, linear_ratio)
    else:
        # Far above the cutoff the linear trace is kept.
        assert abs(sharp_ratio / linear_ratio - 1) < 0.01, (sharp_ratio, linear_ratio)


# -------------------------------------------------------------- 5. zero phase
def test_symmetric_triangle_stays_symmetric():
    apex = 1250
    # 1250 samples 0.4 px apart, centred on the apex column.
    x = apex - 249.8 + np.arange(SHORT_SAMPLES) * 0.4
    filled = np.arange(apex - 300, apex + 301)
    profile = 700.0 - 3.0 * np.abs(filled - apex)
    sampled, out = _sharpen(filled, profile, x)
    assert np.max(np.abs(out - out[::-1])) <= 1e-9
    # Not symmetric by doing nothing: the apex is lifted.
    centre = SHORT_SAMPLES // 2
    assert out[centre] > sampled[centre] + 0.1


# ----------------------------------------------------- 6. gaps and short runs
def test_gaps_and_short_runs_stay_finite():
    x = _x(1000.3, 500.0)
    filled = np.r_[np.arange(980, 1100), np.arange(1110, 1112), np.arange(1130, 1520)]
    profile = 700.0 + 5.0 * np.sin(filled / 7.0)
    sampled, out = _sharpen(filled, profile, x)
    assert out.shape == sampled.shape == (SHORT_SAMPLES,)
    assert np.all(np.isfinite(out))


def test_single_column_lead_keeps_the_linear_trace():
    x = _x(1000.3, 500.0)
    sampled, out = _sharpen(np.array([1200]), np.array([733.7]), x)
    assert np.array_equal(out, sampled)


# ---------------------------------------------------------- 7. short traces
def test_short_trace_keeps_the_linear_trace():
    filled = np.arange(990, 1010)
    profile = 700.0 + 5.0 * np.sin(filled / 1.3)
    x = _x(1000.3, 500.0, n_samples=16)
    # sosfiltfilt pads 15 samples at each end, so it needs 16.
    for n in (12, 15):
        sampled, out = _sharpen(filled, profile, x[:n])
        assert np.array_equal(out, sampled), n
    sampled, out = _sharpen(filled, profile, x)
    assert not np.array_equal(out, sampled)


# ------------------------------------------------- 9. the approved recipe
def test_bandlimited_sharpen_is_the_approved_recipe_bitwise():
    """SP35R written out from first principles, with none of digitize's helpers."""
    x = _x(1000.3, 500.0)
    # A long run, a gap, islands of 3, 2 and 1 columns, a run of exactly 4 and a long
    # run with a narrow spike, all inside the window of x (999.8 to 1499.4).
    filled = np.r_[
        np.arange(980, 1200),
        np.arange(1210, 1213),
        np.arange(1220, 1222),
        np.arange(1230, 1231),
        np.arange(1240, 1244),
        np.arange(1250, 1561),
    ]
    profile = (
        700.0
        + 20.0 * np.sin(filled / 9.0)
        - 40.0 * np.exp(-0.5 * ((filled - 1300) / 2.5) ** 2)
    )
    # The page shift moves the columns; the runs are those of the integer filled.
    columns = filled + 0.3
    linear, out = _sharpen(filled, profile, x, columns=columns)

    # (1) Maximal runs of consecutive integers of filled, half open index ranges.
    runs, i0 = [], 0
    for i in range(1, filled.size + 1):
        if i == filled.size or filled[i] != filled[i - 1] + 1:
            runs.append((i0, i))
            i0 = i
    assert [i1 - i0 for i0, i1 in runs] == [220, 3, 2, 1, 4, 311]

    # (2) The aperture correction [-a, 1 + 2a, -a], a = 1/8, inside each run, whose
    # ends repeat their own value; a single column stays what it was.
    a = 0.125
    fir = profile.copy()
    for i0, i1 in runs:
        seg = profile[i0:i1]
        if seg.size < 2:
            continue
        left = np.r_[seg[0], seg[:-1]]
        right = np.r_[seg[1:], seg[-1]]
        fir[i0:i1] = seg - a * (left - 2.0 * seg + right)

    # (3) np.interp everywhere, a not-a-knot spline through every run of 4 or more
    # columns between its first and its last column.
    spline_trace = np.interp(x, columns, fir)
    for i0, i1 in runs:
        if i1 - i0 < 4:
            continue
        cx = columns[i0:i1]
        inside = (x >= cx[0]) & (x <= cx[-1])
        spline = CubicSpline(cx, fir[i0:i1], bc_type="not-a-knot")
        spline_trace[inside] = spline(x[inside])

    # (4) The difference to the linear trace, (5) low passed by the zero phase fourth
    # order 35 Hz Butterworth at 500 Hz and added back.
    expected_linear = np.interp(x, columns, profile)
    d = spline_trace - expected_linear
    sos = butter(4, 35.0, fs=500.0, output="sos")
    expected = expected_linear + sosfiltfilt(sos, d)

    assert np.array_equal(linear, expected_linear)
    assert np.array_equal(out, expected), np.max(np.abs(out - expected))
    # Not equal by doing nothing.
    assert not np.array_equal(out, linear)
    assert np.max(np.abs(out - linear)) > 0.1


# ------------------------------------------------------------------ 8. QC
def test_append_qc_row_writes_the_sharpen_column(tmp_path):
    qc = {"sharpen": "bandlimited", "trace_estimator": "ink"}
    digitize.append_qc_row(str(tmp_path), "rec", "column", qc, 0.0)
    with open(tmp_path / "qc.csv", newline="") as f:
        reader = csv.DictReader(f)
        row = next(reader)
        assert "sharpen" in reader.fieldnames
        assert reader.fieldnames[:2] == ["record", "placement"]
    assert row["sharpen"] == "bandlimited"
    assert row["trace_estimator"] == "ink"


def _page_label():
    """A standard 3x4 label mask with a rhythm strip of II, 1 px thick traces."""
    label = np.zeros((HEIGHT, WIDTH), dtype=np.uint8)
    mV_per_pixel = 6.25 / P_PAGE
    for lead, column in list(SHORT_COLUMNS.items()) + [(RHYTHM_LEAD, 0)]:
        is_long = lead == RHYTHM_LEAD
        start = G0 + column * P_PAGE
        end = start + (4 if is_long else 1) * P_PAGE
        columns = np.arange(int(np.ceil(start - 0.5)), int(np.ceil(end - 0.5)))
        t = (columns + 0.5 - start) * SHORT_SIGNAL_LENGTH_SEC / P_PAGE
        signal = 0.3 * np.sin(2 * np.pi * 1.1 * t) + 0.9 * np.exp(
            -0.5 * (((t % 1.0) - 0.5) / 0.02) ** 2
        )
        ratio = Y_SHIFT_RATIO["full" if is_long else lead]
        rows = np.floor(
            digitize.baseline_row(ratio, HEIGHT) - signal / mV_per_pixel
        ).astype(int)
        label[rows, columns] = LEAD_LABEL_MAPPING[lead]
    return label


@pytest.mark.parametrize("mode", ["none", "bandlimited"])
def test_run_passes_the_mode_to_every_lead_and_the_qc_row(tmp_path, monkeypatch, mode):
    data_folder, mask_folder = tmp_path / "data", tmp_path / "masks"
    data_folder.mkdir()
    mask_folder.mkdir()
    write_png(torch.full((3, HEIGHT, WIDTH), 255, dtype=torch.uint8),
              str(data_folder / "rec.png"))
    write_png(torch.from_numpy(_page_label())[None], str(mask_folder / "rec_mask.png"))
    output_folder = tmp_path / "out"

    seen = []
    original = digitize.vectorise_grid

    def spy(*args, **kwargs):
        seen.append(kwargs.get("sharpen"))
        return original(*args, **kwargs)

    monkeypatch.setattr(digitize, "vectorise_grid", spy)
    # A blank page shows no grid lines and no tilt: only the column grid and the
    # trace are left to run, with the plain mask estimator and no page shift.
    args = digitize.get_parser().parse_args([
        "-d", str(data_folder),
        "-o", str(output_folder),
        "--mask_folder", str(mask_folder),
        "--rotation", "hough",
        "--perspective", "off",
        "--resolution", "keep",
        "--trace_estimator", "mask",
        "--trace_shift", "off",
        "--baseline", "page",
        "--sharpen", mode,
        "--no-verbose",
    ])
    digitize.run(args)

    assert seen == [mode] * (len(SHORT_COLUMNS) + 1)
    with open(output_folder / "qc.csv", newline="") as f:
        assert next(csv.DictReader(f))["sharpen"] == mode
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        record = wfdb.rdrecord(str(output_folder / "rec"))
    assert record.p_signal.shape == (4 * SHORT_SAMPLES, len(SHORT_COLUMNS) + 1)
