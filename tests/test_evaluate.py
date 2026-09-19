"""Unit tests for src/run/evaluate.py. No model and no data files required."""
import argparse
import os

import numpy as np
import pytest
import wfdb

from src.run import evaluate


def _sine(n=1000, freq=5.0, fs=500.0, phase=0.0):
    t = np.arange(n) / fs
    return np.sin(2 * np.pi * freq * t + phase)


# ---------------------------------------------------------------- raw metrics
def test_identical_signals_give_infinite_snr_and_unit_correlation():
    x = _sine()
    assert np.isinf(evaluate.compute_raw_snr(x, x))
    assert evaluate.compute_pearson(x, x) == pytest.approx(1.0)


def test_snr_decreases_monotonically_with_noise_sigma():
    rng = np.random.default_rng(0)
    x = _sine(n=5000)
    snrs = [
        evaluate.compute_raw_snr(x, x + rng.normal(0.0, sigma, size=x.size))
        for sigma in (0.01, 0.05, 0.1, 0.5, 1.0)
    ]
    assert all(np.isfinite(snrs))
    assert all(a > b for a, b in zip(snrs, snrs[1:])), snrs


def test_pearson_is_nan_for_constant_or_all_zero_prediction():
    x = _sine()
    assert np.isnan(evaluate.compute_pearson(x, np.zeros_like(x)))
    assert np.isnan(evaluate.compute_pearson(x, np.full_like(x, 3.0)))
    assert np.isnan(evaluate.compute_pearson(np.zeros(10), np.zeros(10)))


def test_pearson_ignores_nan_samples():
    x = _sine(n=200)
    y = x.copy()
    y[:20] = np.nan
    assert evaluate.compute_pearson(x, y) == pytest.approx(1.0)


# ------------------------------------------------------------------ alignment
@pytest.mark.parametrize("shift", [-25, -7, 0, 3, 40])
def test_aligned_snr_recovers_a_pure_time_shift(shift):
    x = _sine(n=2000, freq=3.0)
    # The prediction is x delayed by `shift` samples (NaN-padded at the edges).
    y = np.full_like(x, np.nan)
    if shift >= 0:
        y[shift:] = x[: x.size - shift]
    else:
        y[: x.size + shift] = x[-shift:]

    snr_raw = evaluate.compute_raw_snr(x, y)
    snr_aligned, found_shift, offset = evaluate.compute_aligned_snr(x, y, max_shift=50)
    assert found_shift == shift
    assert offset == pytest.approx(0.0, abs=1e-9)
    assert snr_aligned > 60.0  # essentially perfect after alignment
    if shift != 0:
        assert snr_aligned > snr_raw


def test_aligned_snr_removes_a_constant_vertical_offset():
    x = _sine(n=1000)
    y = x + 0.75
    snr_aligned, found_shift, offset = evaluate.compute_aligned_snr(x, y, max_shift=10)
    assert found_shift == 0
    assert offset == pytest.approx(0.75, abs=1e-6)
    assert snr_aligned > 60.0


def test_aligned_snr_respects_max_shift():
    x = _sine(n=2000, freq=3.0)
    y = np.full_like(x, np.nan)
    y[60:] = x[:-60]
    _, found_shift, _ = evaluate.compute_aligned_snr(x, y, max_shift=10)
    assert abs(found_shift) <= 10


# --------------------------------------------------------------- lead helpers
def test_valid_window_from_nan_mask():
    sig = np.full(5000, np.nan)
    sig[2500:3750] = _sine(n=1250)
    assert evaluate.valid_window(sig) == (2500, 3750)
    assert evaluate.valid_window(np.full(10, np.nan)) is None


def test_match_prediction_prefix_and_identity():
    preds = ["00001_hr-0_0000", "00002_hr", "00010_hr-0_0000"]
    assert evaluate.match_prediction("00001_hr", preds) == "00001_hr-0_0000"
    assert evaluate.match_prediction("00002_hr", preds) == "00002_hr"
    assert evaluate.match_prediction("00003_hr", preds) is None
    # A prefix that is not followed by "-" must not match.
    assert evaluate.match_prediction("00001", preds) is None


def test_evaluate_lead_auto_prefers_start_placement():
    gt = np.full(5000, np.nan)
    gt[2500:3750] = _sine(n=1250)
    pred = np.zeros(5000)
    pred[:1250] = gt[2500:3750]  # digitiser writes every short lead at 0:1250

    res = evaluate.evaluate_lead(gt, pred, (2500, 3750), placement="auto")
    assert res["placement"] == "start"
    assert np.isinf(res["snr_raw"])
    assert res["pearson_r"] == pytest.approx(1.0)

    res_window = evaluate.evaluate_lead(gt, pred, (2500, 3750), placement="window")
    assert res_window["placement"] == "window"
    assert np.isnan(res_window["pearson_r"])  # compared against all zeros


def test_evaluate_lead_auto_prefers_window_placement_when_aligned_there():
    gt = np.full(5000, np.nan)
    gt[2500:3750] = _sine(n=1250)
    pred = np.zeros(5000)
    pred[2500:3750] = gt[2500:3750]

    res = evaluate.evaluate_lead(gt, pred, (2500, 3750), placement="auto")
    assert res["placement"] == "window"
    assert np.isinf(res["snr_raw"])


# ------------------------------------------------------------------ end to end
def _write_record(directory, name, signal, sig_names):
    wfdb.wrsamp(
        name,
        fs=500,
        units=["mV"] * signal.shape[1],
        sig_name=list(sig_names),
        p_signal=signal,
        write_dir=str(directory),
        fmt=["16"] * signal.shape[1],
        adc_gain=[1000.0] * signal.shape[1],
        baseline=[0] * signal.shape[1],
    )


def test_end_to_end_auto_picks_start(tmp_path):
    """Tiny 3-lead records: NaN-masked GT vs a 'start'-placed prediction."""
    n = 600
    leads = ["I", "II", "V1"]
    windows = {"I": (0, 200), "II": (0, n), "V1": (400, 600)}

    rng = np.random.default_rng(7)
    gt = np.full((n, len(leads)), np.nan)
    pred = np.zeros((n, len(leads)))
    for j, lead in enumerate(leads):
        i0, i1 = windows[lead]
        wave = _sine(n=i1 - i0, freq=4.0)
        gt[i0:i1, j] = wave
        # Prediction: same wave written at the start of the record, plus a
        # little noise so the SNR is finite but high.
        pred[: i1 - i0, j] = wave + rng.normal(0.0, 0.01, size=i1 - i0)

    gt_dir = tmp_path / "gt"
    pred_dir = tmp_path / "pred"
    gt_dir.mkdir()
    pred_dir.mkdir()
    _write_record(gt_dir, "rec001", gt, leads)
    _write_record(pred_dir, "rec001-0_0000", pred, leads)

    out_csv = tmp_path / "out.csv"
    args = argparse.Namespace(
        gt_folder=str(gt_dir),
        pred_folder=str(pred_dir),
        output_file=str(out_csv),
        placement="auto",
        max_shift=20,
        verbose=False,
    )
    df = evaluate.run(args)

    assert os.path.exists(out_csv)
    assert len(df) == 3
    assert set(df["lead"]) == set(leads)
    assert not df["missing"].any()
    # Lead II spans the whole record, so both placements are identical there.
    short = df[df["lead"] != "II"]
    assert set(short["placement"]) == {"start"}
    assert (df["snr_raw"] > 20).all()
    assert (df["pearson_r"] > 0.99).all()


def test_end_to_end_missing_lead_is_reported(tmp_path):
    n = 400
    gt = np.full((n, 2), np.nan)
    gt[:, 0] = _sine(n=n)
    gt[:200, 1] = _sine(n=200)
    pred = np.zeros((n, 1))
    pred[:, 0] = gt[:, 0]

    gt_dir = tmp_path / "gt"
    pred_dir = tmp_path / "pred"
    gt_dir.mkdir()
    pred_dir.mkdir()
    _write_record(gt_dir, "rec001", gt, ["I", "II"])
    _write_record(pred_dir, "rec001-0_0000", pred, ["I"])

    args = argparse.Namespace(
        gt_folder=str(gt_dir),
        pred_folder=str(pred_dir),
        output_file=None,
        placement="auto",
        max_shift=5,
        verbose=False,
    )
    df = evaluate.run(args)
    assert int(df["missing"].sum()) == 1
    missing_row = df[df["missing"]].iloc[0]
    assert missing_row["lead"] == "II"
    assert np.isnan(missing_row["snr_raw"])
    assert np.isnan(missing_row["pearson_r"])


def test_parser_defaults():
    args = evaluate.get_parser().parse_args(["-g", "a", "-p", "b"])
    assert args.placement == "auto"
    assert args.max_shift == 50
    assert args.output_file is None
