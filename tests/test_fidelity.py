"""Unit tests for scripts/fidelity.py.

No data/ folder, no torch and no judge weights required.
"""
import math
import os
import re
import subprocess
import sys
import types
import warnings

import numpy as np
import pandas as pd
import pytest
import wfdb

from scripts import fidelity

CLASSES = fidelity.JUDGE_CLASSES
REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


class FakeJudge:
    """Deterministic stand-in for scripts.judge.Judge: a fixed linear map of
    per-lead mean, std and mean |x| of the prepared input."""

    name = "fake"
    classes = CLASSES
    mean = 0.0
    std = 1.0

    def __init__(self, seed=0):
        rng = np.random.default_rng(seed)
        self.weights = rng.normal(size=(36, len(CLASSES)))
        self.bias = rng.normal(size=len(CLASSES))

    def predict_logits(self, X, batch_size=512):
        X = np.asarray(X, dtype=np.float64)
        if X.ndim == 2:
            X = X[None]
        if not np.all(np.isfinite(X)):
            raise ValueError("non-finite judge input")
        features = np.concatenate(
            [X.mean(axis=1), X.std(axis=1), np.abs(X).mean(axis=1)], axis=1
        )
        return features @ self.weights + self.bias

    def predict_proba(self, X, batch_size=512):
        return fidelity.sigmoid(self.predict_logits(X, batch_size))


def _signal(n=5000, fs=500, seed=0):
    """A smooth 12-lead test signal in mV, quantised to the WFDB gain of 1000."""
    rng = np.random.default_rng(seed)
    t = np.arange(n) / fs
    freqs = rng.uniform(0.8, 3.0, size=12)
    amps = rng.uniform(0.2, 1.5, size=12)
    x = amps * np.sin(2 * np.pi * freqs * t[:, None]) + 0.05 * rng.normal(size=(n, 12))
    return np.round(x, 3)


def _write(directory, name, signal, sig_names=fidelity.LEADS):
    os.makedirs(directory, exist_ok=True)
    # wfdb writes NaN as the fmt-16 sentinel, but warns while casting.
    with np.errstate(invalid="ignore"), warnings.catch_warnings():
        warnings.simplefilter("ignore", RuntimeWarning)
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


# ------------------------------------------------------ 1. identical inputs
def test_identical_inputs_give_perfect_agreement():
    rng = np.random.default_rng(3)
    n = 40
    logits = rng.normal(size=(n, 5))
    y = (rng.random((n, 5)) < 0.3).astype(int)
    y[:, 1] = 0  # HYP: no positives -> AUROC undefined
    thr = np.zeros(5)
    res = fidelity.compare(y, logits, logits.copy(), thr, n_boot=200, seed=0)
    assert res["kappa"] == 1.0
    # No flips: the bootstrap CIs are degenerate and reported as NaN.
    assert res["ci_degenerate"] is True
    degenerate_keys = (
        "kappa_lo", "kappa_hi", "flip_lo", "flip_hi", "rflip_boot_lo", "rflip_boot_hi"
    )
    for key in degenerate_keys:
        assert np.isnan(res[key])
    assert res["flip_rate"] == 0.0
    assert res["record_flip_rate"] == 0.0
    assert res["n_flips"] == 0
    assert res["mean_abs_dp"] == 0.0 and res["max_abs_dp"] == 0.0
    assert res["auroc_retention_macro"] == pytest.approx(1.0)
    for c in CLASSES:
        assert res[f"kappa_{c}"] == 1.0
        if c == "HYP":
            assert np.isnan(res[f"ret_{c}"])
        else:
            assert res[f"ret_{c}"] == pytest.approx(1.0)
    assert res["rflip_exact_lo"] == 0.0
    assert res["rflip_exact_hi"] == pytest.approx(1 - 0.025 ** (1 / n))


# ------------------------------------------------------------- 2. kappa
def test_cohen_kappa_matches_sklearn_and_constant_raters():
    metrics = pytest.importorskip("sklearn.metrics")
    rng = np.random.default_rng(7)
    for _ in range(20):
        a = rng.random(60) < rng.uniform(0.2, 0.8)
        b = np.where(rng.random(60) < 0.8, a, ~a)
        expected = metrics.cohen_kappa_score(a, b)
        assert fidelity.cohen_kappa(a, b) == pytest.approx(expected)
    ones = np.ones(10, dtype=bool)
    assert fidelity.cohen_kappa(ones, ones) == 1.0
    assert fidelity.cohen_kappa(~ones, ~ones) == 1.0
    assert fidelity.cohen_kappa(ones, ~ones) == 0.0  # p_o = 0, p_e = 0
    assert np.isnan(fidelity.cohen_kappa([], []))


def test_auroc_matches_sklearn_with_ties():
    metrics = pytest.importorskip("sklearn.metrics")
    rng = np.random.default_rng(11)
    y = rng.random(80) < 0.4
    s = np.round(rng.normal(size=80) + y, 1)  # rounded -> ties
    assert fidelity.auroc(y, s) == pytest.approx(metrics.roc_auc_score(y, s))
    assert np.isnan(fidelity.auroc(np.zeros(5), np.arange(5)))


# ------------------------------------------------------------- 3. flips
def test_hand_built_flip_example():
    ref = np.array(
        [
            [0, 0, 0, 1, 0],
            [1, 0, 0, 0, 1],
            [0, 0, 1, 0, 0],
            [0, 0, 0, 1, 0],
        ],
        dtype=bool,
    )
    test = ref.copy()
    test[0, 3] = False  # record 0: NORM flips
    test[0, 4] = True  # record 0: STTC flips
    test[2, 2] = False  # record 2: MI flips
    assert fidelity.flip_rate(ref, test) == pytest.approx(3 / 20)
    assert fidelity.record_flip_rate(ref, test) == pytest.approx(2 / 4)
    assert list(fidelity.flip_counts(ref, test)) == [0, 0, 1, 1, 1]

    # Same example through logits and thresholds (0 is a strict boundary).
    thr = np.zeros(5)
    logit_ref = np.where(ref, 1.0, -1.0)
    logit_test = np.where(test, 1.0, -1.0)
    res = fidelity.compare(None, logit_ref, logit_test, thr, n_boot=0)
    assert res["n_flips"] == 3
    assert res["n_record_flips"] == 2
    assert res["flip_rate"] == pytest.approx(0.15)
    assert res["record_flip_rate"] == pytest.approx(0.5)
    assert [res[f"flips_{c}"] for c in CLASSES] == [0, 0, 1, 1, 1]
    assert np.isnan(res["kappa_lo"]) and np.isnan(res["auroc_retention_macro"])


def test_decisions_are_strict_and_nan_threshold_is_negative():
    dec = fidelity.decisions(np.array([[0.0, 0.5, 1.0]]), np.array([0.0, 0.0, np.nan]))
    assert dec.tolist() == [[False, True, False]]


# ------------------------------------------------------------- 4. layout
def test_layout_mask_windows():
    m500 = fidelity.layout_mask(5000, 500)
    counts = dict(zip(fidelity.LEADS, m500.sum(axis=0)))
    assert counts["II"] == 5000
    assert all(counts[lead] == 1250 for lead in fidelity.LEADS if lead != "II")
    starts = {
        lead: int(np.flatnonzero(m500[:, j])[0])
        for j, lead in enumerate(fidelity.LEADS)
    }
    assert starts["I"] == starts["III"] == starts["II"] == 0
    assert starts["aVR"] == starts["aVL"] == starts["aVF"] == 1250
    assert starts["V1"] == starts["V2"] == starts["V3"] == 2500
    assert starts["V4"] == starts["V5"] == starts["V6"] == 3750

    m100 = fidelity.layout_mask(1000, 100)
    counts = dict(zip(fidelity.LEADS, m100.sum(axis=0)))
    assert counts["II"] == 1000
    assert all(counts[lead] == 250 for lead in fidelity.LEADS if lead != "II")
    assert np.flatnonzero(m100[:, fidelity.LEADS.index("V4")])[0] == 750


# ------------------------------------------------------- 5. prepare_for_judge
def test_prepare_for_judge_masked_500hz():
    x = fidelity.apply_mask(_signal(), fidelity.layout_mask(5000, 500))
    out = fidelity.prepare_for_judge(x, 500)
    assert out.shape == (1000, 12) and out.dtype == np.float32
    assert np.all(np.isfinite(out))
    window = fidelity.layout_mask(1000, 100)
    assert np.all(out[~window] == 0.0)
    avr = fidelity.LEADS.index("aVR")
    assert np.all(out[250:500, avr] != 0.0)


def test_prepare_for_judge_identity_at_100hz():
    x = _signal(1000, 100, seed=2).astype(np.float32)
    out = fidelity.prepare_for_judge(x, 100)
    assert np.array_equal(out, x)


def test_prepare_for_judge_interpolates_interior_gaps():
    x = np.zeros((1000, 12))
    x[:, 0] = np.linspace(0.0, 1.0, 1000)
    x[100:110, 0] = np.nan
    x[:50, 1] = np.nan  # leading gap -> zero
    x[:, 2] = np.nan  # fully missing lead -> zeros
    out = fidelity.prepare_for_judge(x, 100)
    assert np.allclose(out[100:110, 0], np.linspace(0.0, 1.0, 1000)[100:110], atol=1e-6)
    assert np.all(out[:50, 1] == 0.0)
    assert np.all(out[:, 2] == 0.0)
    assert np.all(np.isfinite(out))


def test_prepare_for_judge_pads_and_truncates():
    assert fidelity.prepare_for_judge(np.ones((400, 12)), 500).shape == (1000, 12)
    short = fidelity.prepare_for_judge(np.ones((400, 12)), 500)
    assert np.all(short[80:] == 0.0)
    assert fidelity.prepare_for_judge(np.ones((6000, 12)), 500).shape == (1000, 12)


def test_window_qc_counts_gaps_and_missing_leads():
    mask = fidelity.layout_mask(5000, 500)
    d = fidelity.apply_mask(_signal(), mask)
    d[0:125, 0] = np.nan  # 10 % of lead I's window
    d[:, fidelity.LEADS.index("V6")] = np.nan
    qc = fidelity.window_qc(d, mask)
    assert qc["nan_frac"][0] == pytest.approx(0.1)
    assert qc["missing"].tolist() == [lead == "V6" for lead in fidelity.LEADS]
    assert qc["n_window"][1] == 5000


# ------------------------------------------------------- 6. symmetric path
def test_symmetric_path_ignores_values_outside_the_window():
    orig = _signal(seed=4)
    mask = fidelity.layout_mask(5000, 500)
    masked = fidelity.prepare_for_judge(fidelity.apply_mask(orig, mask), 500)
    dig = orig.copy()
    rng = np.random.default_rng(5)
    dig[~mask] = 5.0 * rng.normal(size=int((~mask).sum()))  # junk outside the windows
    dig_prepared = fidelity.prepare_for_judge(fidelity.apply_mask(dig, mask), 500)
    assert np.array_equal(masked.view(np.uint32), dig_prepared.view(np.uint32))
    judge = FakeJudge()
    assert np.array_equal(
        judge.predict_logits(masked), judge.predict_logits(dig_prepared)
    )
    # The full original is a different input (layout information loss).
    full = fidelity.prepare_for_judge(orig, 500)
    assert not np.array_equal(full, masked)


# ------------------------------------------------------------- 7. bootstrap
def test_bootstrap_ci_constant_and_deterministic():
    lo, hi, n_valid = fidelity.bootstrap_ci(lambda idx: 0.42, 10, n_boot=100, seed=0)
    assert lo == hi == 0.42 and n_valid == 100
    values = np.random.default_rng(0).normal(size=30)
    ci1 = fidelity.bootstrap_ci(lambda idx: values[idx].mean(), 30, n_boot=300, seed=9)
    ci2 = fidelity.bootstrap_ci(lambda idx: values[idx].mean(), 30, n_boot=300, seed=9)
    assert ci1 == ci2
    assert ci1[0] < values.mean() < ci1[1]
    lo, hi, n_valid = fidelity.bootstrap_ci(lambda idx: np.nan, 5, n_boot=50)
    assert np.isnan(lo) and np.isnan(hi) and n_valid == 0
    assert fidelity.bootstrap_ci(lambda idx: 1.0, 5, n_boot=0)[2] == 0


def test_cluster_bootstrap_resamples_whole_patients():
    values = np.random.default_rng(1).normal(size=12)
    stat = lambda idx: values[idx].mean()
    # One record per cluster, in record order: the record bootstrap exactly.
    singletons = [f"p{i}" for i in range(12)]
    assert fidelity.bootstrap_ci(stat, 12, 200, seed=3, groups=singletons) == (
        fidelity.bootstrap_ci(stat, 12, 200, seed=3)
    )
    # Clusters of 3 records: every draw holds whole clusters, 4 of them.
    groups = [i // 3 for i in range(12)]
    seen = []

    def record(idx):
        seen.append(np.asarray(idx))
        return values[idx].mean()

    lo, hi, n_valid = fidelity.bootstrap_ci(record, 12, 50, seed=3, groups=groups)
    assert n_valid == 50 and lo <= hi
    for idx in seen:
        assert len(idx) == 12
        for k in range(0, 12, 3):
            block = idx[k:k + 3]
            assert block[0] % 3 == 0 and list(block) == list(range(block[0], block[0] + 3))
    members = fidelity.cluster_members(["b", "a", "b", 7], 4)
    assert [m.tolist() for m in members] == [[0, 2], [1], [3]]
    with pytest.raises(ValueError, match="cluster labels"):
        fidelity.cluster_members(["a"], 2)
    with pytest.raises(ValueError, match="must not be missing"):
        fidelity.cluster_members(["a", None], 2)
    # compare: groups=None is the record bootstrap; groups change the CIs only.
    ref, test = _flipping_pair()
    thr = np.zeros(5)
    plain = fidelity.compare(None, ref, test, thr, n_boot=300, seed=4)
    pairs = [i // 2 for i in range(len(ref))]
    clus = fidelity.compare(None, ref, test, thr, n_boot=300, seed=4, groups=pairs)
    for key in ("kappa", "flip_rate", "record_flip_rate", "mean_abs_dp", "n_flips"):
        assert clus[key] == plain[key]
    assert (clus["kappa_lo"], clus["kappa_hi"]) != (plain["kappa_lo"], plain["kappa_hi"])
    dec_ref, dec_test = fidelity.decisions(ref, thr), fidelity.decisions(test, thr)
    manual = fidelity.bootstrap_ci(
        lambda idx: fidelity.cohen_kappa(dec_ref[idx], dec_test[idx]),
        len(ref), 300, seed=4, groups=pairs,
    )
    assert (clus["kappa_lo"], clus["kappa_hi"]) == manual[:2]


# ------------------------------------------------------- 8. Clopper-Pearson
def test_clopper_pearson_edges():
    lo, hi = fidelity.clopper_pearson(0, 8)
    assert lo == 0.0 and hi == pytest.approx(0.369, abs=1e-3)
    lo, hi = fidelity.clopper_pearson(8, 8)
    assert lo == pytest.approx(0.631, abs=1e-3) and hi == 1.0
    lo, hi = fidelity.clopper_pearson(3, 8)
    assert 0.0 < lo < 3 / 8 < hi < 1.0


# ------------------------------------------------------------ 9. end to end
def _make_condition(tmp_path, n_records=6, perturb=None):
    gt_dir, orig_dir, pred_dir = tmp_path / "gt", tmp_path / "orig", tmp_path / "pred"
    mask = fidelity.layout_mask(5000, 500)
    records = [f"{i:05d}_hr" for i in range(1, n_records + 1)]
    rng = np.random.default_rng(12)
    # The PTB-XL originals spell AVR, AVL, AVF.
    orig_names = [s.upper() if s.startswith("aV") else s for s in fidelity.LEADS]
    for i, rec in enumerate(records):
        orig = _signal(seed=100 + i)
        gt = fidelity.apply_mask(orig, mask)
        dig = orig + np.where(mask, 0.0, rng.normal(size=orig.shape))  # junk outside
        if perturb is not None:
            dig = perturb(i, dig)
        _write(orig_dir, rec, orig, orig_names)
        _write(gt_dir, rec, gt)
        _write(pred_dir, rec + fidelity.PRED_SUFFIX, dig)
    return str(gt_dir), str(pred_dir), str(orig_dir), records


def _labels(n_records):
    rows = []
    for i in range(1, n_records + 1):
        row = {c: 0 for c in CLASSES}
        if i <= 3:
            row["NORM"] = 1
        elif i <= 5:
            row["MI"] = 1
        row["strat_fold"] = 10
        rows.append(row)
    return pd.DataFrame(rows, index=pd.Index(range(1, n_records + 1), name="ecg_id"))


def test_end_to_end_identical_digitisation(tmp_path):
    gt_dir, pred_dir, orig_dir, records = _make_condition(tmp_path)
    judge = FakeJudge()
    thr = np.zeros(5)
    row, recs = fidelity.evaluate_condition(
        judge, "toy", gt_dir, pred_dir, orig_dir, thr, thr + 0.5,
        labels=_labels(len(records)), n_boot=100, seed=0, set_name="unit",
    )
    assert row["n_records"] == 6
    assert row["kappa"] == 1.0 and row["kappa_thrfull"] == 1.0
    assert row["n_flips"] == 0 and row["flip_rate"] == 0.0
    assert row["record_flip_rate"] == 0.0
    assert row["mean_abs_dp"] == 0.0
    assert row["missing_leads"] == 0 and row["interior_nan_frac"] == 0.0
    assert row["npos_NORM"] == 3 and row["npos_MI"] == 2
    assert row["ret_NORM"] == pytest.approx(1.0)
    assert row["ci_note"].startswith("6 records: CI wide")
    assert len(recs) == 6 * 5
    assert not recs["flip"].any()
    assert (recs["logit_masked"] == recs["logit_dig"]).all()
    assert (recs["logit_full"] != recs["logit_masked"]).any()


def test_end_to_end_perturbed_digitisation(tmp_path):
    ii = fidelity.LEADS.index("II")

    def perturb(i, dig):
        dig = dig.copy()
        dig[:, ii] += 0.3  # baseline offset on the rhythm lead
        dig[1300:1400, fidelity.LEADS.index("aVL")] = np.nan  # interior gap
        return dig

    gt_dir, pred_dir, orig_dir, records = _make_condition(tmp_path, perturb=perturb)
    thr = np.zeros(5)
    row, recs = fidelity.evaluate_condition(
        FakeJudge(), "toy", gt_dir, pred_dir, orig_dir, thr, thr, n_boot=50, seed=0,
    )
    assert row["mean_abs_dp"] > 0.0
    assert row["interior_nan_frac"] == pytest.approx(6 * 100 / (6 * (11 * 1250 + 5000)))
    assert (recs["abs_dp"] > 0).any()


def test_end_to_end_rejects_an_original_that_differs_from_the_ground_truth(tmp_path):
    gt_dir, pred_dir, orig_dir, records = _make_condition(tmp_path, n_records=2)
    other = _signal(seed=999)
    _write(orig_dir, records[0], other)
    with pytest.raises(ValueError, match="differs from the ground truth"):
        fidelity.condition_inputs(gt_dir, pred_dir, orig_dir)


def _link_originals(orig_dir, store, targets):
    """Move the original records of orig_dir into `store` and replace
    <orig_dir>/<rec>.hea/.dat by relative symlinks to the files of targets[rec]."""
    os.makedirs(store)
    for rec in targets:
        for ext in (".hea", ".dat"):
            os.rename(os.path.join(orig_dir, rec + ext), os.path.join(store, rec + ext))
    for rec, target in targets.items():
        for ext in (".hea", ".dat"):
            relative = os.path.relpath(os.path.join(store, target + ext), orig_dir)
            os.symlink(relative, os.path.join(orig_dir, rec + ext))


def test_originals_read_through_relative_symlinks(tmp_path):
    gt_dir, pred_dir, orig_dir, records = _make_condition(tmp_path, n_records=3)
    plain = fidelity.condition_inputs(gt_dir, pred_dir, orig_dir)
    store = str(tmp_path / "store" / "records500" / "00000")
    _link_originals(orig_dir, store, {rec: rec for rec in records})
    link = os.path.join(orig_dir, records[0] + ".hea")
    assert os.path.islink(link) and not os.path.isabs(os.readlink(link))
    assert os.readlink(link) == os.path.join(
        "..", "store", "records500", "00000", records[0] + ".hea"
    )
    linked = fidelity.condition_inputs(gt_dir, pred_dir, orig_dir)
    assert linked["records"] == plain["records"]
    for kind in ("full", "masked", "dig"):
        for a, b in zip(plain[kind], linked[kind]):
            assert np.array_equal(a, b, equal_nan=True)
    pred_base = os.path.join(pred_dir, records[1] + fidelity.PRED_SUFFIX)
    orig, mask, d, present, fs = fidelity._read_condition_record(
        gt_dir, orig_dir, records[1], pred_base
    )
    direct, _ = fidelity.read_record(os.path.join(store, records[1]))
    assert present and fs == 500 and np.array_equal(orig, direct)


def test_originals_symlinked_to_another_record_are_rejected(tmp_path):
    gt_dir, pred_dir, orig_dir, records = _make_condition(tmp_path, n_records=2)
    store = str(tmp_path / "store" / "records500" / "00000")
    # records[0] points at the files of records[1].
    _link_originals(
        orig_dir, store, {records[0]: records[1], records[1]: records[1]}
    )
    with pytest.raises(ValueError, match="differs from the ground truth inside"):
        fidelity.condition_inputs(gt_dir, pred_dir, orig_dir)


def test_read_record_maps_leads_by_name_case_insensitively(tmp_path):
    x = _signal(seed=8)
    reordered = list(reversed([s.upper() for s in fidelity.LEADS]))
    _write(tmp_path, "rec", x[:, ::-1], reordered)
    signal, fs = fidelity.read_record(str(tmp_path / "rec"))
    assert fs == 500.0
    assert np.allclose(signal, x, atol=1e-9)


# ------------------------------------------------------------ 10. SNR helpers
def test_snr_median_uses_non_missing_rows_only():
    ev = pd.DataFrame(
        {
            "record": ["r1", "r1", "r1", "r2", "r2"],
            "lead": ["I", "II", "III", "I", "II"],
            "snr_raw": [10.0, 20.0, -50.0, 30.0, np.nan],
            "missing": [False, False, True, False, True],
        }
    )
    assert fidelity.condition_snr_median(ev) == pytest.approx(20.0)
    per_record = fidelity.record_snr(ev)
    assert per_record.loc["r1", "snr_rec_median"] == pytest.approx(15.0)
    assert per_record.loc["r1", "snr_rec_min"] == pytest.approx(10.0)
    assert per_record.loc["r2", "snr_rec_median"] == pytest.approx(30.0)
    # 'missing' read back as strings.
    ev_str = ev.assign(missing=ev["missing"].astype(str))
    assert fidelity.condition_snr_median(ev_str) == pytest.approx(20.0)


# ------------------------------------------------------------ plumbing
def test_logit_cache_round_trip_and_invalidation(tmp_path):
    path = str(tmp_path / "cache" / "fid_fake.npz")
    cache = fidelity.LogitCache(path, tag="fake:abc")
    cache.put("c/r/dig", "sha1", np.arange(5.0))
    cache.save()
    again = fidelity.LogitCache(path, tag="fake:abc")
    assert np.array_equal(again.get("c/r/dig", "sha1"), np.arange(5.0))
    assert again.get("c/r/dig", "other-sha") is None
    assert len(fidelity.LogitCache(path, tag="fake:new-weights")) == 0


def test_cached_logits_only_predicts_misses(tmp_path):
    class Counting(FakeJudge):
        calls = 0

        def predict_logits(self, X, batch_size=512):
            Counting.calls += len(X)
            return super().predict_logits(X, batch_size)

    judge = Counting()
    cache = fidelity.LogitCache(str(tmp_path / "c.npz"), tag="t")
    arrays = [
        fidelity.prepare_for_judge(_signal(1000, 100, seed=s), 100) for s in range(3)
    ]
    first = fidelity.cached_logits(judge, arrays, ["a", "b", "c"], cache)
    second = fidelity.cached_logits(judge, arrays, ["a", "b", "c"], cache)
    assert Counting.calls == 3
    assert np.array_equal(first, second)


V0_CONDITIONS = [
    "gen_clean", "gen_aug", "gen_rot",
    "blur", "jpeg", "gridfaint", "gridblue", "margin", "persp05", "persp15",
    "photo", "scale080", "scale088", "scale115", "scale125",
]
B_CONDITIONS = ["b_clean", "b_scan", "b_aug", "b_rot"]


def test_conditions_cover_the_fifteen_absolute_conditions():
    assert len(fidelity.CONDITIONS) == 19
    assert "rot03" not in fidelity.CONDITIONS and "rot15" not in fidelity.CONDITIONS
    # 'all' keeps its v0 meaning: the 15 absolute conditions, no Phase B one.
    v0 = fidelity.resolve_conditions(["all"])
    assert v0 == V0_CONDITIONS
    assert not any(c.startswith("b_") for c in v0)
    assert list(fidelity.CONDITIONS)[:15] == V0_CONDITIONS
    with pytest.raises(ValueError):
        fidelity.resolve_conditions(["nope"])


def test_phase_b_conditions_are_the_fidelity_b_set():
    for c in ("clean", "scan", "aug", "rot"):
        assert fidelity.CONDITIONS[f"b_{c}"] == (
            f"data/fidelity/images/{c}", f"data/fidelity/out/{c}", "fidelity_b"
        )
    assert [c for c, v in fidelity.CONDITIONS.items() if v[2] == "fidelity_b"] == (
        B_CONDITIONS
    )
    assert list(fidelity.CONDITIONS)[15:] == B_CONDITIONS
    groups = {"all", *(v[2] for v in fidelity.CONDITIONS.values())}
    assert groups == {"all", "gen_eval", "quasi_real", "fidelity_b"}
    assert not groups & set(fidelity.CONDITIONS)


def test_resolve_conditions_expands_groups_in_conditions_order():
    resolve = fidelity.resolve_conditions
    assert resolve(["fidelity_b"]) == B_CONDITIONS
    assert resolve(["b_scan"]) == ["b_scan"]
    assert resolve(["gen_eval"]) == ["gen_clean", "gen_aug", "gen_rot"]
    assert resolve(["quasi_real"]) == V0_CONDITIONS[3:]
    assert resolve(["all", "fidelity_b"]) == list(fidelity.CONDITIONS)
    assert len(resolve(["all", "fidelity_b"])) == 19
    # Duplicates are dropped, the first occurrence keeps its place.
    assert resolve(["b_rot", "fidelity_b"]) == ["b_rot", "b_clean", "b_scan", "b_aug"]
    assert resolve(["gen_aug", "all"])[:2] == ["gen_aug", "gen_clean"]
    assert resolve(None) == V0_CONDITIONS
    assert resolve([]) == V0_CONDITIONS
    for bad in (["nope"], ["all", "nope"]):
        with pytest.raises(ValueError, match="nope"):
            resolve(bad)


def test_ev_csv_per_condition(monkeypatch):
    for c in ("clean", "scan", "aug", "rot"):
        assert fidelity.ev_csv(f"b_{c}") == f"data/fidelity/ev/ev_{c}.csv"
    assert fidelity.ev_csv("gen_clean") == (
        "data/perspective_results/ev_gen_clean_inkpage_default.csv"
    )
    # The module globals are read at call time.
    monkeypatch.setattr(fidelity, "EV_CSV", "elsewhere/ev_{cond}.csv")
    assert fidelity.ev_csv("gen_clean") == "elsewhere/ev_gen_clean.csv"
    assert fidelity.ev_csv("b_scan") == "data/fidelity/ev/ev_scan.csv"


def test_importing_fidelity_needs_neither_torch_nor_the_judge():
    code = (
        "import sys; from scripts import fidelity; "
        "assert 'torch' not in sys.modules; assert 'scripts.judge' not in sys.modules"
    )
    subprocess.run([sys.executable, "-c", code], cwd=REPO_ROOT, check=True)


# ------------------------------------------------ 11. metrics that must fail
def test_retention_matches_sklearn_with_different_logits():
    metrics = pytest.importorskip("sklearn.metrics")
    rng = np.random.default_rng(21)
    n = 60
    y = (rng.random((n, 5)) < 0.35).astype(int)
    y[:, 1] = 0  # HYP: no positives -> undefined, left out of the macro
    y[:5] = 0  # 5 unlabelled records: excluded from every AUROC
    y[5, 3] = 1  # keep record 5 labelled
    ref = 1.2 * y + rng.normal(size=y.shape)
    test = ref + 0.8 * rng.normal(size=y.shape)
    ref[:5] = 50.0  # unlabelled rows would dominate the ranks if they were used
    res = fidelity.compare(y, ref, test, np.zeros(5), n_boot=0)
    lab = y.sum(axis=1) > 0
    assert res["n_labelled"] == int(lab.sum())
    aucs_ref, aucs_test = [], []
    for c, name in enumerate(CLASSES):
        if name == "HYP":
            assert np.isnan(res["ret_HYP"]) and np.isnan(res["auroc_ref_HYP"])
            continue
        a_ref = metrics.roc_auc_score(y[lab, c], ref[lab, c])
        a_test = metrics.roc_auc_score(y[lab, c], test[lab, c])
        assert res[f"auroc_ref_{name}"] == pytest.approx(a_ref)
        assert res[f"auroc_test_{name}"] == pytest.approx(a_test)
        assert res[f"ret_{name}"] == pytest.approx(a_test / a_ref)
        aucs_ref.append(a_ref)
        aucs_test.append(a_test)
    assert res["auroc_retention_macro"] == pytest.approx(
        np.mean(aucs_test) / np.mean(aucs_ref)
    )
    assert res["auroc_retention_macro"] != pytest.approx(1.0)


def test_abs_dp_is_on_probabilities():
    assert fidelity.abs_dp(0.0, math.log(3.0)) == pytest.approx(0.25)
    stats = fidelity.abs_dp_stats(np.zeros((2, 1)), np.array([[math.log(3.0)], [0.0]]))
    assert stats["mean"] == pytest.approx(0.125) and stats["max"] == pytest.approx(0.25)


def _flipping_pair(n=32, seed=5):
    rng = np.random.default_rng(seed)
    ref = rng.normal(size=(n, 5))
    test = ref + 0.6 * rng.normal(size=(n, 5))
    return ref, test


def test_compare_bootstrap_shares_resamples_and_is_reproducible():
    ref, test = _flipping_pair()
    thr = np.zeros(5)
    a = fidelity.compare(None, ref, test, thr, n_boot=300, seed=4)
    b = fidelity.compare(None, ref, test, thr, n_boot=300, seed=4)
    assert a["ci_degenerate"] is False and a["n_record_flips"] > 0
    for key in ("kappa_lo", "kappa_hi", "flip_lo", "flip_hi", "dp_lo", "dp_hi"):
        assert np.isfinite(a[key]) and a[key] == b[key]
    dec_ref, dec_test = fidelity.decisions(ref, thr), fidelity.decisions(test, thr)
    flips = dec_ref != dec_test
    manual = fidelity.bootstrap_ci(
        lambda idx: fidelity.cohen_kappa(dec_ref[idx], dec_test[idx]), 32, 300, seed=4
    )
    assert (a["kappa_lo"], a["kappa_hi"]) == manual[:2]
    manual = fidelity.bootstrap_ci(lambda idx: flips[idx].mean(), 32, 300, seed=4)
    assert (a["flip_lo"], a["flip_hi"]) == manual[:2]


def test_retention_bootstrap_keeps_the_class_set_of_the_full_sample():
    rng = np.random.default_rng(8)
    n, n_boot, seed = 16, 400, 3
    y = np.zeros((n, 5), dtype=int)
    y[:10, 3] = 1  # NORM: 10 positives
    y[10:, 4] = 1  # STTC: 6 positives
    y[0, 2] = 1  # MI: a single positive
    ref = 1.5 * y + rng.normal(size=y.shape)
    test = ref + 0.5 * rng.normal(size=y.shape)
    res = fidelity.compare(y, ref, test, np.zeros(5), n_boot=n_boot, seed=seed)
    full = fidelity.auroc_retention(y, ref, test)
    defined = full["defined"]
    assert defined.tolist() == [False, False, True, True, True]
    assert res["auroc_retention_macro"] == pytest.approx(full["macro_retention"])
    # Reference: only draws in which every class defined in the full sample is
    # defined, each averaged over exactly those classes.
    draws = np.random.default_rng(seed).integers(0, n, size=(n_boot, n))
    kept = []
    for idx in draws:
        r = fidelity.auroc_retention(y[idx], ref[idx], test[idx])
        if (r["defined"] == defined).all():
            kept.append(r["macro_retention"])
    assert 0 < len(kept) < n_boot
    assert res["ret_boot_valid"] == len(kept)
    assert res["ret_boot_frac"] == pytest.approx(len(kept) / n_boot)
    assert res["ret_lo"] == pytest.approx(np.percentile(kept, 2.5))
    assert res["ret_hi"] == pytest.approx(np.percentile(kept, 97.5))
    # require: NaN unless every required class is defined.
    miss_mi = np.flatnonzero(y[:, 2] == 0)
    r = fidelity.auroc_retention(y[miss_mi], ref[miss_mi], test[miss_mi], require=defined)
    assert np.isnan(r["macro_retention"])


def test_no_flip_condition_notes_the_exact_bound():
    logits = np.random.default_rng(2).normal(size=(32, 5))
    res = fidelity.compare(None, logits, logits.copy(), np.zeros(5), n_boot=100)
    assert res["ci_degenerate"] is True and np.isnan(res["kappa_lo"])
    assert res["rflip_exact_hi"] == pytest.approx(0.1089, abs=1e-4)
    note = fidelity.ci_note(
        32, no_flips=True, rflip_exact_hi=res["rflip_exact_hi"]
    )
    assert note == "no flips: bootstrap CI degenerate; record flip rate <= 0.109 (exact)"
    row = {
        "judge": "fake", "condition": "toy", "n_records": 32, "snr_median_db": 22.0,
        "n_record_flips": 0, "missing_outputs": 0, **res,
    }
    (line,) = fidelity.format_table([row])[1:]
    assert "[no flips]" in line and "0/32 [0.000, 0.109]" in line


def test_ci_note_lists_undefined_and_few_positive_or_negative_classes():
    note = fidelity.ci_note(
        8, npos=[0, 0, 1, 7, 0], nneg=[8, 8, 7, 1, 8], missing_outputs=2,
        ret_boot_frac=0.35,
    )
    assert note == (
        "8 records: CI wide; AUROC retention not interpretable; "
        "2/8 digitised outputs missing (judged all-zero); "
        "AUROC undefined: CD, HYP, STTC; <5 pos/neg: MI, NORM; "
        "retention CI conditional on 35% of draws"
    )
    assert fidelity.ci_note(32, npos=[6, 5, 7, 12, 8], nneg=[26, 27, 25, 20, 24]) == ""
    assert fidelity.ci_note(32, npos=None) == ""


# ------------------------------------------------ 12. signal path details
def test_masked_dc_offset_has_no_step_at_the_window_boundaries():
    x = np.full((5000, 12), 0.5)
    out = fidelity.prepare_for_judge(fidelity.apply_mask(x, fidelity.layout_mask()), 500)
    window = fidelity.layout_mask(1000, 100)
    assert np.allclose(out[window], 0.5, atol=1e-6)
    assert np.all(out[~window] == 0.0)
    # Unmasked record ends: no zero-padding step either.
    ramp = np.tile(np.linspace(-1.0, 1.0, 5000)[:, None], (1, 12))
    out = fidelity.prepare_for_judge(ramp, 500)
    assert np.allclose(out, ramp[::5], atol=2e-3)


def test_signal_diff_stats_zones():
    a = np.zeros((1000, 12))
    b = a.copy()
    b[0, 0] = 1.0  # head
    b[500, 3] = 0.5  # interior
    b[999, 1] = 2.0  # tail
    stats = fidelity.signal_diff_stats(a, b)
    assert stats["sig_head_max_mv"] == 1.0
    assert stats["sig_interior_max_mv"] == 0.5
    assert stats["sig_tail_max_mv"] == 2.0
    only_lead3 = np.zeros((1000, 12), dtype=bool)
    only_lead3[:, 3] = True
    stats = fidelity.signal_diff_stats(a[None], b[None], mask=only_lead3)
    assert stats["sig_head_max_mv"] == 0.0 and stats["sig_tail_max_mv"] == 0.0
    assert stats["sig_interior_max_mv"] == 0.5


# ------------------------------------------------ 13. end to end, stricter
def _shuffled_labels(n_records):
    """Asymmetric labels, index out of order, strat_fold = 100 + ecg_id."""
    labels = _labels(n_records)
    labels["STTC"] = 0
    labels.loc[n_records, "STTC"] = 1
    labels["strat_fold"] = 100 + labels.index.to_numpy()
    return labels.iloc[::-1]


def _perturb(i, dig):
    dig = dig.copy()
    dig[:, fidelity.LEADS.index("II")] += 0.1 * (i + 1)
    return dig


def test_end_to_end_labels_follow_the_ecg_id(tmp_path):
    gt_dir, pred_dir, orig_dir, records = _make_condition(tmp_path, perturb=_perturb)
    labels = _shuffled_labels(len(records))
    row, recs = fidelity.evaluate_condition(
        FakeJudge(), "toy", gt_dir, pred_dir, orig_dir, np.zeros(5), np.zeros(5),
        labels=labels, n_boot=0,
    )
    for _, r in recs.iterrows():
        assert r["label"] == labels.loc[r["ecg_id"], r["class"]]
        assert r["strat_fold"] == 100 + r["ecg_id"]
        assert r["ecg_id"] == int(r["record"][:5])
    assert row["npos_NORM"] == 3 and row["npos_MI"] == 2 and row["npos_STTC"] == 1


def test_end_to_end_full_thresholds_are_used_for_the_thrfull_columns(tmp_path):
    gt_dir, pred_dir, orig_dir, records = _make_condition(tmp_path, perturb=_perturb)
    judge = FakeJudge()
    _, recs = fidelity.evaluate_condition(
        judge, "toy", gt_dir, pred_dir, orig_dir, np.zeros(5), np.zeros(5), n_boot=0
    )
    l_masked = recs.pivot(index="record", columns="class", values="logit_masked")
    l_dig = recs.pivot(index="record", columns="class", values="logit_dig")
    l_masked = l_masked[list(CLASSES)].to_numpy()
    l_dig = l_dig[list(CLASSES)].to_numpy()
    low = min(l_masked.min(), l_dig.min()) - 1.0
    thr_layout = np.full(5, low)  # everything positive: no flip
    thr_full = np.full(5, low)
    thr_full[0] = 0.5 * (l_masked[0, 0] + l_dig[0, 0])  # record 0 flips on CD
    expected = np.mean(
        fidelity.decisions(l_masked, thr_full) != fidelity.decisions(l_dig, thr_full)
    )
    assert expected > 0
    row, _ = fidelity.evaluate_condition(
        judge, "toy", gt_dir, pred_dir, orig_dir, thr_layout, thr_full, n_boot=0
    )
    assert row["flip_rate"] == 0.0 and row["kappa"] == 1.0
    assert row["flip_rate_thrfull"] == pytest.approx(expected)
    assert row["kappa_thrfull"] < 1.0


def test_end_to_end_missing_digitised_record(tmp_path):
    gt_dir, pred_dir, orig_dir, records = _make_condition(tmp_path)
    for ext in (".hea", ".dat"):
        os.remove(os.path.join(pred_dir, records[2] + fidelity.PRED_SUFFIX + ext))
    row, recs = fidelity.evaluate_condition(
        FakeJudge(), "toy", gt_dir, pred_dir, orig_dir, np.zeros(5), np.zeros(5),
        n_boot=0,
    )
    assert row["missing_outputs"] == 1
    assert row["missing_leads"] == 12
    assert "1/6 digitised outputs missing" in row["ci_note"]
    line = fidelity.format_table([row])[1]
    assert line.rstrip().endswith(" 1")


def test_end_to_end_rejects_wrong_or_empty_folders(tmp_path):
    gt_dir, pred_dir, orig_dir, records = _make_condition(tmp_path, n_records=2)
    with pytest.raises(FileNotFoundError):
        fidelity.condition_inputs(gt_dir, pred_dir + "_does_not_exist", orig_dir)
    empty = tmp_path / "empty"
    empty.mkdir()
    with pytest.raises(ValueError, match="wrong folder"):
        fidelity.condition_inputs(gt_dir, str(empty), orig_dir)
    with pytest.raises(ValueError, match="No .hea records"):
        fidelity.condition_inputs(str(empty), pred_dir, orig_dir)


# ------------------------------------------------ 14. plumbing, stricter
def test_default_records_out_keeps_the_folder():
    assert fidelity.default_records_out("data/fidelity/table_v0.csv") == (
        "data/fidelity/table_v0_records.csv"
    )
    assert fidelity.default_records_out("data/fidelity/table_v0") == (
        "data/fidelity/table_v0_records.csv"
    )
    assert fidelity.default_records_out("x") == "x_records.csv"


def _fake_judge_module(classes=CLASSES):
    class Loaded(FakeJudge):
        pass

    Loaded.classes = tuple(classes)

    def load(name, weights_dir=None, device="cpu"):
        judge = Loaded()
        judge.weights_sha256 = "abc"
        return judge

    return types.SimpleNamespace(
        CLASSES=CLASSES, Judge=types.SimpleNamespace(load=load)
    )


def test_open_judge_checks_the_class_order_and_tags_the_device(tmp_path):
    args = types.SimpleNamespace(
        weights_dir=str(tmp_path), device="mps", no_cache=False,
        cache_dir=str(tmp_path / "cache"),
    )
    judge, cache = fidelity._open_judge(_fake_judge_module(), "fake", args)
    assert cache.tag == "fake:abc:mps"
    # One cache file per device: a device switch never empties the other one.
    assert cache.path == str(tmp_path / "cache" / "fid_fake_mps.npz")
    args.device = "cpu"
    _, cache_cpu = fidelity._open_judge(_fake_judge_module(), "fake", args)
    assert cache_cpu.path == str(tmp_path / "cache" / "fid_fake.npz")
    assert fidelity.cache_file("c", "x", "cuda:0") == os.path.join(
        "c", "fid_x_cuda_0.npz"
    )
    with pytest.raises(ValueError, match="classes"):
        fidelity._open_judge(_fake_judge_module(CLASSES[::-1]), "fake", args)


def test_threshold_device_mismatch_is_warned(tmp_path, capsys):
    path = tmp_path / "thresholds.csv"
    rows = [
        {"judge": "fake", "input": kind, "class": c, "threshold_logit": 0.0,
         "device": "cpu"}
        for kind in ("full", "layout") for c in CLASSES
    ]
    pd.DataFrame(rows).to_csv(path, index=False)
    assert fidelity.check_threshold_device(str(path), "fake", "cpu") == []
    assert fidelity.check_threshold_device(str(path), "fake", "mps") == ["cpu"]
    assert "come from ['cpu'] logits" in capsys.readouterr().err
    pd.DataFrame(rows).drop(columns="device").to_csv(path, index=False)
    assert fidelity.check_threshold_device(str(path), "fake", "mps") == []


def test_fold10_500hz_path_on_a_toy_ptbxl(tmp_path):
    root = tmp_path / "ptbxl"
    ids = [1, 2, 3]
    signals = {i: _signal(seed=40 + i) for i in ids}
    for i in ids:
        _write(root / "records500" / "00000", f"{i:05d}_hr", signals[i])
    db = pd.DataFrame(
        {"filename_hr": [f"records500/00000/{i:05d}_hr" for i in ids]},
        index=pd.Index(ids, name="ecg_id"),
    )

    def load_signals_100(ecg_ids, root=None):
        return np.stack([signals[int(e)][::5] for e in ecg_ids]).astype(np.float32)

    J = types.SimpleNamespace(load_signals_100=load_signals_100)
    judge = FakeJudge()
    logits, stats = fidelity.fold10_500hz(J, judge, db, ids, str(root), chunk=2)
    for kind in fidelity.INPUT_KINDS:
        ours = np.stack([fidelity._prepare_kind(signals[i], 500, kind) for i in ids])
        assert np.allclose(logits[kind], judge.predict_logits(ours), atol=1e-6)
        official = np.stack(
            [fidelity._prepare_kind(signals[i][::5], 100, kind) for i in ids]
        )
        mask = None if kind == "full" else fidelity.layout_mask(1000, 100)
        manual = fidelity.signal_diff_stats(ours, official, mask)
        for key, value in manual.items():
            assert stats[kind][key] == pytest.approx(value, abs=1e-6)


def test_path_floor_reads_the_layout_row(tmp_path):
    path = tmp_path / "layout_loss.csv"
    pd.DataFrame(
        [
            {"judge": "fake", "analysis": "path_floor_fold10", "thresholds": "full",
             "n_records": 10, "n_record_flips": 9, "record_flip_rate": 0.9,
             "flip_rate": 0.3, "mean_abs_dp": 0.2},
            {"judge": "fake", "analysis": "path_floor_fold10", "thresholds": "layout",
             "n_records": 10, "n_record_flips": 1, "record_flip_rate": 0.1,
             "flip_rate": 0.02, "mean_abs_dp": 0.01},
        ]
    ).to_csv(path, index=False)
    floor = fidelity.path_floor(str(path), "fake")
    assert floor["path_floor_record_flip_rate"] == 0.1
    assert floor["path_floor_n_record_flips"] == 1
    assert np.isnan(fidelity.path_floor(str(path), "other")["path_floor_flip_rate"])
    assert np.isnan(fidelity.path_floor(None, "fake")["path_floor_flip_rate"])


# ------------------------------------------------ 15. A3 sensitivity sweeps
def _helper_compute_snr():
    helper = pytest.importorskip("src.utils.helper_code")
    return helper.compute_snr


def test_lead_snr_matches_compute_raw_snr_with_gaps():
    compute_snr = _helper_compute_snr()
    mask = fidelity.layout_mask(5000, 500)
    ref = fidelity.apply_mask(_signal(seed=31), mask)
    rng = np.random.default_rng(32)
    test = ref + 0.05 * rng.normal(size=ref.shape)
    test[1300:1400, fidelity.LEADS.index("aVL")] = np.nan  # gap inside a window
    test[:, fidelity.LEADS.index("V6")] = np.nan  # lead missing entirely
    test[:, fidelity.LEADS.index("V1")] = ref[:, fidelity.LEADS.index("V1")]  # exact
    ours = fidelity.lead_snr(ref, test)
    for j in range(12):
        w = mask[:, j]
        with np.errstate(invalid="ignore"), warnings.catch_warnings():
            warnings.simplefilter("ignore", RuntimeWarning)  # empty lead V6
            expected = compute_snr(ref[w, j], test[w, j])[0]
        if np.isnan(expected):
            assert np.isnan(ours[j])
        else:
            assert ours[j] == pytest.approx(expected, abs=1e-9)
    assert np.isinf(ours[fidelity.LEADS.index("V1")])
    assert np.isnan(ours[fidelity.LEADS.index("V6")])
    # The gap costs the covered fraction (compute_snr's keep_nans penalty).
    avl = fidelity.LEADS.index("aVL")
    w = mask[:, avl] & np.isfinite(test[:, avl])
    full = fidelity.lead_snr(ref[w, avl], test[w, avl])
    assert ours[avl] == pytest.approx(full * 1150 / 1250, abs=1e-12)


def _toy_masked_dig(seed=41):
    mask = fidelity.layout_mask(5000, 500)
    orig = _signal(seed=seed)
    rng = np.random.default_rng(seed + 1)
    masked = fidelity.apply_mask(orig, mask)
    dig = fidelity.apply_mask(orig + 0.04 * rng.normal(size=orig.shape) + 0.02, mask)
    return masked, dig, mask


def test_scale_error_endpoints_are_bit_exact():
    masked, dig, _ = _toy_masked_dig()
    x0 = fidelity.scale_error(masked, dig, 0.0)
    x1 = fidelity.scale_error(masked, dig, 1.0)
    assert np.array_equal(x0, masked, equal_nan=True)
    assert np.array_equal(x1.view(np.uint64), dig.view(np.uint64))
    prep = lambda x: fidelity.input_sha1(fidelity.prepare_for_judge(x, 500))
    assert prep(x0) == prep(masked) and prep(x1) == prep(dig)
    # A gap in dig inside a window: alpha > 0 keeps dig's NaN pattern, alpha 0
    # is the reference itself.
    dig[2000:2100, fidelity.LEADS.index("aVF")] = np.nan
    x_half = fidelity.scale_error(masked, dig, 0.5)
    assert np.array_equal(np.isnan(x_half), np.isnan(dig))
    assert np.array_equal(fidelity.scale_error(masked, dig, 0), masked, equal_nan=True)
    with pytest.raises(ValueError):
        fidelity.scale_error(masked, dig, -1.0)


def test_scaled_error_snr_follows_minus_20_log10_alpha():
    masked, dig, _ = _toy_masked_dig(seed=45)
    snr1 = fidelity.lead_snr(masked, fidelity.scale_error(masked, dig, 1.0))
    assert np.all(np.isfinite(snr1))
    x0 = fidelity.scale_error(masked, dig, 0)
    assert np.all(np.isinf(fidelity.lead_snr(masked, x0)))
    for alpha in (0.5, 0.71, 1.41, 2.0, 5.66):
        snr = fidelity.lead_snr(masked, fidelity.scale_error(masked, dig, alpha))
        assert np.allclose(snr, snr1 - 20.0 * np.log10(alpha), rtol=0, atol=1e-9)


def _band_power_fraction(x, fs, lo, hi):
    spectrum = np.abs(np.fft.rfft(x)) ** 2
    freqs = np.fft.rfftfreq(x.size, 1.0 / fs)
    return spectrum[(freqs >= lo) & (freqs < hi)].sum() / spectrum.sum()


def test_band_noise_hits_the_target_snr_per_lead_and_is_reproducible():
    base = fidelity.apply_mask(_signal(seed=51), fidelity.layout_mask(5000, 500))
    for snr in (50.0, 20.0, 10.0, -3.0):
        noisy = fidelity.add_band_noise(base, snr, seed=0, ecg_id=7, level_index=2)
        assert np.allclose(fidelity.lead_snr(base, noisy), snr, rtol=0, atol=1e-9)
        # NaN outside the windows stays NaN, and nothing else is NaN.
        assert np.array_equal(np.isnan(noisy), np.isnan(base))
    a = fidelity.add_band_noise(base, 20.0, seed=0, ecg_id=7, level_index=2)
    b = fidelity.add_band_noise(base, 20.0, seed=0, ecg_id=7, level_index=2)
    assert np.array_equal(a, b, equal_nan=True)
    for other in (
        dict(seed=0, ecg_id=8, level_index=2),
        dict(seed=0, ecg_id=7, level_index=3),
        dict(seed=1, ecg_id=7, level_index=2),
    ):
        c = fidelity.add_band_noise(base, 20.0, **other)
        assert not np.allclose(a, c, equal_nan=True)
    assert np.array_equal(
        fidelity.add_band_noise(base, np.inf, 0, 7, 0), base, equal_nan=True
    )
    # The exact draw, rebuilt independently: default_rng([seed, ecg_id,
    # level_index]), 4th-order 0.5-40 Hz Butterworth band-pass applied forwards
    # and backwards (sosfiltfilt), scaled per lead over the window.
    from scipy.signal import butter, sosfiltfilt
    white = np.random.default_rng([0, 7, 2]).standard_normal(base.shape)
    sos = butter(4, (0.5, 40.0), btype="bandpass", fs=500, output="sos")
    noise = sosfiltfilt(sos, white, axis=0)
    window = np.isfinite(base)
    p_signal = np.sum(np.where(window, base, 0.0) ** 2, axis=0)
    p_noise = np.sum(np.where(window, noise, 0.0) ** 2, axis=0)
    expected = noise * np.sqrt(p_signal / (p_noise * 10.0 ** (20.0 / 10.0)))
    assert np.allclose((a - base)[window], expected[window], rtol=1e-12, atol=1e-15)
    # Band limited (rhythm lead II, 10 s): > 90 % of the power in 0.3-45 Hz and
    # < 1 % above 60 Hz (white noise: ~17 % and ~76 %).
    ii = fidelity.LEADS.index("II")
    noise = a[:, ii] - base[:, ii]
    assert _band_power_fraction(noise, 500, 0.3, 45.0) > 0.9
    assert _band_power_fraction(noise, 500, 60.0, 251.0) < 0.01


def _pinned_thresholds(judge, inputs, quantile=0.5):
    """Per-class thresholds at the median logit of `inputs` (so flips happen)."""
    return np.quantile(judge.predict_logits(np.stack(inputs)), quantile, axis=0)


def test_error_sweep_on_a_toy_condition(tmp_path):
    def perturb(i, dig):
        rng = np.random.default_rng(60 + i)
        return dig + 0.05 * (i + 1) * rng.normal(size=dig.shape) + 0.03 * i

    gt_dir, pred_dir, orig_dir, records = _make_condition(tmp_path, perturb=perturb)
    signals = fidelity.condition_signals(gt_dir, pred_dir, orig_dir)
    alphas = [0.0, 0.5, 1.0, 2.0, 4.0]
    sweep = fidelity.error_sweep_inputs(signals, alphas, "toy")
    inputs = fidelity.condition_inputs(gt_dir, pred_dir, orig_dir)
    sha = fidelity.input_sha1
    for i in range(len(records)):
        assert sha(sweep["inputs"][0.0][i]) == sha(inputs["masked"][i])
        assert sha(sweep["inputs"][1.0][i]) == sha(inputs["dig"][i])
        assert sha(sweep["ref"][i]) == sha(inputs["masked"][i])
    for a in (0.5, 2.0, 4.0):
        assert np.allclose(
            sweep["snr"][a], sweep["snr"][1.0] - 20 * np.log10(a), rtol=0, atol=1e-9
        )
    assert np.all(np.isinf(sweep["snr"][0.0]))
    assert sweep["keys"][0.0][0] == "toy/00001_hr/masked"
    assert sweep["keys"][1.0][0] == "toy/00001_hr/dig"
    assert sweep["keys"][0.5][0] == "toy/00001_hr/err_a0.5"

    judge = FakeJudge()
    thr = _pinned_thresholds(judge, sweep["ref"])
    cache = fidelity.LogitCache(str(tmp_path / "c.npz"), tag="t")
    rows, recs = fidelity.error_sweep_judge(
        judge, sweep, thr, cache=cache, judge_name="fake", set_name="unit",
        n_boot=50, seed=0,
    )
    assert len(rows) == len(alphas)
    assert list(recs.columns) == list(fidelity.RECORD_ERROR_COLUMNS)
    assert len(recs) == len(records) * len(alphas)
    row0 = rows[0]
    assert row0["alpha"] == 0.0 and row0["n_record_flips"] == 0
    assert row0["kappa"] == 1.0 and row0["mean_abs_dp"] == 0.0
    assert np.isinf(row0["snr_median_db"])
    r0 = recs[recs["alpha"] == 0.0]
    assert not r0["any_flip"].any() and np.all(np.isinf(r0["snr_rec_median"]))
    assert (r0["flipped_classes"] == "").all()
    for row in rows:
        sub = recs[recs["alpha"] == row["alpha"]]
        assert row["n_record_flips"] == int(sub["any_flip"].sum())
        assert row["snr_median_db"] == pytest.approx(
            np.median(sweep["snr"][row["alpha"]])
        )
        assert np.isnan(row["snr_check_max_absdiff"])
    assert sum(r["n_record_flips"] for r in rows) > 0  # pinned thresholds flip
    # Record SNR = median (and min) over the record's leads, for every alpha.
    for a in alphas:
        sub = recs[recs["alpha"] == a].reset_index(drop=True)
        np.testing.assert_array_equal(
            sub["snr_rec_median"], np.median(sweep["snr"][a], axis=1)
        )
        np.testing.assert_array_equal(
            sub["snr_rec_min"], np.min(sweep["snr"][a], axis=1)
        )
        np.testing.assert_array_equal(
            sub["snr_judge_rec_median"], np.median(sweep["snr_judge"][a], axis=1)
        )
    # Judge-band SNR, by hand: the prepared 100 Hz inputs over the 100 Hz windows.
    m100 = fidelity.layout_mask(1000, 100)
    for a in (0.5, 1.0, 4.0):
        for i in range(len(records)):
            ref = fidelity.prepare_for_judge(signals["masked"][i], 500).astype(float)
            x = fidelity.scale_error(signals["masked"][i], signals["dig"][i], a)
            test = fidelity.prepare_for_judge(x, 500).astype(float)
            expected = 10 * np.log10(
                (ref**2 * m100).sum(axis=0) / ((ref - test) ** 2 * m100).sum(axis=0)
            )
            assert np.allclose(sweep["snr_judge"][a][i], expected, rtol=0, atol=1e-9)
    assert np.all(np.isinf(sweep["snr_judge"][0.0]))
    # The toy error is mostly white: the resampler removes most of it.
    assert np.all(sweep["snr_judge"][1.0] > sweep["snr"][1.0] + 3.0)
    assert all(r["n_leads_window_nan"] == 0 for r in rows)
    # alpha 1 = the table's digitised logits; min margin by hand.
    l_ref = judge.predict_logits(np.stack(inputs["masked"]))
    l_dig = judge.predict_logits(np.stack(inputs["dig"]))
    assert np.allclose(cache.get("toy/00001_hr/dig", sha(inputs["dig"][0])), l_dig[0])
    r1 = recs[recs["alpha"] == 1.0].reset_index(drop=True)
    assert np.allclose(r1["max_abs_dlogit"], np.abs(l_dig - l_ref).max(axis=1))
    assert np.allclose(r1["min_margin_ref"], np.abs(l_ref - thr).min(axis=1))
    flips = (l_ref > thr) != (l_dig > thr)
    assert r1["n_class_flips"].tolist() == flips.sum(axis=1).tolist()
    expected = [";".join(c for c, f in zip(CLASSES, row) if f) for row in flips]
    assert r1["flipped_classes"].tolist() == expected


def test_error_sweep_counts_leads_with_nan_inside_their_window(tmp_path):
    avl = fidelity.LEADS.index("aVL")

    def gap(i, dig):
        dig = dig.copy()
        dig[1300:1400, avl] = np.nan  # inside aVL's window [1250, 2500)
        return dig

    for name, perturb, expected in (("plain", None, 0), ("gap", gap, 3)):
        gt_dir, pred_dir, orig_dir, _ = _make_condition(
            tmp_path / name, n_records=3, perturb=perturb
        )
        signals = fidelity.condition_signals(gt_dir, pred_dir, orig_dir)
        sweep = fidelity.error_sweep_inputs(signals, [0.0, 1.0], "toy")
        assert sweep["n_leads_window_nan"] == expected
        rows, _ = fidelity.error_sweep_judge(
            FakeJudge(), sweep, np.zeros(5), n_boot=0
        )
        assert [r["n_leads_window_nan"] for r in rows] == [expected, expected]


def _ev_from_helper(gt_dir, pred_dir, records):
    """An evaluation frame as src/run/evaluate.py writes it (window placement)."""
    compute_snr = _helper_compute_snr()
    rows = []
    for rec in records:
        gt, _ = fidelity.read_record(os.path.join(gt_dir, rec))
        d, _ = fidelity.read_record(os.path.join(pred_dir, rec + fidelity.PRED_SUFFIX))
        for j, lead in enumerate(fidelity.LEADS):
            idx = np.flatnonzero(np.isfinite(gt[:, j]))
            i0, i1 = idx[0], idx[-1] + 1
            rows.append(
                {"record": rec, "lead": lead,
                 "snr_raw": compute_snr(gt[i0:i1, j], d[i0:i1, j])[0],
                 "missing": False}
            )
    return pd.DataFrame(rows)


def test_snr_check_against_an_evaluation_frame(tmp_path):
    gt_dir, pred_dir, orig_dir, records = _make_condition(tmp_path, perturb=_perturb)
    signals = fidelity.condition_signals(gt_dir, pred_dir, orig_dir)
    sweep = fidelity.error_sweep_inputs(signals, [1.0], "toy")
    ev = _ev_from_helper(gt_dir, pred_dir, records)
    check = fidelity.snr_check(records, sweep["snr1"], ev)
    assert check["snr_check_n_leads"] + check["snr_check_n_mismatch"] <= 72
    finite = int(np.isfinite(sweep["snr1"]).sum())
    assert check["snr_check_n_leads"] == finite and check["snr_check_n_mismatch"] == 0
    assert check["snr_check_max_absdiff"] < 1e-9
    row = ev.index[(ev["record"] == records[1]) & (ev["lead"] == "II")][0]
    assert np.isfinite(ev.loc[row, "snr_raw"])  # _perturb only touches lead II
    ev.loc[row, "snr_raw"] += 0.25
    assert fidelity.snr_check(records, sweep["snr1"], ev)[
        "snr_check_max_absdiff"
    ] == pytest.approx(0.25, abs=1e-9)
    assert np.isnan(fidelity.snr_check(records, sweep["snr1"], None)[
        "snr_check_max_absdiff"
    ])


def _toy_ptbxl(tmp_path, n=6, first=1):
    """records500 of ecg_id first..first+n-1 (fold 10, labelled) plus an
    unlabelled fold-10 record (first+n) and a fold-9 record (first+n+1)."""
    root = tmp_path / "ptbxl"
    ids = list(range(first, first + n + 2))
    signals = {i: _signal(seed=70 + i) for i in ids}
    for i in ids:
        _write(root / "records500" / "00000", f"{i:05d}_hr", signals[i])
    rows = []
    for i in ids:
        row = {c: 0 for c in CLASSES}
        if i < first + n:
            row["NORM" if i % 2 else "MI"] = 1
            row["STTC"] = int(i % 3 == 0)
        row["strat_fold"] = 9 if i == first + n + 1 else 10
        row["filename_hr"] = f"records500/00000/{i:05d}_hr"
        rows.append(row)
    db = pd.DataFrame(rows, index=pd.Index(ids, name="ecg_id"))
    return str(root), db, signals


def test_noise_sweep_on_a_toy_ptbxl(tmp_path):
    root, db, signals = _toy_ptbxl(tmp_path)
    ids = fidelity.fold10_ids_500(db, root)
    assert ids == [1, 2, 3, 4, 5, 6]  # unlabelled and fold-9 records left out
    judge = FakeJudge()
    clean = [fidelity._prepare_kind(signals[i], 500, "layout") for i in ids]
    thr = _pinned_thresholds(judge, clean)
    cache = fidelity.LogitCache(str(tmp_path / "c.npz"), tag="t")
    opened = [("fake", judge, cache, thr)]
    snrs = [np.inf, 30.0, 5.0]
    rows, recs = fidelity.noise_sweep(
        opened, db, ids, root, snrs, seed=0, n_boot=30, chunk=4
    )
    assert [r["snr_db"] for r in rows] == snrs
    assert list(recs.columns) == list(fidelity.RECORD_NOISE_COLUMNS)
    assert len(recs) == len(ids) * len(snrs)
    inf_row = rows[0]
    assert inf_row["n_record_flips"] == 0 and inf_row["kappa"] == 1.0
    assert inf_row["auroc_retention_macro"] == pytest.approx(1.0)
    assert np.isfinite(rows[2]["auroc_test_NORM"])
    # Clean logits = the fold10_500hz layout input; noisy = level index 2 noise.
    l_clean = judge.predict_logits(np.stack(clean))
    key = fidelity.noise_key(1, np.inf)
    assert key == "ptbxl500/1/layout"
    assert np.allclose(cache.get(key, fidelity.input_sha1(clean[0])), l_clean[0])
    base = fidelity.apply_mask(signals[3], fidelity.layout_mask(5000, 500))
    noisy = fidelity.prepare_for_judge(
        fidelity.add_band_noise(base, 5.0, 0, 3, 2), 500
    )
    key = fidelity.noise_key(3, 5.0, 0, 2)
    assert key == "ptbxl500/3/noise5.0_s0_l2"
    hit = cache.get(key, fidelity.input_sha1(noisy))
    assert hit is not None
    r = recs[(recs["snr_db"] == 5.0) & (recs["ecg_id"] == 3)].iloc[0]
    ref_row = l_clean[2]
    flips = (ref_row > thr) != (hit > thr)
    assert r["n_class_flips"] == int(flips.sum())
    assert r["min_margin_ref"] == pytest.approx(np.abs(ref_row - thr).min())
    # Judge-band SNR by hand (100 Hz windows); inf at the clean level.  The
    # 0.5-40 Hz noise passes the resampler: close to the 500 Hz target.
    m100 = fidelity.layout_mask(1000, 100)
    ref = clean[2].astype(float)
    err = ((ref - noisy.astype(float)) ** 2 * m100).sum(axis=0)
    expected = np.median(10 * np.log10((ref**2 * m100).sum(axis=0) / err))
    assert r["snr_judge_rec_median"] == pytest.approx(expected, abs=1e-9)
    assert np.all(np.isinf(recs.loc[np.isinf(recs["snr_db"]), "snr_judge_rec_median"]))
    at30 = recs.loc[recs["snr_db"] == 30.0, "snr_judge_rec_median"]
    assert np.all(np.abs(at30 - 30.0) < 0.5)
    assert rows[1]["snr_judge_median_db"] == pytest.approx(30.0, abs=0.5)
    # cache_only: the same keys, filled without any comparison.
    cache2 = fidelity.LogitCache(str(tmp_path / "c2.npz"), tag="t")
    out = fidelity.noise_sweep(
        [("fake", judge, cache2, thr)], db, ids, root, snrs, seed=0, chunk=4,
        cache_only=True,
    )
    assert out[0] == [] and out[1].empty
    assert len(cache2) == len(cache) == len(ids) * len(snrs)
    assert np.array_equal(cache2.get(key, fidelity.input_sha1(noisy)), hit)
    # Deterministic: a second run without the cache gives the same rows.
    rows2, recs2 = fidelity.noise_sweep(
        [("fake", FakeJudge(), None, thr)], db, ids, root, snrs, seed=0, n_boot=30,
        chunk=5,
    )
    pd.testing.assert_frame_equal(recs, recs2)
    assert rows2[2]["kappa"] == rows[2]["kappa"]


def test_snr_bins_counts_and_exact_ci():
    snr = np.array([5.0, 8.0, 8.5, 9.99, 10.0, 25.9, 26.0, 40.0, np.inf, np.nan])
    flip = np.array([1, 0, 1, 1, 0, 0, 0, 1, 1, 1], dtype=bool)
    groups = np.array([1, 1, 2, 3, 3, 4, 5, 5, 6, 6])
    dp = np.arange(10, dtype=float)
    bins = fidelity.snr_bins(snr, flip, groups, dp)
    assert len(bins) == len(fidelity.SNR_BIN_EDGES) - 1
    by_lo = {b["bin_lo"]: b for b in bins}
    assert by_lo[-np.inf]["n_rows"] == 1 and by_lo[-np.inf]["n_flips"] == 1
    b8 = by_lo[8.0]
    assert (b8["n_rows"], b8["n_flips"], b8["n_unique_ecg_ids"]) == (3, 2, 3)
    assert b8["record_flip_rate"] == pytest.approx(2 / 3)
    exact = fidelity.clopper_pearson(2, 3)
    assert (b8["rflip_exact_lo"], b8["rflip_exact_hi"]) == exact
    assert b8["mean_abs_dp"] == pytest.approx(2.0)
    assert by_lo[10.0]["n_rows"] == 1 and by_lo[24.0]["n_rows"] == 1
    top = by_lo[26.0]
    assert (top["n_rows"], top["n_flips"], top["n_unique_ecg_ids"]) == (2, 1, 1)
    assert by_lo[12.0]["n_rows"] == 0 and np.isnan(by_lo[12.0]["record_flip_rate"])
    assert sum(b["n_rows"] for b in bins) == 8  # inf and NaN left out


def _monotone_flips(n_groups=300, per_group=10, b0=7.2, b1=-0.6, seed=3):
    rng = np.random.default_rng(seed)
    groups = np.repeat(np.arange(n_groups), per_group)
    snr = rng.uniform(4.0, 30.0, size=groups.size)
    flip = rng.random(groups.size) < fidelity.sigmoid(b0 + b1 * snr)
    return snr, flip, groups


def test_flip_logistic_recovers_the_snr_threshold():
    snr, flip, groups = _monotone_flips()
    fit = fidelity.flip_logistic(snr, flip, groups, n_boot=60, seed=0)
    truth = {p: (math.log(p / (1 - p)) - 7.2) / -0.6 for p in fidelity.FLIP_PROBS}
    assert truth[0.05] == pytest.approx(16.907, abs=1e-3)
    for p, value in truth.items():
        est = fit[f"snr_p{p:g}"]
        assert abs(est - value) < 1.0
        assert fit[f"snr_p{p:g}_lo"] < est < fit[f"snr_p{p:g}_hi"]
    assert fit["slope"] == pytest.approx(-0.6, abs=0.1) and fit["converged"]
    assert fit["n_boot_valid"] + fit["n_boot_degenerate"] == 60
    assert fit["n_unique_ecg_ids"] == 300 and fit["n_rows"] == 3000
    # Reproducible for the same seed.
    again = fidelity.flip_logistic(snr, flip, groups, n_boot=60, seed=0)
    assert again["snr_p0.05_lo"] == fit["snr_p0.05_lo"]


def test_flip_logistic_degenerate_inputs_and_draws():
    snr = np.linspace(5, 25, 30)
    fit = fidelity.flip_logistic(snr, np.zeros(30, bool), np.arange(30), n_boot=10)
    assert fit["fit"] == "degenerate: no flips" and np.isnan(fit["snr_p0.05"])
    # Three clusters, only one with flips: draws without it are degenerate.
    groups = np.repeat([1, 2, 3], 10)
    snr = np.tile(np.linspace(5, 25, 10), 3)
    flip = (groups == 1) & (snr < 15)
    fit = fidelity.flip_logistic(snr, flip, groups, n_boot=100, seed=1)
    assert fit["n_boot_degenerate"] > 0
    assert fit["n_boot_valid"] + fit["n_boot_degenerate"] == 100


def test_flip_logistic_detects_separation_in_the_data_and_in_the_draws():
    snr = np.linspace(4.0, 30.0, 400)
    groups = np.arange(400) // 2
    for flip in (snr < 12.0, snr > 20.0):  # flips below / above a cut only
        fit = fidelity.flip_logistic(snr, flip, groups, n_boot=20)
        assert fit["fit"] == "degenerate: separated"
        assert np.isnan(fit["slope"]) and np.isnan(fit["snr_p0.05"])
        assert fit["n_boot_valid"] == 0
    one = np.zeros(400, dtype=bool)
    one[0] = True  # a single flip at the lowest SNR
    assert fidelity.flip_logistic(snr, one, groups, n_boot=0)["fit"] == (
        "degenerate: separated"
    )
    x = np.array([1.0, 2.0, 3.0, 3.0, 4.0, 5.0])
    assert fidelity.is_separated(x, [1, 1, 1, 0, 0, 0])  # quasi-complete (tie at 3)
    assert not fidelity.is_separated(x, [1, 0, 1, 0, 0, 0])  # overlap
    # Draws: ecg_id 1 holds every flip; drawn alone its rows are separated,
    # drawn without it nothing flips.  Replay the cluster draws.
    groups = np.repeat([1, 2, 3], 10)
    snr = np.tile(np.linspace(5, 25, 10), 3)
    flip = (groups == 1) & (snr < 15)
    n_boot, seed = 200, 1
    fit = fidelity.flip_logistic(snr, flip, groups, n_boot=n_boot, seed=seed)
    assert fit["fit"] == "ok"
    rng = np.random.default_rng(seed)
    expected = 0
    for _ in range(n_boot):
        drawn = set(rng.integers(0, 3, 3).tolist())
        expected += 0 not in drawn or drawn == {0}
    assert 0 < expected < n_boot
    assert fit["n_boot_degenerate"] == expected
    assert fit["n_boot_valid"] == n_boot - expected


def _newton_logistic(x, y, weight=None, n_iter=60):
    """Reference MLE of y ~ x by Newton's method (no separation)."""
    X = np.c_[np.ones_like(x), x]
    w = np.ones_like(x) if weight is None else np.asarray(weight, dtype=float)
    beta = np.zeros(2)
    for _ in range(n_iter):
        p = fidelity.sigmoid(X @ beta)
        grad = X.T @ (w * (y - p))
        hess = (X * (w * p * (1 - p))[:, None]).T @ X
        beta = beta + np.linalg.solve(hess, grad)
    return beta


def test_fit_flip_logistic_is_the_mle():
    snr, flip, _ = _monotone_flips()
    b0, b1, converged = fidelity._fit_flip_logistic(snr, flip)
    ref = _newton_logistic(snr, flip.astype(float))
    assert converged
    assert b0 == pytest.approx(ref[0], rel=1e-7)
    assert b1 == pytest.approx(ref[1], rel=1e-7)
    w = np.random.default_rng(1).integers(0, 3, snr.size).astype(float)
    b0, b1, _ = fidelity._fit_flip_logistic(snr, flip, weight=w)
    ref = _newton_logistic(snr, flip.astype(float), w)
    assert (b0, b1) == pytest.approx(tuple(ref), rel=1e-7)


def test_flip_logistic_ecg_id_weighting_equals_replicating_small_groups():
    rng = np.random.default_rng(5)
    # 40 ecg_ids with one row, 40 with three rows.
    groups = np.concatenate([np.arange(40), np.repeat(np.arange(40, 80), 3)])
    snr = rng.uniform(5.0, 30.0, groups.size)
    flip = rng.random(groups.size) < fidelity.sigmoid(6.0 - 0.4 * snr)
    w = fidelity.ecg_id_weights(groups)
    assert w.mean() == pytest.approx(1.0) and w[0] == pytest.approx(3 * w[40])
    fit = fidelity.flip_logistic(snr, flip, groups, n_boot=0, weighting="ecg_id")
    assert (fit["rows_per_id_min"], fit["rows_per_id_max"]) == (1, 3)
    rep = np.concatenate([np.repeat(np.arange(40), 3), np.arange(40, groups.size)])
    same = fidelity.flip_logistic(snr[rep], flip[rep], groups[rep], n_boot=0)
    assert fit["slope"] == pytest.approx(same["slope"], rel=1e-6)
    assert fit["intercept"] == pytest.approx(same["intercept"], rel=1e-6)
    rows = fidelity.flip_logistic(snr, flip, groups, n_boot=0)
    assert rows["weighting"] == "rows" and fit["weighting"] == "ecg_id"
    assert abs(rows["slope"] - fit["slope"]) > 1e-3
    boot = fidelity.flip_logistic(snr, flip, groups, n_boot=30, weighting="ecg_id")
    assert boot["snr_p0.05_lo"] < boot["snr_p0.05"] < boot["snr_p0.05_hi"]
    with pytest.raises(ValueError):
        fidelity.flip_logistic(snr, flip, groups, weighting="records")


def test_snr_flip_rows_notes_clustering_and_weighting():
    df = pd.DataFrame(
        {
            "ecg_id": [1, 2, 3, 3, 4, 4, 5],
            "snr": [9.0, 11.0, 30.0, 31.0, 9.5, 28.0, 12.5],
            "any_flip": [True, False, False, True, True, False, False],
            "mean_abs_dp": 0.1,
        }
    )
    rows = fidelity.snr_flip_rows(
        df, "noise", "fake", "snr", n_boot=10, snr_axis="judge_100hz",
        extra_note="extra",
    )
    bins = {r["bin_lo"]: r for r in rows if r["kind"] == "bin"}
    assert bins[8.0]["n_rows"] == 2 and bins[8.0]["n_unique_ecg_ids"] == 2
    assert bins[8.0]["note"] == ""  # one row per ecg_id: the exact CI holds
    assert "clustered" in bins[26.0]["note"]  # ecg_id 3 twice
    logistic = [r for r in rows if r["kind"] == "logistic"]
    assert [(r["weighting"], r["p_flip"]) for r in logistic] == [
        (w, p) for w in fidelity.WEIGHTINGS for p in fidelity.FLIP_PROBS
    ]
    assert all(r["snr_axis"] == "judge_100hz" for r in rows)
    assert all(r["note"].endswith("extra") for r in logistic)
    assert all("only 3 flips" in r["note"] for r in logistic)
    assert "1-2 rows per ecg_id" in logistic[0]["note"]
    assert "every ecg_id weighs 1" in logistic[-1]["note"]


def test_sweep_constants_are_the_spec_values():
    assert fidelity.DEFAULT_ALPHAS == (0.0, 0.5, 0.71, 1.0, 1.41, 2.0, 2.83, 4.0, 5.66)
    assert fidelity.DEFAULT_SNRS == (
        math.inf, 50.0, 40.0, 35.0, 30.0, 25.0, 22.0, 20.0, 18.0, 15.0, 12.0, 10.0
    )
    assert fidelity.NOISE_BAND_HZ == (0.5, 40.0) and fidelity.NOISE_ORDER == 4
    assert fidelity.SNR_BIN_EDGES == (
        -math.inf, 8.0, 10.0, 12.0, 14.0, 16.0, 18.0, 20.0, 22.0, 24.0, 26.0, math.inf
    )
    assert fidelity.FLIP_PROBS == (0.05, 0.10)
    assert fidelity.MARGIN_CUTS == (0.25, 0.5, 1.0)
    parser = fidelity.get_parser()
    assert parser.parse_args(["sweep-error"]).alphas == list(fidelity.DEFAULT_ALPHAS)
    assert parser.parse_args(["sweep-noise"]).snrs == list(fidelity.DEFAULT_SNRS)


def test_cache_keys_are_exact():
    assert fidelity.error_key("c", "r", 0.5) == "c/r/err_a0.5"
    assert fidelity.error_key("c", "r", 0.5000001) != fidelity.error_key("c", "r", 0.5)
    assert fidelity.error_key("c", "r", 2) == "c/r/err_a2.0"
    assert fidelity.error_key("c", "r", 0) == "c/r/masked"
    assert fidelity.error_key("c", "r", 1.0) == "c/r/dig"
    assert fidelity.noise_key(3, 30.0, 0, 4) == "ptbxl500/3/noise30.0_s0_l4"
    keys = {
        fidelity.noise_key(3, 30.0, 0, 4), fidelity.noise_key(3, 30.0, 1, 4),
        fidelity.noise_key(3, 30.0, 0, 5), fidelity.noise_key(3, 30.0000001, 0, 4),
    }
    assert len(keys) == 4
    assert fidelity.noise_key(3, math.inf) == "ptbxl500/3/layout"
    assert fidelity.noise_key(3, math.inf, 1, 0) == "ptbxl500/3/layout"
    with pytest.raises(ValueError):
        fidelity.noise_key(3, 30.0)


def test_partial_runs_never_write_the_v0_name(tmp_path):
    default = fidelity.SWEEP_NOISE_CSV
    part = fidelity.partial_out(default, default, 540, 1080)
    assert part == fidelity._path("data/fidelity/sweep_noise_v0_part540-1080.csv")
    assert fidelity.default_records_out(part).endswith(
        "sweep_noise_v0_part540-1080_records.csv"
    )
    custom = str(tmp_path / "x.csv")
    assert fidelity.partial_out(custom, default, 0, 10) == custom


def test_min_margin_and_margin_summary_hand_example():
    logits = np.array([[1.0, -0.2, 3.0], [0.1, 0.9, -2.0], [np.nan, 5.0, 1.0]])
    thr = np.array([0.5, 0.0, np.nan])
    assert fidelity.min_margin(logits, thr).tolist() == pytest.approx([0.2, 0.4, 5.0])
    summary = fidelity.margin_summary([0.2, 0.4, 0.8, 1.5, np.nan])
    assert summary["n"] == 4
    assert summary["median_min_margin"] == pytest.approx(0.6)
    assert summary["frac_lt_0.25"] == 0.25
    assert summary["frac_lt_0.5"] == 0.5
    assert summary["frac_lt_1"] == 0.75
    at_cut = fidelity.margin_summary([0.2, 0.5, 1.0, 1.5])  # values at the cuts
    assert at_cut["frac_lt_0.5"] == 0.25 and at_cut["frac_lt_1"] == 0.5
    rows = fidelity.margin_rows("fake", [0.2, 0.4], [0.8, 1.5, 2.0], "local_toy")
    assert [r["set"] for r in rows] == ["local_toy", "fold10_500hz"]
    assert 0.0 < rows[0]["mwu_p_vs_fold10"] <= 1.0
    assert np.isnan(rows[1]["mwu_p_vs_fold10"])
    assert rows[1]["frac_lt_1"] == pytest.approx(1 / 3)


# ------------------------------------------------ 16. A3 subcommands end to end
def _cli_world(tmp_path, monkeypatch):
    """A toy condition (ecg_id 1-6), a toy PTB-XL, thresholds and a fake judge.

    PTB-XL: ecg_id 5-10 labelled fold 10 (5 and 6 are also local records),
    11 unlabelled fold 10, 12 fold 9, 1-4 fold 1 with 3 unlabelled.  The
    full thresholds are the layout thresholds + 3, so using the wrong kind
    changes every decision count.
    """
    def perturb(i, dig):
        rng = np.random.default_rng(80 + i)
        return dig + 0.08 * rng.normal(size=dig.shape)

    gt_dir, pred_dir, orig_dir, records = _make_condition(tmp_path, perturb=perturb)
    root, db, ptb_signals = _toy_ptbxl(tmp_path, n=6, first=5)
    local_rows = []
    for i in (1, 2, 3, 4):
        row = {c: 0 for c in CLASSES}
        row["NORM"] = int(i != 3)
        row["strat_fold"] = 1
        row["filename_hr"] = f"records500/00000/{i:05d}_hr"
        local_rows.append(row)
    db = pd.concat(
        [pd.DataFrame(local_rows, index=pd.Index([1, 2, 3, 4], name="ecg_id")), db]
    )
    monkeypatch.setattr(fidelity, "CONDITIONS", {"toy": (gt_dir, pred_dir, "unit")})
    ev_dir = tmp_path / "ev"
    ev_dir.mkdir()
    monkeypatch.setattr(fidelity, "EV_CSV", str(ev_dir / "ev_{cond}.csv"))
    ev = _ev_from_helper(gt_dir, pred_dir, records)
    ev.to_csv(ev_dir / "ev_toy.csv", index=False)

    calls = {"n": 0}

    class Loaded(FakeJudge):
        def predict_logits(self, X, batch_size=512):
            calls["n"] += len(X)
            return super().predict_logits(X, batch_size)

    def load(name, weights_dir=None, device="cpu"):
        judge = Loaded()
        judge.weights_sha256 = "abc"
        return judge

    J = types.SimpleNamespace(
        CLASSES=CLASSES, Judge=types.SimpleNamespace(load=load),
        load_ptbxl_db=lambda root=None: db,
    )
    monkeypatch.setattr(fidelity, "_judge_module", lambda: J)
    signals = fidelity.condition_signals(gt_dir, pred_dir, orig_dir)
    masked = [fidelity.prepare_for_judge(x, 500) for x in signals["masked"]]
    thr = _pinned_thresholds(FakeJudge(), masked)
    thr_csv = tmp_path / "thresholds.csv"
    pd.DataFrame(
        [
            {"judge": "fake", "input": kind, "class": c,
             "threshold_logit": t + (3.0 if kind == "full" else 0.0), "device": "cpu"}
            for kind in ("full", "layout")
            for c, t in zip(CLASSES, thr)
        ]
    ).to_csv(thr_csv, index=False)
    common = [
        "--judges", "fake", "--ptbxl_root", root, "--weights_dir", str(tmp_path),
        "--cache_dir", str(tmp_path / "cache"), "--thresholds", str(thr_csv),
    ]
    return types.SimpleNamespace(
        common=common, orig_dir=orig_dir, records=records, calls=calls, out=tmp_path,
        thr=thr, signals=signals, ptb_signals=ptb_signals,
    )


def _cli(argv):
    return fidelity.run(fidelity.get_parser().parse_args(argv))


def _record_flip_counts(signals, alphas, thr):
    """Records flipping at `thr` per alpha, straight from FakeJudge."""
    judge = FakeJudge()
    prep = lambda xs: np.stack([fidelity.prepare_for_judge(x, 500) for x in xs])
    l_ref = judge.predict_logits(prep(signals["masked"]))
    counts = []
    for a in alphas:
        scaled = [
            fidelity.scale_error(m, d, a)
            for m, d in zip(signals["masked"], signals["dig"])
        ]
        l_test = judge.predict_logits(prep(scaled))
        counts.append(int(((l_ref > thr) != (l_test > thr)).any(axis=1).sum()))
    return counts


def test_sweep_subcommands_end_to_end(tmp_path, monkeypatch):
    w = _cli_world(tmp_path, monkeypatch)
    alphas = [0.0, 0.5, 1.0, 2.0, 4.0]
    err_out = str(w.out / "sweep_error.csv")
    argv = [
        "sweep-error", *w.common, "--original_dir", w.orig_dir, "--conditions", "toy",
        "--alphas", *[str(a) for a in alphas], "--n_boot", "20", "--out", err_out,
    ]
    agg = _cli(argv)
    n_rec = len(w.records)
    assert len(agg) == 5 and agg["alpha"].tolist() == alphas
    assert (agg["snr_check_max_absdiff"] < 1e-9).all()
    assert (agg["snr_check_n_leads"] == 12 * n_rec).all()
    recs = pd.read_csv(fidelity.default_records_out(err_out))
    assert len(recs) == 5 * n_rec
    first_calls = w.calls["n"]
    assert first_calls == 5 * n_rec  # masked, 0.5, dig, 2, 4 (alpha 0 = masked)
    agg2 = _cli(argv)  # second run: every input comes from the cache
    assert w.calls["n"] == first_calls
    pd.testing.assert_frame_equal(agg, agg2)
    assert agg.loc[0, "n_record_flips"] == 0 and agg.loc[0, "kappa"] == 1.0
    # Decisions use the layout thresholds (the full ones give other counts).
    layout = _record_flip_counts(w.signals, alphas, w.thr)
    assert agg["n_record_flips"].tolist() == layout
    assert sum(layout) > 0
    assert layout != _record_flip_counts(w.signals, alphas, w.thr + 3.0)

    noise_out = str(w.out / "sweep_noise.csv")
    noise_argv = [
        "sweep-noise", *w.common, "--snrs", "inf", "30", "8", "--n_boot", "20",
        "--chunk", "4", "--out", noise_out,
    ]
    # A cache-only part (records 2-4 = ecg_id 7-9) writes no CSV ...
    calls = w.calls["n"]
    part = _cli([*noise_argv, "--start", "2", "--max_records", "3", "--cache_only"])
    assert part.empty and w.calls["n"] == calls + 3 * 3
    cached = fidelity.LogitCache(
        fidelity.cache_file(str(w.out / "cache"), "fake"), tag="fake:abc:cpu"
    )
    noise_ids = {k.split("/")[1] for k in cached._data if k.startswith("ptbxl500/")}
    assert noise_ids == {"7", "8", "9"}
    assert not os.path.exists(noise_out)
    assert not os.path.exists(fidelity.default_records_out(noise_out))
    # ... and the full run predicts only the other three records.
    noise = _cli(noise_argv)
    assert w.calls["n"] == calls + 6 * 3
    assert noise["snr_db"].tolist() == [np.inf, 30.0, 8.0]
    assert noise["n_records"].tolist() == [6, 6, 6]
    noise_recs = pd.read_csv(fidelity.default_records_out(noise_out))
    assert len(noise_recs) == 18 and np.isinf(noise_recs["snr_db"]).sum() == 6
    assert sorted(noise_recs["ecg_id"].unique()) == [5, 6, 7, 8, 9, 10]

    flip_out = str(w.out / "snr_flip.csv")
    table = _cli([
        "snr-flip", "--error_records", fidelity.default_records_out(err_out),
        "--noise_records", fidelity.default_records_out(noise_out), "--n_boot", "20",
        "--out", flip_out,
    ])
    assert list(table.columns) == list(fidelity.SNR_FLIP_COLUMNS)
    edges = fidelity.SNR_BIN_EDGES
    err_rows = recs[recs["alpha"] > 0]
    noisy_rows = noise_recs[np.isfinite(noise_recs["snr_db"])]
    columns = {
        ("error_scaling", "window_500hz"): err_rows["snr_rec_median"],
        ("error_scaling", "judge_100hz"): err_rows["snr_judge_rec_median"],
        ("noise", "window_500hz"): noisy_rows["snr_db"],
        ("noise", "judge_100hz"): noisy_rows["snr_judge_rec_median"],
    }
    for (source, axis), values in columns.items():
        values = values.to_numpy(dtype=float)
        sub = table[(table["source"] == source) & (table["snr_axis"] == axis)]
        bins = sub[sub["kind"] == "bin"]
        expected = [
            int(((values >= lo) & (values < hi)).sum())
            for lo, hi in zip(edges[:-1], edges[1:])
        ]
        assert bins["n_rows"].tolist() == expected  # left-closed bins, no reference
        logistic = sub[sub["kind"] == "logistic"]
        assert list(zip(logistic["weighting"], logistic["p_flip"])) == [
            (wt, p) for wt in fidelity.WEIGHTINGS for p in fidelity.FLIP_PROBS
        ]
        assert (logistic["n_rows"] == values.size).all()
        assert np.allclose(logistic["snr_obs_min"], values.min())
        assert np.allclose(logistic["snr_obs_max"], values.max())
        s500 = columns[(source, "window_500hz")].to_numpy(dtype=float)
        s100 = columns[(source, "judge_100hz")].to_numpy(dtype=float)
        gap = np.median(s100 - s500)
        assert logistic["note"].str.contains(
            f"judge-band minus 500 Hz SNR: median {gap:+.2f} dB over "
            f"{values.size} rows",
            regex=False,
        ).all()
        comparable = "not comparable" if axis == "window_500hz" else "comparable across"
        assert logistic["note"].str.contains(comparable).all()
    # The toy digitisation error is mostly white: the judge sees much less of it.
    assert np.median(err_rows["snr_judge_rec_median"] - err_rows["snr_rec_median"]) > 3
    assert table["note"].str.contains("clustered").any()

    margins_out = str(w.out / "margins.csv")
    margins = _cli([
        "margins", *w.common, "--original_dir", w.orig_dir, "--condition", "toy",
        "--out", margins_out,
    ])
    assert margins["set"].tolist() == ["local_toy", "fold10_500hz"]
    # Local: the labelled ecg_ids 1, 2, 4, 5, 6; reference: fold 10 without
    # the local ids 5 and 6.
    assert margins["n"].tolist() == [5, 4]
    assert margins["n_excluded"].tolist() == [1, 2]
    assert "[3]" in margins.loc[0, "note"] and "[5, 6]" in margins.loc[1, "note"]
    judge = FakeJudge()
    labelled = [0, 1, 3, 4, 5]  # positions of ecg_id 1, 2, 4, 5, 6
    local = judge.predict_logits(np.stack([
        fidelity.prepare_for_judge(w.signals["masked"][i], 500) for i in labelled
    ]))
    assert margins.loc[0, "median_min_margin"] == pytest.approx(
        np.median(np.abs(local - w.thr).min(axis=1))
    )
    reference = judge.predict_logits(np.stack([
        fidelity._prepare_kind(w.ptb_signals[i], 500, "layout") for i in (7, 8, 9, 10)
    ]))
    assert margins.loc[1, "median_min_margin"] == pytest.approx(
        np.median(np.abs(reference - w.thr).min(axis=1))
    )


def _ev_only_in_ev_csvs(tmp_path, monkeypatch):
    """Register the toy CSV in EV_CSVS only; EV_CSV points at a folder without
    it, so ignoring EV_CSVS leaves the SNR columns empty."""
    ev_path = tmp_path / "ev_b" / "ev_toy_b.csv"
    ev_path.parent.mkdir()
    os.rename(tmp_path / "ev" / "ev_toy.csv", ev_path)
    empty = tmp_path / "ev_none"
    empty.mkdir()
    monkeypatch.setattr(fidelity, "EV_CSV", str(empty / "ev_{cond}.csv"))
    monkeypatch.setattr(fidelity, "EV_CSVS", {"toy": str(ev_path)})
    assert not os.path.exists(fidelity.EV_CSV.format(cond="toy"))
    return ev_path


def test_table_reads_the_per_condition_evaluation_csv(tmp_path, monkeypatch, capsys):
    w = _cli_world(tmp_path, monkeypatch)
    ev_path = _ev_only_in_ev_csvs(tmp_path, monkeypatch)
    out = str(w.out / "table.csv")
    table = _cli([
        "table", *w.common, "--original_dir", w.orig_dir, "--conditions", "toy",
        "--n_boot", "20", "--layout_loss", str(tmp_path / "no_layout_loss.csv"),
        "--out", out,
    ])
    assert "no evaluation CSV" not in capsys.readouterr().err
    ev = pd.read_csv(ev_path)
    assert len(table) == 1 and table.loc[0, "condition"] == "toy"
    assert np.isfinite(table.loc[0, "snr_median_db"])
    assert table.loc[0, "snr_median_db"] == pytest.approx(
        fidelity.condition_snr_median(ev)
    )
    recs = pd.read_csv(fidelity.default_records_out(out))
    assert len(recs) == len(w.records) * len(CLASSES)
    assert np.isfinite(recs["snr_rec_median"]).all()
    assert np.isfinite(recs["snr_rec_min"]).all()
    per_record = fidelity.record_snr(ev)
    first = recs[recs["record"] == w.records[0]].iloc[0]
    assert first["snr_rec_median"] == pytest.approx(
        per_record.loc[w.records[0], "snr_rec_median"]
    )


def test_sweep_error_reads_the_per_condition_evaluation_csv(
    tmp_path, monkeypatch, capsys
):
    w = _cli_world(tmp_path, monkeypatch)
    _ev_only_in_ev_csvs(tmp_path, monkeypatch)
    agg = _cli([
        "sweep-error", *w.common, "--original_dir", w.orig_dir, "--conditions", "toy",
        "--alphas", "0.5", "1", "--n_boot", "20",
        "--out", str(w.out / "sweep_error.csv"),
    ])
    assert "no evaluation CSV" not in capsys.readouterr().err
    assert (agg["snr_check_n_leads"] == 12 * len(w.records)).all()
    assert (agg["snr_check_max_absdiff"] < 1e-9).all()


# ------------------------------------- 17. thresholds on the records500 path
def _thresholds_world(tmp_path, monkeypatch):
    """Toy PTB-XL: ecg_id 40-45 labelled fold 9 (records500 files), 46 unlabelled,
    47 fold 9 unlabelled; a fake judge module with median 'Youden' thresholds."""
    root, db, signals = _toy_ptbxl(tmp_path, n=6, first=40)
    db.loc[40:45, "strat_fold"] = 9
    calls = {"n": 0, "load500": []}

    class Loaded(FakeJudge):
        def predict_logits(self, X, batch_size=512):
            calls["n"] += len(X)
            return super().predict_logits(X, batch_size)

    def load(name, weights_dir=None, device="cpu"):
        judge = Loaded()
        judge.weights_sha256 = "abc"
        return judge

    def macro_auroc(y, s):
        per = np.array([fidelity.auroc(y[:, k], s[:, k]) for k in range(y.shape[1])])
        return per, float(np.nanmean(per))

    J = types.SimpleNamespace(
        CLASSES=CLASSES, HELD_OUT_ECG_IDS=tuple(range(1, 33)),
        Judge=types.SimpleNamespace(load=load), load_ptbxl_db=lambda root=None: db,
        youden_thresholds=lambda y, s: np.median(s, axis=0), macro_auroc=macro_auroc,
        load_signals_100=lambda ecg_ids, root=None: np.stack(
            [signals[int(e)][::5] for e in ecg_ids]
        ).astype(np.float32),
    )
    monkeypatch.setattr(fidelity, "_judge_module", lambda: J)
    argv = [
        "thresholds", "--judges", "fake", "--ptbxl_root", root,
        "--weights_dir", str(tmp_path), "--cache_dir", str(tmp_path / "cache"),
    ]
    return J, root, db, signals, calls, argv


def test_thresholds_default_to_the_records500_condition_path(tmp_path, monkeypatch):
    J, root, db, signals, calls, argv = _thresholds_world(tmp_path, monkeypatch)
    assert fidelity.get_parser().parse_args(["thresholds"]).source == "records500"
    ids = [40, 41, 42, 43, 44, 45]
    out = tmp_path / "thr.csv"
    df = _cli([*argv, "--out", str(out)])
    written = pd.read_csv(out)
    assert (written["source"] == "records500").all() and len(written) == 10
    for kind in fidelity.INPUT_KINDS:
        prepared = np.stack(
            [fidelity._prepare_kind(signals[i], 500, kind) for i in ids]
        )
        expected = np.median(FakeJudge().predict_logits(prepared), axis=0)
        sub = df[df["input"] == kind]
        assert sub["class"].tolist() == list(CLASSES)
        np.testing.assert_allclose(sub["threshold_logit"], expected, atol=1e-9)
    # load_thresholds keeps its API with the extra column.
    thr = fidelity.load_thresholds(str(out), "fake")
    layout_rows = df[df["input"] == "layout"]
    np.testing.assert_allclose(thr["layout"], layout_rows["threshold_logit"])
    # The logits sit under the ptbxl500 keys that fold10_500hz shares.
    cache = fidelity.LogitCache(
        fidelity.cache_file(str(tmp_path / "cache"), "fake"), tag="fake:abc:cpu"
    )
    assert set(cache._data) == {
        f"ptbxl500/{i}/{kind}" for i in ids for kind in fidelity.INPUT_KINDS
    }
    assert calls["n"] == 12

    # With the judge's cached loader the arrays come from it (and the cache still hits).
    def load_inputs_500(ecg_ids, kind, root=None):
        calls["load500"].append((list(ecg_ids), kind, root))
        return np.stack(fidelity.ptbxl500_inputs(db, ecg_ids, root)[kind])

    J.load_inputs_500 = load_inputs_500
    df2 = _cli([*argv, "--out", str(out)])
    assert calls["load500"] == [(ids, "full", root), (ids, "layout", root)]
    assert calls["n"] == 12  # same inputs, same SHA-1: nothing predicted again
    pd.testing.assert_frame_equal(df2, df)

    # records100 on request: the ptbxl keys, and the column says so.
    df100 = _cli([*argv, "--source", "records100", "--out", str(out)])
    assert (df100["source"] == "records100").all()
    assert (pd.read_csv(out)["source"] == "records100").all()
    cache = fidelity.LogitCache(
        fidelity.cache_file(str(tmp_path / "cache"), "fake"), tag="fake:abc:cpu"
    )
    assert {k for k in cache._data if k.startswith("ptbxl/")} == {
        f"ptbxl/{i}/{kind}" for i in ids for kind in fidelity.INPUT_KINDS
    }


def test_thresholds_records500_abort_on_missing_files(tmp_path, monkeypatch):
    J, root, db, signals, calls, argv = _thresholds_world(tmp_path, monkeypatch)
    os.remove(os.path.join(root, "records500", "00000", "00041_hr.hea"))
    out = tmp_path / "thr.csv"
    with pytest.raises(SystemExit, match=r"records500 missing for 1/6 fold-9"):
        _cli([*argv, "--out", str(out)])
    assert not out.exists() and calls["n"] == 0
    with pytest.raises(ValueError):
        fidelity.ptbxl500_logits(J, FakeJudge(), db, [40], "window", root)


# ------------------------------------------ 18. v0 result files are guarded
V0_OUTPUTS = (
    "data/fidelity/table_v0.csv", "data/fidelity/table_v0_records.csv",
    "data/fidelity/sweep_error_v0.csv", "data/fidelity/sweep_error_v0_records.csv",
    "data/fidelity/margins_v0.csv",
)


def test_guard_v0_outputs_blocks_non_v0_conditions_on_v0_files(tmp_path):
    guard = fidelity.guard_v0_outputs
    assert fidelity.v0_outputs() == V0_OUTPUTS
    for p in V0_OUTPUTS:
        # v0 reruns keep working, relative or absolute.
        guard(V0_CONDITIONS, [p])
        guard(["gen_clean"], [fidelity._path(p)])
        for spelling in (p, fidelity._path(p)):
            with pytest.raises(ValueError, match="b_clean") as err:
                guard(["b_clean"], [spelling])
            assert os.path.basename(p) in str(err.value)
            assert "--out" in str(err.value)
    # One non-v0 condition among v0 ones is enough; only it is named.
    with pytest.raises(ValueError, match="table_v0.csv") as err:
        guard(["gen_clean", "b_clean"], [fidelity.TABLE_CSV])
    assert "'b_clean'" in str(err.value) and "'gen_clean'" not in str(err.value)
    # Non-v0 conditions writing other files are never blocked.
    guard(["b_clean"], [str(tmp_path / "table_v1.csv")])
    guard(B_CONDITIONS, [str(tmp_path / "t.csv"), str(tmp_path / "t_records.csv")])
    guard(["b_clean"], [])
    # Another spelling of a v0 file: '..' and a symlinked folder.
    dotted = os.path.join(
        fidelity.REPO_ROOT, "data", "fidelity", "..", "fidelity", "table_v0.csv"
    )
    with pytest.raises(ValueError, match="b_clean"):
        guard(["b_clean"], [dotted])
    link = tmp_path / "fid_link"
    os.symlink(fidelity._path("data/fidelity"), link)
    with pytest.raises(ValueError, match="margins_v0.csv"):
        guard(["b_rot"], [str(tmp_path / "t.csv"), str(link / "margins_v0.csv")])


def test_guard_v0_outputs_suggests_a_name_for_the_file_hit():
    guard = fidelity.guard_v0_outputs
    expected = {
        "data/fidelity/table_v0.csv": "table_v1.csv",
        "data/fidelity/table_v0_records.csv": "table_v1.csv",
        "data/fidelity/sweep_error_v0.csv": "sweep_error_v1.csv",
        "data/fidelity/sweep_error_v0_records.csv": "sweep_error_v1.csv",
        "data/fidelity/margins_v0.csv": "margins_v1.csv",
    }
    for p, name in expected.items():
        with pytest.raises(ValueError) as err:
            guard(["b_clean"], [p])
        message = str(err.value)
        assert message.endswith(f"e.g. data/fidelity/{name}.")
        # margins has no --records_out.
        assert ("--records_out" in message) == (not name.startswith("margins"))


def test_guard_v0_outputs_catches_the_same_file_under_another_name(
    tmp_path, monkeypatch
):
    v0 = tmp_path / "table_v0.csv"
    v0.write_text("logged v0 result\n")
    monkeypatch.setattr(fidelity, "TABLE_CSV", str(v0))
    hard = tmp_path / "hard_link.csv"
    os.link(v0, hard)
    with pytest.raises(ValueError, match="hard_link.csv"):
        fidelity.guard_v0_outputs(["b_clean"], [str(hard)])
    fidelity.guard_v0_outputs(["gen_clean"], [str(hard)])
    upper = tmp_path / "TABLE_V0.CSV"
    if not upper.exists():
        pytest.skip("case-sensitive file system")
    with pytest.raises(ValueError, match="TABLE_V0.CSV"):
        fidelity.guard_v0_outputs(["b_aug"], [str(upper)])
    assert v0.read_text() == "logged v0 result\n"


def test_v0_outputs_follow_the_module_globals(tmp_path, monkeypatch):
    moved = str(tmp_path / "moved_table.csv")
    monkeypatch.setattr(fidelity, "TABLE_CSV", moved)
    assert fidelity.v0_outputs()[:2] == (moved, str(tmp_path / "moved_table_records.csv"))
    with pytest.raises(ValueError):
        fidelity.guard_v0_outputs(["b_clean"], [moved])
    fidelity.guard_v0_outputs(["b_clean"], ["data/fidelity/table_v0.csv"])


def _no_judge_no_write(monkeypatch):
    """_judge_module raises; _write_csv records its paths instead of writing."""
    def judge_module():
        raise AssertionError("judge loaded before the guard")

    writes = []
    monkeypatch.setattr(fidelity, "_judge_module", judge_module)
    monkeypatch.setattr(fidelity, "_write_csv", lambda df, path: writes.append(path))
    return writes


def test_guard_runs_before_the_judge_is_loaded(tmp_path, monkeypatch):
    writes = _no_judge_no_write(monkeypatch)
    out = tmp_path / "t.csv"
    blocked = (
        (["table", "--conditions", "fidelity_b"], "table_v0.csv"),
        (["table", "--conditions", "b_clean", "--out", str(out),
          "--records_out", "data/fidelity/table_v0_records.csv"],
         "table_v0_records.csv"),
        (["sweep-error", "--conditions", "b_scan"], "sweep_error_v0.csv"),
        (["sweep-error", "--conditions", "b_scan", "--out", str(tmp_path / "s.csv"),
          "--records_out", "data/fidelity/sweep_error_v0_records.csv"],
         "sweep_error_v0_records.csv"),
        (["margins", "--condition", "b_rot"], "margins_v0.csv"),
    )
    for argv, hit in blocked:
        with pytest.raises(ValueError, match="--out") as err:
            _cli(argv)
        assert hit in str(err.value)
    # An unknown margins condition is also rejected before the judge loads.
    with pytest.raises(ValueError, match="Unknown condition 'nope'"):
        _cli(["margins", "--condition", "nope"])
    # The v0 defaults are not blocked: the run gets as far as the judge.
    for argv in (["table"], ["sweep-error"], ["margins"]):
        with pytest.raises(AssertionError, match="judge loaded before the guard"):
            _cli(argv)
    assert writes == []
    assert not out.exists()
    assert not os.path.exists(fidelity.default_records_out(str(out)))
    assert os.listdir(tmp_path) == []


def test_records_out_is_the_path_the_guard_checked(tmp_path, monkeypatch):
    w = _cli_world(tmp_path, monkeypatch)
    extra = {
        "table": ["--layout_loss", str(tmp_path / "no_layout_loss.csv")],
        "sweep-error": ["--alphas", "0.5", "1"],
    }
    n_rows = {"table": len(w.records) * len(CLASSES), "sweep-error": 2 * len(w.records)}
    for command in ("table", "sweep-error"):
        out = str(w.out / f"{command}.csv")
        records_out = str(w.out / "elsewhere" / f"{command}_recs.csv")
        _cli([
            command, *w.common, "--original_dir", w.orig_dir, "--conditions", "toy",
            "--n_boot", "20", *extra[command], "--out", out,
            "--records_out", records_out,
        ])
        assert os.path.exists(out)
        assert len(pd.read_csv(records_out)) == n_rows[command]
        assert not os.path.exists(fidelity.default_records_out(out))


def test_out_help_names_the_guard():
    subparsers = fidelity.get_parser()._subparsers._group_actions[0].choices
    for command in ("table", "sweep-error", "margins"):
        help_text = " ".join(subparsers[command].format_help().split())
        assert "Non-v0 conditions may not write the *_v0 defaults." in help_text
    layout_help = " ".join(subparsers["layout-loss"].format_help().split())
    assert "Non-v0" not in layout_help


# ------------------------------- 19. B4: digitiser run tag and split halves
# The table columns up to the path floor (table_v0 / table_v1) and the
# per-record columns before B4, in their order.
V1_TABLE_COLUMNS = [
    "judge", "condition", "set", "n_records", "lattice_return", "snr_median_db",
    "kappa", "kappa_lo", "kappa_hi", *[f"kappa_{c}" for c in CLASSES],
    "flip_rate", "flip_lo", "flip_hi", "record_flip_rate", "rflip_boot_lo",
    "rflip_boot_hi", "rflip_exact_lo", "rflip_exact_hi", "n_flips", "mean_abs_dp",
    "dp_lo", "dp_hi", "max_abs_dp", "kappa_thrfull", "flip_rate_thrfull",
    "record_flip_rate_thrfull", "auroc_masked_macro", "auroc_dig_macro",
    "auroc_retention_macro", "ret_lo", "ret_hi", *[f"ret_{c}" for c in CLASSES],
    *[f"npos_{c}" for c in CLASSES], "interior_nan_frac", "missing_leads", "ci_note",
    "n_record_flips", "ci_degenerate", "kappa_class_mean", "median_abs_dp",
    *[f"flips_{c}" for c in CLASSES], *[f"nneg_{c}" for c in CLASSES],
    "ret_boot_valid", "ret_boot_frac", "n_labelled", "missing_outputs",
    "path_floor_n_records", "path_floor_n_record_flips", "path_floor_record_flip_rate",
    "path_floor_flip_rate", "path_floor_mean_abs_dp",
]
V1_RECORD_COLUMNS = [
    "judge", "condition", "record", "ecg_id", "strat_fold", "class", "label",
    "labelled", "logit_full", "logit_masked", "logit_dig", "thr_layout", "thr_full",
    "dec_masked", "dec_dig", "flip", "p_masked", "p_dig", "abs_dp", "snr_rec_median",
    "snr_rec_min",
]
SPLIT_HALF_COLUMNS = [
    "n_records_odd", "n_records_even", "n_flips_odd", "n_flips_even",
    "flip_rate_odd", "flip_rate_even", "record_flip_rate_odd", "record_flip_rate_even",
    *[f"mdl_{c}{h}" for c in CLASSES for h in ("", "_odd", "_even")],
]


def test_pred_dir_and_ev_csv_follow_the_run_tag(monkeypatch):
    pred_dir, ev_csv = fidelity.pred_dir, fidelity.ev_csv
    # Default tag: the CONDITIONS folders and the v0 / v1 evaluation CSVs.
    for cond in V0_CONDITIONS + B_CONDITIONS:
        assert pred_dir(cond) == fidelity.CONDITIONS[cond][1]
        assert pred_dir(cond, fidelity.PRED_RUN) == fidelity.CONDITIONS[cond][1]
    out = "data/perspective_results/out"
    assert pred_dir("gen_clean") == f"{out}/gen_clean/inkpage_default"
    assert ev_csv("gen_clean", fidelity.PRED_RUN) == (
        "data/perspective_results/ev_gen_clean_inkpage_default.csv"
    )
    # Another tag: v0 sets under out/<cond>/<tag> and ev_<cond>_<tag>.csv ...
    assert pred_dir("gen_clean", "sp35r") == f"{out}/gen_clean/sp35r"
    assert pred_dir("scale125", "sp35r") == f"{out}/scale125/sp35r"
    assert ev_csv("photo", "sp35r") == "data/perspective_results/ev_photo_sp35r.csv"
    # ... fidelity_b under data/fidelity/<tag>/b_<c>/<tag>.
    for c in ("clean", "scan", "aug", "rot"):
        assert pred_dir(f"b_{c}") == f"data/fidelity/out/{c}"
        assert ev_csv(f"b_{c}") == f"data/fidelity/ev/ev_{c}.csv"
        assert pred_dir(f"b_{c}", "sp35r") == f"data/fidelity/sp35r/b_{c}/sp35r"
        assert ev_csv(f"b_{c}", "sp35r") == f"data/fidelity/sp35r/ev_{c}_sp35r.csv"
    for bad in ("", "a/b", "a@b", " x"):
        with pytest.raises(ValueError, match="Invalid pred_run"):
            pred_dir("gen_clean", bad)
    with pytest.raises(ValueError, match="nope"):
        pred_dir("nope")
    with pytest.raises(ValueError, match="nope"):
        ev_csv("nope", "sp35r")
    # A set without a tagged layout: the default is unchanged, a tag is rejected.
    monkeypatch.setattr(fidelity, "CONDITIONS", {"toy": ("gt", "pred", "unit")})
    monkeypatch.setattr(fidelity, "EV_CSV", "elsewhere/ev_{cond}.csv")
    assert pred_dir("toy") == "pred"
    assert ev_csv("toy") == "elsewhere/ev_toy.csv"
    for fn in (pred_dir, ev_csv):
        with pytest.raises(ValueError, match="'toy'"):
            fn("toy", "sp35r")


def test_condition_key_tags_only_the_digitised_input():
    key = fidelity.condition_key
    assert key("gen_clean", "00001_hr", "dig") == "gen_clean/00001_hr/dig"
    assert key("gen_clean", "00001_hr", "dig", "sp35r") == "gen_clean@sp35r/00001_hr/dig"
    for kind in ("full", "masked"):
        assert key("gen_clean", "00001_hr", kind, "sp35r") == f"gen_clean/00001_hr/{kind}"
    # The default dig key is the one sweep-error shares with the table.
    assert key("c", "r", "dig") == fidelity.error_key("c", "r", 1.0)
    with pytest.raises(ValueError, match="Invalid pred_run"):
        key("c", "r", "dig", "a/b")


def test_evaluate_condition_with_a_run_tag_reuses_the_reference_logits(tmp_path):
    class Counting(FakeJudge):
        calls = 0

        def predict_logits(self, X, batch_size=512):
            Counting.calls += len(X)
            return super().predict_logits(X, batch_size)

    gt_dir, pred_dir, orig_dir, records = _make_condition(tmp_path / "a")
    # Another digitisation of the same pages (same ground truth and originals).
    _, pred_tag, _, _ = _make_condition(tmp_path / "b", perturb=_perturb)
    judge = Counting()
    cache = fidelity.LogitCache(str(tmp_path / "c.npz"), tag="t")
    thr = np.zeros(5)
    row0, recs0 = fidelity.evaluate_condition(
        judge, "toy", gt_dir, pred_dir, orig_dir, thr, thr, cache=cache, n_boot=0
    )
    keys0 = set(cache._data)
    assert keys0 == {
        f"toy/{r}/{kind}" for r in records for kind in ("full", "masked", "dig")
    }
    before = {k: (cache._data[k][0], cache._data[k][1].copy()) for k in keys0}
    calls = Counting.calls
    row, recs = fidelity.evaluate_condition(
        judge, "toy", gt_dir, pred_tag, orig_dir, thr, thr, cache=cache, n_boot=0,
        pred_run="tagx",
    )
    # Only the new digitised inputs are judged; full and masked hit the cache.
    assert Counting.calls == calls + len(records)
    assert set(cache._data) - keys0 == {f"toy@tagx/{r}/dig" for r in records}
    for k, (sha, logits) in before.items():
        assert cache._data[k][0] == sha and np.array_equal(cache._data[k][1], logits)
    assert row0["pred_run"] == fidelity.PRED_RUN and row["pred_run"] == "tagx"
    assert list(row)[-1] == "pred_run"
    assert list(recs0.columns) == [*V1_RECORD_COLUMNS, "pred_run"]
    assert list(recs.columns) == [*V1_RECORD_COLUMNS, "pred_run"]
    assert (recs0["pred_run"] == fidelity.PRED_RUN).all()
    assert (recs["pred_run"] == "tagx").all()
    assert np.array_equal(recs["logit_masked"], recs0["logit_masked"])
    assert np.array_equal(recs["logit_full"], recs0["logit_full"])
    assert not np.array_equal(recs["logit_dig"], recs0["logit_dig"])


def test_split_half_hand_example():
    classes = ("A", "B")
    ids = [1, 2, 3, 4, 7]
    ref = np.tile([1.0, -1.0], (5, 1))
    test = ref.copy()
    test[0] = [-1.0, 1.0]  # ecg_id 1 (odd): both classes flip
    test[1, 1] = 0.5  # ecg_id 2 (even): B flips
    test[4, 0] = 0.5  # ecg_id 7 (odd): A moves but stays positive
    out = fidelity.split_half(ids, ref, test, np.zeros(2), classes)
    assert list(out) == list(fidelity.split_half_columns(classes))
    assert (out["n_records_odd"], out["n_records_even"]) == (3, 2)
    assert (out["n_flips_odd"], out["n_flips_even"]) == (2, 1)
    assert out["flip_rate_odd"] == pytest.approx(2 / 6)
    assert out["flip_rate_even"] == pytest.approx(1 / 4)
    assert out["record_flip_rate_odd"] == pytest.approx(1 / 3)
    assert out["record_flip_rate_even"] == pytest.approx(1 / 2)
    assert out["mdl_A"] == pytest.approx(-0.5)
    assert out["mdl_A_odd"] == pytest.approx(-2.5 / 3)
    assert out["mdl_A_even"] == 0.0
    assert out["mdl_B"] == pytest.approx(0.7)
    assert out["mdl_B_odd"] == pytest.approx(2 / 3)
    assert out["mdl_B_even"] == pytest.approx(0.75)
    # An empty half gives NaN rates and means, zero counts.
    out = fidelity.split_half([1, 3], ref[:2], test[:2], np.zeros(2), classes)
    assert out["n_records_even"] == 0 and out["n_flips_even"] == 0
    assert np.isnan(out["flip_rate_even"]) and np.isnan(out["record_flip_rate_even"])
    assert np.isnan(out["mdl_A_even"]) and np.isfinite(out["mdl_A_odd"])
    with pytest.raises(ValueError):
        fidelity.split_half([1], ref, test, np.zeros(2), classes)
    assert list(fidelity.split_half_columns()) == SPLIT_HALF_COLUMNS


def test_split_half_columns_of_a_condition(tmp_path):
    gt_dir, pred_dir, orig_dir, records = _make_condition(tmp_path, perturb=_perturb)
    judge = FakeJudge()
    inputs = fidelity.condition_inputs(gt_dir, pred_dir, orig_dir)
    thr = _pinned_thresholds(judge, inputs["masked"])
    labels = _labels(len(records))  # ecg_id 6 is unlabelled
    row, recs = fidelity.evaluate_condition(
        judge, "toy", gt_dir, pred_dir, orig_dir, thr, thr, labels=labels, n_boot=0,
    )
    assert row["n_flips"] > 0 and not recs["labelled"].all()
    assert list(row)[-len(SPLIT_HALF_COLUMNS) - 1:] == [*SPLIT_HALF_COLUMNS, "pred_run"]
    # The halves add up to the primary comparison.
    assert row["n_records_odd"] == row["n_records_even"] == 3
    assert row["n_records_odd"] + row["n_records_even"] == row["n_records"]
    assert row["n_flips_odd"] + row["n_flips_even"] == row["n_flips"]
    assert sum(row[f"flips_{c}"] for c in CLASSES) == row["n_flips"]
    assert (row["flip_rate_odd"] + row["flip_rate_even"]) / 2 == pytest.approx(
        row["flip_rate"]
    )
    odd = recs["ecg_id"] % 2 == 1
    assert row["n_flips_odd"] == int(recs.loc[odd, "flip"].sum())
    assert row["n_flips_even"] == int(recs.loc[~odd, "flip"].sum())
    rec_flip = recs.groupby("ecg_id")["flip"].any()
    assert row["record_flip_rate_odd"] == pytest.approx(
        rec_flip[rec_flip.index % 2 == 1].mean()
    )
    # mdl over ALL records (labelled or not) and over each parity.
    for c in CLASSES:
        sub = recs[recs["class"] == c]
        d = sub["logit_dig"] - sub["logit_masked"]
        is_odd = sub["ecg_id"] % 2 == 1
        assert row[f"mdl_{c}"] == pytest.approx(d.mean(), rel=0, abs=1e-12)
        assert row[f"mdl_{c}_odd"] == pytest.approx(d[is_odd].mean(), rel=0, abs=1e-12)
        assert row[f"mdl_{c}_even"] == pytest.approx(d[~is_odd].mean(), rel=0, abs=1e-12)
        assert row[f"mdl_{c}_odd"] != pytest.approx(row[f"mdl_{c}_even"])


def test_table_pred_run_defaults_and_keeps_the_v1_columns(
    tmp_path, monkeypatch, capsys
):
    w = _cli_world(tmp_path, monkeypatch)
    parser = fidelity.get_parser()
    assert parser.parse_args(["table"]).pred_run == fidelity.PRED_RUN
    assert parser.parse_args(["table", "--pred_run", "x"]).pred_run == "x"
    with pytest.raises(SystemExit):  # only the table has run tags
        parser.parse_args(["layout-loss", "--pred_run", "x"])
    out = str(w.out / "table.csv")
    table = _cli([
        "table", *w.common, "--original_dir", w.orig_dir, "--conditions", "toy",
        "--n_boot", "20", "--layout_loss", str(tmp_path / "no_layout_loss.csv"),
        "--out", out,
    ])
    assert f"[pred_run {fidelity.PRED_RUN}]" in capsys.readouterr().out
    # The v0 / v1 columns stay a prefix in their order; the extras follow.
    expected = [*V1_TABLE_COLUMNS, *SPLIT_HALF_COLUMNS, "pred_run"]
    assert list(table.columns) == expected
    assert list(pd.read_csv(out).columns) == expected
    assert table.loc[0, "pred_run"] == fidelity.PRED_RUN
    recs = pd.read_csv(fidelity.default_records_out(out))
    assert list(recs.columns) == [*V1_RECORD_COLUMNS, "pred_run"]
    cache = fidelity.LogitCache(
        fidelity.cache_file(str(w.out / "cache"), "fake"), tag="fake:abc:cpu"
    )
    assert set(cache._data) == {
        f"toy/{r}/{kind}" for r in w.records for kind in ("full", "masked", "dig")
    }


def test_table_with_a_run_tag_end_to_end(tmp_path, monkeypatch, capsys):
    w = _cli_world(tmp_path, monkeypatch)

    def other(i, dig):
        rng = np.random.default_rng(90 + i)
        return dig + 0.12 * rng.normal(size=dig.shape)

    # Run 'tagx' of the same pages: tmp/tagx/pred, its evaluation CSV in ev/.
    gt2, pred2, _, _ = _make_condition(tmp_path / "tagx", perturb=other)
    monkeypatch.setattr(
        fidelity, "RUN_PRED_DIRS", {"unit": str(tmp_path / "{run}" / "pred")}
    )
    monkeypatch.setattr(
        fidelity, "RUN_EV_CSVS", {"unit": str(tmp_path / "ev" / "ev_{cond}_{run}.csv")}
    )
    ev2 = _ev_from_helper(gt2, pred2, w.records)
    ev2.to_csv(tmp_path / "ev" / "ev_toy_tagx.csv", index=False)
    argv = [
        "table", *w.common, "--original_dir", w.orig_dir, "--conditions", "toy",
        "--n_boot", "20", "--layout_loss", str(tmp_path / "no_layout_loss.csv"),
    ]
    out0, out1 = str(w.out / "t0.csv"), str(w.out / "t1.csv")
    base = _cli([*argv, "--out", out0])
    cache_path = fidelity.cache_file(str(w.out / "cache"), "fake")
    keys0 = set(fidelity.LogitCache(cache_path, tag="fake:abc:cpu")._data)
    calls = w.calls["n"]
    capsys.readouterr()
    tagged = _cli([*argv, "--pred_run", "tagx", "--out", out1])
    assert "[pred_run tagx]" in capsys.readouterr().out
    assert w.calls["n"] == calls + len(w.records)
    keys1 = set(fidelity.LogitCache(cache_path, tag="fake:abc:cpu")._data)
    assert keys0 <= keys1
    assert keys1 - keys0 == {f"toy@tagx/{r}/dig" for r in w.records}
    r = tagged.iloc[0]
    assert r["pred_run"] == "tagx" and base.loc[0, "pred_run"] == fidelity.PRED_RUN
    assert r["snr_median_db"] == pytest.approx(fidelity.condition_snr_median(ev2))
    assert r["snr_median_db"] != pytest.approx(base.loc[0, "snr_median_db"])
    assert r["n_flips_odd"] + r["n_flips_even"] == r["n_flips"]
    rec0 = pd.read_csv(fidelity.default_records_out(out0))
    rec1 = pd.read_csv(fidelity.default_records_out(out1))
    assert np.array_equal(rec0["logit_masked"], rec1["logit_masked"])
    assert np.array_equal(rec0["logit_full"], rec1["logit_full"])
    assert not np.array_equal(rec0["logit_dig"], rec1["logit_dig"])
    assert (rec1["pred_run"] == "tagx").all()
    # A set without a tagged layout is rejected before the judge is loaded.
    monkeypatch.setattr(fidelity, "RUN_PRED_DIRS", {})
    out2 = str(w.out / "t2.csv")
    calls = w.calls["n"]
    with pytest.raises(ValueError, match="'toy'"):
        _cli([*argv, "--pred_run", "tagx", "--out", out2])
    assert w.calls["n"] == calls and not os.path.exists(out2)


def test_a_tagged_table_run_never_writes_the_v0_files(tmp_path, monkeypatch):
    guard = fidelity.guard_v0_outputs
    guard(["gen_clean"], [fidelity.TABLE_CSV])  # the default run may rerun v0
    for p in (fidelity.TABLE_CSV, fidelity.default_records_out(fidelity.TABLE_CSV)):
        with pytest.raises(ValueError, match="pred_run 'sp35r'") as err:
            guard(["gen_clean"], [p], pred_run="sp35r")
        assert "'gen_clean'" in str(err.value) and "table_sp35r.csv" in str(err.value)
        assert "table_v1.csv" not in str(err.value)
    guard(["gen_clean", "b_clean"], [str(tmp_path / "t.csv")], pred_run="sp35r")
    writes = _no_judge_no_write(monkeypatch)
    with pytest.raises(ValueError, match="--out"):
        _cli(["table", "--conditions", "gen_clean", "--pred_run", "sp35r"])
    assert writes == [] and os.listdir(tmp_path) == []


# ------------------------------------------ 20. C2: real-image conditions
REAL_NAMES = [
    "real_render", "real_scan_color", "real_scan_gray", "real_scan_mould_color",
    "real_scan_mould_bw", "real_photo", "real_photo_stained", "real_photo_mould",
]
REAL_SPLIT_CSV = os.path.join(REPO_ROOT, fidelity.SPLIT_CSV)
REAL_RENDER_GT = os.path.join(REPO_ROOT, "data/real_images/render")


def test_real_conditions_map_to_the_c2_folders():
    assert list(fidelity.REAL_CONDITIONS) == REAL_NAMES
    assert fidelity.REAL_SET == "real_c2" and fidelity.REAL_PRED_RUN == "bl_o05"
    for name in REAL_NAMES:
        c = name[len("real_"):]
        assert fidelity.REAL_CONDITIONS[name] == (
            f"data/real_images/{c}", f"data/real_results/out/{c}/bl_o05", "real_c2"
        )
        assert fidelity.condition_entry(name) == fidelity.REAL_CONDITIONS[name]
        assert fidelity.is_real(name)
        # A registry of their own: CONDITIONS and its sets are unchanged.
        assert name not in fidelity.CONDITIONS
    assert len(fidelity.CONDITIONS) == 19
    assert not any(fidelity.is_real(c) for c in fidelity.CONDITIONS)
    assert fidelity.condition_entry("gen_clean") == fidelity.CONDITIONS["gen_clean"]
    with pytest.raises(ValueError, match="Unknown condition 'nope'"):
        fidelity.condition_entry("nope")
    # The v0 'photo' and the real 'real_photo' are different conditions.
    assert fidelity.CONDITIONS["photo"][0] == "data/quasi_real/photo"


def test_real_pred_run_paths_and_cache_keys():
    pred_dir, ev_csv, run = fidelity.pred_dir, fidelity.ev_csv, fidelity.condition_run
    for name in REAL_NAMES:
        c = name[len("real_"):]
        default_dir = f"data/real_results/out/{c}/bl_o05"
        default_ev = f"data/real_results/ev/ev_{c}_bl_o05.csv"
        assert pred_dir(name) == default_dir
        assert ev_csv(name) == default_ev
        # The default tag means bl_o05 for a real condition: the same folder,
        # evaluation CSV and dig cache key as an explicit --pred_run bl_o05.
        assert run(name) == run(name, fidelity.PRED_RUN) == "bl_o05"
        assert run(name, "bl_o05") == "bl_o05" and run(name, "none_o0") == "none_o0"
        assert pred_dir(name, run(name)) == default_dir
        assert ev_csv(name, run(name)) == default_ev
        for tag in ("bl_o0", "none_o05", "none_o0", "bl_o05_repro"):
            assert pred_dir(name, tag) == f"data/real_results/out/{c}/{tag}"
            assert ev_csv(name, tag) == f"data/real_results/ev/ev_{c}_{tag}.csv"
    # Synthetic conditions keep their default run.
    assert run("gen_clean") == fidelity.PRED_RUN and run("b_scan", "x") == "x"
    with pytest.raises(ValueError, match="Invalid pred_run"):
        pred_dir("real_render", "a/b")
    key = fidelity.condition_key
    assert key("real_photo", "00022_hr", "dig", "bl_o05") == (
        "real_photo@bl_o05/00022_hr/dig"
    )
    assert key("real_photo", "00022_hr", "masked", "bl_o05") == (
        "real_photo/00022_hr/masked"
    )
    # No real key equals a key of another condition, for any kind or run.
    rec = "00022_hr"
    real_keys = {
        key(c, rec, k, r) for c in REAL_NAMES for k in ("full", "masked", "dig")
        for r in (fidelity.PRED_RUN, "bl_o05", "none_o0")
    }
    other_keys = {
        key(c, rec, k, r) for c in fidelity.CONDITIONS
        for k in ("full", "masked", "dig") for r in (fidelity.PRED_RUN, "bl_o05")
    } | {f"ptbxl/22/{k}" for k in ("full", "layout")} | {
        f"ptbxl500/22/{k}" for k in ("full", "layout")
    } | {fidelity.error_key(c, rec, 0.5) for c in fidelity.CONDITIONS}
    assert not real_keys & other_keys
    assert len(real_keys) == len(REAL_NAMES) * (2 + 3)


def test_all_excludes_the_real_conditions():
    resolve = fidelity.resolve_conditions
    assert resolve(["all"]) == V0_CONDITIONS
    assert resolve(["all"], allow_real=True) == V0_CONDITIONS
    assert resolve(None, allow_real=True) == V0_CONDITIONS
    for group in ("real", "real_c2"):
        assert resolve([group], allow_real=True) == REAL_NAMES
    assert resolve(["real_photo", "real"], allow_real=True) == [
        "real_photo", *[n for n in REAL_NAMES if n != "real_photo"]
    ]
    # Without allow_real (sweep-error) a real token is refused with its message.
    for token in ("real", "real_c2", "real_render"):
        with pytest.raises(ValueError, match="table subcommand only"):
            resolve([token])
    with pytest.raises(ValueError, match="Unknown condition"):
        resolve(["real_nope"], allow_real=True)


def test_real_split_file_has_115_dev_and_84_eval_records():
    if not (os.path.exists(REAL_SPLIT_CSV) and os.path.isdir(REAL_RENDER_GT)):
        pytest.skip("real-image data not present")
    split = fidelity.load_split(REAL_SPLIT_CSV)
    records = fidelity.list_records(REAL_RENDER_GT)
    assert len(records) == 199
    dev = fidelity.split_records(records, split, "dev")
    ev = fidelity.split_records(records, split, "eval")
    assert (len(dev), len(ev)) == (115, 84)
    assert sorted(dev + ev) == records and not set(dev) & set(ev)
    # Patient split: no patient on both sides.
    pid = dict(zip(split["record"], split["patient_id"]))
    assert not {pid[r] for r in dev} & {pid[r] for r in ev}
    assert fidelity.table_records(REAL_RENDER_GT, "dev", split) == dev
    assert fidelity.table_records(REAL_RENDER_GT, "eval", split, 10) == ev[:10]


def _split_csv(path, splits, patients=None):
    rows = [
        {"record": r, "split": s, "patient_id": (patients or {}).get(r, 100.0 + i)}
        for i, (r, s) in enumerate(splits.items())
    ]
    pd.DataFrame(rows).to_csv(path, index=False)
    return str(path)


def test_split_filter_on_a_synthetic_split(tmp_path):
    records = [f"{i:05d}_hr" for i in range(1, 7)]
    splits = dict(zip(records, ["dev", "dev", "eval", "dev", "dev", "eval"]))
    df = fidelity.load_split(_split_csv(tmp_path / "s.csv", splits))
    assert fidelity.split_records(records, df, "dev") == [
        "00001_hr", "00002_hr", "00004_hr", "00005_hr"
    ]
    assert fidelity.split_records(records, df, "eval") == ["00003_hr", "00006_hr"]
    # The order of the given records is kept.
    assert fidelity.split_records(records[::-1], df, "eval") == ["00006_hr", "00003_hr"]
    with pytest.raises(ValueError, match="Unknown split"):
        fidelity.split_records(records, df, "test")
    # The split file must list exactly the given records.
    with pytest.raises(ValueError, match="not in the split file"):
        fidelity.split_records([*records, "00007_hr"], df, "dev")
    with pytest.raises(ValueError, match="not on disk"):
        fidelity.split_records(records[:-1], df, "dev")
    bad = tmp_path / "bad.csv"
    _split_csv(bad, {**splits, "00001_hr": "train"})
    with pytest.raises(ValueError, match="unknown split value"):
        fidelity.load_split(str(bad))
    pd.concat([df, df.iloc[:1]]).to_csv(bad, index=False)
    with pytest.raises(ValueError, match="duplicate record"):
        fidelity.load_split(str(bad))
    df.drop(columns="split").to_csv(bad, index=False)
    with pytest.raises(ValueError, match="lacks column"):
        fidelity.load_split(str(bad))
    # Record names keep their leading zeros.
    assert fidelity.load_split(_split_csv(tmp_path / "s.csv", splits))[
        "record"
    ].tolist() == records
    # table_records: None without split and limit; the limit follows the split.
    gt_dir, _, _, _ = _make_condition(tmp_path / "c")
    assert fidelity.table_records(gt_dir) is None
    assert fidelity.table_records(gt_dir, max_records=2) == records[:2]
    assert fidelity.table_records(gt_dir, "dev", df, 3) == [
        "00001_hr", "00002_hr", "00004_hr"
    ]


def test_ptbxl500_originals_resolve_without_a_copy(tmp_path):
    gt_dir, pred_dir, orig_dir, records = _make_condition(tmp_path, n_records=3)
    root = tmp_path / "ptb"
    store = root / "records500" / "00000"
    store.mkdir(parents=True)
    for rec in records:
        for ext in (".hea", ".dat"):
            os.link(os.path.join(orig_dir, rec + ext), store / (rec + ext))
    db = pd.DataFrame(
        {"filename_hr": [f"records500/00000/{r}" for r in records]},
        index=pd.Index([1, 2, 3], name="ecg_id"),
    )
    originals = fidelity.ptbxl500_originals(db, records, str(root))
    assert originals == {r: str(store / r) for r in records}
    assert fidelity.original_base(originals, records[0]) == str(store / records[0])
    assert fidelity.original_base(orig_dir, records[0]) == os.path.join(
        orig_dir, records[0]
    )
    with pytest.raises(ValueError, match="no original"):
        fidelity.original_base(originals, "00009_hr")
    # The same inputs as the folder of originals; the GT check still runs.
    plain = fidelity.condition_inputs(gt_dir, pred_dir, orig_dir)
    mapped = fidelity.condition_inputs(gt_dir, pred_dir, originals)
    for kind in ("full", "masked", "dig"):
        for a, b in zip(plain[kind], mapped[kind]):
            assert np.array_equal(a, b)
    wrong = dict(originals)
    wrong[records[0]] = str(store / records[1])
    with pytest.raises(ValueError, match="differs from the ground truth inside"):
        fidelity.condition_inputs(gt_dir, pred_dir, wrong)
    # A database row naming another file, an unknown ecg_id, a missing file.
    other = db.copy()
    other.loc[2, "filename_hr"] = "records500/00000/00003_hr"
    with pytest.raises(ValueError, match="another record name"):
        fidelity.ptbxl500_originals(other, records, str(root))
    with pytest.raises(ValueError, match="not in the PTB-XL database"):
        fidelity.ptbxl500_originals(db, ["00009_hr"], str(root))
    os.remove(store / (records[2] + ".hea"))
    with pytest.raises(FileNotFoundError, match="records500"):
        fidelity.ptbxl500_originals(db, records, str(root))


def test_real_originals_are_the_records500_files():
    db_csv = os.path.join(REPO_ROOT, fidelity.PTBXL_ROOT, "ptbxl_database.csv")
    pred = os.path.join(REPO_ROOT, fidelity.REAL_CONDITIONS["real_render"][1])
    if not (os.path.exists(db_csv) and os.path.isdir(REAL_RENDER_GT)
            and os.path.isdir(pred)):
        pytest.skip("real-image data not present")
    db = pd.read_csv(db_csv, index_col="ecg_id")
    records = fidelity.list_records(REAL_RENDER_GT)[:3]
    originals = fidelity.ptbxl500_originals(
        db, records, os.path.join(REPO_ROOT, fidelity.PTBXL_ROOT)
    )
    for rec in records:
        assert originals[rec].endswith(os.path.join("records500", rec[:2] + "000", rec))
        orig, mask, d, present, fs = fidelity._read_condition_record(
            REAL_RENDER_GT, originals, rec, os.path.join(pred, rec + fidelity.PRED_SUFFIX)
        )
        assert present and fs == 500.0 and orig.shape == (5000, 12)
        assert mask[:, fidelity.LEADS.index("II")].all() and not mask.all()


def test_a_digitised_record_without_a_lead_is_missing_there(tmp_path, capsys):
    gt_dir, pred_dir, orig_dir, records = _make_condition(tmp_path, n_records=2)
    base = os.path.join(pred_dir, records[0] + fidelity.PRED_SUFFIX)
    full, _ = fidelity.read_record(base)
    keep = [j for j, lead in enumerate(fidelity.LEADS) if lead != "V3"]
    for ext in (".hea", ".dat"):
        os.remove(base + ext)
    _write(pred_dir, records[0] + fidelity.PRED_SUFFIX, full[:, keep],
           [fidelity.LEADS[j] for j in keep])
    with pytest.raises(ValueError, match="Lead V3 missing"):
        fidelity.read_record(base)
    x, fs = fidelity.read_record(base, allow_missing=True)
    assert "['V3'] absent" in capsys.readouterr().err
    v3 = fidelity.LEADS.index("V3")
    assert np.isnan(x[:, v3]).all()
    assert np.array_equal(x[:, keep], full[:, keep])
    # Only with allow_missing (the real-image conditions): the other sets
    # still abort on an absent lead, in the table and in sweep-error.
    with pytest.raises(ValueError, match="Lead V3 missing"):
        fidelity.condition_inputs(gt_dir, pred_dir, orig_dir)
    with pytest.raises(ValueError, match="Lead V3 missing"):
        fidelity.condition_signals(gt_dir, pred_dir, orig_dir)
    assert fidelity.condition_signals(
        gt_dir, pred_dir, orig_dir, allow_missing=True
    )["present"] == [True, True]
    inputs = fidelity.condition_inputs(gt_dir, pred_dir, orig_dir, allow_missing=True)
    assert inputs["present"] == [True, True]
    assert inputs["qc"][0]["missing"][v3] and inputs["qc"][0]["missing"].sum() == 1
    assert not inputs["qc"][1]["missing"].any()
    # An original without a lead still aborts.
    for ext in (".hea", ".dat"):
        os.remove(os.path.join(orig_dir, records[1] + ext))
    orig_full = _signal(seed=101)
    _write(orig_dir, records[1], orig_full[:, keep], [fidelity.LEADS[j] for j in keep])
    with pytest.raises(ValueError, match="Lead V3 missing"):
        fidelity.condition_inputs(gt_dir, pred_dir, orig_dir, allow_missing=True)


def test_real_table_options_are_checked_before_the_judge_loads(tmp_path, monkeypatch):
    writes = _no_judge_no_write(monkeypatch)
    # Hermetic: data/fidelity (gitignored) is a folder under tmp_path, absent
    # at first; the table and cache defaults live in it as in the repo.
    fid = tmp_path / "fid"
    monkeypatch.setattr(fidelity, "FIDELITY_DIR", str(fid))
    monkeypatch.setattr(fidelity, "TABLE_CSV", str(fid / "table_v0.csv"))
    monkeypatch.setattr(fidelity, "CACHE_DIR", str(fid / "cache"))
    out = str(tmp_path / "t.csv")
    cache = ["--cache_dir", str(tmp_path / "cache")]
    p5 = ["--original_dir", "ptbxl500"]
    under = f"may not write under {re.escape(str(fid))}"
    outputs = (
        (["table", "--conditions", "real_c2", "--split", "dev", *p5, *cache,
          "--out", str(fid / "table_real.csv"), "--records_out", out], under),
        (["table", "--conditions", "real_photo", "--split", "dev", *p5, *cache,
          "--out", out, "--records_out", str(fid / "sub" / "r.csv")], under),
        # The default --out is inside it: the real guard speaks first.
        (["table", "--conditions", "real", "--split", "dev", *p5, *cache],
         under + ".*data/real_results/c2"),
        # So is the default --cache_dir, and any other one in it.
        (["table", "--conditions", "real", "--split", "dev", *p5, "--out", out],
         under + ".*cache"),
        (["table", "--conditions", "real", "--split", "dev", *p5, "--out", out,
          "--cache_dir", str(fid / "c2")], under),
        # A case variant of the folder.
        (["table", "--conditions", "real", "--split", "dev", *p5, *cache,
          "--out", str(tmp_path / "FID" / "t.csv")], under),
    )
    refused = (
        (["table", "--conditions", "real_render"], "need --split"),
        (["table", "--conditions", "real", "--out", out], "need --split"),
        (["table", "--conditions", "gen_clean", "--split", "dev", "--out", out],
         "--split applies"),
        (["table", "--split", "eval"], "--split applies"),
        (["table", "--conditions", "real_render", "gen_clean", "--split", "dev",
          *p5, "--out", out], "cannot share a table run"),
        (["table", "--conditions", "real", "--split", "dev", "--out", out],
         "--original_dir ptbxl500"),
        *outputs,
        (["table", "--conditions", "real_photo", "--split", "dev", *p5,
          "--out", out, "--max_records", "-1"], "--max_records must be >= 0"),
        # A partial run may not write the v0 files either.
        (["table", "--max_records", "2"], "--out"),
        (["sweep-error", "--conditions", "real_render"], "table subcommand only"),
        (["margins", "--condition", "real_render"], "Unknown condition"),
    )
    for argv, message in refused:
        with pytest.raises(ValueError, match=message):
            _cli(argv)
    with pytest.raises(SystemExit):  # only dev and eval
        fidelity.get_parser().parse_args(["table", "--split", "test"])
    parser = fidelity.get_parser()
    args = parser.parse_args(["table"])
    assert args.split is None and args.max_records == 0
    assert args.split_csv == fidelity.SPLIT_CSV
    assert args.original_dir == fidelity.ORIGINAL_DIR
    assert writes == [] and os.listdir(tmp_path) == []
    # guard_real_outputs: real conditions only; the folder absent or present.
    for present in (False, True):
        if present:
            fid.mkdir()
            for argv, message in outputs[:-1]:
                with pytest.raises(ValueError, match=message):
                    _cli(argv)
        fidelity.guard_real_outputs(["gen_clean"], [str(fid / "x.csv")])
        fidelity.guard_real_outputs(["real_render"], [out])
        for bad in (fid / "x.csv", fid, fid / "a" / "b.csv"):
            with pytest.raises(ValueError, match="real_render"):
                fidelity.guard_real_outputs(["real_render"], [str(bad)])
        # A sibling whose name starts like it is not inside.
        fidelity.guard_real_outputs(["real_render"], [str(tmp_path / "fid2" / "t")])
    if os.path.exists(tmp_path / "FID"):  # a case-insensitive disk
        with pytest.raises(ValueError, match="real_render"):
            fidelity.guard_real_outputs(["real_render"], [str(tmp_path / "FID" / "t")])
    assert writes == [] and os.listdir(fid) == []


def _real_world(tmp_path, monkeypatch):
    """A toy real-image condition 'real_toy' (records 00001-00006_hr): its
    default run bl_o05 and a run tagx, originals only in a toy records500
    tree, a synthetic split (dev 1, 2, 4, 5; eval 3, 6), a fake judge."""
    def noisy(i, dig):
        rng = np.random.default_rng(60 + i)
        return dig + 0.03 * (i + 1) * rng.normal(size=dig.shape)

    base = _make_condition(tmp_path / "bl_o05", perturb=noisy)
    gt_dir, pred_dir, orig_dir, records = base

    def other(i, dig):
        rng = np.random.default_rng(90 + i)
        return dig + 0.12 * rng.normal(size=dig.shape)

    _make_condition(tmp_path / "tagx", perturb=other)
    root = tmp_path / "ptb"
    store = root / "records500" / "00000"
    store.mkdir(parents=True)
    for rec in records:
        for ext in (".hea", ".dat"):
            os.link(os.path.join(orig_dir, rec + ext), store / (rec + ext))
    db = pd.DataFrame(
        [{**{c: int(c == ("NORM" if i % 2 else "MI")) for c in CLASSES},
          "strat_fold": 10, "filename_hr": f"records500/00000/{i:05d}_hr"}
         for i in range(1, 7)],
        index=pd.Index(range(1, 7), name="ecg_id"),
    )
    real_set = fidelity.REAL_SET
    monkeypatch.setattr(
        fidelity, "REAL_CONDITIONS", {"real_toy": (gt_dir, pred_dir, real_set)}
    )
    monkeypatch.setattr(
        fidelity, "RUN_PRED_DIRS",
        {**fidelity.RUN_PRED_DIRS, real_set: str(tmp_path / "{run}" / "pred")},
    )
    ev_dir = tmp_path / "ev"
    ev_dir.mkdir()
    monkeypatch.setattr(
        fidelity, "RUN_EV_CSVS",
        {**fidelity.RUN_EV_CSVS, real_set: str(ev_dir / "ev_{c}_{run}.csv")},
    )
    monkeypatch.setattr(
        fidelity, "EV_CSVS",
        {**fidelity.EV_CSVS, "real_toy": str(ev_dir / "ev_toy_bl_o05.csv")},
    )
    ev = {
        run: _ev_from_helper(gt_dir, str(tmp_path / run / "pred"), records)
        for run in ("bl_o05", "tagx")
    }
    for run, frame in ev.items():
        frame.to_csv(ev_dir / f"ev_toy_{run}.csv", index=False)
    splits = dict(zip(records, ["dev", "dev", "eval", "dev", "dev", "eval"]))
    patients = {r: float(200 + i // 2) for i, r in enumerate(records)}
    split_csv = _split_csv(tmp_path / "split.csv", splits, patients)
    calls = {"n": 0}

    class Loaded(FakeJudge):
        def predict_logits(self, X, batch_size=512):
            calls["n"] += len(X)
            return super().predict_logits(X, batch_size)

    def load(name, weights_dir=None, device="cpu"):
        judge = Loaded()
        judge.weights_sha256 = "abc"
        return judge

    J = types.SimpleNamespace(
        CLASSES=CLASSES, Judge=types.SimpleNamespace(load=load),
        load_ptbxl_db=lambda root=None: db,
    )
    monkeypatch.setattr(fidelity, "_judge_module", lambda: J)
    thr = np.zeros(len(CLASSES))
    thr_csv = tmp_path / "thresholds.csv"
    pd.DataFrame(
        [{"judge": "fake", "input": kind, "class": c, "threshold_logit": t,
          "device": "cpu"}
         for kind in ("full", "layout") for c, t in zip(CLASSES, thr)]
    ).to_csv(thr_csv, index=False)
    argv = [
        "table", "--judges", "fake", "--ptbxl_root", str(root),
        "--weights_dir", str(tmp_path), "--cache_dir", str(tmp_path / "cache"),
        "--thresholds", str(thr_csv), "--original_dir", "ptbxl500",
        "--split_csv", split_csv, "--conditions", "real_c2", "--n_boot", "20",
        "--layout_loss", str(tmp_path / "no_layout_loss.csv"),
    ]
    return types.SimpleNamespace(
        argv=argv, records=records, splits=splits, patients=patients, ev=ev,
        calls=calls, gt_dir=gt_dir, orig_dir=orig_dir, db=db, thr=thr,
        cache=fidelity.cache_file(str(tmp_path / "cache"), "fake"),
        out=tmp_path / "outs",
    )


def test_real_table_end_to_end_on_a_split(tmp_path, monkeypatch, capsys):
    w = _real_world(tmp_path, monkeypatch)
    dev = [r for r in w.records if w.splits[r] == "dev"]
    ev_recs = [r for r in w.records if w.splits[r] == "eval"]
    out = str(w.out / "dev.csv")
    table = _cli([*w.argv, "--split", "dev", "--out", out])
    printed = capsys.readouterr().out
    assert "[pred_run bl_o05]" in printed and "[split dev]" in printed
    expected = [*V1_TABLE_COLUMNS, *SPLIT_HALF_COLUMNS, "pred_run", "split"]
    assert list(table.columns) == expected
    assert list(pd.read_csv(out).columns) == expected
    row = table.iloc[0]
    assert row["condition"] == "real_toy" and row["set"] == "real_c2"
    assert row["n_records"] == len(dev) == 4
    assert row["split"] == "dev" and row["pred_run"] == "bl_o05"
    # The SNR median covers the dev records only.
    ev = w.ev["bl_o05"]
    assert row["snr_median_db"] == pytest.approx(
        fidelity.condition_snr_median(ev[ev["record"].isin(dev)])
    )
    assert row["snr_median_db"] != pytest.approx(fidelity.condition_snr_median(ev))
    recs = pd.read_csv(fidelity.default_records_out(out), dtype={"record": str})
    assert list(recs.columns) == [*V1_RECORD_COLUMNS, "pred_run", "split", "patient_id"]
    assert sorted(set(recs["record"])) == dev
    assert (recs["split"] == "dev").all()
    assert recs.groupby("record")["patient_id"].first().to_dict() == {
        r: w.patients[r] for r in dev
    }
    keys = set(fidelity.LogitCache(w.cache, tag="fake:abc:cpu")._data)
    assert keys == {f"real_toy/{r}/{k}" for r in dev for k in ("full", "masked")} | {
        f"real_toy@bl_o05/{r}/dig" for r in dev
    }
    # The same numbers as evaluate_condition on the dev records with the folder
    # of originals.
    direct, direct_recs = fidelity.evaluate_condition(
        FakeJudge(), "real_toy", w.gt_dir, fidelity.pred_dir("real_toy"), w.orig_dir,
        w.thr, w.thr, labels=w.db, n_boot=20, seed=0, records=dev,
    )
    for k in ("kappa", "n_flips", "mean_abs_dp", "record_flip_rate"):
        assert row[k] == pytest.approx(direct[k], rel=0, abs=1e-12)
    assert np.allclose(recs["logit_dig"], direct_recs["logit_dig"], rtol=0, atol=1e-12)
    # The CIs resample patients: dev = 4 records of 3 patients (200, 200, 201, 202).
    assert "CIs resample 3 patients (cluster bootstrap)" in row["ci_note"]
    clustered, _ = fidelity.evaluate_condition(
        FakeJudge(), "real_toy", w.gt_dir, fidelity.pred_dir("real_toy"), w.orig_dir,
        w.thr, w.thr, labels=w.db, n_boot=20, seed=0, records=dev, groups=w.patients,
    )
    for k in ("dp_lo", "dp_hi", "ret_lo", "ret_hi", "ci_note"):
        assert (row[k] == clustered[k]) or (pd.isna(row[k]) and pd.isna(clustered[k]))
    assert (row["dp_lo"], row["dp_hi"]) != (direct["dp_lo"], direct["dp_hi"])
    # An explicit --pred_run bl_o05 is the default run: nothing new is judged.
    calls = w.calls["n"]
    again = _cli([*w.argv, "--split", "dev", "--pred_run", "bl_o05",
                  "--out", str(w.out / "dev2.csv")])
    assert w.calls["n"] == calls
    assert again.iloc[0]["kappa"] == row["kappa"]
    # Another run: only its dig inputs are judged, under their own keys.
    tagged = _cli([*w.argv, "--split", "dev", "--pred_run", "tagx",
                   "--out", str(w.out / "dev_tagx.csv")])
    assert w.calls["n"] == calls + len(dev)
    keys1 = set(fidelity.LogitCache(w.cache, tag="fake:abc:cpu")._data)
    assert keys1 - keys == {f"real_toy@tagx/{r}/dig" for r in dev}
    assert tagged.iloc[0]["pred_run"] == "tagx"
    ev_t = w.ev["tagx"]
    assert tagged.iloc[0]["snr_median_db"] == pytest.approx(
        fidelity.condition_snr_median(ev_t[ev_t["record"].isin(dev)])
    )
    # The eval split and a smoke limit.
    t_eval = _cli([*w.argv, "--split", "eval", "--out", str(w.out / "eval.csv")])
    assert t_eval.iloc[0]["n_records"] == len(ev_recs) == 2
    assert t_eval.iloc[0]["split"] == "eval"
    capsys.readouterr()
    smoke = _cli([*w.argv, "--split", "dev", "--max_records", "1",
                  "--out", str(w.out / "smoke.csv")])
    assert smoke.iloc[0]["n_records"] == 1
    assert "smoke run" in capsys.readouterr().err
    smoke_recs = pd.read_csv(str(w.out / "smoke_records.csv"), dtype={"record": str})
    assert set(smoke_recs["record"]) == {dev[0]}


def test_existing_tables_get_no_split_column(tmp_path, monkeypatch):
    w = _cli_world(tmp_path, monkeypatch)
    out = str(w.out / "table.csv")
    table = _cli([
        "table", *w.common, "--original_dir", w.orig_dir, "--conditions", "toy",
        "--n_boot", "20", "--layout_loss", str(tmp_path / "no_layout_loss.csv"),
        "--out", out, "--max_records", "3",
    ])
    assert list(table.columns) == [*V1_TABLE_COLUMNS, *SPLIT_HALF_COLUMNS, "pred_run"]
    assert table.iloc[0]["n_records"] == 3
    recs = pd.read_csv(fidelity.default_records_out(out))
    assert list(recs.columns) == [*V1_RECORD_COLUMNS, "pred_run"]


def test_real_table_checks_the_evaluation_csv_and_reads_absent_leads(
    tmp_path, monkeypatch, capsys
):
    w = _real_world(tmp_path, monkeypatch)
    dev = [r for r in w.records if w.splits[r] == "dev"]
    ev_csv = tmp_path / "ev" / "ev_toy_bl_o05.csv"
    full_ev = w.ev["bl_o05"]
    pred = fidelity.pred_dir("real_toy")
    # A partial evaluation CSV (a dev record with an output but no rows) aborts.
    full_ev[full_ev["record"] != dev[1]].to_csv(ev_csv, index=False)
    with pytest.raises(ValueError, match="partial or stale"):
        _cli([*w.argv, "--split", "dev", "--out", str(w.out / "a.csv")])
    # A record without a digitised output has no rows (src/run/evaluate.py
    # skips it): the SNR median rests on the others, and the row says so.
    for ext in (".hea", ".dat"):
        os.remove(os.path.join(pred, dev[1] + fidelity.PRED_SUFFIX + ext))
    capsys.readouterr()
    row = _cli([*w.argv, "--split", "dev", "--out", str(w.out / "b.csv")]).iloc[0]
    assert "SNR median over 3/4 records" in row["ci_note"]
    assert "1/4 digitised outputs missing" in row["ci_note"]
    assert "SNR median over 3/4" in capsys.readouterr().err
    kept = full_ev[full_ev["record"].isin([r for r in dev if r != dev[1]])]
    assert row["snr_median_db"] == pytest.approx(fidelity.condition_snr_median(kept))
    # A real digitised file without a lead is read with that lead missing.
    base = os.path.join(pred, dev[0] + fidelity.PRED_SUFFIX)
    x, _ = fidelity.read_record(base)
    keep = [j for j, lead in enumerate(fidelity.LEADS) if lead != "V3"]
    for ext in (".hea", ".dat"):
        os.remove(base + ext)
    _write(pred, dev[0] + fidelity.PRED_SUFFIX, x[:, keep],
           [fidelity.LEADS[j] for j in keep])
    row = _cli([*w.argv, "--split", "dev", "--out", str(w.out / "c.csv")]).iloc[0]
    assert row["missing_leads"] >= 1
    assert "['V3'] absent" in capsys.readouterr().err
    # A split file without patient_id cannot give patient-level CIs.
    split_csv = w.argv[w.argv.index("--split_csv") + 1]
    pd.read_csv(split_csv).drop(columns="patient_id").to_csv(
        tmp_path / "nopid.csv", index=False
    )
    argv = list(w.argv)
    argv[argv.index("--split_csv") + 1] = str(tmp_path / "nopid.csv")
    with pytest.raises(ValueError, match="lacks patient_id"):
        _cli([*argv, "--split", "dev", "--out", str(w.out / "d.csv")])
