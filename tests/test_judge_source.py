"""Tests of the judge signal source (records100 / records500 condition path), the
training augmentation and the per-epoch validation of scripts/judge.py.

Needs torch; no benchmark code, weights or PTB-XL: records are tiny synthetic WFDB files
in tmp dirs, and the training / evaluation smokes use a small stand-in network.
"""
import os
import warnings

import numpy as np
import pandas as pd
import pytest
import wfdb

torch = pytest.importorskip("torch")

from scripts import fidelity  # noqa: E402
from scripts import judge as J  # noqa: E402

PTB_NAMES = ["I", "II", "III", "AVR", "AVL", "AVF", "V1", "V2", "V3", "V4", "V5", "V6"]


def _signal(n, fs, seed):
    """A smooth 12-lead test signal in mV, quantised to the WFDB gain of 1000."""
    rng = np.random.default_rng(seed)
    t = np.arange(n) / fs
    freqs = rng.uniform(0.8, 3.0, size=12)
    amps = rng.uniform(0.2, 1.5, size=12)
    x = amps * np.sin(2 * np.pi * freqs * t[:, None]) + 0.05 * rng.normal(size=(n, 12))
    return np.round(x, 3)


def _wrsamp(directory, name, signal, fs):
    os.makedirs(directory, exist_ok=True)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", RuntimeWarning)
        wfdb.wrsamp(
            name, fs=fs, units=["mV"] * 12, sig_name=PTB_NAMES, p_signal=signal,
            write_dir=str(directory), fmt=["16"] * 12, adc_gain=[1000.0] * 12,
            baseline=[0] * 12,
        )


def _fake_ptbxl(tmp_path, ids, absent=()):
    """PTB-XL-like root: records500 (and records100) WFDB files of `ids` and a
    ptbxl_database.csv that also lists the `absent` ids (no files)."""
    root = tmp_path / "ptbxl"
    rows = []
    for i in list(ids) + list(absent):
        lr, hr = f"records100/00000/{i:05d}_lr", f"records500/00000/{i:05d}_hr"
        if i in ids:
            hr_sig, lr_sig = _signal(5000, 500, i), _signal(1000, 100, i)
            _wrsamp(root / "records500" / "00000", f"{i:05d}_hr", hr_sig, 500)
            _wrsamp(root / "records100" / "00000", f"{i:05d}_lr", lr_sig, 100)
        rows.append(
            {"ecg_id": i, "strat_fold": 9, "filename_lr": lr, "filename_hr": hr}
        )
    db = pd.DataFrame(rows).set_index("ecg_id")
    db.to_csv(root / "ptbxl_database.csv")
    return str(root), db


# ------------------------------------------------------------------ signal caches
def test_load_inputs_500_equals_the_condition_path(tmp_path, monkeypatch):
    monkeypatch.setattr(J, "DEFAULT_CACHE_DIR", str(tmp_path / "cache"))
    root, db = _fake_ptbxl(tmp_path, [40, 41, 42, 43], absent=[44])
    order = [42, 40, 41]
    ref = fidelity.ptbxl500_inputs(db, order, root)
    got = {}
    for kind in J.INPUT_KINDS:
        got[kind] = J.load_inputs_500(order, kind, root=root)
        assert got[kind].dtype == np.float32 and got[kind].shape == (3, 1000, 12)
        np.testing.assert_array_equal(got[kind], np.stack(ref[kind]))  # bit-identical
        name = f"ptbxl_records500_{kind}{J._root_tag(root)}.npz"
        assert J._root_tag(root) and os.path.exists(tmp_path / "cache" / name)
    assert np.abs(got["full"] - got["layout"]).max() > 0.1

    # Cache hits read nothing; a new id reads only that record, in any order.
    reads = []
    real_read = fidelity.read_record

    def counting_read(path):
        reads.append(os.path.basename(path))
        return real_read(path)

    monkeypatch.setattr(fidelity, "read_record", counting_read)
    again = J.load_inputs_500([41, 42], "layout", root=root)
    np.testing.assert_array_equal(again, got["layout"][[2, 0]])
    assert reads == []
    both = J.load_inputs_500([43, 40], "layout", root=root)
    assert reads == ["00043_hr"]
    x43, fs43 = real_read(f"{root}/records500/00000/00043_hr")
    np.testing.assert_array_equal(both[0], fidelity._prepare_kind(x43, fs43, "layout"))
    np.testing.assert_array_equal(both[1], got["layout"][1])
    cache = tmp_path / "cache" / f"ptbxl_records500_layout{J._root_tag(root)}.npz"
    with np.load(cache) as z:
        assert z["ecg_id"].tolist() == [40, 41, 42, 43]  # kept sorted by ecg_id

    with pytest.raises(FileNotFoundError, match="1/1 records500 files missing"):
        J.load_inputs_500([44], "full", root=root)
    with pytest.raises(KeyError):
        J.load_inputs_500([99], "full", root=root)
    with pytest.raises(ValueError, match="kind"):
        J.load_inputs_500([40], "window", root=root)


def test_load_signals_100_reads_and_caches_after_the_refactor(tmp_path, monkeypatch):
    root, _ = _fake_ptbxl(tmp_path, [50, 51, 52])
    cache = str(tmp_path / "c100.npz")
    reads = []
    real = J._read_record_100

    def counting(path):
        reads.append(os.path.basename(path))
        return real(path)

    monkeypatch.setattr(J, "_read_record_100", counting)
    X = J.load_signals_100([52, 50], root=root, cache_path=cache)
    assert reads == ["00050_lr", "00052_lr"] and X.dtype == np.float32
    for row, i in zip(X, (52, 50)):
        sig, _ = wfdb.rdsamp(f"{root}/records100/00000/{i:05d}_lr")
        np.testing.assert_array_equal(row, sig.astype(np.float32))
    Y = J.load_signals_100([50, 51, 52], root=root, cache_path=cache)
    assert reads == ["00050_lr", "00052_lr", "00051_lr"]
    np.testing.assert_array_equal(Y[[2, 0]], X)


def test_parser_defaults_keep_the_old_protocol():
    p = J.get_parser()
    args = p.parse_args(["train", "--model", "inception1d"])
    assert (args.source, args.aug, args.val_every) == ("records100", "none", None)
    assert p.parse_args(["evaluate", "--model", "inception1d"]).source is None
    for bad in (["--source", "records250"], ["--aug", "flip"]):
        with pytest.raises(SystemExit):
            p.parse_args(["train", "--model", "inception1d", *bad])


# ------------------------------------------------------------------ augmentation
def _nonzero(shape, seed):
    rng = np.random.default_rng(seed)
    x = rng.normal(0.0, 0.5, size=shape)
    return np.where(np.abs(x) < 0.1, 0.1, x).astype(np.float32)


def test_shift_zero_fill_moves_without_wrap_around():
    X = _nonzero((4, 1000, 12), 1)
    out = J.shift_zero_fill(X, [3, -5, 0, 1000])
    assert out.dtype == X.dtype and out.shape == X.shape
    assert (out[0, :3] == 0).all()
    np.testing.assert_array_equal(out[0, 3:], X[0, :-3])
    assert (out[1, -5:] == 0).all()
    np.testing.assert_array_equal(out[1, :-5], X[1, 5:])
    np.testing.assert_array_equal(out[2], X[2])
    assert (out[3] == 0).all()
    with pytest.raises(ValueError):
        J.shift_zero_fill(X, [1, 2])


def test_draw_aug_ranges_and_determinism():
    n = 4000
    rng = np.random.default_rng(7)
    state = rng.bit_generator.state
    assert J.draw_aug(rng, n, "none") is None
    assert rng.bit_generator.state == state  # none draws nothing
    gain, shift = J.draw_aug(rng, n, "amp")
    assert gain.shape == (n,) and (shift == 0).all()
    assert gain.min() >= 0.9 and gain.max() < 1.1 and abs(gain.mean() - 1.0) < 0.01
    gain, shift = J.draw_aug(np.random.default_rng(7), n, "amp_shift")
    moved = shift[shift != 0]
    assert shift.dtype == np.int64 and 0.45 < len(moved) / n < 0.55
    assert set(moved.tolist()) == set(range(-25, 0)) | set(range(1, 26))
    assert abs(moved.mean()) < 1.5
    g2, s2 = J.draw_aug(np.random.default_rng(7), n, "amp_shift")
    np.testing.assert_array_equal(g2, gain)
    np.testing.assert_array_equal(s2, shift)
    with pytest.raises(ValueError):
        J.draw_aug(rng, n, "flip")


def test_train_batch_without_aug_is_the_old_batch():
    X = _nonzero((20, 1000, 12), 2)
    mean, std = 0.013, 0.21
    idx = np.random.default_rng(0).permutation(20)[:8]
    Xs = J._standardise(X, mean, std)
    old = np.ascontiguousarray(Xs[idx].transpose(0, 2, 1))  # layout: whole input
    new = J.train_batch(X, idx, mean, std)
    assert new.dtype == np.float32 and new.flags.c_contiguous
    np.testing.assert_array_equal(new, old)
    starts = np.random.default_rng(1).integers(0, 750, size=20)
    cols = np.arange(J.WINDOW)
    old = Xs[idx[:, None], starts[idx][:, None] + cols[None, :], :]
    old = np.ascontiguousarray(old.transpose(0, 2, 1))
    new = J.train_batch(X, idx, mean, std, starts)
    assert new.shape == (8, 12, J.WINDOW)
    np.testing.assert_array_equal(new, old)


def test_train_batch_gain_and_shift_in_mv_then_mask():
    full = _nonzero((6, 1000, 12), 3)
    # Condition-path stand-in: layout input that is NOT apply_layout(full), so an
    # unshifted record must keep its own layout row.
    lay = J.apply_layout(full + 0.01)
    mean, std = 0.013, 0.21
    idx = np.array([4, 1, 5, 0])
    gain = np.array([0.9, 1.05, 1.0999, 1.0])
    shift = np.array([0, 7, -12, 0])
    xs = J.train_batch(lay, idx, mean, std, aug=(gain, shift), Xfull=full)
    mv = xs.transpose(0, 2, 1).astype(np.float64) * std + mean  # back to mV
    expect = lay[idx].astype(np.float64)
    expect[1] = J.apply_layout(J.shift_zero_fill(full[idx[1:2]], [7]))[0]
    expect[2] = J.apply_layout(J.shift_zero_fill(full[idx[2:3]], [-12]))[0]
    expect *= gain[:, None, None]
    np.testing.assert_allclose(mv, expect, rtol=0, atol=1e-5)
    ref = J._standardise(expect, mean, std).transpose(0, 2, 1)
    np.testing.assert_allclose(xs, ref, rtol=1e-6, atol=1e-6)
    # Lead I (window 0-250): the shift is applied to the unmasked signal, then masked.
    lead_i = 0
    assert np.abs(mv[1, :7, lead_i]).max() < 1e-5  # zero fill
    np.testing.assert_allclose(mv[1, 7:250, lead_i], 1.05 * full[1, :243, 0], atol=1e-5)
    assert np.abs(mv[1, 250:, lead_i]).max() < 1e-5  # masked
    # A negative shift pulls samples from outside the printed window into it.
    np.testing.assert_allclose(mv[2, 238:250, lead_i], 1.0999 * full[5, 250:262, 0],
                               atol=1e-5)
    # A gain scales every lead of the record alike.
    ratio = mv[0][J.LAYOUT_MASK] / lay[4][J.LAYOUT_MASK]
    np.testing.assert_allclose(ratio, 0.9, rtol=1e-5)
    starts = np.zeros(6, np.int64)
    with pytest.raises(ValueError, match="layout"):
        J.train_batch(lay, idx, mean, std, starts, aug=(gain, shift), Xfull=full)


def test_frac_delay_matches_a_delayed_sinusoid():
    t = np.arange(1000, dtype=np.float64)
    x = np.stack([np.sin(2 * np.pi * f * t / 100 + 0.3) for f in (1.3, 7.0, 31.0)], 1)
    X = np.stack([x, x, x]).astype(np.float32)  # (3, 1000, 3)
    out = J.frac_delay(X, [0.37, -0.25, 0.0])
    assert out.dtype == X.dtype and out.shape == X.shape
    for i, d in ((0, 0.37), (1, -0.25)):
        want = np.stack(
            [np.sin(2 * np.pi * f * (t - d) / 100 + 0.3) for f in (1.3, 7.0, 31.0)], 1
        )
        # Interior: exact up to float32 and the edge padding, whose ringing decays
        # into the record from both ends (31 Hz: 0.1 % 100 samples in, 0.3 % at 50).
        np.testing.assert_allclose(out[i, 100:900], want[100:900], atol=2e-3)
        np.testing.assert_allclose(out[i, 50:950], want[50:950], atol=5e-3)
    np.testing.assert_array_equal(out[2], X[2])  # 0 = unchanged
    np.testing.assert_array_equal(J.frac_delay(X, [0, 0, 0]), X)
    with pytest.raises(ValueError):
        J.frac_delay(X, [0.1])


def test_draw_aug_frac_keeps_the_amp_shift_stream():
    n = 4000
    gain, shift = J.draw_aug(np.random.default_rng(7), n, "amp_shift")
    g2, s2, frac = J.draw_aug(np.random.default_rng(7), n, "amp_shift_frac")
    np.testing.assert_array_equal(g2, gain)
    np.testing.assert_array_equal(s2, shift)
    moved = frac[frac != 0]
    assert 0.45 < len(moved) / n < 0.55
    assert moved.min() >= -0.5 and moved.max() < 0.5 and abs(moved.mean()) < 0.02


def test_train_batch_frac_delay_before_shift_and_mask():
    full = _nonzero((6, 1000, 12), 4)
    lay = J.apply_layout(full + 0.01)
    mean, std = 0.013, 0.21
    idx = np.array([2, 3, 0])
    gain = np.array([1.0, 0.95, 1.08])
    shift = np.array([0, 5, 0])
    frac = np.array([0.3, -0.2, 0.0])
    xs = J.train_batch(lay, idx, mean, std, aug=(gain, shift, frac), Xfull=full)
    mv = xs.transpose(0, 2, 1).astype(np.float64) * std + mean
    expect = lay[idx].astype(np.float64)
    expect[0] = J.apply_layout(J.frac_delay(full[2:3], [0.3]))[0]
    expect[1] = J.apply_layout(J.shift_zero_fill(J.frac_delay(full[3:4], [-0.2]), [5]))[0]
    expect *= gain[:, None, None]
    np.testing.assert_allclose(mv, expect, rtol=0, atol=1e-5)


# ------------------------------------------------------------------ training smoke
N_TRAIN, N_VAL, N_TEST = 48, 12, 12


def _tiny_build(arch, num_classes=len(J.CLASSES), input_channels=J.N_LEADS, **kw):
    nn = torch.nn
    return nn.Sequential(
        nn.Conv1d(input_channels, 8, 9, padding=4), nn.ReLU(), nn.AdaptiveAvgPool1d(1),
        nn.Flatten(), nn.Dropout(0.3), nn.Linear(8, num_classes),
    )


def _toy_world(tmp_path, monkeypatch):
    """A fake db (folds 1-8 / 9 / 10), fake loaders of both sources and a tiny model."""
    n = N_TRAIN + N_VAL + N_TEST
    ids = np.arange(100, 100 + n)
    folds = [1 + i % 8 for i in range(N_TRAIN)] + [9] * N_VAL + [10] * N_TEST
    y = np.zeros((n, len(J.CLASSES)), int)
    y[np.arange(n), np.arange(n) % len(J.CLASSES)] = 1
    y[np.arange(n) % 3 == 0, 4] = 1
    db = pd.DataFrame(y, index=pd.Index(ids, name="ecg_id"), columns=list(J.CLASSES))
    db["strat_fold"] = folds
    db["patient_id"] = ids + 0.0
    sig = {
        "records100": {int(e): _nonzero((1000, 12), int(e)) for e in ids},
        "full": {int(e): _nonzero((1000, 12), 1000 + int(e)) for e in ids},
    }
    sig["layout"] = {e: J.apply_layout(x + 0.02) for e, x in sig["full"].items()}
    calls = []

    def load_signals_100(ecg_ids, root=None, cache_path=None):
        calls.append(("records100", [int(e) for e in ecg_ids]))
        return np.stack([sig["records100"][int(e)] for e in ecg_ids])

    def load_inputs_500(ecg_ids, kind, root=None, cache_path=None):
        calls.append((kind, [int(e) for e in ecg_ids]))
        return np.stack([sig[kind][int(e)] for e in ecg_ids])

    monkeypatch.setattr(J, "load_ptbxl_db", lambda root=None: db)
    monkeypatch.setattr(J, "load_signals_100", load_signals_100)
    monkeypatch.setattr(J, "load_inputs_500", load_inputs_500)
    monkeypatch.setattr(J, "build_model", _tiny_build)
    monkeypatch.setattr(J, "init_weights", lambda model: model)
    wdir = tmp_path / "w"
    base = ["--ptbxl", str(tmp_path), "--weights_dir", str(wdir)]
    return db, sig, calls, wdir, base


def _train_argv(base, tag, *extra):
    return [*base, "train", "--model", "xresnet1d50", "--epochs", "3", "--bs", "16",
            "--tag", tag, *extra]


def test_train_records500_amp_shift_fields_and_resume(tmp_path, monkeypatch):
    db, sig, calls, wdir, base = _toy_world(tmp_path, monkeypatch)
    opts = ["--input", "layout", "--source", "records500", "--aug", "amp_shift",
            "--val_every", "1"]
    assert J.main(_train_argv(base, "a", *opts)) == 0
    final = torch.load(wdir / "xresnet1d50_layout_a_final.pt", weights_only=False)
    assert (final["source"], final["aug"], final["val_every"]) == (
        "records500", "amp_shift", 1
    )
    train_ids, val_ids = final["train_ids"], final["val_ids"]
    assert len(train_ids) == N_TRAIN and len(val_ids) == N_VAL
    # Standardisation from the unmasked records500 inputs; layout from the condition
    # path itself (load_inputs_500 'layout'), never from records100.
    full_tr = np.stack([sig["full"][e] for e in train_ids])
    assert (final["mean"], final["std"]) == J._global_mean_std(full_tr)
    assert calls == [("full", train_ids), ("layout", train_ids + val_ids)]
    last = torch.load(wdir / "xresnet1d50_layout_a_last.pt", weights_only=False)
    assert last["config"]["source"] == "records500" and last["source"] == "records500"
    assert last["config"]["aug"] == "amp_shift" and last["config"]["val_every"] == 1
    fresh = np.random.default_rng([0, 1]).bit_generator.state
    assert last["rng_aug"] != fresh  # augmentation drew from its own stream
    log = pd.read_csv(wdir / "xresnet1d50_layout_a_log.csv")
    assert np.isfinite(log["val_macro_auroc"]).all() and len(log) == 3

    # Interrupted after epoch 1 and resumed = uninterrupted (weights and losses).
    assert J.main(_train_argv(base, "b", *opts, "--max_minutes", "0")) == 0
    assert torch.load(wdir / "xresnet1d50_layout_b_last.pt", weights_only=False)[
        "epoch"
    ] == 1
    assert not os.path.exists(wdir / "xresnet1d50_layout_b_final.pt")
    assert J.main(_train_argv(base, "b", *opts, "--resume")) == 0
    resumed = torch.load(wdir / "xresnet1d50_layout_b_final.pt", weights_only=False)
    for k, v in final["state_dict"].items():
        assert torch.equal(v, resumed["state_dict"][k]), k
    log_b = pd.read_csv(wdir / "xresnet1d50_layout_b_log.csv")
    np.testing.assert_array_equal(log_b["train_loss"], log["train_loss"])

    # --resume refuses another source or augmentation.
    other = ["--input", "layout", "--source", "records100", "--aug", "amp_shift"]
    with pytest.raises(SystemExit, match="--source records500"):
        J.main(_train_argv(base, "b", *other, "--resume"))
    other = ["--input", "layout", "--source", "records500", "--aug", "amp"]
    with pytest.raises(SystemExit, match="--aug amp_shift"):
        J.main(_train_argv(base, "b", *other, "--resume"))
    with pytest.raises(SystemExit, match="layout"):
        J.main(_train_argv(base, "c", "--aug", "amp_shift"))

    # Judge.load exposes the source; the augmentation is in meta.
    judge = J.Judge.load("xresnet1d50_layout_a", weights_dir=str(wdir))
    assert judge.source == "records500" and judge.meta["aug"] == "amp_shift"
    assert judge.input_mode == "layout"


def test_train_records100_default_keeps_the_old_protocol(tmp_path, monkeypatch):
    db, sig, calls, wdir, base = _toy_world(tmp_path, monkeypatch)
    assert J.main(_train_argv(base, "d", "--input", "layout")) == 0
    final = torch.load(wdir / "xresnet1d50_layout_d_final.pt", weights_only=False)
    fields = (final["source"], final["aug"], final["val_every"])
    assert fields == ("records100", "none", 5)
    train_ids = final["train_ids"]
    assert [c[0] for c in calls] == ["records100"]
    x_tr = np.stack([sig["records100"][e] for e in train_ids])
    assert (final["mean"], final["std"]) == J._global_mean_std(x_tr)
    last = torch.load(wdir / "xresnet1d50_layout_d_last.pt", weights_only=False)
    # aug none never touches its generator.
    assert last["rng_aug"] == np.random.default_rng([0, 1]).bit_generator.state
    log = pd.read_csv(wdir / "xresnet1d50_layout_d_log.csv")
    assert np.isfinite(log["val_loss"]).tolist() == [False, False, True]  # last epoch
    # A pre-field run (no source/aug/val_every, no rng_aug) resumes as records100/none.
    ck = torch.load(wdir / "xresnet1d50_layout_d_last.pt", weights_only=False)
    ck["epoch"] = 2
    ck["log"] = ck["log"][:2]
    for key in ("source", "aug", "val_every"):
        ck["config"].pop(key)
    for key in ("rng_aug", "source", "aug"):
        ck.pop(key)
    torch.save(ck, wdir / "xresnet1d50_layout_d_last.pt")
    with pytest.raises(SystemExit, match="--source records100"):
        J.main(_train_argv(base, "d", "--input", "layout", "--source", "records500",
                           "--resume"))
    assert J.main(_train_argv(base, "d", "--input", "layout", "--resume")) == 0
    old = J.Judge.load("xresnet1d50_layout_d", weights_dir=str(wdir))
    assert old.source == "records100"


# ------------------------------------------------------------------ evaluation
def _save_final(path, model, **extra):
    ck = {
        "format": J.CKPT_FORMAT,
        "arch": "xresnet1d50",
        "arch_kwargs": {},
        "classes": list(J.CLASSES),
        "state_dict": model.state_dict(),
        "mean": 0.01,
        "std": 0.2,
        "fs": J.FS,
        "input_mode": "layout",
    }
    ck.update(extra)
    torch.save(ck, path)


def test_fold_logits_source_selects_inputs_and_cache_file(tmp_path, monkeypatch):
    db, sig, calls, wdir, base = _toy_world(tmp_path, monkeypatch)
    torch.manual_seed(0)
    model = _tiny_build("x").eval()
    judge = J.Judge(model, 0.01, 0.2, J.CLASSES, "toy_layout", "cpu", "layout",
                    source="records500")
    judge.weights_sha256 = "abc"
    ids = db.index[db["strat_fold"] == 10].to_numpy()
    z, y = J.fold_logits(judge, db, 10, ids, cache_dir=str(tmp_path / "c"))
    x500 = np.stack([sig["layout"][int(e)] for e in ids])
    np.testing.assert_allclose(z, judge.predict_logits(x500), rtol=1e-5, atol=1e-6)
    assert os.path.exists(tmp_path / "c" / "judge_toy_layout_layout_fold10_r500.npz")
    z100, _ = J.fold_logits(
        judge, db, 10, ids, str(tmp_path / "c"), source="records100"
    )
    x100 = J.apply_layout(np.stack([sig["records100"][int(e)] for e in ids]))
    np.testing.assert_allclose(z100, judge.predict_logits(x100), rtol=1e-5, atol=1e-6)
    assert os.path.exists(tmp_path / "c" / "judge_toy_layout_layout_fold10.npz")
    assert [c[0] for c in calls] == ["layout", "records100"]
    J.fold_logits(judge, db, 10, ids, cache_dir=str(tmp_path / "c"))  # cache hit
    assert len(calls) == 2
    with pytest.raises(ValueError, match="source"):
        J.fold_logits(judge, db, 10, ids, str(tmp_path / "c"), source="records250")
    with pytest.raises(ValueError, match="source"):
        J.Judge(model, 0.0, 1.0, source="records250")


def test_evaluate_rows_are_kept_per_model_and_source(tmp_path, monkeypatch):
    db, sig, calls, wdir, base = _toy_world(tmp_path, monkeypatch)
    os.makedirs(wdir)
    torch.manual_seed(1)
    _save_final(wdir / "toy_layout_final.pt", _tiny_build("x"), source="records500")
    out = tmp_path / "judge_fold10.csv"
    old_cols = ["model", "class", "n_test", "n_pos", "auroc", "ci95_lo", "ci95_hi",
                "ci90_lo", "ci90_hi", "threshold_logit_fold9", "sens_fold10",
                "spec_fold10", "pass_090", "n_boot_valid"]
    old = pd.DataFrame(
        [[m, c, 9, 1, 0.5, 0, 1, 0, 1, 0.0, 0.5, 0.5, "", 10]
         for m in ("other", "toy_layout") for c in list(J.CLASSES) + ["macro"]],
        columns=old_cols,
    )
    old.to_csv(out, index=False)  # schema before the source column
    ev = [*base, "evaluate", "--model", "toy_layout", "--out", str(out),
          "--cache_dir", str(tmp_path / "c"), "--n_boot", "20"]
    assert J.main(ev) == 0
    df = pd.read_csv(out)
    assert list(df.columns[:3]) == ["model", "source", "class"]
    got = df.groupby(["model", "source"]).size().to_dict()
    assert got == {("other", "records100"): 6, ("toy_layout", "records100"): 6,
                   ("toy_layout", "records500"): 6}
    tag = J._root_tag(str(tmp_path))  # --ptbxl is not the default root
    for fold in (9, 10):
        name = f"judge_toy_layout_layout_fold{fold}_r500{tag}.npz"
        assert os.path.exists(tmp_path / "c" / name)
    r500 = df[(df["model"] == "toy_layout") & (df["source"] == "records500")]
    assert r500["n_test"].eq(N_TEST).all()
    # The same judge on records100 replaces only the (toy_layout, records100) rows.
    assert J.main([*ev, "--source", "records100"]) == 0
    df2 = pd.read_csv(out)
    assert df2.groupby(["model", "source"]).size().to_dict() == got
    r100 = df2[(df2["model"] == "toy_layout") & (df2["source"] == "records100")]
    assert r100["n_test"].eq(N_TEST).all()  # new rows (the old ones had n_test 9)
    pd.testing.assert_frame_equal(
        df2[df2["source"] == "records500"].reset_index(drop=True),
        r500.reset_index(drop=True),
    )
    assert (df2.loc[df2["model"] == "other", "n_test"] == 9).all()
    assert os.path.exists(tmp_path / "c" / f"judge_toy_layout_layout_fold10{tag}.npz")
