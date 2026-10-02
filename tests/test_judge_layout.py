"""Tests of the layout-aware judge path in scripts/judge.py (--input layout).

Needs torch; the model tests also need the benchmark code in data/judge/code (no
weights, no PTB-XL).
"""
import os

import numpy as np
import pandas as pd
import pytest

torch = pytest.importorskip("torch")

from scripts import fidelity  # noqa: E402
from scripts import judge as J  # noqa: E402

needs_code = pytest.mark.skipif(
    not os.path.isfile(os.path.join(J.BENCH_CODE, "models", "xresnet1d.py")),
    reason="benchmark code (data/judge/code) not present",
)


def _signals(n, seed=0):
    rng = np.random.default_rng(seed)
    # |x| >= 0.1 everywhere, so a zero in the output can only come from the mask.
    x = rng.normal(0.0, 0.5, size=(n, J.N_SAMPLES, J.N_LEADS))
    return np.where(np.abs(x) < 0.1, 0.1, x)


def _tiny_model(seed=0):
    torch.manual_seed(seed)
    model = J.init_weights(J.build_model("xresnet1d50", len(J.CLASSES)))
    # BatchNorm running statistics of one batch, so the random logits stay O(1).
    for m in model.modules():
        if isinstance(m, torch.nn.modules.batchnorm._BatchNorm):
            m.momentum = None
    with torch.no_grad():
        model.train()(torch.randn(16, J.N_LEADS, J.N_SAMPLES))
    return model.eval()


def _judge(model, mode, name="toy"):
    return J.Judge(model, 0.01, 0.2, J.CLASSES, name, "cpu", input_mode=mode)


# CPU convolutions of different batch shapes differ in float32 rounding; the random
# model's logits are O(10), so compare with a small relative and absolute tolerance.
def _close(a, b):
    np.testing.assert_allclose(a, b, rtol=1e-4, atol=1e-3)


# ------------------------------------------------------------------ layout mask
def test_layout_mask_equals_fidelity_layout_mask():
    ref = fidelity.layout_mask(1000, 100)
    assert J.LAYOUT_MASK.dtype == bool and J.LAYOUT_MASK.shape == (1000, 12)
    np.testing.assert_array_equal(J.LAYOUT_MASK, ref)
    np.testing.assert_array_equal(J.layout_mask(), ref)
    assert not J.LAYOUT_MASK.flags.writeable


def test_layout_windows_per_lead():
    counts = dict(zip(J.LEADS, J.LAYOUT_MASK.sum(axis=0)))
    assert counts["II"] == 1000
    assert all(counts[lead] == 250 for lead in J.LEADS if lead != "II")
    col = {lead: j for j, lead in enumerate(J.LEADS)}
    assert J.LAYOUT_MASK[0:250, col["I"]].all()
    assert not J.LAYOUT_MASK[250:, col["I"]].any()
    assert J.LAYOUT_MASK[250:500, col["aVF"]].all()
    assert J.LAYOUT_MASK[500:750, col["V3"]].all()
    assert J.LAYOUT_MASK[750:1000, col["V6"]].all()
    assert not J.LAYOUT_MASK[:750, col["V6"]].any()


def test_apply_layout_zeroes_exactly_outside_the_windows():
    x = _signals(3)
    before = x.copy()
    out = J.apply_layout(x)
    assert out.dtype == np.float32 and out.shape == x.shape
    m = np.broadcast_to(J.LAYOUT_MASK, x.shape)
    np.testing.assert_array_equal(out[m], x.astype(np.float32)[m])
    assert (out[~m] == 0).all()
    assert (out[m] != 0).all()
    np.testing.assert_array_equal(x, before)  # input untouched
    np.testing.assert_array_equal(out, (x * J.LAYOUT_MASK).astype(np.float32))
    # one record, float32 in, and non-finite values outside the windows
    y = x[0].astype(np.float32)
    y[~J.LAYOUT_MASK] = np.nan
    one = J.apply_layout(y)
    assert one.shape == (1000, 12)
    np.testing.assert_array_equal(one, out[0])
    with pytest.raises(ValueError):
        J.apply_layout(np.zeros((2, 500, 12)))


def test_run_names():
    assert J._run_name("xresnet1d50", None) == "xresnet1d50"
    assert J._run_name("xresnet1d50", "smoke") == "xresnet1d50_smoke"
    assert J._run_name("xresnet1d50", None, "layout") == "xresnet1d50_layout"
    assert J._run_name("xresnet1d50", "smoke", "layout") == "xresnet1d50_layout_smoke"
    with pytest.raises(ValueError):
        J._run_name("xresnet1d50", None, "crop")


def test_parser_input_default_is_window():
    p = J.get_parser()
    assert p.parse_args(["train", "--model", "xresnet1d50"]).input == "window"
    args = p.parse_args(["train", "--model", "xresnet1d50", "--input", "layout"])
    assert args.input == "layout"
    with pytest.raises(SystemExit):
        p.parse_args(["train", "--model", "xresnet1d50", "--input", "crop"])


# ------------------------------------------------------------------ layout judge
@needs_code
def test_layout_predict_logits_shape_batch_invariance_and_one_pass():
    model = _tiny_model()
    judge = _judge(model, "layout")
    x = J.apply_layout(_signals(5, seed=1))
    z = judge.predict_logits(x)
    assert z.shape == (5, len(J.CLASSES)) and np.isfinite(z).all()
    for bs in (1, 2, 3):
        _close(judge.predict_logits(x, batch_size=bs), z)
    _close(judge.predict_logits(x[2]), z[2:3])
    # one forward pass over the whole standardised (N, 12, 1000) input
    xs = ((x.astype(np.float64) - 0.01) / 0.2).astype(np.float32).transpose(0, 2, 1)
    with torch.no_grad():
        ref = model(torch.from_numpy(np.ascontiguousarray(xs))).numpy()
    _close(z, ref)
    p = judge.predict_proba(x)
    np.testing.assert_allclose(p, 1.0 / (1.0 + np.exp(-z.astype(np.float64))))
    # the window path of the same weights is the max over 7 windows, not the one pass
    zw = _judge(model, "window").predict_logits(x)
    assert zw.shape == z.shape and np.abs(zw - z).max() > 0.1
    with pytest.raises(ValueError):
        _judge(model, "crop")


def _save_ckpt(path, model, **extra):
    ck = {
        "format": J.CKPT_FORMAT,
        "arch": "xresnet1d50",
        "arch_kwargs": J.ARCHS["xresnet1d50"][2],
        "classes": list(J.CLASSES),
        "state_dict": model.state_dict(),
        "mean": 0.01,
        "std": 0.2,
        "fs": J.FS,
        "window": J.WINDOW,
        "window_starts": list(J.WINDOW_STARTS),
    }
    ck.update(extra)
    torch.save(ck, path)


@needs_code
def test_checkpoint_input_mode_field(tmp_path):
    model = _tiny_model(seed=3)
    _save_ckpt(tmp_path / "old_final.pt", model)  # pre-field checkpoint
    _save_ckpt(tmp_path / "lay_final.pt", model, input_mode="layout")
    _save_ckpt(tmp_path / "bad_final.pt", model, input_mode="crop")
    old = J.Judge.load("old", weights_dir=str(tmp_path))
    assert old.input_mode == "window" and "input_mode" not in old.meta
    lay = J.Judge.load("lay", weights_dir=str(tmp_path))
    assert lay.input_mode == "layout" and lay.meta["input_mode"] == "layout"
    x = J.apply_layout(_signals(4, seed=2))
    _close(old.predict_logits(x), _judge(model, "window").predict_logits(x))
    _close(lay.predict_logits(x), _judge(model, "layout").predict_logits(x))
    with pytest.raises(ValueError):
        J.Judge.load("bad", weights_dir=str(tmp_path))


@needs_code
def test_fold_logits_layout_masks_inputs_and_uses_its_own_cache(tmp_path, monkeypatch):
    ids = np.array([101, 102, 103], np.int64)
    raw = _signals(3, seed=4).astype(np.float32)
    calls = []

    def fake_load(ecg_ids, root=J.DEFAULT_PTBXL, cache_path=None):
        calls.append(list(ecg_ids))
        return raw[np.searchsorted(ids, ecg_ids)]

    monkeypatch.setattr(J, "load_signals_100", fake_load)
    db = pd.DataFrame(np.eye(3, len(J.CLASSES), dtype=int), index=ids, columns=J.CLASSES)
    model = _tiny_model(seed=5)
    lay = _judge(model, "layout", name="toy_layout")
    lay.weights_sha256 = "abc"
    z, y = J.fold_logits(lay, db, 10, ids, cache_dir=str(tmp_path))
    _close(z, lay.predict_logits(J.apply_layout(raw)))
    np.testing.assert_array_equal(y, db.to_numpy())
    assert os.path.exists(tmp_path / "judge_toy_layout_layout_fold10.npz")
    z2, _ = J.fold_logits(lay, db, 10, ids, cache_dir=str(tmp_path))  # from the cache
    np.testing.assert_array_equal(z2, z)
    assert len(calls) == 1
    win = _judge(model, "window", name="toy")
    win.weights_sha256 = "def"
    zw, _ = J.fold_logits(win, db, 10, ids, cache_dir=str(tmp_path))
    _close(zw, win.predict_logits(raw))  # a window judge sees the unmasked records
    assert os.path.exists(tmp_path / "judge_toy_full_fold10.npz")
