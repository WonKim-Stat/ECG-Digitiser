# PTB-XL superdiagnostic "judge" classifier for the diagnosis-fidelity harness (plan A1).
#
# Re-runs the fastai v1 protocol of Strodthoff et al. (ecg_ptbxl_benchmarking) at 100 Hz
# in plain PyTorch: strat folds 1-8 train, fold 9 validation and thresholds, fold 10 test;
# one random 2.5 s crop per record and epoch for training (fit_one_cycle(50, 1e-2), AdamW,
# batch 128); prediction = 7 windows of 2.5 s (stride 1.25 s), max over windows.
# One deliberate deviation: train and validation leave out the local evaluation records
# (HELD_OUT_ECG_IDS = ecg_ids 1-32, the gen_eval/quasi_real records) and every other
# record of their patients (held_out_patient_records); fold 10 is the benchmark's.
#
# The network definitions are imported at runtime from the GPL-3.0 clone in
# data/judge/code; none of that code is copied here. A minimal in-memory stand-in for the
# fastai v1 helpers they import is installed only while importing them, and only when
# fastai is not importable.
#
# Supplementary layout-aware judge (train --input layout, run <model>_layout[_<tag>]):
# same architecture, split, standardisation (global mean/std of the unmasked training
# values), optimiser and one-cycle schedule, but every training, validation and test input
# is the 3x4 + rhythm II paper layout of the 10 s record (apply_layout: each lead zero
# outside its printed 2.5 s window, II the full 10 s) and the model sees the whole
# 1000-sample input (no random crop, no sliding windows); prediction is one forward pass.
# Its checkpoint carries input_mode "layout" (checkpoints without the field are "window").
# The judge never masks internally: a layout judge is only meaningful on layout-masked
# input (apply_layout(X), or scripts/fidelity's layout-prepared signals).
#
# Signal source (train --source, stored in the checkpoint; checkpoints without the field
# are records100): records100 (default) = the official 100 Hz files; records500 = the
# digitisation condition path of scripts/fidelity (records500 -> NaN layout mask at 500 Hz
# -> prepare_for_judge to 100 Hz; load_inputs_500), so a records500 layout judge is
# trained, validated and thresholded on exactly the inputs the conditions give it. The
# standardisation of a records500 run comes from its unmasked ("full") training inputs.
# Optional training augmentation (--aug, default none = the protocol above): amp = one
# gain U(0.9, 1.1) per record over all leads; amp_shift (layout only) = amp plus, for
# half of the records, a zero-filled shift of the unmasked 10 s input by 1-25 samples
# either way before the 100 Hz layout mask; amp_shift_frac = amp_shift plus, for half of
# the records (drawn independently), a global fractional delay U(-0.5, 0.5) samples
# (+-5 ms; FFT phase shift of the edge-padded unmasked input) before the integer shift
# and the mask: the records500 path is records100 delayed by 0.12 samples (1.2 ms), and
# a judge should not flip on such a delay. --val_every N validates every N epochs
# (default 5).
#
#   python scripts/judge.py params
#   python scripts/judge.py train --model inception1d --device mps --max_minutes 9
#   python scripts/judge.py train --model inception1d --device mps --max_minutes 9 \
#       --resume
#   python scripts/judge.py evaluate --model inception1d
#   python scripts/judge.py train --model xresnet1d50 --input layout --device mps
#   python scripts/judge.py evaluate --model xresnet1d50_layout   # mode from checkpoint
#   python scripts/judge.py train --model inception1d --input layout --source records500 \
#       --aug amp_shift --val_every 1 --tag r500 --device mps
#   python scripts/judge.py evaluate --model inception1d_layout --tag r500  # its source
#   python scripts/judge.py check-benchmark     # end-to-end check against exp0 outputs
import argparse
import ast
import hashlib
import importlib
import os
import pickle
import random
import re
import sys
import time
import types
import typing
import warnings

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
BENCH_DIR = os.path.join(REPO_ROOT, "data", "judge")
BENCH_CODE = os.path.join(BENCH_DIR, "code")

CLASSES = ("CD", "HYP", "MI", "NORM", "STTC")  # MultiLabelBinarizer (alphabetical) order
FS = 100
N_SAMPLES = 1000
N_LEADS = 12
LEADS = ("I", "II", "III", "aVR", "aVL", "aVF", "V1", "V2", "V3", "V4", "V5", "V6")
# Local gen_eval / quasi_real records: never used for training or thresholds.
HELD_OUT_ECG_IDS = tuple(range(1, 33))

WINDOW = 250  # 2.5 s model input
STRIDE = 125  # benchmark chunk stride for validation/prediction
WINDOW_STARTS = tuple(range(0, N_SAMPLES - WINDOW + 1, STRIDE))  # 0, 125, ..., 750
TRAIN_FOLDS = (1, 2, 3, 4, 5, 6, 7, 8)
VAL_FOLD = 9
TEST_FOLD = 10

# Judge input modes: "window" = benchmark protocol (2.5 s crops / 7 sliding windows),
# "layout" = whole 10 s input in the 3x4 + rhythm II paper layout (see apply_layout).
INPUT_MODES = ("window", "layout")
# Signal sources: the official 100 Hz files, or records500 through the condition path.
SOURCES = ("records100", "records500")
# Prepared records500 input kinds (= scripts/fidelity.INPUT_KINDS): unmasked, 3x4 masked.
INPUT_KINDS = ("full", "layout")
# Training augmentation (mV space, training batches only).
AUGS = ("none", "amp", "amp_shift", "amp_shift_frac")
AMP_RANGE = (0.9, 1.1)  # per-record gain, uniform, all leads
SHIFT_MAX = 25  # amp_shift: shift in samples at 100 Hz (0.25 s), uniform over +-1..25
SHIFT_PROB = 0.5  # amp_shift: probability that a record is shifted
SHIFT_AUGS = ("amp_shift", "amp_shift_frac")  # rebuild records from the full input
FRAC_MAX = 0.5  # amp_shift_frac: fractional delay in samples at 100 Hz, uniform +-0.5
FRAC_PROB = 0.5  # amp_shift_frac: probability that a record is delayed
FRAC_PAD = 100  # amp_shift_frac: edge padding (samples) around the FFT phase shift
# 3x4 + rhythm II layout at 100 Hz: [start, end) samples of each lead's printed window
# (I/III 0-2.5 s, aVR/aVL/aVF 2.5-5 s, V1-V3 5-7.5 s, V4-V6 7.5-10 s, II the 10 s rhythm
# strip). Equals scripts/fidelity.layout_mask(1000, 100) (checked in tests).
LAYOUT_WINDOWS = {
    "I": (0, 250),
    "II": (0, 1000),
    "III": (0, 250),
    "aVR": (250, 500),
    "aVL": (250, 500),
    "aVF": (250, 500),
    "V1": (500, 750),
    "V2": (500, 750),
    "V3": (500, 750),
    "V4": (750, 1000),
    "V5": (750, 1000),
    "V6": (750, 1000),
}


# Bool (N_SAMPLES, N_LEADS) mask, True inside each lead's printed window.
def layout_mask():
    mask = np.zeros((N_SAMPLES, N_LEADS), dtype=bool)
    for j, lead in enumerate(LEADS):
        start, end = LAYOUT_WINDOWS[lead]
        mask[start:end, j] = True
    return mask


LAYOUT_MASK = layout_mask()
LAYOUT_MASK.flags.writeable = False


# X * LAYOUT_MASK as float32: (N, 1000, 12) or (1000, 12) mV at 100 Hz -> a new array,
# X (cast to float32) inside each lead's window and exactly 0 outside it.
def apply_layout(X):
    X = np.asarray(X)
    if X.ndim not in (2, 3) or X.shape[-2:] != (N_SAMPLES, N_LEADS):
        raise ValueError(f"expected (N, {N_SAMPLES}, {N_LEADS}), got {X.shape}")
    if X.dtype.kind not in "fiu":
        raise ValueError(f"numeric input required, got {X.dtype}")
    out = np.zeros(X.shape, np.float32)
    np.copyto(out, X, casting="unsafe", where=LAYOUT_MASK)
    return out

# Benchmark fastai_model._get_learner arguments (kernel_size 5; inception uses 8 * 5).
ARCHS = {
    "inception1d": (
        "inception1d",
        "inception1d",
        {"use_residual": True, "ps_head": 0.5, "lin_ftrs_head": [128], "kernel_size": 40},
    ),
    "xresnet1d50": (
        "xresnet1d",
        "xresnet1d50",
        {"kernel_size": 5, "ps_head": 0.5, "lin_ftrs_head": [128]},
    ),
    "xresnet1d101": (
        "xresnet1d",
        "xresnet1d101",
        {"kernel_size": 5, "ps_head": 0.5, "lin_ftrs_head": [128]},
    ),
}
TRAINABLE = ("inception1d", "xresnet1d50")

DEFAULT_PTBXL = "data/ptbxl"
DEFAULT_WEIGHTS_DIR = "data/judge/weights"
DEFAULT_CACHE_DIR = "data/fidelity/cache"
DEFAULT_EVAL_CSV = "data/fidelity/judge_fold10.csv"
SIGNAL_CACHE_NAME = "ptbxl_records100.npz"
INPUTS500_CACHE_NAME = "ptbxl_records500_{kind}.npz"
CKPT_FORMAT = "ecg-digitiser-judge-v1"


# Relative paths are taken relative to the repository root, not the working directory.
def _resolve(path):
    return path if os.path.isabs(path) else os.path.join(REPO_ROOT, path)


# scripts/fidelity, imported lazily (it imports this module lazily too; needs no torch).
def _fidelity_module():
    try:
        from scripts import fidelity
    except ImportError:
        if REPO_ROOT not in sys.path:
            sys.path.insert(0, REPO_ROOT)
        from scripts import fidelity
    return fidelity


def _sha256(path):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def _atomic_torch_save(obj, path):
    tmp = path + ".tmp"
    torch.save(obj, tmp)
    os.replace(tmp, path)


# ---------------------------------------------------------------------------------------
# Benchmark architectures (imported at runtime, never copied).
# ---------------------------------------------------------------------------------------


# In-memory stand-in for the three fastai v1 helpers the benchmark models import.
def _fastai_shim_modules():
    def listify(p=None, q=None):
        if p is None:
            p = []
        elif isinstance(p, str):
            p = [p]
        elif not isinstance(p, typing.Iterable):
            p = [p]
        return list(p)

    def bn_drop_lin(n_in, n_out, bn=True, p=0.0, actn=None):
        layers = [nn.BatchNorm1d(n_in)] if bn else []
        if p != 0:
            layers.append(nn.Dropout(p))
        layers.append(nn.Linear(n_in, n_out))
        if actn is not None:
            layers.append(actn)
        return layers

    class Flatten(nn.Module):
        def __init__(self, full=False):
            super().__init__()
            self.full = full

        def forward(self, x):
            return x.view(-1) if self.full else x.view(x.size(0), -1)

    fa = types.ModuleType("fastai")
    fl = types.ModuleType("fastai.layers")
    fc = types.ModuleType("fastai.core")
    for m in (fl, fc):
        m.Optional = typing.Optional
        m.Collection = typing.Collection
        m.Floats = typing.Union[float, typing.Collection[float]]
        m.listify = listify
        m.bn_drop_lin = bn_drop_lin
        m.Flatten = Flatten
        m.__all__ = [
            "Optional", "Collection", "Floats", "listify", "bn_drop_lin", "Flatten"
        ]
    fa.layers = fl
    fa.core = fc
    return {"fastai": fa, "fastai.layers": fl, "fastai.core": fc}


def _real_fastai_v1():
    try:
        layers = importlib.import_module("fastai.layers")
        importlib.import_module("fastai.core")
    except ImportError:
        return False
    return hasattr(layers, "bn_drop_lin")


_BENCH_MODULES = {}


# Import models.{basic_conv1d,xresnet1d,inception1d} from the benchmark clone. The repo
# root has an unrelated `models/` folder, so any `models*` entries are set aside during
# the import and restored afterwards; no bytecode is written into the clone.
def _benchmark_modules():
    if _BENCH_MODULES:
        return _BENCH_MODULES
    if not os.path.isfile(os.path.join(BENCH_CODE, "models", "xresnet1d.py")):
        raise FileNotFoundError(f"benchmark code not found under {BENCH_CODE}")
    shim = {} if _real_fastai_v1() else _fastai_shim_modules()

    def ours(k):
        return k in shim or k == "models" or k.startswith("models.")

    saved = {k: sys.modules[k] for k in list(sys.modules) if ours(k)}
    for k in saved:
        del sys.modules[k]
    sys.modules.update(shim)
    old_dont_write = sys.dont_write_bytecode
    sys.dont_write_bytecode = True
    sys.path.insert(0, BENCH_CODE)
    try:
        mods = {
            n: importlib.import_module("models." + n)
            for n in ("basic_conv1d", "xresnet1d", "inception1d")
        }
    finally:
        sys.path.remove(BENCH_CODE)
        sys.dont_write_bytecode = old_dont_write
        for k in [k for k in list(sys.modules) if ours(k)]:
            del sys.modules[k]
        sys.modules.update(saved)
    _BENCH_MODULES.update(mods)
    return _BENCH_MODULES


# Build one benchmark architecture (untrained, PyTorch default init).
def build_model(arch, num_classes=len(CLASSES), input_channels=N_LEADS, **overrides):
    if arch not in ARCHS:
        raise ValueError(f"unknown architecture {arch!r}; choose from {sorted(ARCHS)}")
    module_name, fn_name, kwargs = ARCHS[arch]
    kw = {k: (list(v) if isinstance(v, list) else v) for k, v in kwargs.items()}
    kw.update(overrides)
    fn = getattr(_benchmark_modules()[module_name], fn_name)
    return fn(num_classes=num_classes, input_channels=input_channels, **kw)


# Benchmark fit() re-initialises every model with basic_conv1d.weight_init.
def init_weights(model):
    model.apply(_benchmark_modules()["basic_conv1d"].weight_init)
    return model


def count_parameters(model):
    return int(sum(p.numel() for p in model.parameters()))


# ---------------------------------------------------------------------------------------
# PTB-XL labels and signals.
# ---------------------------------------------------------------------------------------


# ptbxl_database.csv with the benchmark 'superdiagnostic' aggregation: the
# diagnostic_class of every scp_codes key that is a diagnostic statement (any
# likelihood, including 0).
def load_ptbxl_db(root=DEFAULT_PTBXL):
    root = _resolve(root)
    db = pd.read_csv(os.path.join(root, "ptbxl_database.csv"), index_col="ecg_id")
    db["scp_codes"] = db["scp_codes"].apply(ast.literal_eval)
    statements = pd.read_csv(os.path.join(root, "scp_statements.csv"), index_col=0)
    diagnostic = statements[statements["diagnostic"] == 1.0]
    code_class = {
        code: cls
        for code, cls in diagnostic["diagnostic_class"].items()
        if isinstance(cls, str)
    }
    db["superdiagnostic"] = db["scp_codes"].apply(
        lambda codes: sorted({code_class[c] for c in codes if c in code_class})
    )
    found = set().union(*db["superdiagnostic"])
    if found != set(CLASSES):
        raise ValueError(f"superdiagnostic classes {sorted(found)} != {list(CLASSES)}")
    for cls in CLASSES:
        db[cls] = db["superdiagnostic"].apply(lambda s, c=cls: int(c in s)).astype(int)
    return db


def _labelled(db):
    return db[list(CLASSES)].sum(axis=1) > 0


# ecg_ids of every record whose patient has a record in HELD_OUT_ECG_IDS (those ids
# included). PTB-XL folds are patient-stratified, so siblings share the held-out record's
# fold; with v1.0.3 the siblings are 34 (fold 9, unlabelled), 39 (fold 9) and 200 (fold 8).
def held_out_patient_records(db):
    patients = db.loc[db.index.isin(HELD_OUT_ECG_IDS), "patient_id"]
    return np.sort(db.index[db["patient_id"].isin(patients)].to_numpy())


# Benchmark selection (>= 1 superclass) split by strat_fold. The held-out records and all
# records of their patients are excluded from train and validation only, so neither the
# weights nor the fold-9 thresholds have seen a held-out patient.
def split_ids(db, limit=None, seed=0):
    has = _labelled(db)
    held = db.index.isin(held_out_patient_records(db))
    train = db.index[has & db["strat_fold"].isin(TRAIN_FOLDS) & ~held].to_numpy()
    val = db.index[has & (db["strat_fold"] == VAL_FOLD) & ~held].to_numpy()
    test = db.index[has & (db["strat_fold"] == TEST_FOLD)].to_numpy()
    if limit:
        sub = np.random.default_rng(seed)
        train = np.sort(sub.choice(train, min(limit, len(train)), replace=False))
        val = np.sort(sub.choice(val, min(limit, len(val)), replace=False))
    return train, val, test


def _read_record_100(path):
    import wfdb

    sig, fields = wfdb.rdsamp(path)
    names = [str(s).upper() for s in fields["sig_name"]]
    if fields["fs"] != FS or sig.shape != (N_SAMPLES, N_LEADS):
        raise ValueError(f"{path}: fs={fields['fs']} shape={sig.shape}")
    if names != [s.upper() for s in LEADS]:
        raise ValueError(f"{path}: lead order {names}")
    return sig.astype(np.float32)


_SIGNAL_MEMO = {}  # cache path -> (mtime, ecg_ids, X)


def _read_signal_cache(path):
    if not os.path.exists(path):
        return np.zeros(0, np.int64), np.zeros((0, N_SAMPLES, N_LEADS), np.float32)
    mtime = os.path.getmtime(path)
    memo = _SIGNAL_MEMO.get(path)
    if memo is not None and memo[0] == mtime:
        return memo[1], memo[2]
    with np.load(path) as z:
        ids, X = z["ecg_id"].astype(np.int64), z["X"]
    _SIGNAL_MEMO[path] = (mtime, ids, X)
    return ids, X


# Cache-name suffix for a PTB-XL root: "" for the default data/ptbxl, else a hash of the
# real path, so caches built from different roots never mix.
def _root_tag(root):
    real = os.path.realpath(_resolve(root))
    if real == os.path.realpath(_resolve(DEFAULT_PTBXL)):
        return ""
    return "_" + hashlib.sha256(real.encode()).hexdigest()[:12]


# Record paths (root/<column>, no extension) of the given ecg_ids from ptbxl_database.csv.
def _record_paths(root, ecg_ids, column):
    files = pd.read_csv(
        os.path.join(root, "ptbxl_database.csv"),
        index_col="ecg_id",
        usecols=["ecg_id", column],
    )[column]
    unknown = [e for e in ecg_ids if e not in files.index]
    if unknown:
        raise KeyError(f"ecg_id not in PTB-XL: {unknown[:10]}")
    return [os.path.join(root, files.at[e]) for e in ecg_ids]


# read_one(path) -> (1000, 12) of every path, stacked as float32, with progress lines.
def _read_records(paths, read_one):
    t0 = time.time()
    out = np.empty((len(paths), N_SAMPLES, N_LEADS), np.float32)
    for i, path in enumerate(paths):
        out[i] = read_one(path)
        if (i + 1) % 2000 == 0:
            print(f"  read {i + 1}/{len(paths)} records ({time.time() - t0:.0f} s)")
    return out


# Rows of an npz cache keyed by ecg_id (arrays ecg_id, X) in the given ecg_id order. The
# ids not cached yet are produced by read_missing(sorted ids) -> (n, 1000, 12) float32,
# merged in (the file stays sorted by ecg_id) and saved before returning; the loaded file
# is memoised per path and mtime.
def _cached_rows(ids, cache_path, read_missing):
    cached_ids, cached_X = _read_signal_cache(cache_path)
    pos = {int(e): i for i, e in enumerate(cached_ids)}
    missing = sorted({int(e) for e in ids if int(e) not in pos})
    if missing:
        new_X = read_missing(missing)
        all_ids = np.concatenate([cached_ids, np.asarray(missing, np.int64)])
        all_X = np.concatenate([cached_X, new_X])
        order = np.argsort(all_ids, kind="stable")
        cached_ids, cached_X = all_ids[order], all_X[order]
        del all_X, new_X
        os.makedirs(os.path.dirname(cache_path), exist_ok=True)
        tmp = cache_path[: -len(".npz")] + ".tmp.npz"
        np.savez(tmp, ecg_id=cached_ids, X=cached_X)
        os.replace(tmp, cache_path)
        _SIGNAL_MEMO[cache_path] = (os.path.getmtime(cache_path), cached_ids, cached_X)
        print(f"  signal cache {cache_path}: +{len(missing)} -> {len(cached_ids)}")
        pos = {int(e): i for i, e in enumerate(cached_ids)}
    return cached_X[np.fromiter((pos[int(e)] for e in ids), np.int64, len(ids))]


# records100 signals (mV, float32) in the given ecg_id order, shape (N, 1000, 12). Records
# are read with wfdb once and kept in an npz cache keyed by ecg_id
# (data/fidelity/cache/ptbxl_records100.npz for the default root; another root gets its own
# file, see _root_tag).
def load_signals_100(ecg_ids, root=DEFAULT_PTBXL, cache_path=None):
    ids = np.asarray(ecg_ids, dtype=np.int64).reshape(-1)
    if cache_path is None:
        stem = SIGNAL_CACHE_NAME[: -len(".npz")]
        cache_path = os.path.join(DEFAULT_CACHE_DIR, f"{stem}{_root_tag(root)}.npz")
    root = _resolve(root)

    def read_missing(missing):
        paths = _record_paths(root, missing, "filename_lr")
        return _read_records(paths, _read_record_100)

    return _cached_rows(ids, _resolve(cache_path), read_missing)


# records500 judge inputs through the digitisation condition path, float32 (N, 1000, 12)
# mV at 100 Hz in the given ecg_id order: fidelity._prepare_kind(x, fs, kind) of
# fidelity.read_record(<root>/<filename_hr>), kind "full" (unmasked) or "layout" (NaN
# 3x4 mask at 500 Hz, then prepare_for_judge), i.e. exactly fidelity.ptbxl500_inputs.
# Cached like load_signals_100 in data/fidelity/cache/ptbxl_records500_<kind>.npz (another
# root: + _root_tag). Every missing records500 file is reported before anything is read.
def load_inputs_500(ecg_ids, kind, root=DEFAULT_PTBXL, cache_path=None):
    if kind not in INPUT_KINDS:
        raise ValueError(f"unknown input kind {kind!r}; choose from {INPUT_KINDS}")
    ids = np.asarray(ecg_ids, dtype=np.int64).reshape(-1)
    if cache_path is None:
        stem = INPUTS500_CACHE_NAME.format(kind=kind)[: -len(".npz")]
        cache_path = os.path.join(DEFAULT_CACHE_DIR, f"{stem}{_root_tag(root)}.npz")
    root = _resolve(root)

    def read_missing(missing):
        fid = _fidelity_module()
        paths = _record_paths(root, missing, "filename_hr")
        absent = [p for p in paths if not os.path.exists(p + ".hea")]
        if absent:
            raise FileNotFoundError(
                f"{len(absent)}/{len(paths)} records500 files missing under {root}, "
                f"e.g. {absent[:3]}"
            )
        return _read_records(
            paths, lambda p: fid._prepare_kind(*fid.read_record(p), kind)
        )

    return _cached_rows(ids, _resolve(cache_path), read_missing)


# ---------------------------------------------------------------------------------------
# Metrics.
# ---------------------------------------------------------------------------------------


def _as_2d(y_true, scores):
    y = np.asarray(y_true)
    s = np.asarray(scores, dtype=np.float64)
    if y.ndim == 1:
        y, s = y[:, None], s[:, None]
    if y.shape != s.shape:
        raise ValueError(f"shape mismatch {y.shape} vs {s.shape}")
    return y.astype(int), s


# Per class argmax(tpr - fpr) over sklearn roc_curve thresholds (= benchmark
# utils.find_optimal_cutoff_threshold); NaN if a class lacks positives or negatives.
# A score > threshold is a positive call (benchmark apply_thresholds); see binarize().
def youden_thresholds(y_true, scores):
    from sklearn.metrics import roc_curve

    y, s = _as_2d(y_true, scores)
    out = np.full(y.shape[1], np.nan)
    for k in range(y.shape[1]):
        if y[:, k].min() == y[:, k].max():
            continue
        fpr, tpr, thr = roc_curve(y[:, k], s[:, k])
        out[k] = thr[np.argmax(tpr - fpr)]
    return out


def binarize(scores, thresholds):
    # Strict '>' as in the benchmark apply_thresholds and fidelity.decisions.
    return np.asarray(scores) > np.asarray(thresholds)[None, :]


# Per-class AUROC (NaN when a class lacks positives or negatives) and the macro mean over
# the defined classes only. On a small record set (e.g. the 8/32 local records) that macro
# may rest on 1-2 classes with a single minority record each and is then not comparable
# with the 5-class fold-10 macro: check np.isfinite(per).sum() or auroc_support().
def macro_auroc(y_true, scores):
    from sklearn.metrics import roc_auc_score

    y, s = _as_2d(y_true, scores)
    per = np.full(y.shape[1], np.nan)
    for k in range(y.shape[1]):
        if y[:, k].min() != y[:, k].max():
            per[k] = roc_auc_score(y[:, k], s[:, k])
    macro = float(np.nanmean(per)) if np.isfinite(per).any() else float("nan")
    return per, macro


# Per-class support of an AUROC: (n_pos, n_neg, defined) arrays, defined = both > 0.
def auroc_support(y_true):
    y = np.asarray(y_true).astype(int)
    if y.ndim == 1:
        y = y[:, None]
    n_pos = y.sum(axis=0)
    n_neg = len(y) - n_pos
    return n_pos, n_neg, (n_pos > 0) & (n_neg > 0)


# Rank (Mann-Whitney) AUROC with average ranks for ties; equals roc_auc_score.
def _auroc_rank(y, s):
    from scipy.stats import rankdata

    n_pos = int(y.sum())
    n_neg = len(y) - n_pos
    if n_pos == 0 or n_neg == 0:
        return np.nan
    r = rankdata(s)
    return (r[y == 1].sum() - n_pos * (n_pos + 1) / 2.0) / (n_pos * n_neg)


# Record bootstrap: (B, C) per-class AUROC draws, NaN where a class is undefined.
def bootstrap_auroc(y_true, scores, n_boot=1000, seed=0):
    y, s = _as_2d(y_true, scores)
    rng = np.random.default_rng(seed)
    n = len(y)
    draws = np.full((n_boot, y.shape[1]), np.nan)
    for b in range(n_boot):
        idx = rng.integers(0, n, n)
        yb, sb = y[idx], s[idx]
        for k in range(y.shape[1]):
            draws[b, k] = _auroc_rank(yb[:, k], sb[:, k])
    return draws


# ---------------------------------------------------------------------------------------
# Prediction (benchmark predict: valid chunks of 250 with stride 125, max aggregation).
# ---------------------------------------------------------------------------------------


# (x - mean) / std in float64, returned as float32 (benchmark StandardScaler + astype).
def _standardise(X, mean, std, chunk=2048):
    out = np.empty(X.shape, np.float32)
    for i in range(0, len(X), chunk):
        out[i : i + chunk] = (X[i : i + chunk].astype(np.float64) - mean) / std
    return out


def _windows(Xs):
    w = np.stack([Xs[:, s : s + WINDOW, :] for s in WINDOW_STARTS], axis=1)  # n,7,250,12
    return np.ascontiguousarray(w.transpose(0, 1, 3, 2).reshape(-1, Xs.shape[2], WINDOW))


# Standardised (n, 1000, 12) -> (n, 7, C) window logits; model left in eval mode.
@torch.no_grad()
def _window_logits(model, Xs, device, n_out, batch_size=512):
    model.eval()
    n, n_win = len(Xs), len(WINDOW_STARTS)
    if n == 0:
        return np.zeros((0, n_win, n_out), np.float32)
    per = max(1, batch_size // n_win)
    out = []
    for i in range(0, n, per):
        w = torch.from_numpy(_windows(Xs[i : i + per])).to(device)
        out.append(model(w).float().cpu().numpy())
    return np.concatenate(out).reshape(n, n_win, -1)


# Standardised (n, 1000, 12) -> (n, C) logits of one forward pass over the whole input
# (layout judge); batch_size records per batch; model left in eval mode.
@torch.no_grad()
def _full_logits(model, Xs, device, n_out, batch_size=512):
    model.eval()
    n = len(Xs)
    if n == 0:
        return np.zeros((0, n_out), np.float32)
    per = max(1, int(batch_size))
    out = []
    for i in range(0, n, per):
        x = np.ascontiguousarray(Xs[i : i + per].transpose(0, 2, 1))  # b,12,1000
        out.append(model(torch.from_numpy(x).to(device)).float().cpu().numpy())
    return np.concatenate(out)


def _check_input_mode(mode):
    if mode not in INPUT_MODES:
        raise ValueError(f"unknown input_mode {mode!r}; choose from {INPUT_MODES}")
    return mode


def _check_source(source):
    if source not in SOURCES:
        raise ValueError(f"unknown source {source!r}; choose from {SOURCES}")
    return source


class Judge:
    # input_mode "window" (benchmark: max over 7 windows of 2.5 s) or "layout" (one pass
    # over the whole 10 s input; meaningful only on apply_layout-masked input). source:
    # the signal source it was trained on (records100 or records500, see SOURCES); it
    # only selects the default fold-9/10 inputs of evaluate, prediction is the same.
    def __init__(
        self,
        model,
        mean,
        std,
        classes=CLASSES,
        name="judge",
        device="cpu",
        input_mode="window",
        source="records100",
    ):
        self.device = torch.device(device)
        self.model = model.to(self.device).eval()
        self.mean = float(mean)
        self.std = float(std)
        self.classes = tuple(classes)
        self.name = name
        self.input_mode = _check_input_mode(input_mode)
        self.source = _check_source(source)
        self.weights_path = None
        self.weights_sha256 = None
        self.meta = {}

    @classmethod
    def load(cls, name="inception1d", weights_dir=DEFAULT_WEIGHTS_DIR, device="cpu"):
        path = os.path.join(_resolve(weights_dir), f"{name}_final.pt")
        ck = torch.load(path, map_location="cpu", weights_only=False)
        if ck.get("format") != CKPT_FORMAT:
            raise ValueError(f"{path}: not a judge checkpoint")
        model = build_model(ck["arch"], len(ck["classes"]), **ck["arch_kwargs"])
        model.load_state_dict(ck["state_dict"], strict=True)
        mode = ck.get("input_mode", "window")  # checkpoints before the field: window
        source = ck.get("source", "records100")  # checkpoints before the field
        judge = cls(
            model, ck["mean"], ck["std"], ck["classes"], name, device, mode, source
        )
        judge.weights_path = path
        judge.weights_sha256 = _sha256(path)
        judge.meta = {k: v for k, v in ck.items() if k != "state_dict"}
        return judge

    # X: (N, 1000, 12) or (1000, 12), mV at 100 Hz, finite. Returns (N, C) logits: for
    # input_mode "window" the max over the 7 windows per class; for "layout" one forward
    # pass over the whole standardised input (batch_size records per batch). The judge
    # does not mask: a layout judge is meaningful only on layout-masked input
    # (apply_layout(X), or equivalent).
    def predict_logits(self, X, batch_size=512):
        X = np.asarray(X)
        if X.ndim == 2:
            X = X[None]
        if X.ndim != 3 or X.shape[1:] != (N_SAMPLES, N_LEADS):
            raise ValueError(f"expected (N, {N_SAMPLES}, {N_LEADS}), got {X.shape}")
        if X.dtype.kind not in "fiu":
            raise ValueError(f"numeric input required, got {X.dtype}")
        if not np.isfinite(X).all():
            raise ValueError("input contains NaN or inf")
        Xs = _standardise(X, self.mean, self.std)
        if self.input_mode == "layout":
            n_out = len(self.classes)
            return _full_logits(self.model, Xs, self.device, n_out, batch_size)
        wl = _window_logits(self.model, Xs, self.device, len(self.classes), batch_size)
        return wl.max(axis=1)

    def predict_proba(self, X, batch_size=512):
        z = self.predict_logits(X, batch_size)
        return 1.0 / (1.0 + np.exp(-z.astype(np.float64)))


# ---------------------------------------------------------------------------------------
# Training (fastai v1 fit_one_cycle(50, 1e-2) as used by the benchmark).
# ---------------------------------------------------------------------------------------


# <model>[_layout][_<tag>]; a window run's name is unchanged (<model>[_<tag>]).
def _run_name(model, tag, input_mode="window"):
    base = f"{model}_layout" if _check_input_mode(input_mode) == "layout" else model
    return f"{base}_{tag}" if tag else base


def _seed_everything(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)


def _rng_state(device):
    st = {
        "python": random.getstate(),
        "numpy_global": np.random.get_state(),
        "torch_cpu": torch.get_rng_state(),
    }
    if device.type == "mps":
        st["torch_mps"] = torch.mps.get_rng_state()
    return st


def _set_rng_state(st, device):
    random.setstate(st["python"])
    np.random.set_state(st["numpy_global"])
    torch.set_rng_state(st["torch_cpu"])
    if device.type == "mps" and "torch_mps" in st:
        torch.mps.set_rng_state(st["torch_mps"])


def _sync(device):
    if device.type == "mps":
        torch.mps.synchronize()


# One global mean/std over all training values in float64 (benchmark preprocess_signals).
def _global_mean_std(X, chunk=2048):
    n = X.size
    total = 0.0
    for i in range(0, len(X), chunk):
        total += X[i : i + chunk].astype(np.float64).sum()
    mean = total / n
    ss = 0.0
    for i in range(0, len(X), chunk):
        ss += np.square(X[i : i + chunk].astype(np.float64) - mean).sum()
    return float(mean), float(np.sqrt(ss / n))


def _check_aug(aug):
    if aug not in AUGS:
        raise ValueError(f"unknown aug {aug!r}; choose from {AUGS}")
    return aug


# Every record of X (n, T, L) moved by shift[i] samples with zero fill, no wrap-around:
# out[i, t] = X[i, t - shift[i]] where 0 <= t - shift[i] < T, else 0 (a positive shift
# moves the signal later, a negative one earlier).
def shift_zero_fill(X, shift):
    X = np.asarray(X)
    shift = np.asarray(shift, dtype=np.int64).reshape(-1)
    if len(shift) != len(X):
        raise ValueError(f"{len(shift)} shifts for {len(X)} records")
    out = np.zeros_like(X)
    n_t = X.shape[1]
    for i, d in enumerate(shift):
        if d >= n_t or d <= -n_t:
            continue  # moved out entirely
        if d >= 0:
            out[i, d:] = X[i, : n_t - d]
        else:
            out[i, : n_t + d] = X[i, -d:]
    return out


# Every record of X (n, T, L) delayed by frac[i] samples (fractional; a positive delay
# moves the signal later), by an FFT phase shift of the record padded with FRAC_PAD edge
# values on both sides (so the circular wrap stays in the padding); frac 0 = unchanged.
def frac_delay(X, frac):
    X = np.asarray(X)
    frac = np.asarray(frac, dtype=np.float64).reshape(-1)
    if len(frac) != len(X):
        raise ValueError(f"{len(frac)} delays for {len(X)} records")
    out = np.array(X, copy=True)
    moved = np.flatnonzero(frac)
    if moved.size == 0:
        return out
    n_t = X.shape[1]
    padded = np.pad(
        X[moved].astype(np.float64), ((0, 0), (FRAC_PAD, FRAC_PAD), (0, 0)), mode="edge"
    )
    f = np.fft.rfftfreq(padded.shape[1])
    phase = np.exp(-2j * np.pi * f[None, :] * frac[moved][:, None])[:, :, None]
    y = np.fft.irfft(np.fft.rfft(padded, axis=1) * phase, n=padded.shape[1], axis=1)
    out[moved] = y[:, FRAC_PAD : FRAC_PAD + n_t]
    return out


# Augmentation draws of one training batch of n records: gain (n,) uniform in AMP_RANGE;
# shift (n,) int64, 0 = not shifted, else (amp_shift*, probability SHIFT_PROB per
# record) uniform over the integers +-1..SHIFT_MAX; amp_shift_frac adds frac (n,), 0 =
# not delayed, else (probability FRAC_PROB) uniform in +-FRAC_MAX, drawn after the
# amp_shift draws so those keep their stream. aug "none" draws nothing: None.
def draw_aug(rng, n, aug):
    if _check_aug(aug) == "none":
        return None
    gain = rng.uniform(AMP_RANGE[0], AMP_RANGE[1], size=n)
    shift = np.zeros(n, np.int64)
    if aug in SHIFT_AUGS:
        moved = rng.random(n) < SHIFT_PROB
        k = rng.integers(0, 2 * SHIFT_MAX, size=n)  # 0..24 -> -25..-1, 25..49 -> 1..25
        shift[moved] = (k - SHIFT_MAX + (k >= SHIFT_MAX))[moved]
    if aug != "amp_shift_frac":
        return gain, shift
    delayed = rng.random(n) < FRAC_PROB
    frac = np.where(delayed, rng.uniform(-FRAC_MAX, FRAC_MAX, size=n), 0.0)
    return gain, shift, frac


# One standardised training batch (bs, 12, T) float32 of the records idx. X: training
# inputs in mV (n, 1000, 12), layout-masked in layout mode; starts: per-record crop starts
# of window mode ([s, s + WINDOW)), None in layout mode (whole input). aug: draw_aug
# output or None. A record with shift != 0 is rebuilt from Xfull (its unmasked mV input)
# as apply_layout(shift_zero_fill(.)); every other record keeps its X row unchanged; each
# is then scaled by its gain. Standardising per batch is elementwise, so aug None gives
# exactly the rows of the standardised array (the unaugmented protocol, bit for bit).
def train_batch(X, idx, mean, std, starts=None, aug=None, Xfull=None):
    idx = np.asarray(idx)
    if starts is None:
        xb = X[idx]  # bs,1000,12: the whole (masked) record, no crop
    else:  # bs,250,12
        xb = X[idx[:, None], starts[idx][:, None] + np.arange(WINDOW)[None, :], :]
    if aug is not None:
        gain, shift, *rest = aug
        frac = rest[0] if rest else np.zeros(len(idx))
        moved = np.flatnonzero((np.asarray(shift) != 0) | (np.asarray(frac) != 0))
        if moved.size:
            if starts is not None or Xfull is None:
                raise ValueError("a shift needs layout mode and the unmasked inputs")
            full = frac_delay(Xfull[idx[moved]], np.asarray(frac)[moved])
            xb[moved] = apply_layout(shift_zero_fill(full, np.asarray(shift)[moved]))
        xb = xb * np.asarray(gain, np.float64)[:, None, None]
    return np.ascontiguousarray(_standardise(xb, mean, std).transpose(0, 2, 1))


# fold-9 loss as fastai validate() computes it (mean BCE over all 7 windows of every
# record) and the macro AUROC of the max-aggregated logits.
def _val_metrics(model, Xva, Yva, device):
    wl = _window_logits(model, Xva, device, Yva.shape[1])
    y = np.repeat(Yva[:, None, :], wl.shape[1], axis=1)
    loss = F.binary_cross_entropy_with_logits(torch.from_numpy(wl), torch.from_numpy(y))
    per, macro = macro_auroc(Yva, wl.max(axis=1))
    return float(loss), per, macro


# Layout judge: fold-9 mean BCE and macro AUROC of the one-pass logits of the (already
# layout-masked, standardised) validation inputs.
def _val_metrics_layout(model, Xva, Yva, device):
    lg = _full_logits(model, Xva, device, Yva.shape[1])
    loss = F.binary_cross_entropy_with_logits(torch.from_numpy(lg), torch.from_numpy(Yva))
    per, macro = macro_auroc(Yva, lg)
    return float(loss), per, macro


# fastai 1.0.61 OneCycleScheduler (fit_one_cycle defaults: moms (0.95, 0.85), div_factor
# 25, pct_start 0.3, final_div 25 * 1e4): (lr, beta1) of batch i (0-based) of n. Batch 0
# runs at the start values; phase 1 covers batches 0..a1 with a1 = int(0.3 n) (fastai
# Scheduler n_iter >= 1), phase 2 the rest at pct (i - a1) / (n - a1); cosine annealing.
# (torch OneCycleLR ends phase 1 one batch earlier, at 0.3 n - 1.)
def fastai_one_cycle(i, n, lr_max, moms=(0.95, 0.85), div=25.0, pct_start=0.3):
    a1 = int(n * pct_start)
    a2 = n - a1
    p1 = max(1, a1)

    def cos(start, end, pct):
        return end + (start - end) / 2.0 * (np.cos(np.pi * pct) + 1.0)

    if i <= p1:
        pct = i / p1
        return float(cos(lr_max / div, lr_max, pct)), float(cos(moms[0], moms[1], pct))
    pct = (i - p1) / max(1, a2)
    return (
        float(cos(lr_max, lr_max / (div * 1e4), pct)),
        float(cos(moms[1], moms[0], pct)),
    )


def _set_lr_beta1(opt, lr, beta1):
    for g in opt.param_groups:
        g["lr"] = lr
        g["betas"] = (beta1, g["betas"][1])


def _write_log(rows, path):
    pd.DataFrame(rows).to_csv(path, index=False, float_format="%.6g")


def cmd_train(args):
    if args.model not in TRAINABLE:
        raise SystemExit(f"--model must be one of {TRAINABLE}")
    if args.input == "window" and args.tag and args.tag.split("_")[0] == "layout":
        raise SystemExit("a window run's --tag must not start with 'layout'")
    name = _run_name(args.model, args.tag, args.input)
    wdir = _resolve(args.weights_dir)
    os.makedirs(wdir, exist_ok=True)
    last_path = os.path.join(wdir, f"{name}_last.pt")
    final_path = os.path.join(wdir, f"{name}_final.pt")
    log_path = os.path.join(wdir, f"{name}_log.csv")
    device = torch.device(args.device)
    t_start = time.time()
    db = load_ptbxl_db(args.ptbxl)

    if args.input == "window" and args.aug in SHIFT_AUGS:
        raise SystemExit(f"--aug {args.aug} needs --input layout")
    if args.resume:
        ck = torch.load(last_path, map_location="cpu", weights_only=False)
        cfg = ck["config"]
        if cfg["arch"] != args.model:
            raise SystemExit(f"{last_path} is a {cfg['arch']} run")
        saved_mode = cfg.get("input_mode", "window")  # runs before the field: window
        if saved_mode != args.input:
            raise SystemExit(f"{last_path} is a --input {saved_mode} run")
        saved_source = cfg.get("source", "records100")  # runs before the field
        if saved_source != args.source:
            raise SystemExit(f"{last_path} is a --source {saved_source} run")
        saved_aug = cfg.get("aug", "none")  # runs before the field
        if saved_aug != args.aug:
            raise SystemExit(f"{last_path} is an --aug {saved_aug} run")
        if args.epochs is not None and args.epochs != cfg["epochs"]:
            print(f"note: resuming with the saved epochs={cfg['epochs']}")
        # Validation draws no random numbers, so its frequency may change on resume.
        if args.val_every is not None and args.val_every != cfg.get("val_every", 5):
            print(f"note: validating every {args.val_every} epochs from now on")
            cfg["val_every"] = args.val_every
        train_ids, val_ids = np.asarray(ck["train_ids"]), np.asarray(ck["val_ids"])
        print(f"resume {name}: {ck['epoch']}/{cfg['epochs']} epochs done")
    else:
        for p in (last_path, final_path):
            if os.path.exists(p) and not args.fresh:
                raise SystemExit(f"{p} exists; use --resume, or --fresh to start over")
        cfg = {
            "arch": args.model,
            "arch_kwargs": ARCHS[args.model][2],
            "epochs": args.epochs if args.epochs is not None else 50,
            "lr": args.lr,
            "bs": args.bs,
            "wd": args.wd,
            "seed": args.seed,
            "limit": args.limit,
            "input_mode": args.input,
            "source": args.source,
            "aug": args.aug,
            "val_every": args.val_every if args.val_every is not None else 5,
        }
        _seed_everything(cfg["seed"])
        train_ids, val_ids, _ = split_ids(db, cfg["limit"], cfg["seed"])
        ck = None

    layout = cfg.get("input_mode", "window") == "layout"
    source = _check_source(cfg.get("source", "records100"))
    aug = _check_aug(cfg.get("aug", "none"))
    val_every = int(cfg.get("val_every", 5))
    if val_every < 1:
        raise SystemExit("--val_every must be >= 1")
    n_train = len(train_ids)
    steps = n_train // cfg["bs"]  # drop_last=True
    if steps < 1:
        raise SystemExit(f"{n_train} training records < batch size {cfg['bs']}")
    print(f"{name}: train {n_train} records ({steps} batches/epoch), val {len(val_ids)}")
    if layout:
        print("  input layout: 3x4 + rhythm II masked 10 s records, whole-input steps")
    if source != "records100" or aug != "none":
        print(f"  source {source}, aug {aug}, validation every {val_every} epochs")

    # X: training + validation inputs in mV (layout-masked in layout mode); Xfull: the
    # unmasked inputs of the same source, for the standardisation and amp_shift.
    all_ids = np.concatenate([train_ids, val_ids])
    if source == "records500":
        # Layout mode: the records500 condition path itself, not apply_layout(full).
        need_full = not layout or ck is None or aug in SHIFT_AUGS
        Xfull = None
        if need_full:
            full_ids = train_ids if layout else all_ids
            Xfull = load_inputs_500(full_ids, "full", args.ptbxl)
        X = load_inputs_500(all_ids, "layout", args.ptbxl) if layout else Xfull
    else:
        Xfull = load_signals_100(all_ids, args.ptbxl)
        X = apply_layout(Xfull) if layout else Xfull
    if ck is None:
        mean, std = _global_mean_std(Xfull[:n_train])  # unmasked values in both modes
    else:
        mean, std = ck["mean"], ck["std"]
    # Training inputs stay in mV (standardised per batch, after augmentation).
    Xtr = X[:n_train]
    Xtr_full = Xfull[:n_train] if aug in SHIFT_AUGS else None
    Xva = _standardise(X[n_train:], mean, std)
    del X, Xfull
    Ytr = db.loc[train_ids, list(CLASSES)].to_numpy(np.float32)
    Yva = db.loc[val_ids, list(CLASSES)].to_numpy(np.float32)
    print(f"  standardisation mean={mean:.8f} std={std:.8f}")

    model = init_weights(build_model(cfg["arch"], len(CLASSES), **cfg["arch_kwargs"]))
    model.to(device)
    opt = torch.optim.AdamW(
        model.parameters(),
        lr=cfg["lr"],
        betas=(0.9, 0.99),
        eps=1e-8,
        weight_decay=cfg["wd"],
    )
    total_steps = cfg["epochs"] * steps  # lr / beta1 set per batch by fastai_one_cycle
    rng = np.random.default_rng(cfg["seed"])  # epoch order and crop starts
    # Augmentation draws only (none: never drawn), so the streams above stay untouched.
    rng_aug = np.random.default_rng([cfg["seed"], 1])
    start_epoch, log_rows = 0, []
    if ck is not None:
        if "scheduler" in ck:
            raise SystemExit(f"{last_path} uses the old OneCycleLR schedule; use --fresh")
        model.load_state_dict(ck["model"])
        opt.load_state_dict(ck["optimizer"])
        rng.bit_generator.state = ck["rng_data"]
        if "rng_aug" in ck:
            rng_aug.bit_generator.state = ck["rng_aug"]
        _set_rng_state(ck["rng"], device)
        start_epoch, log_rows = ck["epoch"], list(ck["log"])
        del ck

    bs = cfg["bs"]
    epoch = start_epoch
    starts = None  # layout mode: no crop
    for epoch in range(start_epoch, cfg["epochs"]):
        # Reseed torch per epoch: MPS dropout ignores torch.mps.set_rng_state, so this is
        # what makes a resumed run match an uninterrupted one on MPS as well as on CPU.
        torch.manual_seed(cfg["seed"] * 100_003 + epoch + 1)
        model.train()
        perm = rng.permutation(n_train)
        if not layout:
            starts = rng.integers(0, N_SAMPLES - WINDOW, size=n_train)  # randint(0, 749)
        _sync(device)
        t0 = time.time()
        t_first = None
        loss_sum = 0.0
        for b in range(steps):
            idx = perm[b * bs : (b + 1) * bs]
            draws = draw_aug(rng_aug, len(idx), aug)
            xb = train_batch(Xtr, idx, mean, std, starts, draws, Xtr_full)
            xb = torch.from_numpy(xb).to(device)
            yb = torch.from_numpy(Ytr[idx]).to(device)
            loss = F.binary_cross_entropy_with_logits(model(xb), yb)
            opt.zero_grad(set_to_none=True)
            loss.backward()
            lr_b, beta1_b = fastai_one_cycle(epoch * steps + b, total_steps, cfg["lr"])
            _set_lr_beta1(opt, lr_b, beta1_b)
            opt.step()
            loss_sum += loss.item()
            if b == 0:
                t_first = time.time()
        _sync(device)
        t1 = time.time()
        lr_next, beta1_next = fastai_one_cycle((epoch + 1) * steps, total_steps, cfg["lr"])
        row = {
            "epoch": epoch + 1,
            "step": (epoch + 1) * steps,
            "train_loss": loss_sum / steps,
            "lr_next": lr_next,
            "beta1_next": beta1_next,
            "seconds_train": t1 - t0,
            "sec_per_batch_steady": (t1 - t_first) / (steps - 1) if steps > 1 else np.nan,
            "val_loss": np.nan,
            "val_macro_auroc": np.nan,
            **{f"val_auroc_{c}": np.nan for c in CLASSES},
        }
        if (epoch + 1) % val_every == 0 or epoch + 1 == cfg["epochs"]:
            val_fn = _val_metrics_layout if layout else _val_metrics
            vloss, vper, vmacro = val_fn(model, Xva, Yva, device)
            row.update({"val_loss": vloss, "val_macro_auroc": vmacro})
            row.update({f"val_auroc_{c}": v for c, v in zip(CLASSES, vper)})
        row["seconds_epoch"] = time.time() - t0
        log_rows.append(row)
        _write_log(log_rows, log_path)
        _atomic_torch_save(
            {
                "config": cfg,
                "epoch": epoch + 1,
                "model": model.state_dict(),
                "optimizer": opt.state_dict(),
                "rng_data": rng.bit_generator.state,
                "rng_aug": rng_aug.bit_generator.state,
                "rng": _rng_state(device),
                "mean": mean,
                "std": std,
                "train_ids": [int(e) for e in train_ids],
                "val_ids": [int(e) for e in val_ids],
                "log": log_rows,
                "device": str(device),
                "source": source,
                "aug": aug,
            },
            last_path,
        )
        msg = (
            f"epoch {epoch + 1}/{cfg['epochs']} loss {row['train_loss']:.4f} "
            f"train {row['seconds_train']:.1f} s "
            f"({row['sec_per_batch_steady']:.3f} s/batch)"
        )
        if np.isfinite(row["val_loss"]):
            msg += (
                f" | val loss {row['val_loss']:.4f} "
                f"macro AUROC {row['val_macro_auroc']:.4f}"
            )
        print(msg, flush=True)
        elapsed = (time.time() - t_start) / 60.0
        if (
            epoch + 1 < cfg["epochs"]
            and args.max_minutes is not None
            and elapsed > args.max_minutes
        ):
            print(f"stop after epoch {epoch + 1} ({elapsed:.1f} min); use --resume")
            return 0

    # Move the model itself so the state dict keeps xresnet's shared (aliased) parameters.
    model.to("cpu")
    final = {
        "format": CKPT_FORMAT,
        "arch": cfg["arch"],
        "arch_kwargs": cfg["arch_kwargs"],
        "classes": list(CLASSES),
        "state_dict": model.state_dict(),
        "mean": float(mean),
        "std": float(std),
        "fs": FS,
        "input_mode": "layout" if layout else "window",
        "source": source,
        "aug": aug,
        "val_every": val_every,
        # layout: one window, the whole (masked) 10 s input
        "window": N_SAMPLES if layout else WINDOW,
        "window_starts": [0] if layout else list(WINDOW_STARTS),
        "train_ids": [int(e) for e in train_ids],
        "val_ids": [int(e) for e in val_ids],
        "held_out_ecg_ids": list(HELD_OUT_ECG_IDS),
        "epochs": cfg["epochs"],
        "seed": cfg["seed"],
        "lr": cfg["lr"],
        "bs": cfg["bs"],
        "wd": cfg["wd"],
        "limit": cfg["limit"],
        "device": str(device),
        "torch_version": torch.__version__,
        "final_val": {k: v for k, v in log_rows[-1].items() if k.startswith("val_")},
        "created": time.strftime("%Y-%m-%d %H:%M:%S"),
    }
    if layout:
        final["layout_windows"] = {k: list(v) for k, v in LAYOUT_WINDOWS.items()}
    _atomic_torch_save(final, final_path)
    print(f"wrote {final_path}")
    return 0


# ---------------------------------------------------------------------------------------
# Evaluation on fold 10 with fold-9 thresholds.
# ---------------------------------------------------------------------------------------


# Full-signal (10 s) logits for one fold, cached by ecg_id, checkpoint hash and PTB-XL
# root (_root_tag). A layout judge (judge.input_mode "layout") gets layout-masked inputs
# and its own cache file, judge_<name>_layout_fold<k>.npz instead of ..._full_...
# source (default: the judge's): records100 = load_signals_100 (layout: apply_layout of
# it); records500 = load_inputs_500 "full" / "layout" (the condition path), cached as
# judge_<name>_<kind>_fold<k>_r500.npz.
def fold_logits(
    judge,
    db,
    fold,
    ecg_ids,
    cache_dir=DEFAULT_CACHE_DIR,
    root=DEFAULT_PTBXL,
    source=None,
):
    ecg_ids = np.asarray(ecg_ids, dtype=np.int64)
    layout = getattr(judge, "input_mode", "window") == "layout"
    kind = "layout" if layout else "full"
    if source is None:
        source = getattr(judge, "source", "records100")
    r500 = _check_source(source) == "records500"
    suffix = "_r500" if r500 else ""
    path = os.path.join(
        _resolve(cache_dir),
        f"judge_{judge.name}_{kind}_fold{fold}{suffix}{_root_tag(root)}.npz",
    )
    if os.path.exists(path):
        with np.load(path) as z:
            same = str(z["weights_sha256"]) == judge.weights_sha256 and np.array_equal(
                z["ecg_id"], ecg_ids
            )
            if same:
                return z["logits"], z["labels"]
    if r500:
        X = load_inputs_500(ecg_ids, kind, root)
    else:
        X = load_signals_100(ecg_ids, root)
        if layout:
            X = apply_layout(X)
    logits = judge.predict_logits(X)
    labels = db.loc[ecg_ids, list(judge.classes)].to_numpy(np.int8)
    os.makedirs(os.path.dirname(path), exist_ok=True)
    np.savez(
        path,
        ecg_id=ecg_ids,
        logits=logits.astype(np.float32),
        labels=labels,
        weights_sha256=np.array(judge.weights_sha256),
    )
    print(f"  cached {path} ({len(ecg_ids)} records)")
    return logits, labels


def cmd_evaluate(args):
    name = _run_name(args.model, args.tag)
    judge = Judge.load(name, args.weights_dir, args.device)
    source = args.source or judge.source
    if judge.input_mode != "window" or source != "records100":
        how = {
            ("window", "records100"): "records100",
            ("layout", "records100"): "apply_layout(records100)",
            ("window", "records500"): "records500 full path",
            ("layout", "records500"): "records500 layout path",
        }[(judge.input_mode, source)]
        print(
            f"{name}: input_mode {judge.input_mode}, trained on {judge.source} "
            f"(fold 9/10 inputs: {how})"
        )
    db = load_ptbxl_db(args.ptbxl)
    _, val_ids, test_ids = split_ids(db)
    t0 = time.time()
    fold_args = (args.cache_dir, args.ptbxl, source)
    l9, y9 = fold_logits(judge, db, VAL_FOLD, val_ids, *fold_args)
    l10, y10 = fold_logits(judge, db, TEST_FOLD, test_ids, *fold_args)
    print(f"  logits fold 9 {l9.shape} fold 10 {l10.shape} ({time.time() - t0:.1f} s)")
    thr = youden_thresholds(y9, l9)
    per, macro = macro_auroc(y10, l10)
    draws = bootstrap_auroc(y10, l10, args.n_boot, seed=0)
    pred = binarize(l10, thr)
    # ci95_*: 2.5/97.5 percentiles of n_boot record resamples (draws where the class, or
    # for macro every class, is undefined are skipped). ci90_*: 5/95 percentiles of the same
    # draws, the interval the benchmark publishes (it uses 100 draws redrawn until every
    # class has a positive); compare the paper's +/- with ci90, not ci95.
    rows = []
    sens_all, spec_all = [], []
    for k, c in enumerate(judge.classes):
        v = draws[:, k][np.isfinite(draws[:, k])]
        q = np.percentile(v, [2.5, 97.5, 5.0, 95.0]) if len(v) else [np.nan] * 4
        pos, neg = y10[:, k] == 1, y10[:, k] == 0
        sens = pred[pos, k].mean() if pos.any() else np.nan
        spec = (~pred[neg, k]).mean() if neg.any() else np.nan
        sens_all.append(sens)
        spec_all.append(spec)
        rows.append(
            {
                "model": name,
                "source": source,
                "class": c,
                "n_test": len(y10),
                "n_pos": int(pos.sum()),
                "auroc": per[k],
                "ci95_lo": q[0],
                "ci95_hi": q[1],
                "ci90_lo": q[2],
                "ci90_hi": q[3],
                "threshold_logit_fold9": thr[k],
                "sens_fold10": sens,
                "spec_fold10": spec,
                "pass_090": "",
                "n_boot_valid": len(v),
            }
        )
    ok = np.isfinite(draws).all(axis=1)
    mv = draws[ok].mean(axis=1)
    q = np.percentile(mv, [2.5, 97.5, 5.0, 95.0]) if len(mv) else [np.nan] * 4
    rows.append(
        {
            "model": name,
            "source": source,
            "class": "macro",
            "n_test": len(y10),
            "n_pos": np.nan,  # per class only (a sum here would count label cells)
            "auroc": macro,
            "ci95_lo": q[0],
            "ci95_hi": q[1],
            "ci90_lo": q[2],
            "ci90_hi": q[3],
            "threshold_logit_fold9": np.nan,
            "sens_fold10": float(np.nanmean(sens_all)),
            "spec_fold10": float(np.nanmean(spec_all)),
            "pass_090": bool(macro >= 0.90),
            "n_boot_valid": int(ok.sum()),
        }
    )
    new = pd.DataFrame(rows)
    # Metrics rounded to 6 decimals; the fold-9 threshold is written at full precision
    # (fidelity's thresholds.csv stays the operational source of thresholds).
    metric_cols = ["auroc", "ci95_lo", "ci95_hi", "ci90_lo", "ci90_hi"]
    metric_cols += ["sens_fold10", "spec_fold10"]
    new[metric_cols] = new[metric_cols].astype(float).round(6)
    out = _resolve(args.out)
    os.makedirs(os.path.dirname(out), exist_ok=True)
    if os.path.exists(out):
        old = pd.read_csv(out)
        # Older schema: ci_lo/ci_hi were the 95% interval; macro n_pos was a cell count.
        old = old.rename(columns={"ci_lo": "ci95_lo", "ci_hi": "ci95_hi"})
        old.loc[old["class"] == "macro", "n_pos"] = np.nan
        # Rows before the source column were records100; one row set per (model, source).
        if "source" not in old.columns:
            old["source"] = "records100"
        old["source"] = old["source"].fillna("records100")
        same = (old["model"] == name) & (old["source"] == source)
        old = old[~same].reindex(columns=new.columns)
        if len(old):
            new = pd.concat([old, new], ignore_index=True)
    new["n_pos"] = new["n_pos"].astype("Int64")  # integers, empty on the macro rows
    new.to_csv(out, index=False)
    print(f"{name} on {source}:")
    with pd.option_context("display.width", 160, "display.max_columns", 20):
        print(pd.DataFrame(rows).drop(columns=["model", "source"]).to_string(index=False))
    print(f"wrote {out}")
    return 0


# ---------------------------------------------------------------------------------------
# End-to-end check of the prediction path against the authors' exp0 outputs.
# ---------------------------------------------------------------------------------------


def _unpickle_quiet(path):
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        with open(path, "rb") as f:
            return pickle.load(f)


# ecg_ids dropped from PTB-XL after the benchmark's release (v1.0.2/1.0.3 changelogs).
def _dropped_ids(root, fold=None):
    out = set()
    pat = re.compile(r"drop row with ecg_id=(\d+) \[strat_fold=(\d+)\]")
    for name in ("ptbxl_v102_changelog.txt", "ptbxl_v103_changelog.txt"):
        path = os.path.join(root, name)
        if os.path.exists(path):
            with open(path) as f:
                for e, fo in pat.findall(f.read()):
                    if fold is None or int(fo) == fold:
                        out.add(int(e))
    return out


def cmd_check_benchmark(args):
    exp = os.path.join(BENCH_DIR, "output", "exp0")
    mdir = os.path.join(exp, "models", "fastai_xresnet1d101")
    mlb = _unpickle_quiet(os.path.join(exp, "data", "mlb.pkl"))
    classes = [str(c) for c in mlb.classes_]
    scaler = _unpickle_quiet(os.path.join(exp, "data", "standard_scaler.pkl"))
    mean, std = float(scaler.mean_[0]), float(scaler.scale_[0])
    state = torch.load(
        os.path.join(mdir, "models", "fastai_xresnet1d101.pth"),
        map_location="cpu",
        weights_only=False,
    )
    state = state.get("model", state)
    model = build_model("xresnet1d101", len(classes))
    print("load_state_dict(strict=True):", model.load_state_dict(state, strict=True))
    judge = Judge(model, mean, std, classes, "exp0_fastai_xresnet1d101_all", args.device)
    print(f"exp0 'all': {len(classes)} classes, scaler mean {mean:.8f} scale {std:.8f}")

    root = _resolve(args.ptbxl)
    db = load_ptbxl_db(root)
    cls_set = set(classes)
    sel = db[
        (db["strat_fold"] == TEST_FOLD)
        & db["scp_codes"].apply(lambda d: len(set(d) & cls_set) > 0)
    ].sort_index()
    y_ours = np.array(
        [[int(c in d) for c in classes] for d in sel["scp_codes"]], np.int64
    )
    y_test = np.load(os.path.join(exp, "data", "y_test.npy"), allow_pickle=True)
    p_bench = np.load(os.path.join(mdir, "y_test_pred.npy"), allow_pickle=True)
    dropped = _dropped_ids(root, TEST_FOLD)
    bench_ids = sorted(set(sel.index.tolist()) | dropped)
    print(
        f"fold-10 selection v1.0.3: {len(sel)}; + dropped fold-10 ids {sorted(dropped)} "
        f"= {len(bench_ids)}; y_test rows {len(y_test)}"
    )
    if len(bench_ids) != len(y_test):
        raise SystemExit("cannot align benchmark rows to ecg_ids")
    row_of = {e: i for i, e in enumerate(bench_ids)}
    rows = np.array([row_of[e] for e in sel.index])
    yb = np.asarray(y_test[rows], np.int64)
    rec_mis = int((y_ours != yb).any(axis=1).sum())
    cell_mis = int((y_ours != yb).sum())
    print(f"label mismatches vs y_test: {rec_mis} records, {cell_mis} cells")
    if rec_mis:
        bad = sel.index[(y_ours != yb).any(axis=1)].tolist()
        print(f"  mismatching ecg_ids: {bad[:20]}")

    t0 = time.time()
    X = load_signals_100(sel.index.to_numpy(), root)
    z = judge.predict_logits(X).astype(np.float64)
    print(f"predicted {z.shape} on {args.device} in {time.time() - t0:.1f} s")
    p_ours = 1.0 / (1.0 + np.exp(-z))
    pb = np.asarray(p_bench[rows], np.float64)
    print(
        f"y_test_pred: dtype {p_bench.dtype}, min {p_bench.min():.3g}, "
        f"max {p_bench.max():.3g},"
        f" negative values present: {bool((p_bench < 0).any())}"
    )
    d = np.abs(p_ours - pb)
    print(f"probability |diff|: max {d.max():.3g}, median {np.median(d):.3g}, "
          f"99.9th pct {np.percentile(d, 99.9):.3g}")
    inside = (pb > 0) & (pb < 1)
    zb = np.full_like(pb, np.nan)
    zb[inside] = np.log(pb[inside]) - np.log1p(-pb[inside])
    well = inside & (np.abs(zb) < 10)
    r_all = np.corrcoef(z[inside], zb[inside])[0, 1]
    r_well = np.corrcoef(z[well], zb[well])[0, 1]
    dz = np.abs(z[well] - zb[well])
    print(
        f"logit(y_test_pred) vs our logits: Pearson r {r_all:.6f} "
        f"over {inside.sum()} cells "
        f"with 0<p<1 ({(~inside).sum()} saturated); |z|<10: r {r_well:.6f}, max |diff| "
        f"{dz.max():.3g}, median {np.median(dz):.3g} over {well.sum()} cells"
    )
    # Discrimination: other window/aggregation choices must match clearly worse. A residual
    # |dz| present in every record (not float precision: fp32 vs fp64 is far smaller) is a
    # global difference in inputs or weights (PTB-XL signal version, the authors' GPU conv
    # kernels); the path is confirmed when every alternative below matches clearly worse.
    wl = _window_logits(
        judge.model, _standardise(X, mean, std), judge.device, len(classes)
    ).astype(np.float64)
    variants = {
        "max over 7 windows (this path)": wl.max(axis=1),
        "mean logit over 7 windows": wl.mean(axis=1),
        "max over 4 disjoint windows": wl[:, ::2].max(axis=1),
        "first window only": wl[:, 0],
    }
    for label, zv in variants.items():
        dv = np.abs(zv[well] - zb[well])
        dp = np.abs(1.0 / (1.0 + np.exp(-zv)) - pb)
        print(f"  {label:32s} median |dz| {np.median(dv):.3g}  max |dp| {dp.max():.3g}")
    per_ours, m_ours = macro_auroc(yb, p_ours)
    _, m_bench = macro_auroc(yb, pb)
    per_all, m_bench_all = macro_auroc(
        np.asarray(y_test, np.int64), np.asarray(p_bench, np.float64)
    )
    print(
        f"macro AUROC on y_test labels ({len(rows)} aligned rows, "
        f"{int(np.isfinite(per_ours).sum())}/{len(classes)} classes defined): "
        f"ours {m_ours:.6f}, benchmark {m_bench:.6f}; benchmark on all {len(y_test)} rows "
        f"({int(np.isfinite(per_all).sum())}/{len(classes)} defined) {m_bench_all:.6f}"
    )
    return 0


def cmd_params(args):
    for arch in ("inception1d", "xresnet1d50", "xresnet1d101"):
        n_cls = 71 if arch == "xresnet1d101" else len(CLASSES)
        model = init_weights(build_model(arch, n_cls))
        with torch.no_grad():
            out = model.eval()(torch.zeros(2, N_LEADS, WINDOW))
        n = count_parameters(model)
        print(f"{arch}: {n:,} parameters, output {tuple(out.shape)}")
    return 0


def get_parser():
    parser = argparse.ArgumentParser(description="PTB-XL superdiagnostic judge.")
    parser.add_argument("--ptbxl", default=DEFAULT_PTBXL, help="PTB-XL 1.0.3 folder.")
    parser.add_argument("--weights_dir", default=DEFAULT_WEIGHTS_DIR)
    sub = parser.add_subparsers(dest="cmd", required=True)

    p = sub.add_parser("train", help="Train on folds 1-8 (fastai fit_one_cycle).")
    p.add_argument("--model", required=True, choices=TRAINABLE)
    p.add_argument("--epochs", type=int, default=None, help="Default 50.")
    p.add_argument("--lr", type=float, default=1e-2)
    p.add_argument("--bs", type=int, default=128)
    p.add_argument("--wd", type=float, default=1e-2)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--limit", type=int, default=None, help="Random subset (smoke runs).")
    p.add_argument("--tag", default=None,
                   help="Run name suffix: <model>_<tag> (layout: <model>_layout_<tag>).")
    p.add_argument("--input", default="window", choices=INPUT_MODES,
                   help="window (default): benchmark protocol, random 2.5 s crops, "
                        "run name <model>[_<tag>]. layout: supplementary layout-aware "
                        "judge on 3x4 + rhythm II masked 10 s inputs (whole input per "
                        "step, one-pass prediction), run name <model>_layout[_<tag>].")
    p.add_argument("--source", default="records100", choices=SOURCES,
                   help="records100 (default): the official 100 Hz files. records500: "
                        "the digitisation condition path (records500 -> 3x4 NaN mask at "
                        "500 Hz -> 100 Hz, as scripts/fidelity), cached in "
                        "data/fidelity/cache/ptbxl_records500_{full,layout}.npz.")
    p.add_argument("--aug", default="none", choices=AUGS,
                   help="Training augmentation: none (default); amp: one gain "
                        "U(0.9, 1.1) per record; amp_shift (--input layout only): amp "
                        "and, for half of the records, a zero-filled shift of 1-25 "
                        "samples either way of the unmasked input before the layout "
                        "mask.")
    p.add_argument("--val_every", type=int, default=None,
                   help="Validate every N epochs and at the last one (default 5; may be "
                        "changed on --resume).")
    p.add_argument("--device", default="cpu", choices=("cpu", "mps"))
    p.add_argument("--max_minutes", type=float, default=None,
                   help="Stop cleanly after the first epoch that ends past this time.")
    p.add_argument("--resume", action="store_true", help="Continue from <name>_last.pt.")
    p.add_argument("--fresh", action="store_true", help="Overwrite an existing run.")
    p.set_defaults(func=cmd_train)

    p = sub.add_parser(
        "evaluate",
        help="Fold-10 AUROC with fold-9 Youden thresholds.",
        description="Fold-10 AUROC with fold-9 Youden thresholds of one trained run. "
        "The run name is <model>[_<tag>] (loads <weights_dir>/<run>_final.pt); a "
        "layout judge is named with its _layout suffix, e.g. --model xresnet1d50_layout "
        "(or --model xresnet1d50_layout --tag smoke). The input mode and the signal "
        "source are read from the checkpoint: a records100 layout judge is evaluated on "
        "apply_layout(records100) inputs and cached as "
        "judge_<run>_layout_fold{9,10}.npz; records500 inputs come from the condition "
        "path and are cached as judge_<run>_<kind>_fold{9,10}_r500.npz. Rows of "
        "judge_fold10.csv are replaced per (model, source).",
    )
    p.add_argument("--model", required=True,
                   help="Run name, e.g. inception1d, xresnet1d50, xresnet1d50_layout.")
    p.add_argument("--tag", default=None, help="Appended as <model>_<tag>.")
    p.add_argument("--source", default=None, choices=SOURCES,
                   help="Fold-9/10 input source (default: the judge's training source).")
    p.add_argument("--device", default="cpu", choices=("cpu", "mps"))
    p.add_argument("--out", default=DEFAULT_EVAL_CSV)
    p.add_argument("--cache_dir", default=DEFAULT_CACHE_DIR)
    p.add_argument("--n_boot", type=int, default=1000)
    p.set_defaults(func=cmd_evaluate)

    p = sub.add_parser("check-benchmark", help="exp0 xresnet1d101 'all' fold-10 check.")
    p.add_argument("--device", default="cpu", choices=("cpu", "mps"))
    p.set_defaults(func=cmd_check_benchmark)

    p = sub.add_parser("params", help="Instantiate the architectures, count parameters.")
    p.set_defaults(func=cmd_params)
    return parser


def main(argv=None):
    args = get_parser().parse_args(argv)
    return args.func(args)


if __name__ == "__main__":
    sys.exit(main())
