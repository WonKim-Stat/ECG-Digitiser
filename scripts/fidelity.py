# Diagnosis-fidelity harness: do digitised ECGs keep the judge decisions of the originals?
"""Diagnosis-fidelity harness (plan item A2).

A frozen PTB-XL superdiagnostic classifier (the "judge", scripts/judge.py)
reads three versions of every record of a digitisation condition:

  full    the original 10 s, 12-lead signal (data/ptbxl_eval, 500 Hz);
  masked  the original with every sample outside the lead's printed window set
          to NaN, using the condition's ground-truth NaN mask (3x4 layout:
          2.5 s per short lead, the full 10 s for the rhythm strip II);
  dig     the digitised signal with the SAME ground-truth mask applied.

Symmetric processing: `masked` and `dig` take the identical path (mask at the
native 500 Hz, then `prepare_for_judge`), so any difference between their judge
outputs is caused by the digitisation alone.  `full` vs `masked` is the
separate layout information loss: what a 3x4 print throws away before any
digitisation happens.

Metrics (label free unless stated): Cohen's kappa of the thresholded
decisions (the headline `kappa` pools all (record, class) decisions under one
marginal, so it reads higher than a class-stratified kappa; `kappa_class_mean`
and the per-class flip counts are reported next to it), the flip rate (class
decisions that change), the record flip rate (a record flips when any class
flips), |delta p| of the sigmoid scores and, with labels, the per-class AUROC
retention.  Confidence intervals come from a bootstrap over records (all
statistics of one call share the same resamples); the record flip rate also
gets the exact Clopper-Pearson interval.  When nothing flips, the bootstrap
intervals of kappa and the flip rates are degenerate and are reported as NaN
(ci_degenerate); the exact interval is then the one to quote.

Thresholds are fold-9 Youden cut-offs on the same path as every condition
input: the records500 files through the 500 Hz mask and prepare_for_judge
(`thresholds --source records500`, the default; thresholds.csv records the
source).  The official records100 files remain available (--source
records100) but cannot be reproduced from records500: different anti-alias
filter, and records500 holds a constant value over its last ~0.1 s.  The
`path_floor_fold10` rows of layout-loss (records100 vs the records500 path at
the same thresholds) measure how robust the decisions are to that source
change; they are no longer a bias of the table.

Subcommands (all write under data/fidelity/):
  thresholds   fold-9 Youden thresholds per judge on the full and on the
               layout-masked input of the records500 condition path
               (--source records100: the official records100) -> thresholds.csv
  layout-loss  full vs layout-masked on fold 10 (records100 and the 500 Hz
               path) and on ecg_id 1-32, the records100 vs 500 Hz path noise
               floor on fold 10 and a resampling check on ecg_id 1-32
                                                            -> layout_loss.csv
  table        judge x condition fidelity next to the SNR median
                                   -> table_v0.csv and table_v0_records.csv
               (--pred_run TAG: another digitiser run of the same pages, see
               pred_dir / ev_csv; its dig logits get their own cache keys, the
               full and masked ones are shared with the default run)
  sweep-error  the realistic digitisation error of every condition scaled by
               alpha (x = masked + alpha (dig - masked) at 500 Hz, so the
               per-lead SNR is SNR(1) - 20 log10 alpha); flips against alpha 0
                         -> sweep_error_v0.csv and sweep_error_v0_records.csv
  sweep-noise  0.5-40 Hz band-limited Gaussian noise at exact per-lead window
               SNRs on the labelled fold-10 records500 (layout-masked 500 Hz
               path, labels available)
                         -> sweep_noise_v0.csv and sweep_noise_v0_records.csv
  snr-flip     record flip rate per record-SNR bin and a logistic fit
               flip ~ SNR (SNR at 5 % / 10 % flip probability, cluster
               bootstrap over ecg_ids) of both sweeps, on the 500 Hz and on
               the judge-band SNR axis, with every row weighing the same and
               with every ecg_id weighing the same         -> snr_flip_v0.csv
  margins      per-record minimum |logit - threshold| of the labelled local
               records vs the other labelled fold-10 records: is the local
               set typical?                                -> margins_v0.csv

SNR (sweeps) = src/run/evaluate.py compute_raw_snr per lead over its window
at 500 Hz: 10 log10(sum gt^2 / sum (gt - x)^2), no mean removal (lead_snr);
record SNR = median over its leads.  The judge-band SNR (snr_judge_*) is the
same formula on the prepared 100 Hz judge inputs over the resampled window
(judge_snr).  The two differ for the digitisation error, part of whose power
lies above the resampler's 50 Hz band, but not for the 0.5-40 Hz noise, so
500 Hz thresholds of the two sweeps must not be compared with each other.

Long runs.  The logit cache is saved after every chunk and a noise draw
depends only on (seed, ecg_id, level_index), so sweep-noise can be split:
`--start S --max_records N --cache_only` fills the cache for records S..S+N-1
without writing any CSV; a final run without them then reads every logit from
the cache.  A partial run that writes CSVs to the default v0 name writes
<name>_part<start>-<stop>.csv instead.  The cache is per judge and device
(fid_<judge>.npz on cpu, fid_<judge>_<device>.npz otherwise) and its tag holds
the weights SHA; run thresholds and every later subcommand on one device
(thresholds.csv records it and a mismatch is warned about).

table, sweep-error and margins refuse to write the *_v0 files for a non-v0
condition (guard_v0_outputs) before anything is loaded; such runs need --out
(table and sweep-error also --records_out) with a new name.  A table run with
a --pred_run other than the default counts as non-v0 for every condition.

Table rows end with split-half and bias extras (split_half: flips and mean
logit shift mdl_<class> per ecg_id parity) and the run tag pred_run, after
the v0 columns and the path floor; the per-record frame ends with pred_run.

Real images (Phase C2, set real_c2, table only).  The conditions real_<c> of
REAL_CONDITIONS (the 8 ECG-Image-Database D0 conditions, 199 PTB-XL records
each; CC BY-ND 4.0, local only) live outside CONDITIONS: 'all' and every
other subcommand ignore them; --conditions real / real_c2 / real_<c> selects
them, never mixed with other sets.  They need --split dev|eval (records of
SPLIT_CSV; refused for the other sets), --original_dir ptbxl500 (each record's
records500 file, no copy; the original = ground truth check stays) and an
--out / --records_out outside data/fidelity (guard_real_outputs).  Their
default run is REAL_PRED_RUN (bl_o05, the C2 queue); --pred_run TAG reads
data/real_results/out/<c>/<TAG> and data/real_results/ev/ev_<c>_<TAG>.csv.
Their rows end with 'split', their record rows with 'split' and 'patient_id';
the SNR median covers the rows of the selected records only (a selected
record with an output but no evaluation row aborts).  Their bootstrap CIs
resample patients (patient_id of SPLIT_CSV, paired cluster bootstrap); those
of the other sets resample records, as before.  A lead absent from a real
digitised file is read as missing (all NaN, judged zero-filled); for the
other sets it still aborts.  A real run also needs a --cache_dir (or
--no_cache) outside data/fidelity.  --max_records N (smoke only) keeps the
first N records of every condition.

The judge is imported lazily, so importing this module (and its unit tests)
needs neither torch nor trained weights.
"""
import argparse
import hashlib
import math
import os
import sys
import warnings
from collections.abc import Mapping

import numpy as np
import pandas as pd
import wfdb
from scipy.signal import butter, resample_poly, sosfiltfilt
from scipy.special import expit
from scipy.stats import beta as beta_dist
from scipy.stats import mannwhitneyu
from scipy.stats import rankdata

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

# ----------------------------------------------------------------------------
# Constants
# ----------------------------------------------------------------------------
LEADS = ("I", "II", "III", "aVR", "aVL", "aVF", "V1", "V2", "V3", "V4", "V5", "V6")

# Printed window of every lead in seconds (3x4 grid + rhythm strip II).
LAYOUT_3X4_II = {
    "I": (0.0, 2.5),
    "II": (0.0, 10.0),
    "III": (0.0, 2.5),
    "aVR": (2.5, 5.0),
    "aVL": (2.5, 5.0),
    "aVF": (2.5, 5.0),
    "V1": (5.0, 7.5),
    "V2": (5.0, 7.5),
    "V3": (5.0, 7.5),
    "V4": (7.5, 10.0),
    "V5": (7.5, 10.0),
    "V6": (7.5, 10.0),
}

# Mirror of scripts/judge.CLASSES (MultiLabelBinarizer order); run code checks that
# judge.classes has this order (_open_judge).
JUDGE_CLASSES = ("CD", "HYP", "MI", "NORM", "STTC")
FS_JUDGE = 100
N_JUDGE = 1000
DEFAULT_JUDGES = ("inception1d", "xresnet1d50")

ORIGINAL_DIR = "data/ptbxl_eval"
PTBXL_ROOT = "data/ptbxl"
WEIGHTS_DIR = "data/judge/weights"
FIDELITY_DIR = "data/fidelity"
CACHE_DIR = "data/fidelity/cache"
THRESHOLDS_CSV = "data/fidelity/thresholds.csv"
# Fold-9 input sources of `thresholds` (mirror of scripts/judge.SOURCES).
THRESHOLD_SOURCES = ("records100", "records500")
LAYOUT_LOSS_CSV = "data/fidelity/layout_loss.csv"
TABLE_CSV = "data/fidelity/table_v0.csv"
PRED_RUN = "inkpage_default"
PRED_SUFFIX = "-0_0000"
# Evaluation CSV of a v0 condition; {run} is the digitiser run tag (--pred_run).
EV_CSV = "data/perspective_results/ev_{cond}_{run}.csv"

_QUASI_REAL = (
    "blur", "jpeg", "gridfaint", "gridblue", "margin", "persp05", "persp15",
    "photo", "scale080", "scale088", "scale115", "scale125",
)
# Phase B conditions (500 fold-10 records each), in queue order.
B_CONDITIONS = ("clean", "scan", "aug", "rot")
# condition -> (ground-truth dir, digitised dir, set name).  v0 (gen_eval,
# quasi_real) first; rot03/rot15 are left out on purpose: they are
# lattice-return conditions, not absolute evidence.  The Phase B conditions
# (set fidelity_b) follow as b_<cond>, so their logit-cache keys
# (<cond>/<rec>/<kind>) never collide with the v0 ones.
CONDITIONS = {
    **{
        f"gen_{s}": (
            f"data/gen_eval/{s}",
            f"data/perspective_results/out/gen_{s}/{PRED_RUN}",
            "gen_eval",
        )
        for s in ("clean", "aug", "rot")
    },
    **{
        c: (
            f"data/quasi_real/{c}",
            f"data/perspective_results/out/{c}/{PRED_RUN}",
            "quasi_real",
        )
        for c in _QUASI_REAL
    },
    **{
        f"b_{c}": (f"data/fidelity/images/{c}", f"data/fidelity/out/{c}", "fidelity_b")
        for c in B_CONDITIONS
    },
}
# Sets that 'all' stands for (the v0 table).
V0_SETS = ("gen_eval", "quasi_real")
# Evaluation CSVs of the Phase B conditions; the v0 ones follow EV_CSV.
EV_CSVS = {f"b_{c}": f"data/fidelity/ev/ev_{c}.csv" for c in B_CONDITIONS}
# Another digitiser run of the same pages (table --pred_run TAG, TAG != PRED_RUN):
# digitised folder and evaluation CSV per set; {cond} = condition name, {c} =
# the name without its 'b_' prefix, {run} = TAG.  The v0 sets keep the layout of
# the perspective drivers (out/<cond>/<TAG>, EV_CSV); a set without an entry has
# no tagged layout (pred_dir / ev_csv raise).  CONDITIONS keeps the default run.
RUN_PRED_DIRS = {
    "gen_eval": "data/perspective_results/out/{cond}/{run}",
    "quasi_real": "data/perspective_results/out/{cond}/{run}",
    "fidelity_b": "data/fidelity/{run}/b_{c}/{run}",
}
RUN_EV_CSVS = {"fidelity_b": "data/fidelity/{run}/ev_{c}_{run}.csv"}

# Phase C2 real images (set real_c2): the ECG-Image-Database v2 D0 pages of 199
# PTB-XL records per condition (CC BY-ND 4.0: local evaluation only).  They live
# in their own registry, not in CONDITIONS, so 'all', the set list and every
# existing subcommand keep their meaning; only `table` reads them (with --split).
# Condition real_<c>: ground truth data/real_images/<c> (500 Hz, NaN outside the
# printed windows); the default digitiser run is the C2 queue's current-default
# run REAL_PRED_RUN; --pred_run TAG reads data/real_results/out/<c>/<TAG> and
# data/real_results/ev/ev_<c>_<TAG>.csv (RUN_PRED_DIRS / RUN_EV_CSVS, {c} = the
# name without 'real_').  Originals: the PTB-XL records500 files
# (--original_dir ptbxl500, resolved per record through ptbxl_database.csv).
REAL_SET = "real_c2"
REAL_PREFIX = "real_"
REAL_SOURCE_CONDITIONS = (
    "render", "scan_color", "scan_gray", "scan_mould_color", "scan_mould_bw",
    "photo", "photo_stained", "photo_mould",
)
REAL_PRED_RUN = "bl_o05"
REAL_CONDITIONS = {
    f"{REAL_PREFIX}{c}": (
        f"data/real_images/{c}",
        f"data/real_results/out/{c}/{REAL_PRED_RUN}",
        REAL_SET,
    )
    for c in REAL_SOURCE_CONDITIONS
}
# Group tokens of --conditions that stand for every real_c2 condition.
REAL_GROUPS = ("real", REAL_SET)
RUN_PRED_DIRS[REAL_SET] = "data/real_results/out/{c}/{run}"
RUN_EV_CSVS[REAL_SET] = "data/real_results/ev/ev_{c}_{run}.csv"
EV_CSVS.update(
    {
        f"{REAL_PREFIX}{c}": f"data/real_results/ev/ev_{c}_{REAL_PRED_RUN}.csv"
        for c in REAL_SOURCE_CONDITIONS
    }
)
# Patient split of the 199 real records (user decision 2026-09-27): every choice
# on dev, eval once for the headline numbers.
SPLIT_CSV = "data/real_images/c1/split_c2.csv"
SPLITS = ("dev", "eval")
# --original_dir token: resolve each record to its PTB-XL records500 file.
ORIGINAL_PTBXL500 = "ptbxl500"

SWEEP_ERROR_CSV = "data/fidelity/sweep_error_v0.csv"
SWEEP_NOISE_CSV = "data/fidelity/sweep_noise_v0.csv"
SNR_FLIP_CSV = "data/fidelity/snr_flip_v0.csv"
MARGINS_CSV = "data/fidelity/margins_v0.csv"
# Error-scaling factors; SNR(alpha) = SNR(1) - 20 log10(alpha), i.e. ~3 dB steps.
DEFAULT_ALPHAS = (0.0, 0.5, 0.71, 1.0, 1.41, 2.0, 2.83, 4.0, 5.66)
# Target window SNRs (dB) of the band-limited noise sweep; inf = no noise.
DEFAULT_SNRS = (math.inf, 50.0, 40.0, 35.0, 30.0, 25.0, 22.0, 20.0, 18.0, 15.0, 12.0, 10.0)
NOISE_BAND_HZ = (0.5, 40.0)
NOISE_ORDER = 4  # scipy.signal.butter N of the band-pass (applied with sosfiltfilt)
# Record-SNR bins of snr-flip, left closed: [lo, hi).
SNR_BIN_EDGES = (-math.inf, 8.0, 10.0, 12.0, 14.0, 16.0, 18.0, 20.0, 22.0, 24.0, 26.0,
                 math.inf)
FLIP_PROBS = (0.05, 0.10)  # flip probabilities whose SNR the logistic fit reports
MARGIN_CUTS = (0.25, 0.5, 1.0)  # logit margins of the margins subcommand
MIN_FIT_FLIPS = 10  # logistic fits resting on fewer flips get a note
# snr_axis of snr_flip_v0.csv: the 500 Hz window SNR (as the evaluation CSVs)
# and the SNR of the prepared 100 Hz judge inputs (judge_snr).
SNR_AXES = ("window_500hz", "judge_100hz")
WEIGHTINGS = ("rows", "ecg_id")  # logistic fit: every row / every ecg_id weighs 1

SMALL_CONDITION = 8  # conditions with <= this many records get the wide-CI note
SMALL_POSITIVES = 5  # classes with fewer positives or negatives are listed in the note
MIN_BOOT_FRAC = 0.9  # retention CIs resting on fewer valid draws are flagged
EDGE_SAMPLES = 10  # head / tail zone of the signal-difference checks (100 Hz samples)


def _path(path):
    """Resolve a repo-relative path against the repository root."""
    return path if os.path.isabs(path) else os.path.join(REPO_ROOT, path)


def _judge_module():
    """Import scripts/judge.py lazily (it needs torch and the weights)."""
    try:
        from scripts import judge
    except ImportError:
        if REPO_ROOT not in sys.path:
            sys.path.insert(0, REPO_ROOT)
        from scripts import judge
    return judge


# ----------------------------------------------------------------------------
# Signals
# ----------------------------------------------------------------------------
def layout_mask(n_samples=5000, fs=500, layout=LAYOUT_3X4_II):
    """Boolean (n_samples, 12) mask of the printed window of every lead."""
    mask = np.zeros((int(n_samples), len(LEADS)), dtype=bool)
    for j, lead in enumerate(LEADS):
        t0, t1 = layout[lead]
        start = int(round(t0 * fs))
        end = min(int(round(t1 * fs)), int(n_samples))
        mask[start:end, j] = True
    return mask


def gt_mask(gt_signal):
    """Valid-sample mask of a ground-truth signal (NaN outside the windows)."""
    return np.isfinite(np.asarray(gt_signal, dtype=float))


def apply_mask(x, mask):
    """Copy of `x` (float64) with NaN wherever `mask` is False."""
    out = np.array(x, dtype=np.float64, copy=True)
    mask = np.asarray(mask, dtype=bool)
    if out.shape != mask.shape:
        raise ValueError(
            f"Signal shape {out.shape} does not match mask shape {mask.shape}."
        )
    out[~mask] = np.nan
    return out


def read_record(path, allow_missing=False):
    """Read a WFDB record; returns (signal (n, 12) in LEADS order, fs).

    Leads are matched by sig_name, case-insensitively (the PTB-XL originals
    spell AVR/AVL/AVF, the digitiser writes aVR/aVL/aVF).  A lead absent from
    the record aborts, unless allow_missing (digitised records only: the
    digitiser drops a lead it cannot trace on some real photos): the lead is
    then all NaN, i.e. missing as in src/run/evaluate.py, with a warning.
    """
    base = path[:-4] if path.endswith((".hea", ".dat")) else path
    record = wfdb.rdrecord(base)
    names = [str(s).upper() for s in record.sig_name]
    columns, absent = [], []
    for lead in LEADS:
        if lead.upper() not in names:
            if not allow_missing:
                raise ValueError(
                    f"Lead {lead} missing in {base} (sig_name {record.sig_name})."
                )
            absent.append(lead)
            columns.append(None)
            continue
        columns.append(names.index(lead.upper()))
    p_signal = np.asarray(record.p_signal, dtype=np.float64)
    if not absent:
        return p_signal[:, columns], float(record.fs)
    signal = np.full((p_signal.shape[0], len(LEADS)), np.nan)
    for j, col in enumerate(columns):
        if col is not None:
            signal[:, j] = p_signal[:, col]
    print(
        f"WARNING: lead(s) {absent} absent in {base}; treated as missing.",
        file=sys.stderr,
    )
    return signal, float(record.fs)


def _rate_ratio(fs_in, fs_out):
    """Integer (up, down) of fs_out / fs_in, reduced by their gcd."""
    fi, fo = int(round(fs_in)), int(round(fs_out))
    if fi <= 0 or fo <= 0 or abs(fi - fs_in) > 1e-9 or abs(fo - fs_out) > 1e-9:
        raise ValueError(
            f"Sampling rates must be positive integers, got {fs_in} and {fs_out}."
        )
    g = math.gcd(fi, fo)
    return fo // g, fi // g


def fill_gaps(x):
    """Per lead: fill every NaN by linear interpolation of the finite samples.

    Interior NaN gaps are interpolated linearly; samples before the first and
    after the last finite sample hold that edge value (np.interp), so the
    anti-alias filter of the resampler sees no artificial step at a window
    boundary.  The valid span of a lead is [first finite, last finite sample].
    Returns (filled float64 (n, L), span bool (n, L)); a lead without any
    finite sample is all zero with an empty span.
    """
    x = np.asarray(x, dtype=np.float64)
    filled = np.zeros_like(x)
    span = np.zeros(x.shape, dtype=bool)
    t = np.arange(x.shape[0])
    for j in range(x.shape[1]):
        idx = np.flatnonzero(np.isfinite(x[:, j]))
        if idx.size == 0:
            continue
        span[idx[0] : idx[-1] + 1, j] = True
        if idx.size == x.shape[0]:
            filled[:, j] = x[:, j]
        else:
            filled[:, j] = np.interp(t, idx, x[idx, j])
    return filled, span


def resample_span(span, up, down):
    """Boolean span mask on the resample_poly output grid.

    Output sample k sits at input position k * down / up; we take the input
    sample at or before it (index k * down // up).  At 500 -> 100 Hz a window
    [1250, 2500) becomes exactly [250, 500).
    """
    n = span.shape[0]
    n_out = -(-n * up // down)  # ceil(n * up / down) = len(resample_poly output)
    source = np.minimum(np.arange(n_out) * down // up, n - 1)
    return span[source]


def prepare_for_judge(x, fs_in, fs_out=FS_JUDGE, n_out=N_JUDGE):
    """Turn a (possibly NaN-masked) signal into the judge input.

    Per lead: interior NaN gaps are interpolated and the edge values held
    outside the valid span (`fill_gaps`); the signal is resampled with
    scipy.signal.resample_poly(up, down, padtype='line') (identity when the
    rates are equal), so neither a window boundary nor the record ends
    produce a step; output samples outside the resampled span are set to 0;
    the result is zero-padded or truncated to n_out.  Returns float32
    (n_out, L) in mV, always finite.  The judge applies its own
    normalisation (the benchmark's global scaler).
    """
    x = np.asarray(x, dtype=np.float64)
    if x.ndim != 2:
        raise ValueError(
            f"Expected a (n_samples, n_leads) signal, got shape {x.shape}."
        )
    up, down = _rate_ratio(fs_in, fs_out)
    filled, span = fill_gaps(x)
    if up == down:
        y, span_out = filled, span
    else:
        y = resample_poly(filled, up, down, axis=0, padtype="line")
        span_out = resample_span(span, up, down)
    y = np.where(span_out, y, 0.0)
    out = np.zeros((int(n_out), x.shape[1]), dtype=np.float32)
    m = min(int(n_out), y.shape[0])
    out[:m] = y[:m]
    return out


def window_qc(x, mask):
    """Missing data of a (digitised) signal inside the ground-truth windows.

    Returns per-lead arrays: n_window (window samples), n_nan (non-finite
    samples inside the window), nan_frac (n_nan / n_window, NaN for a lead
    without a window) and missing (the lead has no finite sample in its window).
    """
    x = np.asarray(x, dtype=float)
    mask = np.asarray(mask, dtype=bool)
    n_window = mask.sum(axis=0)
    n_valid = (np.isfinite(x) & mask).sum(axis=0)
    n_nan = n_window - n_valid
    with np.errstate(invalid="ignore", divide="ignore"):
        nan_frac = np.where(n_window > 0, n_nan / np.maximum(n_window, 1), np.nan)
    return {
        "n_window": n_window,
        "n_nan": n_nan,
        "nan_frac": nan_frac,
        "missing": (n_window > 0) & (n_valid == 0),
    }


def input_sha1(x):
    """SHA-1 of a prepared judge input (float32 bytes + shape)."""
    x = np.ascontiguousarray(x, dtype=np.float32)
    h = hashlib.sha1(str(x.shape).encode())
    h.update(x.tobytes())
    return h.hexdigest()


def _zone_masks(n, n_leads, mask=None, edge=EDGE_SAMPLES):
    """Head (k < edge), interior and tail (k >= n - edge) masks, within `mask`."""
    m = np.ones((n, n_leads), dtype=bool) if mask is None else np.asarray(mask, bool)
    k = np.arange(n)[:, None]
    return {
        "head": (k < edge) & m,
        "interior": (k >= edge) & (k < n - edge) & m,
        "tail": (k >= n - edge) & m,
    }


def _zone_summary(values):
    """sig_<zone>_{max,median,p99}_mv of {zone: absolute differences}."""
    out = {}
    for zone, v in values.items():
        v = np.asarray(v).ravel()
        empty = v.size == 0
        out[f"sig_{zone}_max_mv"] = float("nan") if empty else float(v.max())
        out[f"sig_{zone}_median_mv"] = float("nan") if empty else float(np.median(v))
        out[f"sig_{zone}_p99_mv"] = (
            float("nan") if empty else float(np.percentile(v, 99))
        )
    return out


def signal_diff_stats(a, b, mask=None, edge=EDGE_SAMPLES):
    """|a - b| of prepared inputs, split into head, interior and tail.

    a, b: (N, n, L) or (n, L).  Zones along the time axis: head k < edge,
    tail k >= n - edge, interior the rest; only positions where `mask`
    (n, L) is True count (default all).  Returns sig_<zone>_{max,median,p99}_mv
    (the records500 hold tail and the resampler's record ends sit in head and
    tail, so the interior shows the filter mismatch alone).
    """
    d = np.abs(np.asarray(a, dtype=np.float64) - np.asarray(b, dtype=np.float64))
    if d.ndim == 2:
        d = d[None]
    zones = _zone_masks(d.shape[1], d.shape[2], mask, edge)
    return _zone_summary({zone: d[:, m] for zone, m in zones.items()})


# ----------------------------------------------------------------------------
# Metrics (pure numpy / scipy)
# ----------------------------------------------------------------------------
def sigmoid(z):
    """Logistic function, elementwise."""
    return expit(np.asarray(z, dtype=float))


def decisions(logits, thr):
    """Binary decisions logits > thr per class.

    Strict, as the benchmark's utils.apply_thresholds (`p > thresholds`); its
    argmax fallback for all-negative rows is not applied, so classes stay
    independent.  A NaN threshold gives all False.  scripts/judge.binarize
    uses the same convention.
    """
    return np.asarray(logits, dtype=float) > np.asarray(thr, dtype=float)


def cohen_kappa(a, b):
    """Cohen's kappa of two binary raters (flattened).

    kappa = (p_o - p_e) / (1 - p_e) with p_e from the marginals; 1.0 when
    p_e == 1 (both raters constant and identical); NaN for empty input.
    """
    a = np.asarray(a, dtype=bool).ravel()
    b = np.asarray(b, dtype=bool).ravel()
    if a.size != b.size:
        raise ValueError("Raters must have the same number of decisions.")
    if a.size == 0:
        return float("nan")
    p_o = float(np.mean(a == b))
    pa, pb = float(np.mean(a)), float(np.mean(b))
    p_e = pa * pb + (1.0 - pa) * (1.0 - pb)
    if p_e >= 1.0 - 1e-12:
        return 1.0
    return (p_o - p_e) / (1.0 - p_e)


def flip_matrix(dec_ref, dec_test):
    """Boolean matrix of the decisions that differ."""
    return np.asarray(dec_ref, dtype=bool) != np.asarray(dec_test, dtype=bool)


def flip_rate(dec_ref, dec_test):
    """Fraction of (record, class) decisions that differ."""
    flips = flip_matrix(dec_ref, dec_test)
    return float(flips.mean()) if flips.size else float("nan")


def flip_counts(dec_ref, dec_test):
    """Number of flipped records per class."""
    return flip_matrix(dec_ref, dec_test).sum(axis=0).astype(int)


def record_flip_rate(dec_ref, dec_test):
    """Fraction of records with at least one flipped class."""
    flips = flip_matrix(dec_ref, dec_test)
    if flips.shape[0] == 0:
        return float("nan")
    return float(flips.reshape(flips.shape[0], -1).any(axis=1).mean())


def clopper_pearson(k, n, level=0.95):
    """Exact binomial CI of k successes out of n."""
    if n <= 0:
        return float("nan"), float("nan")
    alpha = 1.0 - level
    lo = 0.0 if k <= 0 else float(beta_dist.ppf(alpha / 2.0, k, n - k + 1))
    hi = 1.0 if k >= n else float(beta_dist.ppf(1.0 - alpha / 2.0, k + 1, n - k))
    return lo, hi


def abs_dp(logit_ref, logit_test):
    """|sigmoid(ref) - sigmoid(test)| elementwise."""
    return np.abs(sigmoid(logit_ref) - sigmoid(logit_test))


def abs_dp_stats(logit_ref, logit_test):
    """Mean, median and max of |delta p| over all (record, class) pairs."""
    d = abs_dp(logit_ref, logit_test)
    if d.size == 0:
        return {"mean": float("nan"), "median": float("nan"), "max": float("nan")}
    return {"mean": float(d.mean()), "median": float(np.median(d)), "max": float(d.max())}


def auroc(y_true, scores):
    """Rank-based AUROC (= sklearn roc_auc_score, ties averaged); NaN if a
    class lacks positives or negatives."""
    y = np.asarray(y_true).astype(bool).ravel()
    s = np.asarray(scores, dtype=float).ravel()
    n_pos = int(y.sum())
    n_neg = y.size - n_pos
    if n_pos == 0 or n_neg == 0:
        return float("nan")
    ranks = rankdata(s)
    return float((ranks[y].sum() - n_pos * (n_pos + 1) / 2.0) / (n_pos * n_neg))


def auroc_retention(y, logit_ref, logit_test, labelled=None, require=None):
    """Per-class AUROC of both systems on the labelled records and their ratio.

    Labelled records are those with >= 1 class unless `labelled` is given.
    A class is undefined (NaN) when it lacks positives or negatives there.
    Macro retention = macro_test / macro_ref over the classes defined in both
    (`defined`).  With `require` (bool per class) the macro averages exactly
    those classes and is NaN unless all of them are defined, which keeps a
    bootstrap draw on the class set of the full sample.
    """
    y = np.asarray(y, dtype=int)
    ref = np.asarray(logit_ref, dtype=float)
    test = np.asarray(logit_test, dtype=float)
    if labelled is None:
        labelled = y.sum(axis=1) > 0
    labelled = np.asarray(labelled, dtype=bool)
    yl, rl, tl = y[labelled], ref[labelled], test[labelled]
    n_classes = y.shape[1]
    auc_ref = np.array([auroc(yl[:, c], rl[:, c]) for c in range(n_classes)])
    auc_test = np.array([auroc(yl[:, c], tl[:, c]) for c in range(n_classes)])
    with np.errstate(invalid="ignore", divide="ignore"):
        retention = auc_test / auc_ref
    both = np.isfinite(auc_ref) & np.isfinite(auc_test)
    if require is None:
        use, ok = both, bool(both.any())
    else:
        use = np.asarray(require, dtype=bool)
        if use.shape != both.shape:
            raise ValueError(f"require has shape {use.shape}, expected {both.shape}.")
        ok = bool(use.any()) and bool(both[use].all())
    macro_ref = float(auc_ref[use].mean()) if ok else float("nan")
    macro_test = float(auc_test[use].mean()) if ok else float("nan")
    macro_ret = macro_test / macro_ref if ok and macro_ref > 0 else float("nan")
    n_pos = yl.sum(axis=0).astype(int)
    return {
        "auroc_ref": auc_ref,
        "auroc_test": auc_test,
        "retention": retention,
        "defined": both,
        "n_pos": n_pos,
        "n_neg": (yl.shape[0] - n_pos).astype(int),
        "macro_ref": macro_ref,
        "macro_test": macro_test,
        "macro_retention": macro_ret,
    }


def cluster_members(groups, n_records):
    """Record indices of each cluster of `groups` (one label per record), the
    clusters in order of first appearance (pd.factorize); ValueError on a
    length mismatch or a missing label."""
    labels = pd.Series(list(groups), dtype=object)
    if len(labels) != int(n_records):
        raise ValueError(f"{len(labels)} cluster labels for {n_records} records.")
    if labels.isna().any():
        raise ValueError("Cluster labels (patient_id) must not be missing.")
    codes, uniques = pd.factorize(labels)
    return [np.flatnonzero(codes == k) for k in range(len(uniques))]


def bootstrap_ci(
    stat_fn, n_records, n_boot=2000, seed=0, level=0.95, groups=None
):
    """Percentile CI of stat_fn(idx) over record resamples with replacement.

    stat_fn receives an index array of length n_records.  Draws giving a
    non-finite statistic are skipped.  Returns (lo, hi, n_valid_draws).
    The same seed gives the same resamples for every statistic.

    groups (one cluster label per record, e.g. patient_id) resamples clusters
    instead (cluster bootstrap): each draw picks as many clusters as there are
    and stat_fn gets the indices of all their records, so its length varies.
    With one record per cluster in record order this is the record bootstrap
    draw for draw.  groups None = the record bootstrap, unchanged.
    """
    if n_boot is None or n_boot <= 0 or n_records <= 0:
        return float("nan"), float("nan"), 0
    rng = np.random.default_rng(seed)
    if groups is None:
        draws = rng.integers(0, n_records, size=(int(n_boot), int(n_records)))
    else:
        members = cluster_members(groups, n_records)
        picks = rng.integers(0, len(members), size=(int(n_boot), len(members)))
        draws = [np.concatenate([members[k] for k in row]) for row in picks]
    values = np.array([stat_fn(idx) for idx in draws], dtype=float)
    values = values[np.isfinite(values)]
    if values.size == 0:
        return float("nan"), float("nan"), 0
    alpha = (1.0 - level) / 2.0
    lo = float(np.percentile(values, 100.0 * alpha))
    hi = float(np.percentile(values, 100.0 * (1.0 - alpha)))
    return lo, hi, int(values.size)


def compare(
    y,
    logit_ref,
    logit_test,
    thr,
    n_boot=2000,
    seed=0,
    thr_test=None,
    labelled=None,
    classes=JUDGE_CLASSES,
    level=0.95,
    groups=None,
):
    """All fidelity metrics of a test system against a reference, as a flat dict.

    logit_ref / logit_test: (N, C) logits of the same N records.  Decisions use
    `thr` for the reference and `thr_test` (default `thr`) for the test system.
    y: (N, C) 0/1 labels or None (label-free; AUROC fields NaN).
    The headline kappa pools all (record, class) decisions under one marginal;
    kappa_<class> is per class and kappa_class_mean their mean.  CIs resample
    records (n_boot <= 0 skips them), or with groups (one cluster label per
    record, e.g. patient_id) whole clusters (bootstrap_ci), paired as before:
    reference and test share every draw.  Without any flip the bootstrap CIs of
    kappa, flip rate and record flip rate are degenerate: they are NaN and
    ci_degenerate is True (quote rflip_exact_lo/hi instead).  The CI of the
    macro AUROC retention averages, in every draw, the classes defined in the
    full sample and skips draws where one of them is undefined
    (ret_boot_valid / ret_boot_frac say how many draws it rests on).
    """
    ref = np.asarray(logit_ref, dtype=float)
    test = np.asarray(logit_test, dtype=float)
    if ref.shape != test.shape or ref.ndim != 2:
        raise ValueError(
            f"Logit shapes differ or are not 2-D: {ref.shape} vs {test.shape}."
        )
    n, n_classes = ref.shape
    classes = tuple(classes)
    if len(classes) != n_classes:
        raise ValueError(f"{len(classes)} class names for {n_classes} logit columns.")
    thr = np.asarray(thr, dtype=float)
    thr_test = thr if thr_test is None else np.asarray(thr_test, dtype=float)
    dec_ref = decisions(ref, thr)
    dec_test = decisions(test, thr_test)
    flips = flip_matrix(dec_ref, dec_test)
    rec_flips = flips.any(axis=1)
    dp = abs_dp(ref, test)
    degenerate = bool(n > 0 and not rec_flips.any())

    out = {"n_records": int(n)}
    boot = dict(n_records=n, n_boot=n_boot, seed=seed, level=level)
    if groups is not None:
        boot["groups"] = groups

    out["kappa"] = cohen_kappa(dec_ref, dec_test)
    out["kappa_lo"], out["kappa_hi"], out["kappa_boot_valid"] = bootstrap_ci(
        lambda idx: cohen_kappa(dec_ref[idx], dec_test[idx]), **boot
    )
    per_class = [cohen_kappa(dec_ref[:, c], dec_test[:, c]) for c in range(n_classes)]
    for name, value in zip(classes, per_class):
        out[f"kappa_{name}"] = value
    finite = [v for v in per_class if np.isfinite(v)]
    out["kappa_class_mean"] = float(np.mean(finite)) if finite else float("nan")

    out["flip_rate"] = float(flips.mean()) if flips.size else float("nan")
    out["flip_lo"], out["flip_hi"], _ = bootstrap_ci(
        lambda idx: flips[idx].mean(), **boot
    )
    out["record_flip_rate"] = float(rec_flips.mean()) if n else float("nan")
    out["rflip_boot_lo"], out["rflip_boot_hi"], _ = bootstrap_ci(
        lambda idx: rec_flips[idx].mean(), **boot
    )
    out["rflip_exact_lo"], out["rflip_exact_hi"] = clopper_pearson(
        int(rec_flips.sum()), n, level=level
    )
    out["n_flips"] = int(flips.sum())
    out["n_record_flips"] = int(rec_flips.sum())
    for c, name in enumerate(classes):
        out[f"flips_{name}"] = int(flips[:, c].sum())
    out["ci_degenerate"] = degenerate
    if degenerate:
        for key in (
            "kappa_lo", "kappa_hi", "flip_lo", "flip_hi", "rflip_boot_lo",
            "rflip_boot_hi",
        ):
            out[key] = float("nan")

    stats = abs_dp_stats(ref, test)
    out["mean_abs_dp"] = stats["mean"]
    out["dp_lo"], out["dp_hi"], _ = bootstrap_ci(lambda idx: dp[idx].mean(), **boot)
    out["median_abs_dp"] = stats["median"]
    out["max_abs_dp"] = stats["max"]

    if y is None:
        out["n_labelled"] = 0
        for key in (
            "auroc_ref_macro", "auroc_test_macro", "auroc_retention_macro",
            "ret_lo", "ret_hi", "ret_boot_frac",
        ):
            out[key] = float("nan")
        out["ret_boot_valid"] = 0
        for name in classes:
            out[f"auroc_ref_{name}"] = out[f"auroc_test_{name}"] = float("nan")
            out[f"ret_{name}"] = float("nan")
            out[f"npos_{name}"] = 0
            out[f"nneg_{name}"] = 0
        return out

    y = np.asarray(y, dtype=int)
    if y.shape != ref.shape:
        raise ValueError(f"Label shape {y.shape} does not match logits {ref.shape}.")
    lab = y.sum(axis=1) > 0 if labelled is None else np.asarray(labelled, dtype=bool)
    ret = auroc_retention(y, ref, test, labelled=lab)
    defined = ret["defined"]
    out["n_labelled"] = int(lab.sum())
    out["auroc_ref_macro"] = ret["macro_ref"]
    out["auroc_test_macro"] = ret["macro_test"]
    out["auroc_retention_macro"] = ret["macro_retention"]
    out["ret_lo"], out["ret_hi"], out["ret_boot_valid"] = bootstrap_ci(
        lambda idx: auroc_retention(
            y[idx], ref[idx], test[idx], labelled=lab[idx], require=defined
        )["macro_retention"],
        **boot,
    )
    n_draws = int(n_boot) if n_boot and n_boot > 0 and n > 0 else 0
    out["ret_boot_frac"] = (
        out["ret_boot_valid"] / n_draws if n_draws else float("nan")
    )
    for c, name in enumerate(classes):
        out[f"auroc_ref_{name}"] = float(ret["auroc_ref"][c])
        out[f"auroc_test_{name}"] = float(ret["auroc_test"][c])
        out[f"ret_{name}"] = float(ret["retention"][c])
        out[f"npos_{name}"] = int(ret["n_pos"][c])
        out[f"nneg_{name}"] = int(ret["n_neg"][c])
    return out


HALVES = ("odd", "even")  # ecg_id % 2 == 1 / == 0


def split_half_columns(classes=JUDGE_CLASSES):
    """Keys of split_half in their order (the table columns after the path floor)."""
    cols = [
        f"{stat}_{half}"
        for stat in ("n_records", "n_flips", "flip_rate", "record_flip_rate")
        for half in HALVES
    ]
    for c in classes:
        cols += [f"mdl_{c}", f"mdl_{c}_odd", f"mdl_{c}_even"]
    return tuple(cols)


def split_half(ecg_ids, logit_ref, logit_test, thr, classes=JUDGE_CLASSES):
    """Flips and mean logit shift of test vs ref, overall and per ecg_id parity.

    The halves (odd / even ecg_id) are fixed by the record alone, so a choice
    made on one half can be checked on the other.  Flips are those of compare
    at the same thresholds (ref > thr != test > thr), so the halves add up to
    its n_flips.  mdl_<c> = mean(test - ref) of class c over ALL records,
    labelled or not (the definition of the B4 low-pass screen), and _odd /
    _even over the records of one parity.  An empty half gives NaN rates and
    means.  Returns a dict in split_half_columns order.
    """
    ids = np.asarray(ecg_ids, dtype=int)
    ref = np.asarray(logit_ref, dtype=float)
    test = np.asarray(logit_test, dtype=float)
    if ref.shape != test.shape or ref.ndim != 2 or ref.shape[0] != ids.size:
        raise ValueError(
            f"{ids.size} ecg_ids for logits of shapes {ref.shape} and {test.shape}."
        )
    flips = flip_matrix(decisions(ref, thr), decisions(test, thr))
    dl = test - ref
    parts = {"odd": ids % 2 == 1, "even": ids % 2 == 0}
    nan = float("nan")
    out = {}
    for half in HALVES:
        out[f"n_records_{half}"] = int(parts[half].sum())
    for half in HALVES:
        out[f"n_flips_{half}"] = int(flips[parts[half]].sum())
    for half in HALVES:
        f = flips[parts[half]]
        out[f"flip_rate_{half}"] = float(f.mean()) if f.size else nan
    for half in HALVES:
        f = flips[parts[half]]
        out[f"record_flip_rate_{half}"] = float(f.any(axis=1).mean()) if len(f) else nan
    for c, name in enumerate(classes):
        out[f"mdl_{name}"] = float(dl[:, c].mean()) if ids.size else nan
        for half in HALVES:
            d = dl[parts[half], c]
            out[f"mdl_{name}_{half}"] = float(d.mean()) if d.size else nan
    return out


# ----------------------------------------------------------------------------
# SNR
# ----------------------------------------------------------------------------
def _as_bool(series):
    """A 'missing' column as booleans (it may be read back as strings)."""
    if series.dtype == bool:
        return series
    return series.astype(str).str.strip().str.lower().isin(("true", "1", "1.0"))


def _present(ev):
    """Rows of an evaluation frame whose lead was not missing."""
    if "missing" not in ev.columns:
        return ev
    return ev[~_as_bool(ev["missing"]).to_numpy()]


def condition_snr_median(ev, metric="snr_raw"):
    """Condition SNR: median of `metric` over the rows with missing == False."""
    values = pd.to_numeric(_present(ev)[metric], errors="coerce")
    return float(values.median()) if len(values) else float("nan")


def record_snr(ev, metric="snr_raw"):
    """Per record: median and min of `metric` over its non-missing leads."""
    kept = _present(ev).copy()
    kept[metric] = pd.to_numeric(kept[metric], errors="coerce")
    grouped = kept.groupby("record")[metric]
    return pd.DataFrame(
        {"snr_rec_median": grouped.median(), "snr_rec_min": grouped.min()}
    )


# ----------------------------------------------------------------------------
# Judge plumbing: logit cache, labels, thresholds
# ----------------------------------------------------------------------------
def _file_sha1(path, chunk=1 << 20):
    h = hashlib.sha1()
    with open(path, "rb") as f:
        for block in iter(lambda: f.read(chunk), b""):
            h.update(block)
    return h.hexdigest()


def weights_tag(weights_dir, name):
    """Cache tag of a judge: its name plus the SHA-1 of its weights file."""
    path = os.path.join(weights_dir, f"{name}_final.pt")
    return f"{name}:{_file_sha1(path)}" if os.path.exists(path) else name


class LogitCache:
    """Judge logits keyed by strings (e.g. 'gen_clean/00001_hr/dig').

    Each entry stores the SHA-1 of the prepared input it was computed from, so
    a changed input is recomputed.  The whole file is discarded when its tag
    (judge name + weights SHA-1) differs.  Stored as plain arrays in one .npz.
    """

    def __init__(self, path=None, tag=""):
        self.path = path
        self.tag = str(tag)
        self._data = {}
        self._dirty = False
        if path and os.path.exists(path):
            with np.load(path, allow_pickle=False) as z:
                if str(z["tag"]) == self.tag:
                    for key, sha, logits in zip(z["keys"], z["sha1"], z["logits"]):
                        self._data[str(key)] = (
                            str(sha),
                            np.asarray(logits, dtype=np.float64),
                        )

    def __len__(self):
        return len(self._data)

    def get(self, key, sha):
        hit = self._data.get(key)
        if hit is None or hit[0] != sha:
            return None
        return hit[1].copy()

    def put(self, key, sha, logits):
        self._data[key] = (sha, np.asarray(logits, dtype=np.float64).copy())
        self._dirty = True

    def save(self):
        if not self.path or not self._dirty:
            return
        keys = sorted(self._data)
        os.makedirs(os.path.dirname(os.path.abspath(self.path)), exist_ok=True)
        tmp = self.path + ".tmp"
        with open(tmp, "wb") as f:
            np.savez(
                f,
                keys=np.array(keys, dtype=str),
                sha1=np.array([self._data[k][0] for k in keys], dtype=str),
                logits=np.stack([self._data[k][1] for k in keys]),
                tag=np.array(self.tag),
            )
        os.replace(tmp, self.path)
        self._dirty = False


def cached_logits(judge, arrays, keys, cache=None, batch_size=512):
    """Judge logits of prepared inputs; cached entries with the same input SHA-1
    are reused."""
    arrays = list(arrays)
    if len(arrays) != len(keys):
        raise ValueError("One key per input is required.")
    n_classes = len(judge.classes)
    out = np.full((len(arrays), n_classes), np.nan)
    shas = [input_sha1(a) for a in arrays]
    todo = []
    for i, (key, sha) in enumerate(zip(keys, shas)):
        hit = cache.get(key, sha) if cache is not None else None
        if hit is None:
            todo.append(i)
        else:
            out[i] = hit
    if todo:
        batch = np.stack([np.asarray(arrays[i], dtype=np.float32) for i in todo])
        logits = judge.predict_logits(batch, batch_size=batch_size)
        logits = np.asarray(logits, dtype=np.float64).reshape(len(todo), n_classes)
        for row, i in enumerate(todo):
            out[i] = logits[row]
            if cache is not None:
                cache.put(keys[i], shas[i], logits[row])
    return out


def labels_for(db, ecg_ids, classes=JUDGE_CLASSES):
    """(y (N, C) int, labelled (N,) bool, strat_fold (N,) int) of the given ecg_ids.

    db is judge.load_ptbxl_db() (index ecg_id, 0/1 class columns, strat_fold);
    None gives (None, None, all -1).  Unknown ids are unlabelled with fold -1.
    """
    ids = [int(i) for i in ecg_ids]
    if db is None:
        return None, None, np.full(len(ids), -1, dtype=int)
    sub = db.reindex(ids)
    y = sub[list(classes)].fillna(0).to_numpy(dtype=int)
    labelled = y.sum(axis=1) > 0
    folds = sub["strat_fold"].fillna(-1).to_numpy(dtype=int)
    return y, labelled, folds


def ecg_id_of(record):
    """PTB-XL ecg_id of a record name such as '00001_hr'."""
    return int(str(record)[:5])


def load_thresholds(path, judge_name, classes=JUDGE_CLASSES):
    """{'full': (C,), 'layout': (C,)} logit thresholds of one judge (thresholds.csv)."""
    if not os.path.exists(path):
        raise FileNotFoundError(
            f"{path} not found; run `python -m scripts.fidelity thresholds` first."
        )
    df = pd.read_csv(path)
    out = {}
    for kind in ("full", "layout"):
        sub = df[(df["judge"] == judge_name) & (df["input"] == kind)]
        sub = sub.set_index("class")
        missing = [c for c in classes if c not in sub.index]
        if missing:
            raise ValueError(
                f"No {kind} thresholds for {judge_name} classes {missing} in {path}."
            )
        out[kind] = sub.loc[list(classes), "threshold_logit"].to_numpy(dtype=float)
    return out


def check_threshold_device(path, judge_name, device):
    """Warn when thresholds.csv says a judge's thresholds were fitted on logits
    of another device (files without a device column are not checked).
    Returns the other devices found."""
    if not path or not os.path.exists(path):
        return []
    df = pd.read_csv(path)
    if "device" not in df.columns:
        return []
    found = df.loc[df["judge"] == judge_name, "device"].dropna().astype(str)
    other = sorted(set(found) - {str(device)})
    if other:
        print(
            f"WARNING: {judge_name} thresholds in {path} come from {other} logits, "
            f"this run uses {device}; run everything on one device.",
            file=sys.stderr,
        )
    return other


def cache_file(cache_dir, name, device="cpu"):
    """Logit-cache file of a judge on a device: fid_<name>.npz on cpu,
    fid_<name>_<device>.npz otherwise, so a device switch never empties the
    other device's cache (LogitCache drops a file whose tag differs)."""
    device = str(device)
    suffix = "" if device == "cpu" else "_" + "".join(
        ch if ch.isalnum() else "_" for ch in device
    )
    return os.path.join(cache_dir, f"fid_{name}{suffix}.npz")


INPUT_KINDS = ("full", "layout")


def _prepare_kind(x, fs, kind):
    """Prepared judge input of one kind: 'full' or 'layout' (3x4 masked)."""
    if kind == "full":
        return prepare_for_judge(x, fs)
    if kind == "layout":
        return prepare_for_judge(apply_mask(x, layout_mask(x.shape[0], fs)), fs)
    raise ValueError(f"Unknown input kind {kind!r}.")


def ptbxl_inputs(J, ecg_ids, kind, root):
    """Prepared records100 inputs: 'full' (identity path) or 'layout' (3x4 masked)."""
    if kind not in INPUT_KINDS:
        raise ValueError(f"Unknown input kind {kind!r}.")
    signals = J.load_signals_100(list(ecg_ids), root=root)
    return [_prepare_kind(x, FS_JUDGE, kind) for x in signals]


def ptbxl_logits(J, judge, ecg_ids, kind, root, cache=None, batch_size=512):
    """Judge logits of records100 inputs of one kind (cache key 'ptbxl/<id>/<kind>')."""
    arrays = ptbxl_inputs(J, ecg_ids, kind, root)
    keys = [f"ptbxl/{int(i)}/{kind}" for i in ecg_ids]
    return cached_logits(judge, arrays, keys, cache=cache, batch_size=batch_size)


def records500_path(db, ecg_id, root):
    """Path (without extension) of the records500 file of an ecg_id."""
    return os.path.join(root, str(db.at[int(ecg_id), "filename_hr"]))


def ptbxl500_inputs(db, ecg_ids, root, kinds=INPUT_KINDS):
    """Prepared records500 inputs through the 500 -> 100 Hz condition path.

    Returns {kind: [(1000, 12) float32, ...]} in the order of ecg_ids; this
    is exactly the path of the `full` and `masked` condition inputs.
    """
    out = {kind: [] for kind in kinds}
    for ecg_id in ecg_ids:
        x, fs = read_record(records500_path(db, ecg_id, root))
        for kind in kinds:
            out[kind].append(_prepare_kind(x, fs, kind))
    return out


def ptbxl500_logits(J, judge, db, ecg_ids, kind, root, cache=None, batch_size=512):
    """Judge logits of records500 condition-path inputs of one kind (cache key
    'ptbxl500/<id>/<kind>', shared with fold10_500hz).

    The inputs come from J.load_inputs_500 (the judge's npz cache of exactly
    `ptbxl500_inputs`) when the judge module has it, else from ptbxl500_inputs.
    """
    if kind not in INPUT_KINDS:
        raise ValueError(f"Unknown input kind {kind!r}.")
    ids = [int(i) for i in ecg_ids]
    load = getattr(J, "load_inputs_500", None)
    if load is not None:
        arrays = list(load(ids, kind, root=root))
    else:
        arrays = ptbxl500_inputs(db, ids, root, kinds=(kind,))[kind]
    keys = [f"ptbxl500/{i}/{kind}" for i in ids]
    return cached_logits(judge, arrays, keys, cache=cache, batch_size=batch_size)


def _open_judge(J, name, args):
    """Load a judge and its logit cache (tag: name, weights SHA, device)."""
    weights_dir = _path(args.weights_dir)
    judge = J.Judge.load(name, weights_dir=weights_dir, device=args.device)
    if tuple(judge.classes) != tuple(J.CLASSES):
        raise ValueError(
            f"Judge {name} predicts classes {tuple(judge.classes)}, but labels and "
            f"thresholds are indexed by {tuple(J.CLASSES)}."
        )
    cache = None
    if not args.no_cache:
        cache_path = cache_file(_path(args.cache_dir), name, args.device)
        sha = getattr(judge, "weights_sha256", None)
        tag = f"{name}:{sha}" if sha else weights_tag(weights_dir, name)
        cache = LogitCache(cache_path, tag=f"{tag}:{args.device}")
    return judge, cache


# ----------------------------------------------------------------------------
# Conditions
# ----------------------------------------------------------------------------
def list_records(gt_dir):
    """Sorted record names (header basenames) of a ground-truth folder."""
    return sorted(f[:-4] for f in os.listdir(gt_dir) if f.endswith(".hea"))


def _condition_records(gt_dir, pred_dir, records=None, pred_suffix=PRED_SUFFIX):
    """Checked (records, digitised base paths) of a condition.

    Aborts when a folder is missing, the ground-truth folder has no records or
    none of the records has a digitised output (wrong folder).
    """
    for label, folder in (("ground-truth", gt_dir), ("digitised", pred_dir)):
        if not os.path.isdir(folder):
            raise FileNotFoundError(f"{label} folder {folder} not found.")
    records = list(records) if records is not None else list_records(gt_dir)
    if not records:
        raise ValueError(f"No .hea records in {gt_dir}.")
    pred_bases = [os.path.join(pred_dir, rec + pred_suffix) for rec in records]
    if not any(os.path.exists(base + ".hea") for base in pred_bases):
        raise ValueError(
            f"None of the {len(records)} records of {gt_dir} has a digitised output "
            f"<record>{pred_suffix}.hea in {pred_dir}; wrong folder?"
        )
    return records, pred_bases


def original_base(original_dir, rec):
    """Base path of the original of a record: <original_dir>/<rec> for a folder,
    original_dir[rec] for a mapping (ptbxl500_originals)."""
    if isinstance(original_dir, Mapping):
        if rec not in original_dir:
            raise ValueError(f"{rec}: no original in the given record -> path map.")
        return original_dir[rec]
    return os.path.join(original_dir, rec)


def ptbxl500_originals(db, records, root):
    """{record: records500 base path} of PTB-XL record names ('00022_hr').

    The path comes from ptbxl_database.csv (records500_path); aborts when an
    ecg_id is not in the database, the file's name is not the record name or
    its header is missing.  Nothing is copied; the original = ground truth
    check of _read_condition_record still runs on every record.
    """
    out, absent = {}, []
    for rec in records:
        ecg_id = ecg_id_of(rec)
        if ecg_id not in db.index:
            raise ValueError(f"{rec}: ecg_id {ecg_id} is not in the PTB-XL database.")
        base = records500_path(db, ecg_id, root)
        if os.path.basename(base) != str(rec):
            raise ValueError(
                f"{rec}: the records500 file of ecg_id {ecg_id} is {base}, "
                "another record name."
            )
        if not os.path.exists(base + ".hea"):
            absent.append(base)
        out[str(rec)] = base
    if absent:
        raise FileNotFoundError(
            f"{len(absent)} records500 file(s) missing, e.g. {absent[:3]}."
        )
    return out


def _read_condition_record(
    gt_dir, original_dir, rec, pred_base, tol=1e-6, allow_missing=False
):
    """(original, GT mask, digitised, present, fs) of one record, checked.

    original_dir is a folder of <rec>.hea/.dat or a {rec: base path} mapping
    (original_base).  allow_missing (the real-image conditions only, see
    run_table) reads a digitised lead absent from its file as missing
    (read_record); otherwise an absent lead aborts, as it always did.

    Aborts when the original differs from the ground truth inside the windows
    or a shape / rate differs.  A missing digitised record is returned all NaN
    (present False) with a warning.
    """
    gt, fs = read_record(os.path.join(gt_dir, rec))
    orig, fs_orig = read_record(original_base(original_dir, rec))
    if fs_orig != fs or orig.shape != gt.shape:
        raise ValueError(
            f"{rec}: original {orig.shape} @ {fs_orig} Hz vs ground truth "
            f"{gt.shape} @ {fs} Hz."
        )
    mask = gt_mask(gt)
    diff = float(np.max(np.abs(orig[mask] - gt[mask]))) if mask.any() else 0.0
    if not diff <= tol:
        raise ValueError(
            f"{rec}: original differs from the ground truth inside the windows "
            f"(max |diff| {diff}); wrong original or ground-truth folder?"
        )
    if os.path.exists(pred_base + ".hea"):
        d, fs_pred = read_record(pred_base, allow_missing=allow_missing)
        if fs_pred != fs or d.shape != gt.shape:
            raise ValueError(
                f"{rec}: digitised {d.shape} @ {fs_pred} Hz vs ground truth "
                f"{gt.shape} @ {fs} Hz."
            )
        present = True
    else:
        print(
            f"WARNING: no digitised record {pred_base}; treated as fully missing.",
            file=sys.stderr,
        )
        d = np.full_like(gt, np.nan)
        present = False
    return orig, mask, d, present, fs


def condition_inputs(
    gt_dir,
    pred_dir,
    original_dir,
    records=None,
    pred_suffix=PRED_SUFFIX,
    tol=1e-6,
    allow_missing=False,
):
    """Read, check and prepare the full, masked and digitised input of every record.

    masked = original with the GT mask; dig = digitised with the SAME GT mask;
    both then go through prepare_for_judge at the native rate.  Aborts when a
    folder is missing, the ground-truth folder has no records, none of the
    records has a digitised output (wrong folder), or the original differs
    from the ground truth inside the windows.  An isolated missing digitised
    record is treated as fully missing (all leads NaN) with a warning; a lead
    absent from a digitised file aborts unless allow_missing.
    """
    records, pred_bases = _condition_records(gt_dir, pred_dir, records, pred_suffix)
    full, masked, dig, qc, present = [], [], [], [], []
    for rec, pred_base in zip(records, pred_bases):
        orig, mask, d, ok, fs = _read_condition_record(
            gt_dir, original_dir, rec, pred_base, tol, allow_missing=allow_missing
        )
        present.append(ok)
        full.append(prepare_for_judge(orig, fs))
        masked.append(prepare_for_judge(apply_mask(orig, mask), fs))
        dig.append(prepare_for_judge(apply_mask(d, mask), fs))
        qc.append(window_qc(d, mask))
    return {
        "records": records,
        "ecg_ids": [ecg_id_of(r) for r in records],
        "full": full,
        "masked": masked,
        "dig": dig,
        "qc": qc,
        "present": present,
    }


def condition_signals(
    gt_dir,
    pred_dir,
    original_dir,
    records=None,
    pred_suffix=PRED_SUFFIX,
    tol=1e-6,
    allow_missing=False,
):
    """Native-rate masked original and masked digitised signal of every record.

    The same reading and checks as condition_inputs, without preparing:
    returns records, ecg_ids, fs, masked and dig (lists of (n, 12) float64,
    NaN outside the ground-truth windows) and present.
    """
    records, pred_bases = _condition_records(gt_dir, pred_dir, records, pred_suffix)
    masked, dig, present, rates = [], [], [], set()
    for rec, pred_base in zip(records, pred_bases):
        orig, mask, d, ok, fs = _read_condition_record(
            gt_dir, original_dir, rec, pred_base, tol, allow_missing=allow_missing
        )
        masked.append(apply_mask(orig, mask))
        dig.append(apply_mask(d, mask))
        present.append(ok)
        rates.add(fs)
    if len(rates) != 1:
        raise ValueError(f"Records of {gt_dir} have different rates {sorted(rates)}.")
    return {
        "records": records,
        "ecg_ids": [ecg_id_of(r) for r in records],
        "fs": rates.pop(),
        "masked": masked,
        "dig": dig,
        "present": present,
    }


def ci_note(
    n_records,
    npos=None,
    classes=JUDGE_CLASSES,
    nneg=None,
    missing_outputs=0,
    no_flips=False,
    rflip_exact_hi=float("nan"),
    ret_boot_frac=float("nan"),
    n_clusters=None,
    snr_records=None,
):
    """Warning text for a table row.

    Flags: small conditions, missing digitised outputs, degenerate bootstrap
    CIs (no flips; the exact record-flip bound is quoted), classes whose AUROC
    is undefined (no positives or no negatives), classes with fewer than
    SMALL_POSITIVES positives or negatives, and a retention CI resting on
    fewer than MIN_BOOT_FRAC of the bootstrap draws.  npos None = label free.
    n_clusters (a cluster bootstrap, real-image rows) says the CIs resample
    that many patients; snr_records (k < n_records) says the SNR median
    rests on the evaluation rows of k records only.
    """
    parts = []
    if n_records <= SMALL_CONDITION:
        parts.append(f"{n_records} records: CI wide; AUROC retention not interpretable")
    if missing_outputs:
        parts.append(
            f"{missing_outputs}/{n_records} digitised outputs missing (judged all-zero)"
        )
    if no_flips:
        parts.append(
            "no flips: bootstrap CI degenerate; record flip rate <= "
            f"{_fmt(rflip_exact_hi)} (exact)"
        )
    if npos is not None:
        npos = [int(k) for k in npos]
        nneg = [0] * len(npos) if nneg is None else [int(k) for k in nneg]
        if len(nneg) != len(npos):
            raise ValueError("npos and nneg must have one entry per class.")
        undefined = [c for c, p, q in zip(classes, npos, nneg) if p == 0 or q == 0]
        few = [
            c
            for c, p, q in zip(classes, npos, nneg)
            if p > 0 and q > 0 and min(p, q) < SMALL_POSITIVES
        ]
        if undefined:
            parts.append("AUROC undefined: " + ", ".join(undefined))
        if few:
            parts.append(f"<{SMALL_POSITIVES} pos/neg: " + ", ".join(few))
    if np.isfinite(ret_boot_frac) and ret_boot_frac < MIN_BOOT_FRAC:
        parts.append(
            f"retention CI conditional on {100.0 * ret_boot_frac:.0f}% of draws"
        )
    if n_clusters is not None:
        parts.append(f"CIs resample {int(n_clusters)} patients (cluster bootstrap)")
    if snr_records is not None and snr_records < n_records:
        parts.append(f"SNR median over {int(snr_records)}/{n_records} records")
    return "; ".join(parts)


def check_pred_run(pred_run):
    """A digitiser run tag: non-empty, without '/' or '@' (it goes into cache
    keys '<cond>@<tag>/<rec>/dig' and folder names)."""
    tag = str(pred_run)
    if not tag or "/" in tag or "@" in tag or tag != tag.strip():
        raise ValueError(f"Invalid pred_run tag {pred_run!r}.")
    return tag


def condition_key(condition, record, kind, pred_run=PRED_RUN):
    """Logit-cache key of a condition input: '<cond>/<rec>/<kind>'.

    Only 'dig' depends on the digitiser run: another run tag gets
    '<cond>@<tag>/<rec>/dig', so it never overwrites the default run's entry
    (LogitCache keeps one input per key), while full and masked keep the
    default keys and are reused from the cache.
    """
    if kind == "dig" and pred_run != PRED_RUN:
        return f"{condition}@{check_pred_run(pred_run)}/{record}/{kind}"
    return f"{condition}/{record}/{kind}"


def evaluate_condition(
    judge,
    condition,
    gt_dir,
    pred_dir,
    original_dir,
    thr_layout,
    thr_full,
    labels=None,
    ev=None,
    cache=None,
    n_boot=2000,
    seed=0,
    set_name="",
    judge_name=None,
    batch_size=512,
    pred_run=PRED_RUN,
    records=None,
    groups=None,
    allow_missing=False,
):
    """Fidelity of one condition for one judge.

    Primary comparison: masked original vs digitised at the 'layout'
    thresholds; secondary: the same at the 'full' thresholds (_thrfull).
    pred_run tags the digitiser run pred_dir holds (condition_key; the
    'pred_run' column of the row and of every record row).  records (None =
    every ground-truth record) restricts the condition to those records, and
    the evaluation frame ev to their rows (a split, a smoke subset); a
    selected record with a digitised output but no evaluation row aborts (a
    partial or stale CSV), one without an output is noted in ci_note.
    groups ({record: patient_id}, real-image rows) makes the bootstrap CIs
    resample patients (compare); allow_missing reads a digitised lead absent
    from its file as missing (condition_inputs).
    Returns (table row dict, per-record-per-class DataFrame).
    """
    classes = tuple(judge.classes)
    name = judge_name or getattr(judge, "name", "judge")
    selected = records is not None
    inputs = condition_inputs(
        gt_dir, pred_dir, original_dir, records=records, allow_missing=allow_missing
    )
    records = inputs["records"]
    snr_records = None
    if ev is not None and selected:
        ev = ev[ev["record"].astype(str).isin(set(map(str, records)))]
        covered = set(ev["record"].astype(str))
        uncovered = [r for r in records if str(r) not in covered]
        stale = [
            r for r, ok in zip(records, inputs["present"])
            if ok and str(r) not in covered
        ]
        if stale:
            raise ValueError(
                f"{condition}: the evaluation CSV has no rows for {len(stale)} "
                f"record(s) with a digitised output (e.g. {stale[:3]}); partial "
                "or stale evaluation CSV?"
            )
        if uncovered:
            print(
                f"WARNING: {condition}: SNR median over {len(records) - len(uncovered)}"
                f"/{len(records)} records (no evaluation rows for {uncovered[:3]}).",
                file=sys.stderr,
            )
            snr_records = len(records) - len(uncovered)
    cluster = None
    if groups is not None:
        unknown = [r for r in records if r not in groups]
        if unknown:
            raise ValueError(
                f"{condition}: no patient_id for record(s) {unknown[:3]}."
            )
        cluster = [groups[r] for r in records]

    def logits(kind):
        keys = [condition_key(condition, r, kind, pred_run) for r in records]
        return cached_logits(
            judge, inputs[kind], keys, cache=cache, batch_size=batch_size
        )

    l_full, l_masked, l_dig = logits("full"), logits("masked"), logits("dig")
    y, labelled, folds = labels_for(labels, inputs["ecg_ids"], classes)
    thr_layout = np.asarray(thr_layout, dtype=float)
    thr_full = np.asarray(thr_full, dtype=float)
    res = compare(
        y, l_masked, l_dig, thr_layout, n_boot=n_boot, seed=seed,
        labelled=labelled, classes=classes, groups=cluster,
    )
    res_full = compare(None, l_masked, l_dig, thr_full, n_boot=0, classes=classes)
    missing_outputs = int(np.sum(~np.asarray(inputs["present"], dtype=bool)))

    qc = inputs["qc"]
    missing = np.array([q["missing"] for q in qc])
    n_nan = np.array([q["n_nan"] for q in qc])
    n_window = np.array([q["n_window"] for q in qc])
    kept = ~missing
    window_total = n_window[kept].sum()
    nan_frac = float(n_nan[kept].sum() / window_total) if window_total else float("nan")

    snr_median = float("nan")
    rec_snr = pd.DataFrame(columns=["snr_rec_median", "snr_rec_min"])
    if ev is not None:
        snr_median = condition_snr_median(ev)
        rec_snr = record_snr(ev)

    row = {
        "judge": name,
        "condition": condition,
        "set": set_name,
        "n_records": len(records),
        "lattice_return": False,
        "snr_median_db": snr_median,
        "kappa": res["kappa"],
        "kappa_lo": res["kappa_lo"],
        "kappa_hi": res["kappa_hi"],
    }
    row.update({f"kappa_{c}": res[f"kappa_{c}"] for c in classes})
    for key in (
        "flip_rate", "flip_lo", "flip_hi", "record_flip_rate", "rflip_boot_lo",
        "rflip_boot_hi", "rflip_exact_lo", "rflip_exact_hi", "n_flips",
        "mean_abs_dp", "dp_lo", "dp_hi", "max_abs_dp",
    ):
        row[key] = res[key]
    row["kappa_thrfull"] = res_full["kappa"]
    row["flip_rate_thrfull"] = res_full["flip_rate"]
    row["record_flip_rate_thrfull"] = res_full["record_flip_rate"]
    row["auroc_masked_macro"] = res["auroc_ref_macro"]
    row["auroc_dig_macro"] = res["auroc_test_macro"]
    row["auroc_retention_macro"] = res["auroc_retention_macro"]
    row["ret_lo"] = res["ret_lo"]
    row["ret_hi"] = res["ret_hi"]
    row.update({f"ret_{c}": res[f"ret_{c}"] for c in classes})
    row.update({f"npos_{c}": res[f"npos_{c}"] for c in classes})
    row["interior_nan_frac"] = nan_frac
    row["missing_leads"] = int(missing.sum())
    row["ci_note"] = ci_note(
        len(records),
        [res[f"npos_{c}"] for c in classes] if y is not None else None,
        classes,
        nneg=[res[f"nneg_{c}"] for c in classes] if y is not None else None,
        missing_outputs=missing_outputs,
        no_flips=res["ci_degenerate"],
        rflip_exact_hi=res["rflip_exact_hi"],
        ret_boot_frac=res["ret_boot_frac"],
        n_clusters=(
            len(cluster_members(cluster, len(records))) if cluster is not None
            else None
        ),
        snr_records=snr_records,
    )
    # Extras (after the v0 columns).
    row["n_record_flips"] = res["n_record_flips"]
    row["ci_degenerate"] = res["ci_degenerate"]
    row["kappa_class_mean"] = res["kappa_class_mean"]
    row["median_abs_dp"] = res["median_abs_dp"]
    row.update({f"flips_{c}": res[f"flips_{c}"] for c in classes})
    row.update({f"nneg_{c}": res[f"nneg_{c}"] for c in classes})
    row["ret_boot_valid"] = res["ret_boot_valid"]
    row["ret_boot_frac"] = res["ret_boot_frac"]
    row["n_labelled"] = res["n_labelled"]
    row["missing_outputs"] = missing_outputs
    # Split-half and bias extras (B4), then the run tag last.
    row.update(split_half(inputs["ecg_ids"], l_masked, l_dig, thr_layout, classes))
    row["pred_run"] = pred_run

    dec_masked = decisions(l_masked, thr_layout)
    dec_dig = decisions(l_dig, thr_layout)
    p_masked, p_dig = sigmoid(l_masked), sigmoid(l_dig)
    rows = []
    for i, rec in enumerate(records):
        snr_med = rec_snr["snr_rec_median"].get(rec, np.nan)
        snr_min = rec_snr["snr_rec_min"].get(rec, np.nan)
        for c, cls in enumerate(classes):
            rows.append(
                {
                    "judge": name,
                    "condition": condition,
                    "record": rec,
                    "ecg_id": inputs["ecg_ids"][i],
                    "strat_fold": int(folds[i]),
                    "class": cls,
                    "label": int(y[i, c]) if y is not None else np.nan,
                    "labelled": bool(labelled[i]) if labelled is not None else False,
                    "logit_full": l_full[i, c],
                    "logit_masked": l_masked[i, c],
                    "logit_dig": l_dig[i, c],
                    "thr_layout": thr_layout[c],
                    "thr_full": thr_full[c],
                    "dec_masked": bool(dec_masked[i, c]),
                    "dec_dig": bool(dec_dig[i, c]),
                    "flip": bool(dec_masked[i, c] != dec_dig[i, c]),
                    "p_masked": p_masked[i, c],
                    "p_dig": p_dig[i, c],
                    "abs_dp": abs(p_masked[i, c] - p_dig[i, c]),
                    "snr_rec_median": snr_med,
                    "snr_rec_min": snr_min,
                    "pred_run": pred_run,
                }
            )
    return row, pd.DataFrame(rows)


# ----------------------------------------------------------------------------
# Reporting
# ----------------------------------------------------------------------------
def _fmt(value, spec=".3f"):
    try:
        return format(float(value), spec) if np.isfinite(float(value)) else "nan"
    except (TypeError, ValueError):
        return str(value)


def _kappa_cell(r):
    """'kappa [lo, hi]', or 'kappa [no flips]' when the bootstrap is degenerate."""
    if bool(r.get("ci_degenerate", False)):
        return f"{_fmt(r['kappa'])} [no flips]"
    return f"{_fmt(r['kappa'])} [{_fmt(r['kappa_lo'])}, {_fmt(r['kappa_hi'])}]"


def format_table(rows, pred_run=None, split=None):
    """Compact printed view of table rows (miss = missing digitised outputs);
    with pred_run the header line names the digitiser run, with split the
    record split (real-image conditions)."""
    header = (
        f"{'judge':<12} {'condition':<10} {'n':>3} {'SNR':>6} "
        f"{'kappa [95% CI]':<22} {'flip':>6} {'rec flips [exact CI]':<22} "
        f"{'mean|dp|':>9} {'AUROC ret':>9} {'miss':>4}"
    )
    if pred_run is not None:
        header += f"  [pred_run {pred_run}]"
    if split is not None:
        header += f"  [split {split}]"
    lines = [header]
    for r in rows:
        rec_flips = (
            f"{int(r.get('n_record_flips', 0))}/{int(r['n_records'])} "
            f"[{_fmt(r['rflip_exact_lo'])}, {_fmt(r['rflip_exact_hi'])}]"
        )
        lines.append(
            f"{r['judge']:<12} {r['condition']:<10} {int(r['n_records']):>3} "
            f"{_fmt(r['snr_median_db'], '.2f'):>6} {_kappa_cell(r):<22} "
            f"{_fmt(r['flip_rate']):>6} {rec_flips:<22} "
            f"{_fmt(r['mean_abs_dp'], '.4f'):>9} "
            f"{_fmt(r['auroc_retention_macro']):>9} "
            f"{int(r.get('missing_outputs', 0)):>4}"
        )
    return lines


def _write_csv(df, path):
    os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
    df.to_csv(path, index=False)
    print(f"Wrote {len(df)} row(s) to {path}")


def default_records_out(out):
    """Per-record CSV next to a table CSV: <out without extension>_records.csv."""
    return os.path.splitext(out)[0] + "_records.csv"


PATH_FLOOR_KEYS = ("n_records", "n_record_flips", "record_flip_rate", "flip_rate",
                   "mean_abs_dp")


def path_floor(layout_loss_csv, judge_name):
    """Path noise floor of one judge from layout_loss.csv.

    The fold-10 row records100_layout vs records500_layout at the layout
    thresholds: what the source change alone (records100 vs the 500 Hz
    condition path) does to the decisions.  Returns path_floor_<key> values
    (NaN when the file or the row is absent).
    """
    out = {f"path_floor_{k}": float("nan") for k in PATH_FLOOR_KEYS}
    if not layout_loss_csv or not os.path.exists(layout_loss_csv):
        return out
    df = pd.read_csv(layout_loss_csv)
    sub = df[
        (df["judge"] == judge_name)
        & (df["analysis"] == "path_floor_fold10")
        & (df["thresholds"] == "layout")
    ]
    if sub.empty:
        return out
    first = sub.iloc[0]
    return {f"path_floor_{k}": float(first[k]) for k in PATH_FLOOR_KEYS}


# ----------------------------------------------------------------------------
# Sensitivity sweeps (plan item A3): below what SNR does the diagnosis wobble?
# ----------------------------------------------------------------------------
def _nan_stat(fn, values, axis=None):
    """fn (np.nanmedian, np.nanmin, ...) without the all-NaN RuntimeWarning."""
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", RuntimeWarning)
        return fn(np.asarray(values, dtype=float), axis=axis)


def lead_snr(ref, test, mask=None):
    """Per-lead raw SNR in dB of `test` against `ref` over the lead windows.

    Mirrors src/run/evaluate.py compute_raw_snr (helper_code.compute_snr with
    keep_nans): the window of a lead is where `ref` is finite (and `mask` is
    True, if given); SNR = 10 log10(sum ref^2 / sum (ref - test)^2) over the
    window samples where `test` is finite too, no mean removal, multiplied by
    that covered fraction of the window.  inf when the error is zero, NaN
    without a covered sample or with a zero signal.  ref, test: (n, L) or
    (n,); returns (L,) (a float for 1-D input).
    """
    ref = np.asarray(ref, dtype=np.float64)
    test = np.asarray(test, dtype=np.float64)
    if ref.shape != test.shape:
        raise ValueError(f"Signal shapes differ: {ref.shape} vs {test.shape}.")
    one_d = ref.ndim == 1
    if one_d:
        ref, test = ref[:, None], test[:, None]
    window = np.isfinite(ref)
    if mask is not None:
        window &= np.asarray(mask, dtype=bool).reshape(ref.shape)
    both = window & np.isfinite(test)
    signal = np.where(both, ref, 0.0)
    error = np.where(both, ref - test, 0.0)
    p_signal = np.sum(signal * signal, axis=0)
    p_error = np.sum(error * error, axis=0)
    n_window = window.sum(axis=0)
    n_both = both.sum(axis=0)
    with np.errstate(divide="ignore", invalid="ignore"):
        snr = 10.0 * np.log10(p_signal / p_error)
        coverage = n_both / n_window
    snr = np.where(p_error == 0, np.inf, snr)
    snr = np.where((p_error > 0) & ~(p_signal > 0), np.nan, snr)
    snr = np.where(n_both == 0, np.nan, snr) * coverage
    return float(snr[0]) if one_d else snr


def judge_window(x, fs, n_out=N_JUDGE):
    """(n_out, L) bool: the samples of prepare_for_judge(x, fs) that carry x.

    The valid span of every lead (fill_gaps) on the output grid
    (resample_span), zero-padded or truncated like the prepared input; for a
    layout-masked 500 Hz record it equals layout_mask(1000, 100).
    """
    x = np.asarray(x, dtype=np.float64)
    up, down = _rate_ratio(fs, FS_JUDGE)
    _, span = fill_gaps(x)
    if up != down:
        span = resample_span(span, up, down)
    out = np.zeros((int(n_out), x.shape[1]), dtype=bool)
    m = min(int(n_out), span.shape[0])
    out[:m] = span[:m]
    return out


def judge_snr(ref_prepared, test_prepared, window):
    """Per-lead SNR as the judge sees it: lead_snr of the prepared (100 Hz)
    test input against the prepared reference over `window` (judge_window of
    the reference).  Error power above the resampler's band is gone here, so
    this is the SNR axis on which band-limited noise and digitisation error
    are comparable."""
    ref = np.where(window, np.asarray(ref_prepared, dtype=np.float64), np.nan)
    return lead_snr(ref, np.asarray(test_prepared, dtype=np.float64))


def scale_error(masked, dig, alpha):
    """The digitisation error scaled by alpha: masked + alpha * (dig - masked).

    Computed as (1 - alpha) * masked + alpha * dig, which is `dig` bit for bit
    at alpha 1.  alpha 0 returns a copy of `masked` (the reference: no
    digitisation error and none of its gaps).  For alpha > 0 the result is
    NaN wherever `masked` or `dig` is NaN, i.e. it has the NaN pattern of dig
    (which carries the ground-truth mask).  Both are (n, L) at the native rate.
    """
    masked = np.asarray(masked, dtype=np.float64)
    dig = np.asarray(dig, dtype=np.float64)
    if masked.shape != dig.shape:
        raise ValueError(f"Signal shapes differ: {masked.shape} vs {dig.shape}.")
    alpha = float(alpha)
    if not (np.isfinite(alpha) and alpha >= 0.0):
        raise ValueError(f"alpha must be finite and >= 0, got {alpha}.")
    if alpha == 0.0:
        return masked.copy()
    return (1.0 - alpha) * masked + alpha * dig


def add_band_noise(
    x, snr_db, seed, ecg_id, level_index, fs=500, band=NOISE_BAND_HZ, order=NOISE_ORDER
):
    """`x` plus band-limited Gaussian noise at an exact per-lead window SNR.

    x: (n, L), NaN outside the windows (the window of a lead = its finite
    samples).  White N(0, 1) noise (n, L) from
    np.random.default_rng([seed, ecg_id, level_index]) is band-passed along
    time over the whole record (butter(order, band, 'bandpass', fs=fs,
    output='sos') applied with sosfiltfilt) and scaled per lead by
    sqrt(sum x^2 / (sum n^2 10^(snr/10))), both sums over the lead's window
    samples, so lead_snr(x, result) == snr_db.  NaN outside the windows stays
    NaN; a lead with a zero signal gets no noise; snr_db = inf returns a copy.
    """
    x = np.asarray(x, dtype=np.float64)
    if x.ndim != 2:
        raise ValueError(f"Expected a (n_samples, n_leads) signal, got {x.shape}.")
    snr_db = float(snr_db)
    if snr_db == math.inf:
        return x.copy()
    if not np.isfinite(snr_db):
        raise ValueError(f"Target SNR must be finite or +inf, got {snr_db}.")
    rng = np.random.default_rng([int(seed), int(ecg_id), int(level_index)])
    white = rng.standard_normal(x.shape)
    sos = butter(order, band, btype="bandpass", fs=fs, output="sos")
    noise = sosfiltfilt(sos, white, axis=0)
    window = np.isfinite(x)
    p_signal = np.sum(np.where(window, x, 0.0) ** 2, axis=0)
    p_noise = np.sum(np.where(window, noise, 0.0) ** 2, axis=0)
    with np.errstate(divide="ignore", invalid="ignore"):
        scale = np.sqrt(p_signal / (p_noise * 10.0 ** (snr_db / 10.0)))
    scale = np.where(np.isfinite(scale), scale, 0.0)
    return x + noise * scale


def error_key(condition, record, alpha):
    """Logit-cache key of a scaled-error input; alpha 0 / 1 reuse 'masked' / 'dig'.
    Other alphas are written exactly (repr), so distinct alphas never share a key."""
    alpha = float(alpha)
    if alpha == 0.0:
        kind = "masked"
    elif alpha == 1.0:
        kind = "dig"
    else:
        kind = f"err_a{alpha!r}"
    return f"{condition}/{record}/{kind}"


def noise_key(ecg_id, snr_db, seed=None, level_index=None):
    """Logit-cache key of a noise-sweep input; the clean level reuses 'layout'
    (the same input as the fold10_500hz layout entries).  A noisy key holds
    the exact SNR, the seed and the level index, which fix the draw."""
    if float(snr_db) == math.inf:
        return f"ptbxl500/{int(ecg_id)}/layout"
    if seed is None or level_index is None:
        raise ValueError("A noisy input's key needs the seed and the level index.")
    return (
        f"ptbxl500/{int(ecg_id)}/noise{float(snr_db)!r}_s{int(seed)}"
        f"_l{int(level_index)}"
    )


def check_alphas(alphas):
    """Finite, non-negative, de-duplicated alphas (order kept)."""
    out = []
    for a in alphas:
        a = float(a)
        if not (np.isfinite(a) and a >= 0.0):
            raise ValueError(f"alpha must be finite and >= 0, got {a}.")
        if a not in out:
            out.append(a)
    if not out:
        raise ValueError("At least one alpha is required.")
    return out


def check_snrs(snrs):
    """Finite or +inf, de-duplicated SNR levels (order kept; it sets level_index)."""
    out = []
    for s in snrs:
        s = float(s)
        if np.isnan(s) or s == -math.inf:
            raise ValueError(f"SNR level must be finite or +inf, got {s}.")
        if s not in out:
            out.append(s)
    if not out:
        raise ValueError("At least one SNR level is required.")
    return out


def min_margin(logits, thr):
    """Per record: min over classes of |logit - thr| (NaN thresholds ignored)."""
    d = np.abs(np.atleast_2d(np.asarray(logits, dtype=float)) - np.asarray(thr, float))
    d = np.where(np.isfinite(d), d, np.inf).min(axis=1)
    return np.where(np.isfinite(d), d, np.nan)


def margin_summary(margins, cuts=MARGIN_CUTS):
    """n, quartiles and the fraction of records below each cut of min-margins."""
    m = np.asarray(margins, dtype=float)
    m = m[np.isfinite(m)]
    empty = m.size == 0
    out = {
        "n": int(m.size),
        "median_min_margin": float("nan") if empty else float(np.median(m)),
        "q25_min_margin": float("nan") if empty else float(np.percentile(m, 25)),
        "q75_min_margin": float("nan") if empty else float(np.percentile(m, 75)),
    }
    for cut in cuts:
        out[f"frac_lt_{cut:g}"] = float("nan") if empty else float(np.mean(m < cut))
    return out


def record_flips(logit_ref, logit_test, thr, classes=JUDGE_CLASSES):
    """Per-record flip summary of a test system against a reference.

    One row per record: n_class_flips, any_flip, flipped_classes (e.g.
    'MI;NORM', '' without a flip), mean_abs_dp, max_abs_dp, max_abs_dlogit
    and min_margin_ref (min over classes of |logit_ref - thr|).
    """
    ref = np.asarray(logit_ref, dtype=float)
    test = np.asarray(logit_test, dtype=float)
    thr = np.asarray(thr, dtype=float)
    flips = flip_matrix(decisions(ref, thr), decisions(test, thr))
    dp = abs_dp(ref, test)
    return pd.DataFrame(
        {
            "n_class_flips": flips.sum(axis=1).astype(int),
            "any_flip": flips.any(axis=1),
            "flipped_classes": [
                ";".join(c for c, f in zip(classes, row) if f) for row in flips
            ],
            "mean_abs_dp": dp.mean(axis=1),
            "max_abs_dp": dp.max(axis=1),
            "max_abs_dlogit": np.abs(test - ref).max(axis=1),
            "min_margin_ref": min_margin(ref, thr),
        }
    )


def snr_check(records, snr_leads, ev, leads=LEADS):
    """Per-lead SNR against the evaluation CSV's snr_raw of the same record/lead.

    Returns snr_check_max_absdiff (over the pairs finite on both sides),
    snr_check_n_leads (their number) and snr_check_n_mismatch (pairs finite on
    one side only); NaN / 0 / 0 without an evaluation frame.
    """
    out = {
        "snr_check_max_absdiff": float("nan"),
        "snr_check_n_leads": 0,
        "snr_check_n_mismatch": 0,
    }
    if ev is None:
        return out
    kept = _present(ev)
    values = pd.to_numeric(kept["snr_raw"], errors="coerce").to_numpy(dtype=float)
    lookup = {
        (str(r), str(lead).upper()): v
        for r, lead, v in zip(kept["record"], kept["lead"], values)
    }
    snr_leads = np.asarray(snr_leads, dtype=float)
    diffs = []
    for i, rec in enumerate(records):
        for j, lead in enumerate(leads):
            theirs = lookup.get((str(rec), lead.upper()))
            if theirs is None:
                continue
            ours = snr_leads[i, j]
            if np.isfinite(ours) and np.isfinite(theirs):
                diffs.append(abs(ours - theirs))
            elif np.isfinite(ours) != np.isfinite(theirs):
                out["snr_check_n_mismatch"] += 1
    out["snr_check_n_leads"] = len(diffs)
    if diffs:
        out["snr_check_max_absdiff"] = float(max(diffs))
    return out


def error_sweep_inputs(signals, alphas, condition):
    """Judge-independent part of the error sweep of one condition.

    signals: condition_signals().  For every record and alpha:
    x_alpha = scale_error(masked, dig, alpha) at the native rate, its prepared
    judge input and cache key (error_key), the per-lead window SNR of
    x_alpha against masked (lead_snr at the native rate, 'snr'; inf at alpha
    0) and the per-lead SNR of the prepared inputs as the judge sees them
    (judge_snr over judge_window(masked), 'snr_judge').  The prepared alpha-0
    input must equal the prepared masked input and the alpha-1 input the
    prepared dig input bit for bit (input_sha1), otherwise AssertionError.
    Also returns the reference inputs ('ref', = prepared masked), snr1 (the
    per-lead SNR of dig, for the check against the evaluation CSV) and the
    number of leads of dig with NaN inside their window.
    """
    alphas = check_alphas(alphas)
    fs = signals["fs"]
    records = list(signals["records"])
    out = {
        "condition": condition,
        "records": records,
        "ecg_ids": list(signals["ecg_ids"]),
        "alphas": alphas,
        "ref": [],
        "ref_keys": [error_key(condition, r, 0.0) for r in records],
        "inputs": {a: [] for a in alphas},
        "keys": {a: [error_key(condition, r, a) for r in records] for a in alphas},
        "snr": {a: [] for a in alphas},
        "snr_judge": {a: [] for a in alphas},
        "snr1": [],
        "n_leads_window_nan": 0,
    }
    for rec, masked, dig in zip(records, signals["masked"], signals["dig"]):
        ref = prepare_for_judge(masked, fs)
        window = judge_window(masked, fs)
        out["ref"].append(ref)
        gaps = np.isfinite(masked) & ~np.isfinite(dig)
        out["n_leads_window_nan"] += int(gaps.any(axis=0).sum())
        out["snr1"].append(lead_snr(masked, dig))
        for a in alphas:
            x = scale_error(masked, dig, a)
            prepared = prepare_for_judge(x, fs)
            if a == 0.0 and input_sha1(prepared) != input_sha1(ref):
                raise AssertionError(f"{condition}/{rec}: alpha 0 input != masked.")
            if a == 1.0 and input_sha1(prepared) != input_sha1(
                prepare_for_judge(dig, fs)
            ):
                raise AssertionError(f"{condition}/{rec}: alpha 1 input != dig.")
            out["inputs"][a].append(prepared)
            out["snr"][a].append(lead_snr(masked, x))
            out["snr_judge"][a].append(judge_snr(ref, prepared, window))
    n = len(records)
    out["snr1"] = np.asarray(out["snr1"], dtype=float).reshape(n, -1)
    for name in ("snr", "snr_judge"):
        out[name] = {
            a: np.asarray(v, dtype=float).reshape(n, -1) for a, v in out[name].items()
        }
    return out


RECORD_ERROR_COLUMNS = (
    "judge", "condition", "record", "ecg_id", "alpha", "snr_rec_median",
    "snr_rec_min", "n_class_flips", "any_flip", "flipped_classes", "mean_abs_dp",
    "max_abs_dp", "max_abs_dlogit", "min_margin_ref", "snr_judge_rec_median",
)


def error_sweep_judge(
    judge,
    sweep,
    thr_layout,
    cache=None,
    judge_name=None,
    set_name="",
    n_boot=1000,
    seed=0,
    check=None,
    batch_size=512,
):
    """Error-sweep rows of one condition (error_sweep_inputs) for one judge.

    Reference = alpha 0 (the masked original); decisions at `thr_layout`.
    Returns (aggregate rows, one per alpha; per-record DataFrame with
    RECORD_ERROR_COLUMNS).  snr_median_db is the median over record x lead,
    snr_rec_median / snr_rec_min over the record's leads (500 Hz window SNR);
    snr_judge_median_db / snr_judge_rec_median are the same medians of the
    judge-band SNR (judge_snr).
    """
    classes = tuple(judge.classes)
    name = judge_name or getattr(judge, "name", "judge")
    cond = sweep["condition"]
    thr = np.asarray(thr_layout, dtype=float)
    check = check or snr_check(sweep["records"], sweep["snr1"], None)
    l_ref = cached_logits(judge, sweep["ref"], sweep["ref_keys"], cache, batch_size)
    rows, frames = [], []
    for a in sweep["alphas"]:
        if a == 0.0:
            l_test = l_ref.copy()
        else:
            l_test = cached_logits(
                judge, sweep["inputs"][a], sweep["keys"][a], cache, batch_size
            )
        res = compare(
            None, l_ref, l_test, thr, n_boot=n_boot, seed=seed, classes=classes
        )
        snr = sweep["snr"][a]
        row = {
            "judge": name,
            "condition": cond,
            "set": set_name,
            "alpha": a,
            "n_records": res["n_records"],
            "snr_median_db": float(_nan_stat(np.nanmedian, snr)),
        }
        for key in (
            "record_flip_rate", "rflip_exact_lo", "rflip_exact_hi", "n_record_flips",
            "flip_rate", "kappa", "mean_abs_dp", "max_abs_dp",
        ):
            row[key] = res[key]
        row["snr_check_max_absdiff"] = check["snr_check_max_absdiff"]
        # Extras (after the requested columns).
        for key in (
            "kappa_lo", "kappa_hi", "dp_lo", "dp_hi", "n_flips", "ci_degenerate",
        ):
            row[key] = res[key]
        row.update({f"flips_{c}": res[f"flips_{c}"] for c in classes})
        row["snr_check_n_leads"] = check["snr_check_n_leads"]
        row["snr_check_n_mismatch"] = check["snr_check_n_mismatch"]
        row["n_leads_window_nan"] = sweep["n_leads_window_nan"]
        snr_judge = sweep["snr_judge"][a]
        row["snr_judge_median_db"] = float(_nan_stat(np.nanmedian, snr_judge))
        rows.append(row)

        rec = record_flips(l_ref, l_test, thr, classes)
        rec["judge"] = name
        rec["condition"] = cond
        rec["record"] = sweep["records"]
        rec["ecg_id"] = sweep["ecg_ids"]
        rec["alpha"] = a
        rec["snr_rec_median"] = _nan_stat(np.nanmedian, snr, axis=1)
        rec["snr_rec_min"] = _nan_stat(np.nanmin, snr, axis=1)
        rec["snr_judge_rec_median"] = _nan_stat(np.nanmedian, snr_judge, axis=1)
        frames.append(rec[list(RECORD_ERROR_COLUMNS)])
    return rows, pd.concat(frames, ignore_index=True)


def fold10_ids_500(db, root, classes=JUDGE_CLASSES):
    """Fold-10 ecg_ids with >= 1 superclass whose records500 file exists."""
    labelled = db[list(classes)].sum(axis=1) > 0
    ids = [int(i) for i in db.index[(db["strat_fold"] == 10) & labelled]]
    have = [i for i in ids if os.path.exists(records500_path(db, i, root) + ".hea")]
    if len(have) < len(ids):
        print(
            f"WARNING: records500 found for {len(have)}/{len(ids)} labelled fold-10 "
            "records; using those only.",
            file=sys.stderr,
        )
    return have


RECORD_NOISE_COLUMNS = (
    "judge", "ecg_id", "snr_db", "any_flip", "n_class_flips", "mean_abs_dp",
    "min_margin_ref", "flipped_classes", "max_abs_dp", "snr_judge_rec_median",
)


def noise_sweep(
    opened,
    db,
    ecg_ids,
    root,
    snrs=DEFAULT_SNRS,
    seed=0,
    n_boot=1000,
    classes=JUDGE_CLASSES,
    batch_size=512,
    chunk=256,
    cache_only=False,
):
    """Band-limited noise sweep on PTB-XL records500 through the 500 Hz path.

    opened: [(name, judge, cache, thr_layout)].  The clean input of a record
    is its layout-masked records500 signal (the 'layout' input of
    fold10_500hz); level i of `snrs` adds add_band_noise(base, snrs[i], seed,
    ecg_id, i) (+inf: no noise).  Reference = clean; decisions at the layout
    thresholds; labels from `db` (judge.load_ptbxl_db).  Returns (rows, one
    per judge and level; per-record DataFrame with RECORD_NOISE_COLUMNS).
    snr_judge_rec_median / snr_judge_median_db: the judge-band SNR
    (judge_snr) of the prepared noisy input, median over the record's leads
    / over record x lead (snr_db is exact per lead at 500 Hz).  cache_only:
    only fill the judges' caches (saved after every chunk); returns ([], an
    empty frame).
    """
    snrs = check_snrs(snrs)
    ids = [int(i) for i in ecg_ids]
    level_of = {s: level for level, s in enumerate(snrs)}
    y, labelled, _ = labels_for(db, ids, classes)
    ref = {name: [] for name, *_ in opened}
    test = {name: {s: [] for s in snrs} for name, *_ in opened}
    noisy_levels = [s for s in snrs if s != math.inf]
    snr_judge = {s: [] for s in [math.inf] + noisy_levels}  # per record: (L,)
    for start in range(0, len(ids), chunk):
        part = ids[start : start + chunk]
        arrays = {s: [] for s in [math.inf] + noisy_levels}
        for ecg_id in part:
            x, fs = read_record(records500_path(db, ecg_id, root))
            base = apply_mask(x, layout_mask(x.shape[0], fs))
            clean = prepare_for_judge(base, fs)
            window = judge_window(base, fs)
            arrays[math.inf].append(clean)
            snr_judge[math.inf].append(np.full(x.shape[1], np.inf))
            for level, s in enumerate(snrs):
                if s != math.inf:
                    noisy = add_band_noise(base, s, seed, ecg_id, level, fs)
                    prepared = prepare_for_judge(noisy, fs)
                    arrays[s].append(prepared)
                    snr_judge[s].append(judge_snr(clean, prepared, window))
        batch, keys = [], []
        for s, prepared in arrays.items():
            batch.extend(prepared)
            keys.extend(noise_key(i, s, seed, level_of.get(s)) for i in part)
        for name, judge, cache, _ in opened:
            logits = cached_logits(judge, batch, keys, cache, batch_size)
            logits = logits.reshape(len(arrays), len(part), -1)
            by_level = dict(zip(arrays, logits))
            ref[name].append(by_level[math.inf])
            for s in snrs:
                test[name][s].append(by_level[s])
            if cache is not None:
                cache.save()
        print(f"  noise sweep: {min(start + chunk, len(ids))}/{len(ids)} records")
    if cache_only:
        return [], pd.DataFrame(columns=list(RECORD_NOISE_COLUMNS))
    jsnr = {
        s: np.asarray(v, dtype=float).reshape(len(ids), -1)
        for s, v in snr_judge.items()
    }

    rows, frames = [], []
    for name, judge, cache, thr in opened:
        thr = np.asarray(thr, dtype=float)
        l_ref = np.concatenate(ref[name])
        for s in snrs:
            l_test = np.concatenate(test[name][s])
            res = compare(
                y, l_ref, l_test, thr, n_boot=n_boot, seed=seed, labelled=labelled,
                classes=classes,
            )
            row = {"judge": name, "snr_db": s, "n_records": res["n_records"]}
            for key in (
                "kappa", "kappa_lo", "kappa_hi", "flip_rate", "record_flip_rate",
                "rflip_exact_lo", "rflip_exact_hi", "n_record_flips",
            ):
                row[key] = res[key]
            row.update({f"flips_{c}": res[f"flips_{c}"] for c in classes})
            for key in (
                "mean_abs_dp", "dp_lo", "dp_hi", "auroc_ref_macro", "auroc_test_macro",
                "auroc_retention_macro", "ret_lo", "ret_hi",
            ):
                row[key] = res[key]
            row.update({f"auroc_test_{c}": res[f"auroc_test_{c}"] for c in classes})
            # Extras.
            for key in (
                "flip_lo", "flip_hi", "n_flips", "max_abs_dp", "ci_degenerate",
                "n_labelled", "ret_boot_frac",
            ):
                row[key] = res[key]
            row.update({f"auroc_ref_{c}": res[f"auroc_ref_{c}"] for c in classes})
            row["noise_seed"] = seed
            row["snr_judge_median_db"] = float(_nan_stat(np.nanmedian, jsnr[s]))
            rows.append(row)
            rec = record_flips(l_ref, l_test, thr, classes)
            rec["judge"] = name
            rec["ecg_id"] = ids
            rec["snr_db"] = s
            rec["snr_judge_rec_median"] = _nan_stat(np.nanmedian, jsnr[s], axis=1)
            frames.append(rec[list(RECORD_NOISE_COLUMNS)])
    return rows, pd.concat(frames, ignore_index=True)


def fold10_layout_logits(opened, db, ecg_ids, root, batch_size=512, chunk=512):
    """{judge name: (N, C) logits} of the layout-masked records500 inputs
    (cache keys 'ptbxl500/<id>/layout', shared with fold10_500hz)."""
    out = {name: [] for name, *_ in opened}
    ids = [int(i) for i in ecg_ids]
    for start in range(0, len(ids), chunk):
        part = ids[start : start + chunk]
        prepared = ptbxl500_inputs(db, part, root, kinds=("layout",))["layout"]
        keys = [noise_key(i, math.inf) for i in part]
        for name, judge, cache, _ in opened:
            out[name].append(cached_logits(judge, prepared, keys, cache, batch_size))
            if cache is not None:
                cache.save()
        print(f"  fold-10 layout: {min(start + chunk, len(ids))}/{len(ids)} records")
    return {name: np.concatenate(v) for name, v in out.items()}


def snr_bins(snr, flip, groups, abs_dp_values=None, edges=SNR_BIN_EDGES, level=0.95):
    """Record flip rate per SNR bin [lo, hi) with the exact binomial CI.

    Rows with a non-finite SNR are left out.  One dict per bin (empty bins
    included): bin_lo, bin_hi, n_rows, n_unique_ecg_ids, n_flips,
    record_flip_rate, rflip_exact_lo, rflip_exact_hi, mean_abs_dp.
    """
    snr = np.asarray(snr, dtype=float)
    flip = np.asarray(flip, dtype=bool)
    groups = np.asarray(groups)
    dp = (
        np.full(snr.shape, np.nan)
        if abs_dp_values is None
        else np.asarray(abs_dp_values, dtype=float)
    )
    edges = np.asarray(edges, dtype=float)
    keep = np.isfinite(snr)
    index = np.searchsorted(edges, snr, side="right") - 1
    rows = []
    for b in range(len(edges) - 1):
        sel = keep & (index == b)
        n, k = int(sel.sum()), int(flip[sel].sum())
        lo, hi = clopper_pearson(k, n, level=level)
        rows.append(
            {
                "bin_lo": float(edges[b]),
                "bin_hi": float(edges[b + 1]),
                "n_rows": n,
                "n_unique_ecg_ids": int(np.unique(groups[sel]).size),
                "n_flips": k,
                "record_flip_rate": k / n if n else float("nan"),
                "rflip_exact_lo": lo,
                "rflip_exact_hi": hi,
                "mean_abs_dp": float(np.mean(dp[sel])) if n else float("nan"),
            }
        )
    return rows


def _fit_flip_logistic(x, y, weight=None, max_iter=1000, tol=1e-10):
    """Unpenalised logistic fit y ~ x; (intercept, slope, converged) in x units.

    C=inf is sklearn's unpenalised fit (penalty=None is deprecated since 1.8);
    tol is tight so the result is the MLE (the default 1e-4 can stop ~0.01 dB
    short in the SNR at a flip probability).  x is standardised for the solver
    and the coefficients mapped back.  The MLE exists only without separation
    (is_separated); callers check that first.
    """
    from sklearn.exceptions import ConvergenceWarning
    from sklearn.linear_model import LogisticRegression

    x = np.asarray(x, dtype=float)
    mu, sd = float(np.mean(x)), float(np.std(x))
    sd = sd if sd > 0 else 1.0
    model = LogisticRegression(C=np.inf, max_iter=max_iter, tol=tol)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", ConvergenceWarning)
        model.fit(
            ((x - mu) / sd)[:, None], np.asarray(y, dtype=int), sample_weight=weight
        )
    w1, w0 = float(model.coef_[0, 0]), float(model.intercept_[0])
    return w0 - w1 * mu / sd, w1 / sd, int(np.max(model.n_iter_)) < max_iter


def is_separated(x, y):
    """True when a threshold on x splits the outcomes (complete or quasi-complete
    separation in 1-D: max x of the flips <= min x of the others, or the
    reverse); the logistic MLE then does not exist (infinite slope).  Needs
    both outcomes present."""
    x = np.asarray(x, dtype=float)
    y = np.asarray(y, dtype=bool)
    x1, x0 = x[y], x[~y]
    return bool(x1.max() <= x0.min() or x1.min() >= x0.max())


def ecg_id_weights(groups):
    """Row weights under which every group (ecg_id) weighs the same: 1 / rows
    of the group, renormalised to mean 1."""
    _, inverse, counts = np.unique(
        np.asarray(groups), return_inverse=True, return_counts=True
    )
    w = 1.0 / counts[inverse].astype(float)
    return w * (w.size / w.sum()) if w.size else w


def snr_at_flip_prob(intercept, slope, p):
    """SNR at which the fitted flip probability equals p (NaN for slope 0)."""
    if not (np.isfinite(intercept) and np.isfinite(slope)) or slope == 0:
        return float("nan")
    return (math.log(p / (1.0 - p)) - intercept) / slope


def flip_logistic(
    snr, flip, groups, probs=FLIP_PROBS, n_boot=1000, seed=0, level=0.95,
    weighting="rows",
):
    """Logistic model flip ~ SNR and the SNR at each flip probability in `probs`.

    weighting 'rows': every row weighs 1; 'ecg_id': every group (ecg_id)
    weighs 1 in total (ecg_id_weights over the rows kept), so groups with
    more rows do not dominate the point estimate.  CIs: cluster bootstrap
    resampling the unique groups (ecg_ids) with replacement (a drawn group
    brings all its rows with their weights; implemented as integer
    multiplicities, identical to replicating the rows).  No fit when the
    outcomes are separated by a threshold on the SNR (is_separated: the MLE
    does not exist); draws in which only one outcome is present, the SNR is
    constant or the drawn rows are separated are skipped and counted
    (n_boot_degenerate).  Rows with a non-finite SNR are left out.
    rows_per_id_min / _max: rows per group among those kept.
    """
    if weighting not in WEIGHTINGS:
        raise ValueError(f"weighting must be one of {WEIGHTINGS}, got {weighting!r}.")
    x = np.asarray(snr, dtype=float)
    y = np.asarray(flip, dtype=bool)
    g = np.asarray(groups)
    keep = np.isfinite(x)
    x, y, g = x[keep], y[keep].astype(int), g[keep]
    uniq, inverse, per_id = np.unique(g, return_inverse=True, return_counts=True)
    out = {
        "weighting": weighting,
        "n_rows": int(x.size),
        "n_unique_ecg_ids": int(uniq.size),
        "rows_per_id_min": int(per_id.min()) if per_id.size else 0,
        "rows_per_id_max": int(per_id.max()) if per_id.size else 0,
        "n_flips": int(y.sum()),
        "snr_obs_min": float(x.min()) if x.size else float("nan"),
        "snr_obs_max": float(x.max()) if x.size else float("nan"),
        "intercept": float("nan"),
        "slope": float("nan"),
        "slope_lo": float("nan"),
        "slope_hi": float("nan"),
        "converged": False,
        "n_boot": int(n_boot or 0),
        "n_boot_valid": 0,
        "n_boot_degenerate": 0,
        "fit": "ok",
    }
    for p in probs:
        for suffix in ("", "_lo", "_hi"):
            out[f"snr_p{p:g}{suffix}"] = float("nan")
    if x.size == 0 or y.min() == y.max() or np.ptp(x) == 0:
        out["fit"] = "degenerate: " + (
            "no rows" if x.size == 0
            else "no flips" if y.max() == 0
            else "all rows flip" if y.min() == 1
            else "constant SNR"
        )
        return out
    if is_separated(x, y):
        out["fit"] = "degenerate: separated"
        return out
    base = ecg_id_weights(g) if weighting == "ecg_id" else np.ones(x.size)
    b0, b1, converged = _fit_flip_logistic(
        x, y, weight=None if weighting == "rows" else base
    )
    out.update(intercept=b0, slope=b1, converged=bool(converged))
    for p in probs:
        out[f"snr_p{p:g}"] = snr_at_flip_prob(b0, b1, p)
    if not n_boot or n_boot <= 0:
        return out
    rng = np.random.default_rng(seed)
    slopes, values = [], {p: [] for p in probs}
    for _ in range(int(n_boot)):
        counts = np.bincount(rng.integers(0, uniq.size, uniq.size), minlength=uniq.size)
        w = counts[inverse] * base
        drawn = w > 0
        if (
            w[y == 1].sum() == 0
            or w[y == 0].sum() == 0
            or np.ptp(x[drawn]) == 0
            or is_separated(x[drawn], y[drawn])
        ):
            out["n_boot_degenerate"] += 1
            continue
        c0, c1, _ = _fit_flip_logistic(x, y, weight=w)
        snrs = [snr_at_flip_prob(c0, c1, p) for p in probs]
        if not np.all(np.isfinite(snrs)):
            out["n_boot_degenerate"] += 1
            continue
        slopes.append(c1)
        for p, v in zip(probs, snrs):
            values[p].append(v)
    out["n_boot_valid"] = len(slopes)
    if slopes:
        q = 100.0 * (1.0 - level) / 2.0
        lo, hi = np.percentile(slopes, [q, 100 - q])
        out["slope_lo"], out["slope_hi"] = float(lo), float(hi)
        for p in probs:
            lo, hi = np.percentile(values[p], [q, 100 - q])
            out[f"snr_p{p:g}_lo"], out[f"snr_p{p:g}_hi"] = float(lo), float(hi)
    return out


SNR_FLIP_COLUMNS = (
    "source", "kind", "judge", "snr_axis", "weighting", "bin_lo", "bin_hi", "p_flip",
    "snr_db", "snr_lo", "snr_hi", "n_rows", "n_unique_ecg_ids", "n_flips",
    "record_flip_rate", "rflip_exact_lo", "rflip_exact_hi", "mean_abs_dp",
    "intercept", "slope", "slope_lo", "slope_hi", "converged", "snr_obs_min",
    "snr_obs_max", "n_boot", "n_boot_valid", "n_boot_degenerate", "note",
)
_CLUSTER_NOTE = {
    "error_scaling": "each ecg_id appears at several alphas and conditions",
    "noise": "each ecg_id appears at several SNR levels",
}


def snr_flip_rows(
    df, source, judge, snr_col, probs=FLIP_PROBS, n_boot=1000, seed=0,
    edges=SNR_BIN_EDGES, snr_axis="", extra_note="",
):
    """Bin rows and logistic rows (snr_flip_v0.csv) of one judge, source and
    SNR axis (`snr_col`, labelled `snr_axis`).

    df: per-record rows with ecg_id, any_flip, mean_abs_dp and `snr_col`.
    Bins count rows (weighting 'rows'); a bin in which an ecg_id has several
    rows notes that its exact CI is too narrow.  Logistic rows: one per
    weighting in WEIGHTINGS and flip probability (flip_logistic); when every
    ecg_id has the same number of rows the 'ecg_id' fit equals the 'rows' fit
    and is reused.  extra_note is appended to the logistic rows' notes.
    """
    snr = pd.to_numeric(df[snr_col], errors="coerce").to_numpy(dtype=float)
    flip = _as_bool(df["any_flip"]).to_numpy(dtype=bool)
    groups = df["ecg_id"].to_numpy()
    dp = pd.to_numeric(df["mean_abs_dp"], errors="coerce").to_numpy(dtype=float)
    cluster = _CLUSTER_NOTE.get(source, "rows are clustered by ecg_id")
    head = {"source": source, "judge": judge, "snr_axis": snr_axis}
    rows = []
    for b in snr_bins(snr, flip, groups, dp, edges=edges):
        clustered = b["n_rows"] > b["n_unique_ecg_ids"]
        rows.append(
            {
                **head, "kind": "bin", "weighting": "rows", **b,
                "note": (
                    f"rows clustered ({cluster}); the exact CI treats rows as "
                    "independent and is too narrow" if clustered else ""
                ),
            }
        )
    boot = dict(probs=probs, n_boot=n_boot, seed=seed)
    fits = {"rows": flip_logistic(snr, flip, groups, weighting="rows", **boot)}
    lo_ids, hi_ids = fits["rows"]["rows_per_id_min"], fits["rows"]["rows_per_id_max"]
    equal = lo_ids == hi_ids
    if equal:
        fits["ecg_id"] = {**fits["rows"], "weighting": "ecg_id"}
    else:
        fits["ecg_id"] = flip_logistic(snr, flip, groups, weighting="ecg_id", **boot)
    for weighting in WEIGHTINGS:
        fit = fits[weighting]
        for p in probs:
            value = fit[f"snr_p{p:g}"]
            notes = [
                f"CI: cluster bootstrap over {fit['n_unique_ecg_ids']} ecg_ids "
                f"({fit['n_boot_degenerate']} of {fit['n_boot']} draws degenerate, "
                "skipped)"
            ]
            if fit["fit"] != "ok":
                notes.insert(0, fit["fit"])
            if 0 < fit["n_flips"] < MIN_FIT_FLIPS:
                notes.append(f"only {fit['n_flips']} flips")
            if equal:
                if weighting == "ecg_id":
                    notes.append("equal rows per ecg_id: same fit as weighting rows")
            elif weighting == "rows":
                notes.append(
                    f"{lo_ids}-{hi_ids} rows per ecg_id: ids with more rows weigh "
                    "more (see weighting ecg_id)"
                )
            else:
                notes.append(f"every ecg_id weighs 1 ({lo_ids}-{hi_ids} rows each)")
            if np.isfinite(fit["slope"]) and fit["slope"] >= 0:
                notes.append("slope >= 0: flips do not fall with SNR")
            if np.isfinite(fit["slope"]) and not fit["converged"]:
                notes.append("fit not converged")
            if np.isfinite(value) and not (
                fit["snr_obs_min"] <= value <= fit["snr_obs_max"]
            ):
                notes.append(
                    f"extrapolated beyond the observed SNR {fit['snr_obs_min']:.1f}-"
                    f"{fit['snr_obs_max']:.1f} dB"
                )
            if extra_note:
                notes.append(extra_note)
            rows.append(
                {
                    **head, "kind": "logistic", "weighting": weighting, "p_flip": p,
                    "snr_db": value, "snr_lo": fit[f"snr_p{p:g}_lo"],
                    "snr_hi": fit[f"snr_p{p:g}_hi"],
                    **{
                        k: fit[k]
                        for k in (
                            "n_rows", "n_unique_ecg_ids", "n_flips", "intercept",
                            "slope", "slope_lo", "slope_hi", "converged",
                            "snr_obs_min", "snr_obs_max", "n_boot", "n_boot_valid",
                            "n_boot_degenerate",
                        )
                    },
                    "note": "; ".join(notes),
                }
            )
    return rows


# ----------------------------------------------------------------------------
# Subcommands
# ----------------------------------------------------------------------------
def run_thresholds(args):
    """Fold-9 Youden thresholds on the full and the layout-masked input of --source:
    records500 (default) = the 500 -> 100 Hz condition path (`ptbxl500_logits`),
    records100 = the official 100 Hz files (`ptbxl_logits`)."""
    J = _judge_module()
    root = _path(args.ptbxl_root)
    db = J.load_ptbxl_db(root)
    classes = tuple(J.CLASSES)
    # The held-out records and every other record of their patients (as the judge's own
    # split_ids; ecg_id 39 is a fold-9 sibling), or the ids alone for a judge module
    # without held_out_patient_records.
    siblings = getattr(J, "held_out_patient_records", None)
    held_ids = siblings(db) if siblings is not None else J.HELD_OUT_ECG_IDS
    held_out = set(int(i) for i in held_ids)
    labelled = db[list(classes)].sum(axis=1) > 0
    fold9 = db[(db["strat_fold"] == 9) & labelled & ~db.index.isin(held_out)]
    ids = [int(i) for i in fold9.index]
    y = fold9[list(classes)].to_numpy(dtype=int)
    source = args.source
    print(f"fold 9: {len(ids)} labelled records (held-out patients excluded), {source}")
    if source == "records500":
        absent = [
            i for i in ids if not os.path.exists(records500_path(db, i, root) + ".hea")
        ]
        if absent:
            raise SystemExit(
                f"records500 missing for {len(absent)}/{len(ids)} fold-9 records under "
                f"{root} (e.g. ecg_id {absent[:5]}); complete the download or use "
                "--source records100."
            )
    rows = []
    for name in args.judges:
        judge, cache = _open_judge(J, name, args)
        for kind in INPUT_KINDS:
            if source == "records500":
                logits = ptbxl500_logits(
                    J, judge, db, ids, kind, root, cache, args.batch_size
                )
            else:
                logits = ptbxl_logits(J, judge, ids, kind, root, cache, args.batch_size)
            thr = np.asarray(J.youden_thresholds(y, logits), dtype=float)
            per_class, macro = J.macro_auroc(y, logits)
            print(f"  {name:<12} {kind:<6} fold-9 macro AUROC {macro:.4f}")
            for c, cls in enumerate(classes):
                rows.append(
                    {
                        "judge": name,
                        "input": kind,
                        "class": cls,
                        "threshold_logit": float(thr[c]),
                        "n_pos": int(y[:, c].sum()),
                        "n_neg": int(len(ids) - y[:, c].sum()),
                        "auroc_fold9": float(per_class[c]),
                        "device": str(args.device),
                        "source": source,
                    }
                )
        if cache is not None:
            cache.save()
    out = pd.DataFrame(rows)
    _write_csv(out, _path(args.out))
    return out


def _layout_row(name, analysis, ref_input, test_input, thresholds, res, extra=None):
    row = {
        "judge": name,
        "analysis": analysis,
        "ref_input": ref_input,
        "test_input": test_input,
        "thresholds": thresholds,
    }
    row.update(res)
    row.update(extra or {})
    return row


def _logit_diff(a, b):
    d = np.abs(np.asarray(a, dtype=float) - np.asarray(b, dtype=float))
    return {
        "logit_absdiff_max": float(d.max()) if d.size else float("nan"),
        "logit_absdiff_median": float(np.median(d)) if d.size else float("nan"),
    }


def fold10_500hz(J, judge, db, ids, root, cache=None, batch_size=512, chunk=512):
    """records500 logits of fold-10 records and |records500 path - records100|.

    Every record goes through the condition path (`ptbxl500_inputs`) as 'full'
    and 'layout' input; the records100 inputs of the same kind give the
    signal difference per zone (head / interior / tail, layout: inside the
    windows only).  Returns ({kind: (N, C) logits}, {kind: zone stats}).
    """
    logits = {kind: [] for kind in INPUT_KINDS}
    values = {kind: {} for kind in INPUT_KINDS}
    zones = {
        "full": _zone_masks(N_JUDGE, len(LEADS)),
        "layout": _zone_masks(N_JUDGE, len(LEADS), layout_mask(N_JUDGE, FS_JUDGE)),
    }
    for start in range(0, len(ids), chunk):
        part = [int(i) for i in ids[start : start + chunk]]
        r500 = ptbxl500_inputs(db, part, root)
        for kind in INPUT_KINDS:
            keys = [f"ptbxl500/{i}/{kind}" for i in part]
            logits[kind].append(
                cached_logits(judge, r500[kind], keys, cache, batch_size)
            )
            r100 = np.stack(ptbxl_inputs(J, part, kind, root))
            d = np.abs(np.stack(r500[kind]) - r100)
            for zone, zone_mask in zones[kind].items():
                values[kind].setdefault(zone, []).append(d[:, zone_mask].ravel())
        if cache is not None:
            cache.save()
        print(f"  records500: {min(start + chunk, len(ids))}/{len(ids)} records")
    out_logits = {kind: np.concatenate(v) for kind, v in logits.items()}
    stats = {
        kind: _zone_summary({z: np.concatenate(v) for z, v in values[kind].items()})
        for kind in INPUT_KINDS
    }
    return out_logits, stats


def run_layout_loss(args):
    """Layout information loss (10 s vs the 3x4 windows), independent of digitisation.

    Rows (analysis / thresholds):
      fold10                  records100 full vs layout (own = full thresholds for
                              the full input, layout thresholds for the layout
                              input; full = full thresholds for both)
      fold10_500hz            the same through the 500 Hz condition path
      path_floor_fold10       records100 vs records500 path, per input kind at
                              that kind's thresholds: how robust the decisions
                              are to the source change (with records500
                              thresholds no longer a bias of the table)
      ids1_32_500hz           layout loss of the 32 condition originals
      resample_check_ids1_32  records100 vs our 500 -> 100 Hz path, per kind
    Fold-10 analyses use the labelled records (as judge_fold10.csv).
    """
    J = _judge_module()
    root = _path(args.ptbxl_root)
    db = J.load_ptbxl_db(root)
    classes = tuple(J.CLASSES)
    labelled = db[list(classes)].sum(axis=1) > 0
    fold10_ids = [int(i) for i in db.index[(db["strat_fold"] == 10) & labelled]]
    print(f"fold 10: {len(fold10_ids)} labelled records")
    y10, lab10, _ = labels_for(db, fold10_ids, classes)
    ids500 = [
        i for i in fold10_ids if os.path.exists(records500_path(db, i, root) + ".hea")
    ]
    if len(ids500) < len(fold10_ids):
        print(
            f"WARNING: records500 found for {len(ids500)}/{len(fold10_ids)} fold-10 "
            "records; the 500 Hz fold-10 analyses use those only.",
            file=sys.stderr,
        )
    pos10 = {e: i for i, e in enumerate(fold10_ids)}
    sel500 = np.array([pos10[e] for e in ids500], dtype=int)
    y5, lab5, _ = labels_for(db, ids500, classes)
    original_dir = _path(args.original_dir)
    orig_records = list_records(original_dir)
    orig_ids = [ecg_id_of(r) for r in orig_records]
    y32, lab32, _ = labels_for(db, orig_ids, classes)
    thr_path = _path(args.thresholds)
    boot = dict(n_boot=args.n_boot, seed=args.seed, classes=classes)

    # Full and layout-masked originals through the 500 Hz path (ecg_id 1-32).
    ours = {kind: [] for kind in INPUT_KINDS}
    for rec in orig_records:
        x, fs = read_record(os.path.join(original_dir, rec))
        for kind in INPUT_KINDS:
            ours[kind].append(_prepare_kind(x, fs, kind))

    rows = []
    for name in args.judges:
        judge, cache = _open_judge(J, name, args)
        thr = load_thresholds(thr_path, name, classes)
        check_threshold_device(thr_path, name, args.device)

        # (i) fold 10, records100.
        lf = ptbxl_logits(J, judge, fold10_ids, "full", root, cache, args.batch_size)
        ll = ptbxl_logits(J, judge, fold10_ids, "layout", root, cache, args.batch_size)
        if cache is not None:
            cache.save()
        res = compare(
            y10, lf, ll, thr["full"], thr_test=thr["layout"], labelled=lab10, **boot
        )
        rows.append(_layout_row(
            name, "fold10", "records100_full", "records100_layout", "own", res
        ))
        res = compare(y10, lf, ll, thr["full"], labelled=lab10, **boot)
        rows.append(_layout_row(
            name, "fold10", "records100_full", "records100_layout", "full", res
        ))

        # (ii) fold 10 through the 500 Hz path and (iii) the path noise floor.
        if ids500:
            l500, stats = fold10_500hz(
                J, judge, db, ids500, root, cache, args.batch_size
            )
            f5, lay5 = l500["full"], l500["layout"]
            res = compare(
                y5, f5, lay5, thr["full"], thr_test=thr["layout"], labelled=lab5,
                **boot,
            )
            rows.append(_layout_row(
                name, "fold10_500hz", "records500_full", "records500_layout", "own",
                res,
            ))
            res = compare(y5, f5, lay5, thr["full"], labelled=lab5, **boot)
            rows.append(_layout_row(
                name, "fold10_500hz", "records500_full", "records500_layout", "full",
                res,
            ))
            r100 = {"full": lf[sel500], "layout": ll[sel500]}
            for kind in INPUT_KINDS:
                res = compare(
                    y5, r100[kind], l500[kind], thr[kind], labelled=lab5, **boot
                )
                extra = {**_logit_diff(r100[kind], l500[kind]), **stats[kind]}
                rows.append(_layout_row(
                    name, "path_floor_fold10", f"records100_{kind}",
                    f"records500_{kind}", kind, res, extra,
                ))

        # (iv) ecg_id 1-32 through the 500 Hz path.
        k500f = [f"ids/{r}/full" for r in orig_records]
        k500l = [f"ids/{r}/layout" for r in orig_records]
        of = cached_logits(judge, ours["full"], k500f, cache, args.batch_size)
        ol = cached_logits(judge, ours["layout"], k500l, cache, args.batch_size)
        res = compare(
            y32, of, ol, thr["full"], thr_test=thr["layout"], labelled=lab32, **boot
        )
        rows.append(_layout_row(
            name, "ids1_32_500hz", "original_full", "original_layout", "own", res
        ))
        res = compare(y32, of, ol, thr["full"], labelled=lab32, **boot)
        rows.append(_layout_row(
            name, "ids1_32_500hz", "original_full", "original_layout", "full", res
        ))

        # (v) resampling check: our 500 -> 100 Hz path vs the official records100.
        for kind, l_ours in (("full", of), ("layout", ol)):
            official = ptbxl_inputs(J, orig_ids, kind, root)
            l_off = ptbxl_logits(J, judge, orig_ids, kind, root, cache, args.batch_size)
            res = compare(y32, l_off, l_ours, thr[kind], labelled=lab32, **boot)
            mask = None if kind == "full" else layout_mask(N_JUDGE, FS_JUDGE)
            extra = {
                **_logit_diff(l_off, l_ours),
                **signal_diff_stats(np.stack(ours[kind]), np.stack(official), mask),
            }
            rows.append(_layout_row(
                name, "resample_check_ids1_32", f"records100_{kind}",
                f"original_{kind}_500to100", kind, res, extra,
            ))
        if cache is not None:
            cache.save()

    out = pd.DataFrame(rows)
    for r in rows:
        line = (
            f"  {r['judge']:<12} {r['analysis']:<22} {r['ref_input']:>17} vs "
            f"{r['test_input']:<24} thr={r['thresholds']:<6} n={r['n_records']:<5} "
            f"kappa {_kappa_cell(r)} flip {_fmt(r['flip_rate'])} "
            f"rec-flip {r['n_record_flips']}/{r['n_records']} "
            f"[{_fmt(r['rflip_exact_lo'])}, {_fmt(r['rflip_exact_hi'])}] "
            f"AUROC {_fmt(r['auroc_ref_macro'])} -> {_fmt(r['auroc_test_macro'])} "
            f"(ret {_fmt(r['auroc_retention_macro'])})"
        )
        if "sig_interior_max_mv" in r:
            line += (
                f" |dx| mV head/int/tail max {_fmt(r['sig_head_max_mv'])}/"
                f"{_fmt(r['sig_interior_max_mv'])}/{_fmt(r['sig_tail_max_mv'])}"
            )
        print(line)
    _write_csv(out, _path(args.out))
    return out


def condition_entry(cond):
    """(ground-truth dir, default digitised dir, set name) of a condition:
    the CONDITIONS entry, else the REAL_CONDITIONS one; ValueError otherwise."""
    if cond in CONDITIONS:
        return CONDITIONS[cond]
    if cond in REAL_CONDITIONS:
        return REAL_CONDITIONS[cond]
    raise ValueError(
        f"Unknown condition {cond!r}; known conditions: "
        f"{[*CONDITIONS, *REAL_CONDITIONS]}."
    )


def is_real(cond):
    """True for a condition of the real-image set (REAL_SET)."""
    return condition_entry(cond)[2] == REAL_SET


def condition_run(cond, pred_run=PRED_RUN):
    """The digitiser run a table row of `cond` reads: pred_run, except that the
    default tag PRED_RUN (a synthetic-set run) means REAL_PRED_RUN for a real
    condition, so the default and an explicit --pred_run bl_o05 read the same
    folder under the same cache key and pred_run column."""
    if pred_run == PRED_RUN and is_real(cond):
        return REAL_PRED_RUN
    return pred_run


def _run_fields(cond, pred_run):
    """(set name, format fields) of a condition under a non-default run tag."""
    set_name = condition_entry(cond)[2]
    if set_name == REAL_SET and cond.startswith(REAL_PREFIX):
        short = cond[len(REAL_PREFIX):]
    else:
        short = cond[len("b_"):] if cond.startswith("b_") else cond
    return set_name, dict(cond=cond, c=short, run=check_pred_run(pred_run))


def pred_dir(cond, pred_run=PRED_RUN):
    """Digitised folder of a condition for a digitiser run tag.

    The default tag gives the CONDITIONS entry (the table_v0 / table_v1
    outputs); another tag the RUN_PRED_DIRS template of the condition's set:
    data/perspective_results/out/<cond>/<tag> (v0 sets),
    data/fidelity/<tag>/b_<c>/<tag> (fidelity_b),
    data/real_results/out/<c>/<tag> (real_c2, whose default entry is the
    REAL_PRED_RUN folder).  ValueError for a set without a template (e.g. a
    test set) or an unknown condition.
    """
    set_name, fields = _run_fields(cond, pred_run)
    if pred_run == PRED_RUN:
        return condition_entry(cond)[1]
    if set_name not in RUN_PRED_DIRS:
        raise ValueError(
            f"Condition {cond!r} (set {set_name!r}) has no digitised folder for "
            f"pred_run {pred_run!r}; tagged runs exist for sets "
            f"{sorted(RUN_PRED_DIRS)}."
        )
    return RUN_PRED_DIRS[set_name].format(**fields)


def ev_csv(cond, pred_run=PRED_RUN):
    """Evaluation CSV of a condition for a digitiser run tag.

    Default tag: the EV_CSVS entry, else the EV_CSV template (unchanged).
    Another tag: EV_CSV with the tag for the v0 sets
    (data/perspective_results/ev_<cond>_<tag>.csv), the RUN_EV_CSVS template
    for the other sets (fidelity_b: data/fidelity/<tag>/ev_<c>_<tag>.csv,
    real_c2: data/real_results/ev/ev_<c>_<tag>.csv; its default entry is the
    REAL_PRED_RUN one); ValueError for a set without one.
    """
    if pred_run == PRED_RUN:
        return EV_CSVS.get(cond, EV_CSV.format(cond=cond, run=PRED_RUN))
    set_name, fields = _run_fields(cond, pred_run)
    if set_name in V0_SETS:
        return EV_CSV.format(**fields)
    if set_name not in RUN_EV_CSVS:
        raise ValueError(
            f"Condition {cond!r} (set {set_name!r}) has no evaluation CSV for "
            f"pred_run {pred_run!r}."
        )
    return RUN_EV_CSVS[set_name].format(**fields)


def resolve_conditions(names, allow_real=False):
    """Condition tokens -> validated list of condition names.

    None / [] mean ['all'].  Each token expands in place: 'all' -> the v0
    conditions (sets V0_SETS), a set name (gen_eval, quasi_real, fidelity_b)
    -> the conditions of that set, a condition name -> itself; always in
    CONDITIONS order within a token.  Duplicates are dropped (first
    occurrence kept); unknown tokens abort.  With allow_real (the table
    subcommand) the real-image conditions (REAL_CONDITIONS) are known too:
    'real' or 'real_c2' -> all of them in REAL_CONDITIONS order, real_<c> ->
    itself; 'all' never includes them.  Without allow_real a real token
    aborts with its own message.
    """
    names = list(names) if names else ["all"]
    sets = {s for _, _, s in CONDITIONS.values()}
    out, unknown, real = [], [], []
    for name in names:
        if name == "all":
            expanded = [c for c, v in CONDITIONS.items() if v[2] in V0_SETS]
        elif name in sets:
            expanded = [c for c, v in CONDITIONS.items() if v[2] == name]
        elif name in CONDITIONS:
            expanded = [name]
        elif name in REAL_GROUPS or name in REAL_CONDITIONS:
            if not allow_real:
                real.append(name)
                continue
            expanded = list(REAL_CONDITIONS) if name in REAL_GROUPS else [name]
        else:
            unknown.append(name)
            continue
        for c in expanded:
            if c not in out:
                out.append(c)
    if unknown:
        known = [*CONDITIONS, *REAL_CONDITIONS] if allow_real else list(CONDITIONS)
        groups = ["all", *sorted(sets), *(REAL_GROUPS if allow_real else ())]
        raise ValueError(
            f"Unknown condition(s) {unknown}; known conditions: {known}; "
            f"groups: {groups}."
        )
    if real:
        raise ValueError(
            f"Real-image condition(s) {real} ({REAL_SET}) are read by the table "
            "subcommand only (with --split)."
        )
    return out


def v0_outputs():
    """The logged v0 result files (default outputs of table, sweep-error and
    margins), read from the module globals at call time."""
    return (
        TABLE_CSV, default_records_out(TABLE_CSV),
        SWEEP_ERROR_CSV, default_records_out(SWEEP_ERROR_CSV),
        MARGINS_CSV,
    )


def guard_v0_outputs(conditions, paths, pred_run=PRED_RUN, partial=False):
    """Abort before any work when a non-v0 condition would write a v0 result.

    A condition is non-v0 when its set (condition_entry(c)[2]) is not in
    V0_SETS, or when pred_run is not the default (another digitiser run is not
    the logged v0 result either), or when the run is partial (a record subset,
    table --max_records); paths are compared after _path and
    os.path.realpath, and existing files also by os.path.samefile, so another
    spelling of a v0 file is caught too (a case variant on a case-insensitive
    disk, a hard link).  v0-only conditions of the default run are never
    blocked, nor are non-v0 conditions writing other files.  The suggested new
    name follows the v0 file that was hit.
    """
    tagged = pred_run != PRED_RUN
    non_v0 = [
        c for c in conditions
        if tagged or partial or condition_entry(c)[2] not in V0_SETS
    ]
    if not non_v0:
        return

    def v0_match(p):
        for q in v0_outputs():
            q_abs = _path(q)
            if os.path.realpath(p) == os.path.realpath(q_abs) or (
                os.path.exists(p) and os.path.exists(q_abs)
                and os.path.samefile(p, q_abs)
            ):
                return q
        return None

    hits = [(p, q) for p in map(_path, paths) for q in [v0_match(p)] if q]
    if hits:
        mains = (TABLE_CSV, SWEEP_ERROR_CSV)
        base = hits[0][1]
        base = next((m for m in mains if base == default_records_out(m)), base)
        stem, ext = os.path.splitext(base)
        # A tagged run gets a name that carries its tag, so that it can never be
        # mistaken for the logged v1 reference of the default run.
        suffix = f"_{pred_run}" if tagged else ("_part" if partial else "_v1")
        example = (stem[:-3] if stem.endswith("_v0") else stem) + suffix + ext
        records = " (and --records_out)" if base in mains else ""
        run = f" of pred_run {pred_run!r}" if tagged else ""
        raise ValueError(
            f"Non-v0 condition(s) {non_v0}{run} would overwrite the v0 result(s) "
            f"{[p for p, _ in hits]}; pass --out{records} with a new name, "
            f"e.g. {example}."
        )


def _inside(path, folder):
    """True when `path` (existing or not) lies in `folder` or is it (existing
    or not): every ancestor of its realpath is compared with the folder, by
    os.path.samefile when both exist (a case variant or another spelling of
    the folder is caught too), else by realpath ignoring case when neither
    exists (a run may not create the folder either; ignoring case may refuse
    a case variant on a case-sensitive disk, never let one through)."""
    folder = os.path.realpath(_path(folder))
    folder_exists = os.path.isdir(folder)
    p = os.path.realpath(_path(path))
    while True:
        exists = os.path.exists(p)
        if exists and folder_exists:
            if os.path.samefile(p, folder):
                return True
        elif not exists and not folder_exists:
            if os.path.normcase(p).casefold() == os.path.normcase(folder).casefold():
                return True
        parent = os.path.dirname(p)
        if parent == p:
            return False
        p = parent


def guard_real_outputs(conditions, paths):
    """Abort before any work when a real-image condition would write under
    FIDELITY_DIR (the v0 / v1 / v2 tables and the Phase B results live
    there): real_c2 tables go elsewhere, e.g. data/real_results/c2/.  The
    default --out of table is inside FIDELITY_DIR, so a real run needs --out;
    paths includes the logit-cache folder (--cache_dir, unless --no_cache),
    whose default is inside FIDELITY_DIR too, so a real run needs --cache_dir
    (e.g. data/real_results/c2/cache) or --no_cache."""
    real = [c for c in conditions if is_real(c)]
    if not real:
        return
    hits = [p for p in map(_path, paths) if _inside(p, FIDELITY_DIR)]
    if hits:
        raise ValueError(
            f"Real-image condition(s) {real} may not write under {FIDELITY_DIR}: "
            f"{hits}; pass --out (and --records_out) elsewhere, e.g. "
            "data/real_results/c2/table_<tag>_<split>.csv, and --cache_dir "
            "elsewhere, e.g. data/real_results/c2/cache (or --no_cache)."
        )


def load_split(path=None):
    """The real-image patient split (default SPLIT_CSV): DataFrame with at
    least record and split (dev / eval); aborts on a missing column, an
    unknown split value or a duplicate record."""
    path = SPLIT_CSV if path is None else path
    df = pd.read_csv(_path(path), dtype={"record": str})
    missing = [c for c in ("record", "split") if c not in df.columns]
    if missing:
        raise ValueError(f"Split file {path} lacks column(s) {missing}.")
    bad = sorted(set(df["split"].astype(str)) - set(SPLITS))
    if bad:
        raise ValueError(f"Split file {path}: unknown split value(s) {bad}.")
    dup = sorted(df.loc[df["record"].duplicated(), "record"])
    if dup:
        raise ValueError(f"Split file {path}: duplicate record(s) {dup[:5]}.")
    return df


def split_records(records, split_df, split):
    """The records of one split, in the given order.

    The split file must cover exactly the given records (the ground-truth
    folder of a condition): a record missing on either side aborts, so a
    stale or wrong split file cannot silently shrink a condition.
    """
    if split not in SPLITS:
        raise ValueError(f"Unknown split {split!r}; splits: {list(SPLITS)}.")
    of = dict(zip(split_df["record"].astype(str), split_df["split"].astype(str)))
    records = [str(r) for r in records]
    not_in_file = [r for r in records if r not in of]
    not_on_disk = sorted(set(of) - set(records))
    if not_in_file or not_on_disk:
        raise ValueError(
            f"Split file and records differ: {len(not_in_file)} record(s) not in "
            f"the split file (e.g. {not_in_file[:3]}), {len(not_on_disk)} split "
            f"record(s) not on disk (e.g. {not_on_disk[:3]})."
        )
    return [r for r in records if of[r] == split]


def table_records(gt_dir, split=None, split_df=None, max_records=0):
    """Records of a table row: None (every ground-truth record) without split
    and limit, else the sorted ground-truth records of the split (split_df
    required), then the first max_records of them (> 0; smoke runs only)."""
    if split is None and not max_records:
        return None
    records = list_records(gt_dir)
    if split is not None:
        records = split_records(records, split_df, split)
    if max_records:
        records = records[: int(max_records)]
    return records


def _check_table_args(args, conditions):
    """Checks of the real-image options of table, before anything is loaded.

    Real conditions: not mixed with other sets, --split required, the
    originals must be named (--original_dir ptbxl500 or a folder other than
    the v0 default); other sets: --split refused.  --max_records >= 0.
    """
    real = [c for c in conditions if is_real(c)]
    split = getattr(args, "split", None)
    max_records = getattr(args, "max_records", 0) or 0
    if max_records < 0:
        raise ValueError(f"--max_records must be >= 0, got {max_records}.")
    if real and len(real) != len(conditions):
        other = [c for c in conditions if c not in real]
        raise ValueError(
            f"Real-image conditions {real} cannot share a table run with {other}."
        )
    if real and split is None:
        raise ValueError(
            f"Real-image condition(s) {real} need --split ({' or '.join(SPLITS)}): "
            "choices are made on dev, eval is used once for the headline."
        )
    if not real and split is not None:
        raise ValueError(
            f"--split applies to the real-image conditions ({REAL_SET}) only; "
            f"got {conditions}."
        )
    if real and args.original_dir == ORIGINAL_DIR:
        raise ValueError(
            f"Real-image condition(s) {real}: the originals are the PTB-XL "
            f"records500 files; pass --original_dir {ORIGINAL_PTBXL500}."
        )


def run_table(args):
    """Fidelity table: judge x condition, next to the SNR median.

    --pred_run picks the digitiser run (pred_dir / ev_csv); its folders are
    resolved, and a set without a tagged layout rejected, before the judge
    loads.  Real-image conditions (real_c2, see REAL_CONDITIONS) need --split
    and a table of their own (_check_table_args, guard_real_outputs); their
    default run is REAL_PRED_RUN (condition_run); their rows end with a
    'split' column and their record rows with 'split' and 'patient_id'.
    --original_dir ptbxl500 reads each record's records500 file.
    --max_records N (smoke only) keeps the first N records of every condition.
    """
    conditions = resolve_conditions(args.conditions, allow_real=True)
    _check_table_args(args, conditions)
    split = getattr(args, "split", None)
    max_records = getattr(args, "max_records", 0) or 0
    pred_run = args.pred_run
    records_out = _records_out(args)
    # The real guard first: its message names a place outside data/fidelity.
    cache_paths = [] if args.no_cache else [_path(args.cache_dir)]
    guard_real_outputs(conditions, [_path(args.out), records_out, *cache_paths])
    guard_v0_outputs(
        conditions, [_path(args.out), records_out], pred_run=pred_run,
        partial=bool(max_records),
    )
    runs = {cond: condition_run(cond, pred_run) for cond in conditions}
    pred_dirs = {cond: pred_dir(cond, runs[cond]) for cond in conditions}
    ev_paths = {cond: _path(ev_csv(cond, runs[cond])) for cond in conditions}
    split_df = (
        load_split(getattr(args, "split_csv", None)) if split is not None else None
    )
    patients = {}
    if split_df is not None and "patient_id" in split_df.columns:
        patients = dict(zip(split_df["record"].astype(str), split_df["patient_id"]))
    elif split_df is not None:
        raise ValueError(
            "The split file lacks patient_id: the real-image CIs resample patients."
        )
    selected = {
        cond: table_records(
            _path(condition_entry(cond)[0]), split, split_df, max_records
        )
        for cond in conditions
    }
    if max_records:
        print(
            f"WARNING: --max_records {max_records}: first {max_records} records per "
            "condition only (smoke run, not a result).",
            file=sys.stderr,
        )
    J = _judge_module()
    root = _path(args.ptbxl_root)
    db = J.load_ptbxl_db(root)
    classes = tuple(J.CLASSES)
    if args.original_dir == ORIGINAL_PTBXL500:
        originals = {
            cond: ptbxl500_originals(
                db,
                selected[cond] if selected[cond] is not None
                else list_records(_path(condition_entry(cond)[0])),
                root,
            )
            for cond in conditions
        }
    else:
        originals = {cond: _path(args.original_dir) for cond in conditions}
    thr_path = _path(args.thresholds)
    layout_loss_csv = _path(args.layout_loss) if args.layout_loss else None
    rows, record_frames, floors = [], [], {}
    for name in args.judges:
        judge, cache = _open_judge(J, name, args)
        thr = load_thresholds(thr_path, name, classes)
        check_threshold_device(thr_path, name, args.device)
        floors[name] = path_floor(layout_loss_csv, name)
        for cond in conditions:
            gt_dir, _, set_name = condition_entry(cond)
            ev_path = ev_paths[cond]
            ev = pd.read_csv(ev_path) if os.path.exists(ev_path) else None
            if ev is None:
                print(
                    f"WARNING: no evaluation CSV {ev_path}; SNR left empty.",
                    file=sys.stderr,
                )
            row, recs = evaluate_condition(
                judge, cond, _path(gt_dir), _path(pred_dirs[cond]),
                originals[cond], thr["layout"], thr["full"], labels=db, ev=ev,
                cache=cache, n_boot=args.n_boot, seed=args.seed, set_name=set_name,
                judge_name=name, batch_size=args.batch_size, pred_run=runs[cond],
                records=selected[cond],
                # Real-image rows: patient-level CIs, absent leads = missing.
                groups=patients if is_real(cond) else None,
                allow_missing=is_real(cond),
            )
            # The path floor keeps its v0 place: the split-half extras and
            # pred_run follow it, so the v0 / v1 columns stay a prefix of the row.
            tail = {k: row.pop(k) for k in (*split_half_columns(classes), "pred_run")}
            row.update(floors[name])
            row.update(tail)
            if split is not None:
                # Real-image rows only: the split after every existing column.
                row["split"] = split
                recs["split"] = split
                recs["patient_id"] = recs["record"].astype(str).map(patients)
            rows.append(row)
            record_frames.append(recs)
            if cache is not None:
                cache.save()
    header_run = runs[conditions[0]] if conditions else pred_run
    print("\n".join(format_table(rows, pred_run=header_run, split=split)))
    for name, floor in floors.items():
        if np.isfinite(floor["path_floor_record_flip_rate"]):
            print(
                f"path floor {name}: fold-10 records100 vs 500 Hz layout input at the "
                f"layout thresholds: {int(floor['path_floor_n_record_flips'])}/"
                f"{int(floor['path_floor_n_records'])} records flip "
                f"({_fmt(floor['path_floor_record_flip_rate'])}), flip rate "
                f"{_fmt(floor['path_floor_flip_rate'], '.4f')}, mean|dp| "
                f"{_fmt(floor['path_floor_mean_abs_dp'], '.4f')}"
            )
        else:
            print(f"path floor {name}: not available (run layout-loss first)")
    out = pd.DataFrame(rows)
    _write_csv(out, _path(args.out))
    _write_csv(pd.concat(record_frames, ignore_index=True), records_out)
    return out


def _open_judges(J, args, classes):
    """[(name, judge, cache, layout thresholds)] of args.judges."""
    thr_path = _path(args.thresholds)
    opened = []
    for name in args.judges:
        judge, cache = _open_judge(J, name, args)
        thr = load_thresholds(thr_path, name, classes)["layout"]
        check_threshold_device(thr_path, name, args.device)
        opened.append((name, judge, cache, thr))
    return opened


def _records_out(args, out=None):
    """--records_out, or <out without extension>_records.csv."""
    if args.records_out:
        return _path(args.records_out)
    return default_records_out(_path(args.out) if out is None else out)


def partial_out(out, default, start, stop):
    """Output path of a run on records start..stop-1 only: the default v0 path
    becomes <name>_part<start>-<stop>.csv so a partial run never looks like
    (or overwrites) the full result; an explicit other --out is kept."""
    out = _path(out)
    if os.path.abspath(out) != os.path.abspath(_path(default)):
        return out
    stem, ext = os.path.splitext(out)
    return f"{stem}_part{int(start)}-{int(stop)}{ext or '.csv'}"


def format_pooled_flips(records, level_col, snr_col):
    """Printed view of per-record flips pooled per judge and level (all
    conditions together): median record SNR, record flips [exact CI], mean |dp|."""
    lines = [
        f"{'judge':<12} {level_col:>7} {'SNR med':>8} {'rec flips [exact CI]':<26} "
        f"{'mean|dp|':>9}"
    ]
    for (judge, level), sub in records.groupby(["judge", level_col], sort=True):
        k, n = int(_as_bool(sub["any_flip"]).sum()), len(sub)
        lo, hi = clopper_pearson(k, n)
        snr = float(_nan_stat(np.nanmedian, sub[snr_col]))
        lines.append(
            f"{judge:<12} {level:>7g} {_fmt(snr, '.2f'):>8} "
            f"{f'{k}/{n} [{_fmt(lo)}, {_fmt(hi)}]':<26} "
            f"{_fmt(sub['mean_abs_dp'].mean(), '.4f'):>9}"
        )
    return lines


def run_sweep_error(args):
    """Scaled digitisation error: decision flips against the record SNR.

    For every condition and record x_alpha = masked + alpha (dig - masked) at
    500 Hz, judged through prepare_for_judge; reference = alpha 0 (masked),
    decisions at the layout thresholds.
    """
    conditions = resolve_conditions(args.conditions)
    records_out = _records_out(args)
    guard_v0_outputs(conditions, [_path(args.out), records_out])
    J = _judge_module()
    classes = tuple(J.CLASSES)
    alphas = check_alphas(args.alphas)
    opened = _open_judges(J, args, classes)
    rows, frames = [], []
    for cond in conditions:
        gt_dir, pred_dir, set_name = CONDITIONS[cond]
        signals = condition_signals(
            _path(gt_dir), _path(pred_dir), _path(args.original_dir)
        )
        sweep = error_sweep_inputs(signals, alphas, cond)
        ev_path = _path(ev_csv(cond))
        ev = pd.read_csv(ev_path) if os.path.exists(ev_path) else None
        if ev is None:
            print(
                f"WARNING: no evaluation CSV {ev_path}; no SNR check.", file=sys.stderr
            )
        check = snr_check(sweep["records"], sweep["snr1"], ev)
        for name, judge, cache, thr in opened:
            r, recs = error_sweep_judge(
                judge, sweep, thr, cache=cache, judge_name=name, set_name=set_name,
                n_boot=args.n_boot, seed=args.seed, check=check,
                batch_size=args.batch_size,
            )
            rows.extend(r)
            frames.append(recs)
            if cache is not None:
                cache.save()
        print(
            f"  {cond}: {len(sweep['records'])} records; SNR(1) vs ev snr_raw max "
            f"|diff| {_fmt(check['snr_check_max_absdiff'], '.2e')} dB over "
            f"{check['snr_check_n_leads']} leads ({check['snr_check_n_mismatch']} "
            f"finite on one side only); {sweep['n_leads_window_nan']} leads with "
            "NaN inside the window"
        )
    out = pd.DataFrame(rows)
    records = pd.concat(frames, ignore_index=True)
    print("\n".join(format_pooled_flips(records, "alpha", "snr_rec_median")))
    _write_csv(out, _path(args.out))
    _write_csv(records, records_out)
    return out


def run_sweep_noise(args):
    """Band-limited noise on the labelled fold-10 records (500 Hz path).

    --start / --max_records select records start .. start + max_records - 1
    of the sorted fold-10 list (a partial run writes <out>_part<a>-<b>.csv
    when --out is the default); --cache_only only fills the judges' logit
    caches for them and writes no CSV.  A long run is split into
    foreground-sized --cache_only parts and finished by one run without these
    options, which reads every logit from the cache.
    """
    J = _judge_module()
    root = _path(args.ptbxl_root)
    db = J.load_ptbxl_db(root)
    classes = tuple(J.CLASSES)
    snrs = check_snrs(args.snrs)
    cache_only = bool(getattr(args, "cache_only", False))
    if cache_only and args.no_cache:
        raise ValueError("--cache_only fills the logit cache; drop --no_cache.")
    all_ids = fold10_ids_500(db, root, classes)
    start = int(getattr(args, "start", 0) or 0)
    if start < 0:
        raise ValueError(f"--start must be >= 0, got {start}.")
    stop = len(all_ids)
    if args.max_records:
        stop = min(start + int(args.max_records), stop)
    ids = all_ids[start:stop]
    if not ids:
        raise ValueError(
            f"No records selected (--start {start}, {len(all_ids)} records)."
        )
    partial = len(ids) < len(all_ids)
    print(
        f"fold 10: records {start}-{stop - 1} of {len(all_ids)} labelled records500; "
        f"levels {snrs} dB" + ("; cache only" if cache_only else "")
    )
    opened = _open_judges(J, args, classes)
    rows, records = noise_sweep(
        opened, db, ids, root, snrs, seed=args.seed, n_boot=args.n_boot,
        classes=classes, batch_size=args.batch_size, chunk=args.chunk,
        cache_only=cache_only,
    )
    if cache_only:
        print(
            f"cache only: {len(ids)} records x {len(snrs)} levels cached; "
            "no CSV written"
        )
        return pd.DataFrame(rows)
    out = pd.DataFrame(rows)
    for r in rows:
        print(
            f"  {r['judge']:<12} SNR {r['snr_db']:>5g} dB (judge band "
            f"{_fmt(r['snr_judge_median_db'], '.2f')})  n={r['n_records']} "
            f"kappa {_kappa_cell(r)} rec-flip {r['n_record_flips']}/{r['n_records']} "
            f"[{_fmt(r['rflip_exact_lo'])}, {_fmt(r['rflip_exact_hi'])}] "
            f"mean|dp| {_fmt(r['mean_abs_dp'], '.4f')} AUROC "
            f"{_fmt(r['auroc_ref_macro'])} -> {_fmt(r['auroc_test_macro'])} "
            f"(ret {_fmt(r['auroc_retention_macro'])})"
        )
    out_path = partial_out(args.out, SWEEP_NOISE_CSV, start, stop) if partial else (
        _path(args.out)
    )
    _write_csv(out, out_path)
    _write_csv(records, _records_out(args, out_path))
    return out


JUDGE_SNR_COL = "snr_judge_rec_median"
_AXIS_NOTE = {
    "window_500hz": (
        "500 Hz axis: error_scaling and noise thresholds are not comparable here "
        "(part of the digitisation error lies above the judge's band, the noise "
        "does not); compare them on judge_100hz and do not apply a noise "
        "threshold to the 500 Hz condition SNR medians"
    ),
    "judge_100hz": "judge-band axis: comparable across sources",
}


def run_snr_flip(args):
    """SNR threshold of decision flips from the sweep per-record files.

    Every source is fitted on both SNR axes (SNR_AXES): its 500 Hz SNR
    (error_scaling: snr_rec_median; noise: the exact snr_db) and the
    judge-band record SNR (snr_judge_rec_median; skipped with a warning for
    files without it).  The logistic rows note the median of judge-band
    minus 500 Hz SNR over the source's rows.
    """
    sources = (
        ("error_scaling", args.error_records, "snr_rec_median", "alpha"),
        ("noise", args.noise_records, "snr_db", "snr_db"),
    )
    rows = []
    for source, path, snr_col, level_col in sources:
        path = _path(path) if path else None
        if not path or not os.path.exists(path):
            print(
                f"WARNING: no {source} records file {path}; skipped.", file=sys.stderr
            )
            continue
        df = pd.read_csv(path)
        level = pd.to_numeric(df[level_col], errors="coerce").to_numpy(dtype=float)
        # Drop the reference level itself (alpha 0 / no noise = inf dB).
        df = df[np.isfinite(level) & (level > 0)] if source == "error_scaling" else (
            df[np.isfinite(level)]
        )
        has_judge_snr = JUDGE_SNR_COL in df.columns
        if not has_judge_snr:
            print(
                f"WARNING: {path} has no {JUDGE_SNR_COL} column (older file); "
                "judge_100hz axis skipped.",
                file=sys.stderr,
            )
        judges = args.judges or sorted(df["judge"].unique())
        for judge in judges:
            sub = df[df["judge"] == judge]
            if sub.empty:
                print(f"WARNING: no {source} rows of {judge}.", file=sys.stderr)
                continue
            axes = [("window_500hz", snr_col)]
            gap_note = ""
            if has_judge_snr:
                axes.append(("judge_100hz", JUDGE_SNR_COL))
                s500 = pd.to_numeric(sub[snr_col], errors="coerce").to_numpy(float)
                s100 = pd.to_numeric(sub[JUDGE_SNR_COL], errors="coerce")
                s100 = s100.to_numpy(float)
                ok = np.isfinite(s500) & np.isfinite(s100)
                if ok.any():
                    gap = float(np.median(s100[ok] - s500[ok]))
                    gap_note = (
                        f"judge-band minus 500 Hz SNR: median {gap:+.2f} dB over "
                        f"{int(ok.sum())} rows"
                    )
            for axis, col in axes:
                note = "; ".join(n for n in (_AXIS_NOTE[axis], gap_note) if n)
                rows.extend(
                    snr_flip_rows(
                        sub, source, judge, col, n_boot=args.n_boot, seed=args.seed,
                        snr_axis=axis, extra_note=note,
                    )
                )
    out = pd.DataFrame(rows, columns=list(SNR_FLIP_COLUMNS))
    for r in rows:
        if r["kind"] == "logistic":
            print(
                f"  {r['judge']:<12} {r['source']:<13} {r['snr_axis']:<12} "
                f"{r['weighting']:<6} P(flip) = {r['p_flip']:.2f} at "
                f"SNR {_fmt(r['snr_db'], '.2f')} dB [{_fmt(r['snr_lo'], '.2f')}, "
                f"{_fmt(r['snr_hi'], '.2f')}]  slope {_fmt(r['slope'], '.3f')}/dB  "
                f"{r['n_flips']}/{r['n_rows']} rows flip, {r['n_unique_ecg_ids']} ids"
            )
    _write_csv(out, _path(args.out))
    return out


def run_margins(args):
    """Per-record minimum logit margin: labelled local records vs fold 10.

    Both sets hold labelled records only (>= 1 superclass, as
    fold10_ids_500): unlabelled local records are left out, and local ecg_ids
    that are also fold-10 records are removed from the reference so the two
    Mann-Whitney samples are disjoint.  Each row gives the number of records
    left out (n_excluded) and their ecg_ids (note).
    """
    cond = args.condition
    if cond not in CONDITIONS:
        raise ValueError(f"Unknown condition {cond!r}; known: {sorted(CONDITIONS)}.")
    guard_v0_outputs([cond], [_path(args.out)])
    J = _judge_module()
    root = _path(args.ptbxl_root)
    db = J.load_ptbxl_db(root)
    classes = tuple(J.CLASSES)
    gt_dir, pred_dir, _ = CONDITIONS[cond]
    signals = condition_signals(
        _path(gt_dir), _path(pred_dir), _path(args.original_dir)
    )
    local_ids = [int(i) for i in signals["ecg_ids"]]
    _, labelled, _ = labels_for(db, local_ids, classes)
    labelled = np.asarray(labelled, dtype=bool)
    keep = np.flatnonzero(labelled)
    unlabelled = [local_ids[i] for i in np.flatnonzero(~labelled)]
    local_inputs = [
        prepare_for_judge(signals["masked"][i], signals["fs"]) for i in keep
    ]
    local_keys = [error_key(cond, signals["records"][i], 0.0) for i in keep]
    all_ids = fold10_ids_500(db, root, classes)
    local_set = set(local_ids)
    overlap = [i for i in all_ids if i in local_set]
    ids = [i for i in all_ids if i not in local_set]
    partial = bool(args.max_records) and int(args.max_records) < len(ids)
    if args.max_records:
        ids = ids[: int(args.max_records)]
    print(
        f"local {cond}: {len(keep)} labelled records ({len(unlabelled)} unlabelled "
        f"left out); fold 10: {len(ids)} records ({len(overlap)} local ids removed)"
    )
    opened = _open_judges(J, args, classes)
    fold10 = fold10_layout_logits(opened, db, ids, root, args.batch_size, args.chunk)
    notes = (
        (len(unlabelled), f"unlabelled local ecg_ids left out: {unlabelled}"
         if unlabelled else ""),
        (len(overlap), f"local ecg_ids removed from the reference: {overlap}"
         if overlap else ""),
    )
    rows = []
    for name, judge, cache, thr in opened:
        l_local = cached_logits(judge, local_inputs, local_keys, cache, args.batch_size)
        if cache is not None:
            cache.save()
        pair = margin_rows(name, min_margin(l_local, thr),
                           min_margin(fold10[name], thr), f"local_{cond}")
        for row, (n_excluded, note) in zip(pair, notes):
            row.update(n_excluded=n_excluded, note=note)
        rows.extend(pair)
    out = pd.DataFrame(rows)
    for r in rows:
        cuts = " ".join(f"<{c:g}: {_fmt(r[f'frac_lt_{c:g}'])}" for c in MARGIN_CUTS)
        print(
            f"  {r['judge']:<12} {r['set']:<16} n={r['n']:<5} median min-margin "
            f"{_fmt(r['median_min_margin'])}  {cuts}  (excluded {r['n_excluded']})"
        )
    out_path = partial_out(args.out, MARGINS_CSV, 0, len(ids)) if partial else (
        _path(args.out)
    )
    _write_csv(out, out_path)
    return out


def margin_rows(judge_name, local, fold10, local_set="local"):
    """margins_v0.csv rows of one judge: the local set (with the two-sided
    Mann-Whitney p against fold 10) and fold 10."""
    local = np.asarray(local, dtype=float)
    fold10 = np.asarray(fold10, dtype=float)
    a, b = local[np.isfinite(local)], fold10[np.isfinite(fold10)]
    p = (
        float(mannwhitneyu(a, b, alternative="two-sided").pvalue)
        if a.size and b.size else float("nan")
    )
    return [
        {"judge": judge_name, "set": local_set, **margin_summary(local),
         "mwu_p_vs_fold10": p},
        {"judge": judge_name, "set": "fold10_500hz", **margin_summary(fold10),
         "mwu_p_vs_fold10": float("nan")},
    ]


# Parse arguments.
def get_parser():
    description = (
        "Diagnosis-fidelity harness: judge decisions of digitised vs masked "
        "original ECGs."
    )
    parser = argparse.ArgumentParser(description=description)
    common = argparse.ArgumentParser(add_help=False)
    common.add_argument(
        "--judges", nargs="+", default=list(DEFAULT_JUDGES), help="Judge model names."
    )
    common.add_argument("--ptbxl_root", default=PTBXL_ROOT, help="PTB-XL 1.0.3 folder.")
    common.add_argument(
        "--weights_dir", default=WEIGHTS_DIR, help="Folder of <name>_final.pt."
    )
    common.add_argument(
        "--device", default="cpu", help="Torch device of the judge (cpu or mps)."
    )
    common.add_argument(
        "--cache_dir", default=CACHE_DIR,
        help="Folder of the fid_<judge>.npz caches (fid_<judge>_<device>.npz off cpu). "
        f"A table run of the {REAL_SET} conditions needs one outside {FIDELITY_DIR} "
        "(e.g. data/real_results/c2/cache) or --no_cache.",
    )
    common.add_argument(
        "--no_cache", action="store_true", help="Do not read or write the logit cache."
    )
    common.add_argument("--batch_size", type=int, default=512, help="Judge batch size.")
    sub = parser.add_subparsers(dest="command", required=True)

    p = sub.add_parser("thresholds", parents=[common], help="Fold-9 Youden thresholds.")
    p.add_argument(
        "--source", default="records500", choices=THRESHOLD_SOURCES,
        help="Fold-9 inputs: records500 (default) through the 500 -> 100 Hz "
        "condition path, or the official records100.",
    )
    p.add_argument("--out", default=THRESHOLDS_CSV, help="Output CSV.")
    p.set_defaults(func=run_thresholds)

    for command, func, out, helptext in (
        ("layout-loss", run_layout_loss, LAYOUT_LOSS_CSV, "Layout information loss."),
        ("table", run_table, TABLE_CSV, "Fidelity table of the conditions."),
    ):
        p = sub.add_parser(command, parents=[common], help=helptext)
        p.add_argument(
            "--thresholds", default=THRESHOLDS_CSV, help="thresholds.csv (thresholds)."
        )
        original_help = "Full 10 s originals, 500 Hz."
        if command == "table":
            original_help += (
                f" '{ORIGINAL_PTBXL500}': each record's PTB-XL records500 file "
                "(<ptbxl_root>/<filename_hr>, nothing copied; required by the "
                f"{REAL_SET} conditions)."
            )
        p.add_argument("--original_dir", default=ORIGINAL_DIR, help=original_help)
        p.add_argument(
            "--n_boot", type=int, default=2000, help="Bootstrap replicates (records)."
        )
        p.add_argument("--seed", type=int, default=0, help="Bootstrap seed.")
        out_help = "Output CSV."
        if command == "table":
            out_help += (
                " Non-v0 conditions may not write the *_v0 defaults."
                f" The {REAL_SET} conditions may not write under {FIDELITY_DIR}."
            )
        p.add_argument("--out", default=out, help=out_help)
        if command == "table":
            p.add_argument(
                "--conditions", nargs="+", default=["all"],
                help="'all' (the v0 set: gen_eval + quasi_real), a set name "
                "(gen_eval, quasi_real, fidelity_b) or condition names; "
                f"'real' / '{REAL_SET}' or real_<c> for the real-image conditions "
                f"(never in 'all'; not mixed with other sets; need --split): "
                f"{', '.join(REAL_CONDITIONS)}.",
            )
            p.add_argument(
                "--split", default=None, choices=SPLITS,
                help=f"Record split of the {REAL_SET} conditions (required for "
                "them, refused otherwise): dev for every choice, eval once for "
                "the headline; the rows get a 'split' column at the end.",
            )
            p.add_argument(
                "--split_csv", default=SPLIT_CSV,
                help="Patient split file (record, split[, patient_id]); it must "
                "list exactly the ground-truth records of every condition.",
            )
            p.add_argument(
                "--max_records", type=int, default=0,
                help="SMOKE ONLY: first N records of every condition (after "
                "--split; 0 = all). Not a result; may not write the *_v0 defaults.",
            )
            p.add_argument(
                "--records_out", default=None,
                help="Per-record CSV (default <out without extension>_records.csv).",
            )
            p.add_argument(
                "--layout_loss", default=LAYOUT_LOSS_CSV,
                help="layout_loss.csv whose path_floor_fold10 rows go next to the table.",
            )
            p.add_argument(
                "--pred_run", default=PRED_RUN,
                help=f"Digitiser run tag (default {PRED_RUN}: the v0 / v1 outputs). "
                "Another tag reads data/perspective_results/out/<cond>/<tag> and "
                "ev_<cond>_<tag>.csv (gen_eval, quasi_real) or "
                "data/fidelity/<tag>/b_<c>/<tag> and "
                "data/fidelity/<tag>/ev_<c>_<tag>.csv "
                "(fidelity_b) or data/real_results/out/<c>/<tag> and "
                "data/real_results/ev/ev_<c>_<tag>.csv (real_<c>, whose default "
                f"run is {REAL_PRED_RUN}); its dig logits are cached under "
                "<cond>@<tag>/<rec>/dig.",
            )
        p.set_defaults(func=func)

    thresholds_help = "thresholds.csv (thresholds); the layout rows are used."
    p = sub.add_parser(
        "sweep-error", parents=[common],
        help="Scaled digitisation error of the conditions vs decision flips.",
    )
    p.add_argument("--thresholds", default=THRESHOLDS_CSV, help=thresholds_help)
    p.add_argument(
        "--original_dir", default=ORIGINAL_DIR, help="Full 10 s originals, 500 Hz."
    )
    p.add_argument(
        "--conditions", nargs="+", default=["all"],
        help="'all' (the v0 set: gen_eval + quasi_real), a set name "
        "(gen_eval, quasi_real, fidelity_b) or condition names.",
    )
    p.add_argument(
        "--alphas", nargs="+", type=float, default=list(DEFAULT_ALPHAS),
        help="Error scale factors (0 = masked original, 1 = digitised).",
    )
    p.add_argument("--n_boot", type=int, default=1000, help="Bootstrap replicates.")
    p.add_argument("--seed", type=int, default=0, help="Bootstrap seed.")
    p.add_argument(
        "--out", default=SWEEP_ERROR_CSV,
        help="Aggregate output CSV. Non-v0 conditions may not write the *_v0 defaults.",
    )
    p.add_argument(
        "--records_out", default=None,
        help="Per-record CSV (default <out without extension>_records.csv).",
    )
    p.set_defaults(func=run_sweep_error)

    p = sub.add_parser(
        "sweep-noise", parents=[common],
        help="Band-limited noise on the labelled fold-10 records500.",
    )
    p.add_argument("--thresholds", default=THRESHOLDS_CSV, help=thresholds_help)
    p.add_argument(
        "--snrs", nargs="+", type=float, default=list(DEFAULT_SNRS),
        help="Target window SNRs in dB ('inf' = no noise); the position in this "
        "list is the level_index of the noise seed.",
    )
    p.add_argument("--n_boot", type=int, default=1000, help="Bootstrap replicates.")
    p.add_argument(
        "--seed", type=int, default=0, help="Noise seed and bootstrap seed."
    )
    p.add_argument(
        "--max_records", type=int, default=0,
        help="Use N records from --start on (0 = all).",
    )
    p.add_argument(
        "--start", type=int, default=0,
        help="Index of the first record in the sorted fold-10 list (for parts).",
    )
    p.add_argument(
        "--cache_only", action="store_true",
        help="Only fill the logit cache for the selected records; write no CSV.",
    )
    p.add_argument("--chunk", type=int, default=256, help="Records read per chunk.")
    p.add_argument(
        "--out", default=SWEEP_NOISE_CSV,
        help="Aggregate output CSV (a partial run turns the default into "
        "<name>_part<start>-<stop>.csv).",
    )
    p.add_argument(
        "--records_out", default=None,
        help="Per-record CSV (default <out without extension>_records.csv).",
    )
    p.set_defaults(func=run_sweep_noise)

    p = sub.add_parser(
        "snr-flip", help="SNR threshold of decision flips from the sweep records."
    )
    p.add_argument(
        "--judges", nargs="+", default=None,
        help="Judge names (default: every judge in the input files).",
    )
    p.add_argument(
        "--error_records", default=default_records_out(SWEEP_ERROR_CSV),
        help="Per-record CSV of sweep-error.",
    )
    p.add_argument(
        "--noise_records", default=default_records_out(SWEEP_NOISE_CSV),
        help="Per-record CSV of sweep-noise.",
    )
    p.add_argument(
        "--n_boot", type=int, default=1000,
        help="Cluster bootstrap replicates (ecg_id).",
    )
    p.add_argument("--seed", type=int, default=0, help="Bootstrap seed.")
    p.add_argument("--out", default=SNR_FLIP_CSV, help="Output CSV.")
    p.set_defaults(func=run_snr_flip)

    p = sub.add_parser(
        "margins", parents=[common],
        help="Per-record minimum logit margin: local records vs fold 10.",
    )
    p.add_argument("--thresholds", default=THRESHOLDS_CSV, help=thresholds_help)
    p.add_argument(
        "--original_dir", default=ORIGINAL_DIR, help="Full 10 s originals, 500 Hz."
    )
    p.add_argument(
        "--condition", default="gen_clean",
        help="Condition whose masked originals are the local set.",
    )
    p.add_argument(
        "--max_records", type=int, default=0,
        help="Use the first N fold-10 reference records (0 = all).",
    )
    p.add_argument("--chunk", type=int, default=512, help="Records read per chunk.")
    p.add_argument(
        "--out", default=MARGINS_CSV,
        help="Output CSV (with --max_records the default becomes "
        "<name>_part0-<N>.csv). Non-v0 conditions may not write the *_v0 defaults.",
    )
    p.set_defaults(func=run_margins)
    return parser


# Run the code.
def run(args):
    with warnings.catch_warnings():
        warnings.filterwarnings(
            "ignore", message="invalid value encountered in cast", category=RuntimeWarning
        )
        return args.func(args)


if __name__ == "__main__":
    run(get_parser().parse_args(sys.argv[1:]))
