# Evaluate digitised WFDB records against ground-truth WFDB records.
import argparse
import os
import sys

import numpy as np
import pandas as pd
import wfdb

# We reuse the official Challenge SNR implementation. Its signature
# (x_ref, x_est) -> (snr, p_signal, p_noise) fits our per-lead comparison and it
# already handles NaNs the way we need (only samples finite in both signals are
# compared, with an alpha penalty for reference samples the estimate does not
# cover).
from src.utils.helper_code import compute_snr

PLACEMENTS = ("start", "window")


# Parse arguments.
def get_parser():
    description = "Evaluate digitised ECG records against ground-truth records."
    parser = argparse.ArgumentParser(description=description)
    parser.add_argument(
        "-g",
        "--gt_folder",
        type=str,
        required=True,
        help="Folder containing the ground-truth WFDB records.",
    )
    parser.add_argument(
        "-p",
        "--pred_folder",
        type=str,
        required=True,
        help="Folder containing the digitised (predicted) WFDB records.",
    )
    parser.add_argument(
        "-o",
        "--output_file",
        type=str,
        required=False,
        default=None,
        help="Optional CSV file to write the per-(record, lead) metrics to.",
    )
    parser.add_argument(
        "--placement",
        type=str,
        choices=["start", "window", "auto"],
        default="auto",
        help=(
            "How to align a predicted lead with the ground-truth lead window. "
            "'window' compares the same sample window, 'start' compares the "
            "beginning of the prediction with the ground-truth window, 'auto' "
            "keeps the better of the two per lead (by raw SNR)."
        ),
    )
    parser.add_argument(
        "--max_shift",
        type=int,
        default=50,
        help="Maximum absolute time shift (in samples) allowed for the aligned SNR.",
    )
    parser.add_argument(
        "-v",
        "--verbose",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Verbose output. Use --no-verbose to disable.",
    )
    return parser


# ----------------------------------------------------------------------------
# Metrics
# ----------------------------------------------------------------------------
def compute_pearson(x_ref, x_est):
    """Pearson correlation of the samples that are finite in both signals.

    Returns NaN for constant (zero-variance) or all-zero inputs instead of
    emitting a numpy warning.
    """
    x_ref = np.asarray(x_ref, dtype=float)
    x_est = np.asarray(x_est, dtype=float)
    n = min(x_ref.size, x_est.size)
    x_ref, x_est = x_ref[:n], x_est[:n]
    idx = np.isfinite(x_ref) & np.isfinite(x_est)
    if np.sum(idx) < 2:
        return float("nan")
    a, b = x_ref[idx], x_est[idx]
    a = a - a.mean()
    b = b - b.mean()
    denom = np.sqrt(np.sum(a**2) * np.sum(b**2))
    if denom <= 0:
        return float("nan")
    return float(np.dot(a, b) / denom)


def compute_raw_snr(x_ref, x_est):
    """Raw SNR in dB: 10*log10(sum(gt^2) / sum((gt - pred)^2))."""
    snr, _, _ = compute_snr(np.asarray(x_ref, dtype=float), np.asarray(x_est, dtype=float))
    return float(snr)


def _shift_overlap(x_ref, x_est, shift):
    """Return the overlapping parts of x_ref and x_est for a given time shift.

    A positive shift means the prediction is delayed, i.e. x_est[i + shift] is
    compared with x_ref[i].
    """
    n = min(x_ref.size, x_est.size)
    if shift >= 0:
        a = x_ref[: n - shift]
        b = x_est[shift:n]
    else:
        a = x_ref[-shift:n]
        b = x_est[: n + shift]
    return a, b


def compute_aligned_snr(x_ref, x_est, max_shift=50):
    """SNR after the best small time shift and a constant vertical offset.

    helper_code.align_signals() does not fit here: it estimates an
    *unconstrained* horizontal lag from a quantised 2D cross-correlation and a
    vertical offset from the same correlation peak, while we need the lag to be
    bounded by --max_shift and the offset to be the median of (pred - gt).  It
    also pads with NaNs, which then leaks into compute_snr's alpha penalty.  So
    we do the minimal bounded search here.

    Returns (snr, shift, offset).
    """
    x_ref = np.asarray(x_ref, dtype=float)
    x_est = np.asarray(x_est, dtype=float)
    n = min(x_ref.size, x_est.size)
    if n == 0:
        return float("nan"), 0, float("nan")
    max_shift = int(max(0, min(max_shift, n - 1)))

    best = (-np.inf, 0, float("nan"))
    for shift in range(-max_shift, max_shift + 1):
        a, b = _shift_overlap(x_ref, x_est, shift)
        idx = np.isfinite(a) & np.isfinite(b)
        if np.sum(idx) < 2:
            continue
        offset = float(np.median(b[idx] - a[idx]))
        snr = compute_raw_snr(a, b - offset)
        if np.isnan(snr):
            continue
        if snr > best[0]:
            best = (snr, shift, offset)

    if not np.isfinite(best[0]) and best[0] == -np.inf:
        return float("nan"), 0, float("nan")
    return float(best[0]), int(best[1]), float(best[2])


# ----------------------------------------------------------------------------
# Record handling
# ----------------------------------------------------------------------------
def list_records(folder):
    """List the WFDB record names (header basenames) in a folder."""
    if not os.path.isdir(folder):
        raise FileNotFoundError(f"Folder not found: {folder}")
    return sorted(f[: -len(".hea")] for f in os.listdir(folder) if f.endswith(".hea"))


def match_prediction(gt_record, pred_records):
    """Match a ground-truth record name to a prediction record name.

    The digitiser names its outputs '<gt_record>-<image index>_<suffix>', so we
    accept an identical name or a '<gt_record>-' prefix.
    """
    for name in pred_records:
        if name == gt_record:
            return name
    candidates = [n for n in pred_records if n.startswith(gt_record + "-")]
    return sorted(candidates)[0] if candidates else None


def valid_window(signal_1d):
    """Derive the valid sample window of a ground-truth lead from its NaN mask."""
    idx = np.flatnonzero(np.isfinite(signal_1d))
    if idx.size == 0:
        return None
    return int(idx[0]), int(idx[-1]) + 1


def evaluate_lead(gt_lead, pred_signal, window, placement="auto", max_shift=50):
    """Evaluate one lead and return a dict of metrics."""
    i0, i1 = window
    gt_seg = gt_lead[i0:i1]
    length = gt_seg.size

    modes = PLACEMENTS if placement == "auto" else (placement,)
    results = []
    for mode in modes:
        if mode == "window":
            pred_seg = pred_signal[i0:i1]
        else:  # start
            pred_seg = pred_signal[:length]
        if pred_seg.size < length:
            pred_seg = np.concatenate(
                (pred_seg, np.full(length - pred_seg.size, np.nan))
            )
        snr = compute_raw_snr(gt_seg, pred_seg)
        results.append((mode, pred_seg, snr))

    # Keep the placement with the better raw SNR (NaN SNRs lose).
    def _key(item):
        snr = item[2]
        return -np.inf if (snr is None or np.isnan(snr)) else snr

    mode, pred_seg, snr_raw = max(results, key=_key)
    snr_aligned, shift, offset = compute_aligned_snr(gt_seg, pred_seg, max_shift=max_shift)
    return {
        "placement": mode,
        "n_samples": int(length),
        "window_start": int(i0),
        "window_end": int(i1),
        "snr_raw": snr_raw,
        "pearson_r": compute_pearson(gt_seg, pred_seg),
        "snr_aligned": snr_aligned,
        "shift": shift,
        "offset": offset,
        "missing": False,
    }


def evaluate_record(gt_record_path, pred_record_path, placement="auto", max_shift=50):
    """Evaluate all leads of one record pair. Returns a list of row dicts."""
    gt = wfdb.rdrecord(gt_record_path)
    pred = wfdb.rdrecord(pred_record_path)
    gt_signal = np.asarray(gt.p_signal, dtype=float)
    pred_signal = np.asarray(pred.p_signal, dtype=float)
    pred_index = {name: i for i, name in enumerate(pred.sig_name)}

    rows = []
    for i, lead in enumerate(gt.sig_name):
        base = {
            "record": os.path.basename(gt_record_path),
            "prediction": os.path.basename(pred_record_path),
            "lead": lead,
        }
        window = valid_window(gt_signal[:, i])
        if lead not in pred_index or window is None:
            rows.append(
                {
                    **base,
                    "placement": "",
                    "n_samples": 0 if window is None else int(window[1] - window[0]),
                    "window_start": -1 if window is None else int(window[0]),
                    "window_end": -1 if window is None else int(window[1]),
                    "snr_raw": float("nan"),
                    "pearson_r": float("nan"),
                    "snr_aligned": float("nan"),
                    "shift": np.nan,
                    "offset": float("nan"),
                    "missing": True,
                }
            )
            continue
        metrics = evaluate_lead(
            gt_signal[:, i],
            pred_signal[:, pred_index[lead]],
            window,
            placement=placement,
            max_shift=max_shift,
        )
        rows.append({**base, **metrics})
    return rows


def summarise(df):
    """Build the printed summary as a list of text lines."""
    lines = []
    valid = df[~df["missing"]]
    lines.append("Per-record SNR (dB):")
    lines.append(f"  {'record':<28} {'leads':>5} {'median':>9} {'mean':>9} {'missing':>8}")
    for record, group in df.groupby("record", sort=True):
        ok = group[~group["missing"]]
        median = np.nanmedian(ok["snr_raw"]) if len(ok) else float("nan")
        mean = np.nanmean(ok["snr_raw"]) if len(ok) else float("nan")
        lines.append(
            f"  {record:<28} {len(ok):>5} {median:>9.3f} {mean:>9.3f} "
            f"{int(group['missing'].sum()):>8}"
        )

    lines.append("")
    lines.append("Overall:")
    lines.append(f"  records               : {df['record'].nunique()}")
    lines.append(f"  lead comparisons      : {len(valid)}")
    lines.append(f"  missing leads         : {int(df['missing'].sum())}")
    if len(valid):
        lines.append(f"  raw SNR     median/mean: "
                     f"{np.nanmedian(valid['snr_raw']):.3f} / {np.nanmean(valid['snr_raw']):.3f} dB")
        lines.append(f"  aligned SNR median/mean: "
                     f"{np.nanmedian(valid['snr_aligned']):.3f} / {np.nanmean(valid['snr_aligned']):.3f} dB")
        lines.append(f"  Pearson r   median/mean: "
                     f"{np.nanmedian(valid['pearson_r']):.4f} / {np.nanmean(valid['pearson_r']):.4f}")
        counts = valid["placement"].value_counts().to_dict()
        placement_str = ", ".join(f"{k}={v}" for k, v in sorted(counts.items()))
        lines.append(f"  placements chosen     : {placement_str}")
    return lines


# Run the code.
def run(args):
    gt_records = list_records(args.gt_folder)
    pred_records = list_records(args.pred_folder)
    if args.verbose:
        print(
            f"Found {len(gt_records)} ground-truth and {len(pred_records)} "
            f"predicted records."
        )

    rows = []
    for gt_record in gt_records:
        pred_record = match_prediction(gt_record, pred_records)
        if pred_record is None:
            print(f"No prediction found for record {gt_record}, skipping.")
            continue
        rows.extend(
            evaluate_record(
                os.path.join(args.gt_folder, gt_record),
                os.path.join(args.pred_folder, pred_record),
                placement=args.placement,
                max_shift=args.max_shift,
            )
        )

    if not rows:
        print("No records evaluated.")
        return None

    df = pd.DataFrame(rows)
    if args.output_file:
        output_dir = os.path.dirname(os.path.abspath(args.output_file))
        os.makedirs(output_dir, exist_ok=True)
        df.to_csv(args.output_file, index=False)
        if args.verbose:
            print(f"Wrote per-lead metrics to {args.output_file}")

    print("")
    print("\n".join(summarise(df)))
    return df


if __name__ == "__main__":
    run(get_parser().parse_args(sys.argv[1:]))
