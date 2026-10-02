# Evaluate digitised WFDB records against ground-truth WFDB records.
import argparse
import glob
import json
import os
import sys

import cv2
import numpy as np
import pandas as pd
import wfdb
from PIL import Image

from config import LEAD_LABEL_MAPPING

# We reuse the official Challenge SNR implementation. Its signature
# (x_ref, x_est) -> (snr, p_signal, p_noise) fits our per-lead comparison and it
# already handles NaNs the way we need (only samples finite in both signals are
# compared, with an alpha penalty for reference samples the estimate does not
# cover).
from src.utils.helper_code import compute_snr

PLACEMENTS = ("start", "window")
MASK_SUFFIX = "_mask.png"


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
        "--gt_json_folder",
        type=str,
        required=False,
        default=None,
        help=(
            "Folder with the image generator JSONs (<image name>.json). Together "
            "with --pred_mask_folder this enables the per-lead mask metrics."
        ),
    )
    parser.add_argument(
        "--pred_mask_folder",
        type=str,
        required=False,
        default=None,
        help=(
            "Folder with the predicted label masks written by digitize.py "
            "--save_mask (<image name>_mask.png)."
        ),
    )
    parser.add_argument(
        "--resample_factor",
        type=int,
        default=3,
        help=(
            "Linear interpolation factor used to densify the plotted pixels of "
            "the ground-truth JSON. Values <= 1 disable the densification."
        ),
    )
    parser.add_argument(
        "--mask_tolerance",
        type=int,
        default=1,
        help="Chebyshev pixel tolerance for the tolerant mask metrics.",
    )
    parser.add_argument(
        "--mask_output_file",
        type=str,
        required=False,
        default=None,
        help="Optional CSV file to write the per-(record, lead) mask metrics to.",
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


# ----------------------------------------------------------------------------
# Segmentation masks
# ----------------------------------------------------------------------------
def densify_pixels(pixels, resample_factor):
    """Linearly interpolate between consecutive plotted pixels.

    This mirrors replot_pixels.resample_pixels_in_dir (np.linspace between each
    pair of consecutive points, resample_factor points per segment), but works
    in memory: the original function rewrites the JSON files on disk.
    """
    pixels = np.asarray(pixels, dtype=float).reshape(-1, 2)
    if resample_factor is None or resample_factor <= 1 or pixels.shape[0] < 2:
        return pixels
    # linspace over the (N-1) segments at once; axis=1 keeps the per-segment
    # blocks contiguous, i.e. the same order as the reference implementation.
    dense = np.linspace(pixels[:-1], pixels[1:], int(resample_factor), axis=1)
    return dense.reshape(-1, 2)


def _filter_full_mode_leads(leads, full_mode_lead):
    """Drop the short entries of the full mode lead (keep the longest one)."""
    lengths = [
        lead["end_sample"] - lead["start_sample"]
        for lead in leads
        if lead["lead_name"] == full_mode_lead
    ]
    if not lengths:
        return list(leads)
    full_lead_length = max(lengths)
    return [
        lead
        for lead in leads
        if lead["lead_name"] != full_mode_lead
        or lead["end_sample"] - lead["start_sample"] == full_lead_length
    ]


def build_gt_mask(json_path, resample_factor=3):
    """Rasterise the ground-truth label mask from an image generator JSON.

    Follows the training convention of create_train_test.create_mask_from_json:
    plotted pixels are [row, col] floats, truncated with astype(int), labelled
    with config.LEAD_LABEL_MAPPING, the short entries of the full mode lead are
    dropped and, on a collision, the later lead in JSON order wins.  Unlike the
    training code we drop negative coordinates instead of letting them wrap
    around, and we never touch the JSON on disk.

    Returns (mask uint8 [height, width], info dict).
    """
    with open(json_path) as f:
        data = json.load(f)

    height, width = int(data["height"]), int(data["width"])
    leads = _filter_full_mode_leads(data["leads"], data.get("full_mode_lead"))

    mask = np.zeros((height, width), dtype=np.uint8)
    n_points = 0
    n_dropped = 0
    n_negative = 0
    lead_names = []
    for lead in leads:
        name = lead["lead_name"]
        if name not in LEAD_LABEL_MAPPING:
            continue
        pixels = densify_pixels(lead["plotted_pixels"], resample_factor)
        if pixels.size == 0:
            continue
        coords = pixels.astype(int)  # truncation, as in the training code
        rows, cols = coords[:, 0], coords[:, 1]
        negative = (rows < 0) | (cols < 0)
        keep = ~negative & (rows < height) & (cols < width)
        n_points += coords.shape[0]
        n_negative += int(np.sum(negative))
        n_dropped += int(np.sum(~keep))
        if not np.any(keep):
            continue
        # Assigning lead by lead in JSON order makes the later lead win.
        mask[rows[keep], cols[keep]] = LEAD_LABEL_MAPPING[name]
        lead_names.append(name)

    info = {
        "height": height,
        "width": width,
        "n_points": int(n_points),
        "n_points_dropped": int(n_dropped),
        "n_points_negative": int(n_negative),
        "rotate": float(data.get("rotate", 0) or 0),
        "crop": float(data.get("crop", 0) or 0),
        "full_mode_lead": data.get("full_mode_lead"),
        "leads": lead_names,
        "resample_factor": int(resample_factor) if resample_factor else 1,
    }
    return mask, info


def _dilate(binary, tolerance):
    """Chebyshev dilation of a boolean mask by `tolerance` pixels."""
    if tolerance is None or tolerance <= 0:
        return binary
    kernel = np.ones((2 * int(tolerance) + 1, 2 * int(tolerance) + 1), np.uint8)
    return cv2.dilate(binary.astype(np.uint8), kernel) > 0


def dice_per_lead(gt_mask, pred_mask, tolerance=1):
    """Per-lead Dice of two label masks. Returns a list of row dicts.

    dice_raw is the plain 2|A n B| / (|A| + |B|).  The tolerant variants accept
    a match within `tolerance` pixels (Chebyshev): precision_tol is the fraction
    of predicted pixels close to a ground-truth pixel of the same lead,
    recall_tol the fraction of ground-truth pixels close to a predicted one, and
    dice_tol their harmonic mean.  A lead absent from both masks gives NaN, a
    lead present in only one gives 0 (with the undefined ratio left NaN).
    """
    gt_mask = np.asarray(gt_mask)
    pred_mask = np.asarray(pred_mask)
    if gt_mask.shape != pred_mask.shape:
        raise ValueError(
            f"Mask shape mismatch: ground truth {gt_mask.shape} vs "
            f"prediction {pred_mask.shape}."
        )

    rows = []
    for lead, label in LEAD_LABEL_MAPPING.items():
        a = gt_mask == label
        b = pred_mask == label
        n_a, n_b = int(np.sum(a)), int(np.sum(b))
        row = {
            "lead": lead,
            "gt_pixels": n_a,
            "pred_pixels": n_b,
            "dice_raw": float("nan"),
            "precision_tol": float("nan"),
            "recall_tol": float("nan"),
            "dice_tol": float("nan"),
        }
        if n_a == 0 and n_b == 0:
            rows.append(row)
            continue

        row["dice_raw"] = 2.0 * float(np.sum(a & b)) / (n_a + n_b)
        if n_b:
            row["precision_tol"] = float(np.sum(b & _dilate(a, tolerance))) / n_b
        if n_a:
            row["recall_tol"] = float(np.sum(a & _dilate(b, tolerance))) / n_a
        precision, recall = row["precision_tol"], row["recall_tol"]
        if n_a == 0 or n_b == 0:
            row["dice_tol"] = 0.0
        elif precision + recall > 0:
            row["dice_tol"] = 2.0 * precision * recall / (precision + recall)
        else:
            row["dice_tol"] = 0.0
        rows.append(row)
    return rows


def list_pred_masks(folder):
    """List the (record, mask path) pairs of a predicted-mask folder."""
    if not os.path.isdir(folder):
        raise FileNotFoundError(f"Folder not found: {folder}")
    paths = sorted(glob.glob(os.path.join(folder, "*" + MASK_SUFFIX)))
    return [(os.path.basename(p)[: -len(MASK_SUFFIX)], p) for p in paths]


def read_pred_mask(mask_path):
    """Read a predicted label mask plus its frame info (rot_angle if present)."""
    mask = np.asarray(Image.open(mask_path))
    if mask.ndim == 3:  # masks are sometimes stored as RGB copies of one channel
        mask = mask[..., 0]
    meta = {}
    meta_path = mask_path[: -len(".png")] + ".json"
    if os.path.exists(meta_path):
        with open(meta_path) as f:
            meta = json.load(f)
    return mask.astype(np.uint8), meta


def match_gt_json(record, gt_json_folder):
    """Find the generator JSON belonging to a predicted mask record."""
    names = sorted(
        os.path.basename(p)[: -len(".json")]
        for p in glob.glob(os.path.join(gt_json_folder, "*.json"))
    )
    # The mask record is the image name, so usually an exact hit; fall back to
    # the same '<record>-<image index>_<suffix>' prefix rule as the signals.
    name = match_prediction(record, names)
    return os.path.join(gt_json_folder, name + ".json") if name else None


def summarise_masks(df):
    """Build the printed mask summary as a list of text lines."""
    columns = ("dice_raw", "dice_tol", "recall_tol", "precision_tol")
    lines = ["Per-record mask metrics (median over the leads present):"]
    lines.append(
        f"  {'record':<28} {'leads':>5} {'dice_raw':>9} {'dice_tol':>9} "
        f"{'recall':>9} {'precis.':>9}"
    )
    for record, group in df.groupby("record", sort=True):
        present = group[(group["gt_pixels"] > 0) | (group["pred_pixels"] > 0)]
        medians = [np.nanmedian(present[c]) if len(present) else float("nan")
                   for c in columns]
        lines.append(
            f"  {record:<28} {len(present):>5} " + " ".join(f"{m:>9.4f}" for m in medians)
        )

    present = df[(df["gt_pixels"] > 0) | (df["pred_pixels"] > 0)]
    lines.append("")
    lines.append("Overall masks:")
    lines.append(f"  records               : {df['record'].nunique()}")
    lines.append(f"  lead comparisons      : {len(present)}")
    if len(present):
        for column in columns:
            lines.append(f"  {column:<22}: median {np.nanmedian(present[column]):.4f}")
    return lines


def run_mask_metrics(args):
    """Compare the predicted label masks with masks rasterised from the JSONs.

    Returns the per-(record, lead) dataframe, or None if the mask evaluation was
    not requested or nothing could be compared.
    """
    gt_json_folder = getattr(args, "gt_json_folder", None)
    pred_mask_folder = getattr(args, "pred_mask_folder", None)
    if not gt_json_folder or not pred_mask_folder:
        return None

    resample_factor = getattr(args, "resample_factor", 3)
    tolerance = getattr(args, "mask_tolerance", 1)
    verbose = getattr(args, "verbose", True)

    pred_masks = list_pred_masks(pred_mask_folder)
    if verbose:
        print("")
        print(f"Found {len(pred_masks)} predicted mask(s) in {pred_mask_folder}.")

    rows = []
    for record, mask_path in pred_masks:
        json_path = match_gt_json(record, gt_json_folder)
        if json_path is None:
            print(f"No ground-truth JSON found for mask {record}, skipping.")
            continue
        gt_mask, info = build_gt_mask(json_path, resample_factor=resample_factor)
        pred_mask, meta = read_pred_mask(mask_path)
        rot_angle = float(meta.get("rot_angle", float("nan")))

        rotated = np.isfinite(rot_angle) and abs(rot_angle) > 0
        if rotated or info["rotate"] != 0 or info["crop"] != 0:
            print(
                f"Warning: {record}: the coordinate frames may not match "
                f"(prediction rot_angle={rot_angle}, generator rotate="
                f"{info['rotate']}, crop={info['crop']}); reporting the numbers "
                f"anyway."
            )
        if info["n_points_dropped"] and verbose:
            print(
                f"  {record}: dropped {info['n_points_dropped']} of "
                f"{info['n_points']} plotted pixels outside the image."
            )

        try:
            lead_rows = dice_per_lead(gt_mask, pred_mask, tolerance=tolerance)
        except ValueError as e:
            print(f"Skipping mask {record}: {e}")
            continue
        for row in lead_rows:
            rows.append({"record": record, **row, "rot_angle": rot_angle})

    if not rows:
        print("No masks evaluated.")
        return None

    mask_df = pd.DataFrame(rows)
    mask_output_file = getattr(args, "mask_output_file", None)
    if mask_output_file:
        output_dir = os.path.dirname(os.path.abspath(mask_output_file))
        os.makedirs(output_dir, exist_ok=True)
        mask_df.to_csv(mask_output_file, index=False)
        if verbose:
            print(f"Wrote per-lead mask metrics to {mask_output_file}")

    print("")
    print("\n".join(summarise_masks(mask_df)))
    return mask_df


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
        run_mask_metrics(args)
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

    # The mask metrics are optional and reported separately; the signal
    # evaluation above (and its CSV) is unaffected by them.
    run_mask_metrics(args)
    return df


if __name__ == "__main__":
    run(get_parser().parse_args(sys.argv[1:]))
