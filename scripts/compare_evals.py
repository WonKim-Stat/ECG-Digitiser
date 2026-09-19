# Compare two per-lead evaluation CSVs written by `python -m src.run.evaluate`.
import argparse
import os
import sys

import numpy as np
import pandas as pd

KEYS = ["record", "lead"]


# Parse arguments.
def get_parser():
    description = "Paired comparison of two per-lead evaluation CSVs (A=baseline, B=candidate)."
    parser = argparse.ArgumentParser(description=description)
    parser.add_argument(
        "--a",
        type=str,
        required=True,
        help="Per-lead CSV of the baseline run (A).",
    )
    parser.add_argument(
        "--b",
        type=str,
        required=True,
        help="Per-lead CSV of the candidate run (B).",
    )
    parser.add_argument(
        "--metric",
        type=str,
        default="snr_raw",
        help="Column to compare, e.g. snr_raw, snr_aligned or pearson_r.",
    )
    parser.add_argument(
        "--n_boot",
        type=int,
        default=2000,
        help="Number of bootstrap replicates.",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=0,
        help="Seed of the bootstrap random generator.",
    )
    parser.add_argument(
        "--threshold",
        type=float,
        default=-1.0,
        help="Leads with delta <= this value are reported as worse (in dB).",
    )
    parser.add_argument(
        "--output_file",
        type=str,
        required=False,
        default=None,
        help="Optional CSV file to write the worse leads to.",
    )
    return parser


# ----------------------------------------------------------------------------
# Pairing
# ----------------------------------------------------------------------------
def _clean(df, metric):
    """Keep the key and metric columns of the rows that carry a usable value."""
    for column in KEYS + [metric]:
        if column not in df.columns:
            raise ValueError(f"Missing column '{column}' in the evaluation CSV.")
    out = df[KEYS + [metric]].copy()
    out[metric] = pd.to_numeric(out[metric], errors="coerce")
    if "missing" in df.columns:
        out = out[~df["missing"].fillna(False).astype(bool)]
    return out[np.isfinite(out[metric])]


def pair_leads(df_a, df_b, metric="snr_raw"):
    """Inner-join two evaluation frames on (record, lead).

    Rows whose metric is missing, NaN or infinite in either run are dropped.
    Returns a frame with record, lead, a, b and delta = b - a.
    """
    a = _clean(df_a, metric).rename(columns={metric: "a"})
    b = _clean(df_b, metric).rename(columns={metric: "b"})
    paired = a.merge(b, on=KEYS, how="inner")
    paired["delta"] = paired["b"] - paired["a"]
    return paired.reset_index(drop=True)


# ----------------------------------------------------------------------------
# Statistics
# ----------------------------------------------------------------------------
def cluster_bootstrap_median(paired, n_boot=2000, seed=0, level=0.95):
    """Bootstrap CI of the median delta, resampling records (not leads).

    The leads of one record share an image and a segmentation, so they are not
    independent: we resample whole records with replacement (a record drawn
    twice contributes its leads twice) and take the median of the pooled deltas
    of each replicate.
    """
    groups = [
        np.asarray(group["delta"].to_numpy(), dtype=float)
        for _, group in paired.groupby("record", sort=True)
    ]
    n_records = len(groups)
    n_leads = int(sum(g.size for g in groups))
    result = {
        "median": float("nan"),
        "lo": float("nan"),
        "hi": float("nan"),
        "n_records": n_records,
        "n_leads": n_leads,
    }
    if n_leads == 0:
        return result

    result["median"] = float(np.median(np.concatenate(groups)))
    if n_boot is None or n_boot <= 0:
        return result

    rng = np.random.default_rng(seed)
    draws = rng.integers(0, n_records, size=(int(n_boot), n_records))
    medians = np.empty(int(n_boot), dtype=float)
    for i in range(int(n_boot)):
        medians[i] = np.median(np.concatenate([groups[j] for j in draws[i]]))
    alpha = (1.0 - level) / 2.0
    result["lo"] = float(np.percentile(medians, 100.0 * alpha))
    result["hi"] = float(np.percentile(medians, 100.0 * (1.0 - alpha)))
    return result


def worse_leads(paired, threshold=-1.0):
    """Leads that lost at least `-threshold` dB, worst first."""
    worse = paired[paired["delta"] <= threshold]
    return worse.sort_values("delta", ascending=True).reset_index(drop=True)


def summarise_pair(paired):
    """Summary statistics of the two runs and of their per-lead difference."""
    delta = paired["delta"].to_numpy(dtype=float)
    n = delta.size
    return {
        "n_records": int(paired["record"].nunique()) if n else 0,
        "n_leads": int(n),
        "a_median": float(np.median(paired["a"])) if n else float("nan"),
        "a_mean": float(np.mean(paired["a"])) if n else float("nan"),
        "a_min": float(np.min(paired["a"])) if n else float("nan"),
        "b_median": float(np.median(paired["b"])) if n else float("nan"),
        "b_mean": float(np.mean(paired["b"])) if n else float("nan"),
        "b_min": float(np.min(paired["b"])) if n else float("nan"),
        "frac_improved": float(np.mean(delta > 0)) if n else float("nan"),
        "n_worse_1db": int(np.sum(delta <= -1.0)),
    }


# ----------------------------------------------------------------------------
# Reporting
# ----------------------------------------------------------------------------
def format_summary(summary, boot, metric="snr_raw"):
    """Build the printed summary as a list of text lines."""
    lines = [f"Paired comparison on '{metric}' (A = baseline, B = candidate):"]
    lines.append(f"  records               : {summary['n_records']}")
    lines.append(f"  lead comparisons      : {summary['n_leads']}")
    lines.append(
        f"  A median/mean/min     : {summary['a_median']:.3f} / "
        f"{summary['a_mean']:.3f} / {summary['a_min']:.3f}"
    )
    lines.append(
        f"  B median/mean/min     : {summary['b_median']:.3f} / "
        f"{summary['b_mean']:.3f} / {summary['b_min']:.3f}"
    )
    lines.append(f"  leads improved        : {summary['frac_improved']:.3f}")
    lines.append(f"  leads worse than -1 dB: {summary['n_worse_1db']}")
    lines.append("")
    lines.append("Cluster bootstrap of the median delta (records resampled):")
    lines.append(
        f"  median delta          : {boot['median']:.3f} "
        f"[{boot['lo']:.3f}, {boot['hi']:.3f}]"
    )
    return lines


def format_worse(worse, threshold=-1.0):
    """Build the printed worse-leads table as a list of text lines."""
    lines = [f"Leads with delta <= {threshold:g}: {len(worse)}"]
    if len(worse):
        lines.append(
            f"  {'record':<28} {'lead':<6} {'a':>9} {'b':>9} {'delta':>9}"
        )
        for row in worse.itertuples(index=False):
            lines.append(
                f"  {str(row.record):<28} {str(row.lead):<6} {row.a:>9.3f} "
                f"{row.b:>9.3f} {row.delta:>9.3f}"
            )
    return lines


# Run the code.
def run(args):
    df_a = pd.read_csv(args.a)
    df_b = pd.read_csv(args.b)
    paired = pair_leads(df_a, df_b, metric=args.metric)
    if not len(paired):
        print("No lead pairs to compare.")
        return paired

    summary = summarise_pair(paired)
    boot = cluster_bootstrap_median(paired, n_boot=args.n_boot, seed=args.seed)
    worse = worse_leads(paired, threshold=args.threshold)

    print("\n".join(format_summary(summary, boot, metric=args.metric)))
    print("")
    print("\n".join(format_worse(worse, threshold=args.threshold)))

    if args.output_file:
        output_dir = os.path.dirname(os.path.abspath(args.output_file))
        os.makedirs(output_dir, exist_ok=True)
        worse.to_csv(args.output_file, index=False)
        print(f"Wrote {len(worse)} worse lead(s) to {args.output_file}")
    return paired


if __name__ == "__main__":
    run(get_parser().parse_args(sys.argv[1:]))
