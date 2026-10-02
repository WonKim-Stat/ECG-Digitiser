"""Unit tests for scripts/compare_evals.py. No model and no data files required."""
import numpy as np
import pandas as pd
import pytest

from scripts import compare_evals


def _frame(rows, metric="snr_raw"):
    """Build a minimal per-lead evaluation frame from (record, lead, value, missing)."""
    return pd.DataFrame(
        [
            {
                "record": record,
                "prediction": record + "-0",
                "lead": lead,
                "placement": "window",
                "n_samples": 1000,
                "window_start": 0,
                "window_end": 1000,
                metric: value,
                "missing": missing,
            }
            for record, lead, value, missing in rows
        ]
    )


def _paired(rows):
    """Build an already paired frame from (record, lead, delta) triples."""
    df = pd.DataFrame(rows, columns=["record", "lead", "delta"])
    df["a"] = 0.0
    df["b"] = df["delta"]
    return df[["record", "lead", "a", "b", "delta"]]


# --------------------------------------------------------------------- pairing
def test_pair_leads_joins_on_record_and_lead_and_signs_delta():
    a = _frame([("r1", "I", 10.0, False), ("r1", "II", 5.0, False)])
    b = _frame([("r1", "II", 7.0, False), ("r1", "I", 8.0, False)])
    paired = compare_evals.pair_leads(a, b)
    assert list(paired.columns) == ["record", "lead", "a", "b", "delta"]
    deltas = dict(zip(paired["lead"], paired["delta"]))
    assert deltas["I"] == pytest.approx(-2.0)
    assert deltas["II"] == pytest.approx(2.0)


def test_pair_leads_drops_missing_nan_inf_and_unmatched_rows():
    a = _frame(
        [
            ("r1", "I", 10.0, False),
            ("r1", "II", float("nan"), True),  # missing in A
            ("r1", "III", 1.0, False),
            ("r1", "V1", 2.0, False),
            ("r2", "I", 3.0, False),  # no record r2 in B
        ]
    )
    b = _frame(
        [
            ("r1", "I", 12.0, False),
            ("r1", "II", 4.0, False),
            ("r1", "III", float("nan"), False),  # NaN metric in B
            ("r1", "V1", float("inf"), False),  # non-finite metric in B
        ]
    )
    paired = compare_evals.pair_leads(a, b)
    assert list(paired["lead"]) == ["I"]
    assert paired["delta"].iloc[0] == pytest.approx(2.0)


def test_pair_leads_honours_the_metric_argument():
    a = _frame([("r1", "I", 1.0, False)], metric="snr_aligned")
    b = _frame([("r1", "I", 4.0, False)], metric="snr_aligned")
    paired = compare_evals.pair_leads(a, b, metric="snr_aligned")
    assert paired["delta"].iloc[0] == pytest.approx(3.0)


# ------------------------------------------------------------------- bootstrap
def test_bootstrap_is_deterministic_and_its_ci_brackets_the_median():
    rng = np.random.default_rng(1)
    paired = _paired(
        [
            (f"r{i}", lead, float(rng.normal(0.5, 1.0)))
            for i in range(8)
            for lead in ("I", "II", "III")
        ]
    )
    first = compare_evals.cluster_bootstrap_median(paired, n_boot=200, seed=0)
    second = compare_evals.cluster_bootstrap_median(paired, n_boot=200, seed=0)
    assert first == second
    other = compare_evals.cluster_bootstrap_median(paired, n_boot=200, seed=1)
    assert other["median"] == pytest.approx(first["median"])
    assert first["n_records"] == 8
    assert first["n_leads"] == 24
    assert first["lo"] <= first["median"] <= first["hi"]


def test_bootstrap_of_a_constant_delta_collapses_onto_it():
    paired = _paired([(f"r{i}", lead, 3.0) for i in range(5) for lead in ("I", "II")])
    result = compare_evals.cluster_bootstrap_median(paired, n_boot=100, seed=0)
    assert result["median"] == pytest.approx(3.0)
    assert result["lo"] == pytest.approx(3.0)
    assert result["hi"] == pytest.approx(3.0)


def test_bootstrap_with_one_record_collapses_onto_that_record_median():
    paired = _paired([("r1", "I", 1.0), ("r1", "II", 2.0), ("r1", "III", 6.0)])
    result = compare_evals.cluster_bootstrap_median(paired, n_boot=50, seed=0)
    assert result["n_records"] == 1
    assert result["median"] == pytest.approx(2.0)
    assert result["lo"] == pytest.approx(2.0)
    assert result["hi"] == pytest.approx(2.0)


def test_bootstrap_of_an_empty_frame_returns_nans():
    paired = _paired([]).astype({"record": object, "lead": object})
    result = compare_evals.cluster_bootstrap_median(paired, n_boot=10, seed=0)
    assert result["n_records"] == 0
    assert result["n_leads"] == 0
    assert np.isnan(result["median"])


# ----------------------------------------------------------------- worse leads
def test_worse_leads_threshold_is_inclusive_and_sorted_ascending():
    paired = _paired(
        [
            ("r1", "I", -0.5),
            ("r1", "II", -1.0),
            ("r2", "I", -3.0),
            ("r2", "II", 2.0),
            ("r2", "III", -2.0),
        ]
    )
    worse = compare_evals.worse_leads(paired, threshold=-1.0)
    assert list(worse["delta"]) == [-3.0, -2.0, -1.0]
    assert list(worse["record"]) == ["r2", "r2", "r1"]
    assert len(compare_evals.worse_leads(paired, threshold=-2.5)) == 1


# --------------------------------------------------------------------- summary
def test_summarise_pair_reports_both_runs_and_the_improvement_fraction():
    paired = pd.DataFrame(
        {
            "record": ["r1", "r1", "r2", "r2"],
            "lead": ["I", "II", "I", "II"],
            "a": [10.0, 2.0, 6.0, 0.0],
            "b": [12.0, 1.0, 6.5, -4.0],
        }
    )
    paired["delta"] = paired["b"] - paired["a"]
    summary = compare_evals.summarise_pair(paired)
    assert summary["n_records"] == 2
    assert summary["n_leads"] == 4
    assert summary["a_median"] == pytest.approx(4.0)
    assert summary["a_min"] == pytest.approx(0.0)
    assert summary["b_min"] == pytest.approx(-4.0)
    assert summary["frac_improved"] == pytest.approx(0.5)
    assert summary["n_worse_1db"] == 2


# ------------------------------------------------------------------------- CLI
def test_cli_runs_end_to_end_and_writes_the_worse_leads(tmp_path, capsys):
    a_path = tmp_path / "a.csv"
    b_path = tmp_path / "b.csv"
    out_path = tmp_path / "worse.csv"
    _frame(
        [("r1", "I", 10.0, False), ("r1", "II", 5.0, False), ("r2", "I", 8.0, False)]
    ).to_csv(a_path, index=False)
    _frame(
        [("r1", "I", 11.0, False), ("r1", "II", 1.0, False), ("r2", "I", 9.0, False)]
    ).to_csv(b_path, index=False)

    args = compare_evals.get_parser().parse_args(
        [
            "--a",
            str(a_path),
            "--b",
            str(b_path),
            "--metric",
            "snr_raw",
            "--n_boot",
            "20",
            "--seed",
            "0",
            "--threshold",
            "-1.0",
            "--output_file",
            str(out_path),
        ]
    )
    paired = compare_evals.run(args)
    assert len(paired) == 3

    out = capsys.readouterr().out
    assert "Paired comparison" in out
    assert "median delta" in out

    worse = pd.read_csv(out_path)
    assert list(worse["lead"]) == ["II"]
    assert worse["delta"].iloc[0] == pytest.approx(-4.0)
