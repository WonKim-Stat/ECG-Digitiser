"""Unit tests for the mask metrics of src/run/evaluate.py.

Everything is synthetic and written to tmp_path; no data file and no model are
required.
"""
import argparse
import json
import os

import numpy as np
import pytest
import wfdb
from PIL import Image

from config import LEAD_LABEL_MAPPING
from src.run import evaluate


def _write_json(tmp_path, leads, height=20, width=30, full_mode_lead="II",
                name="rec-0_0000", **extra):
    """Write a minimal image generator JSON and return its path."""
    data = {
        "height": height,
        "width": width,
        "full_mode_lead": full_mode_lead,
        "rotate": 0,
        "crop": 0,
        "leads": leads,
    }
    data.update(extra)
    path = tmp_path / f"{name}.json"
    path.write_text(json.dumps(data))
    return str(path)


def _lead(name, pixels, start_sample=0, end_sample=1250):
    return {
        "lead_name": name,
        "start_sample": start_sample,
        "end_sample": end_sample,
        "plotted_pixels": [list(map(float, p)) for p in pixels],
    }


# ------------------------------------------------------------- build_gt_mask
def test_coordinates_are_row_col_and_truncated(tmp_path):
    path = _write_json(tmp_path, [_lead("I", [[10.9, 20.9]])])
    mask, info = evaluate.build_gt_mask(path, resample_factor=1)

    assert mask.shape == (20, 30)
    assert mask.dtype == np.uint8
    assert mask[10, 20] == LEAD_LABEL_MAPPING["I"]
    assert int(mask.sum()) == LEAD_LABEL_MAPPING["I"]
    assert info["n_points_dropped"] == 0
    assert info["rotate"] == 0 and info["crop"] == 0


def test_out_of_range_and_negative_points_are_dropped_and_counted(tmp_path):
    pixels = [
        [5.0, 5.0],      # kept
        [-1.0, 5.0],     # negative row
        [5.0, -1.5],     # negative col (would wrap in the training code)
        [6.0, -0.5],     # truncates to column 0, so it is kept
        [20.0, 5.0],     # row == height
        [5.0, 30.0],     # col == width
        [100.0, 100.0],  # far outside
    ]
    path = _write_json(tmp_path, [_lead("V1", pixels)])
    mask, info = evaluate.build_gt_mask(path, resample_factor=1)

    assert int(np.count_nonzero(mask)) == 2
    assert mask[5, 5] == LEAD_LABEL_MAPPING["V1"]
    assert mask[6, 0] == LEAD_LABEL_MAPPING["V1"]
    assert info["n_points"] == 7
    assert info["n_points_dropped"] == 5
    assert info["n_points_negative"] == 2
    # Nothing wrapped around to the far edge.
    assert mask[-1, 5] == 0 and mask[5, -1] == 0


def test_short_full_mode_lead_is_dropped_and_long_one_labelled(tmp_path):
    leads = [
        _lead("II", [[1.0, 1.0]], start_sample=0, end_sample=1250),   # short
        _lead("I", [[2.0, 2.0]]),
        _lead("II", [[3.0, 3.0]], start_sample=0, end_sample=5000),   # rhythm
    ]
    path = _write_json(tmp_path, leads)
    mask, _ = evaluate.build_gt_mask(path, resample_factor=1)

    assert mask[1, 1] == 0                            # short II removed
    assert mask[2, 2] == LEAD_LABEL_MAPPING["I"]
    assert mask[3, 3] == LEAD_LABEL_MAPPING["II"] == 2
    assert set(np.unique(mask)) == {0, LEAD_LABEL_MAPPING["I"], 2}


def test_collision_later_lead_wins(tmp_path):
    leads = [_lead("I", [[4.0, 4.0]]), _lead("V6", [[4.0, 4.0]])]
    path = _write_json(tmp_path, leads)
    mask, _ = evaluate.build_gt_mask(path, resample_factor=1)
    assert mask[4, 4] == LEAD_LABEL_MAPPING["V6"]

    # The opposite JSON order gives the opposite winner.
    path = _write_json(tmp_path, list(reversed(leads)), name="rec2-0_0000")
    mask, _ = evaluate.build_gt_mask(path, resample_factor=1)
    assert mask[4, 4] == LEAD_LABEL_MAPPING["I"]


def test_densification_fills_the_gap_between_two_points(tmp_path):
    path = _write_json(tmp_path, [_lead("I", [[5.0, 5.0], [5.0, 7.0]])])

    mask1, _ = evaluate.build_gt_mask(path, resample_factor=1)
    assert mask1[5, 6] == 0
    assert int(np.count_nonzero(mask1)) == 2

    mask3, _ = evaluate.build_gt_mask(path, resample_factor=3)
    assert mask3[5, 6] == LEAD_LABEL_MAPPING["I"]
    assert int(np.count_nonzero(mask3)) == 3


def test_densify_pixels_matches_the_reference_loop():
    pixels = np.array([[0.0, 0.0], [3.0, 6.0], [10.0, -2.0]])
    factor = 4
    expected = np.zeros(((len(pixels) - 1) * factor, 2))
    for j in range(len(pixels) - 1):
        for axis in (0, 1):
            expected[j * factor:(j + 1) * factor, axis] = np.linspace(
                pixels[j, axis], pixels[j + 1, axis], factor
            )
    np.testing.assert_allclose(evaluate.densify_pixels(pixels, factor), expected)
    np.testing.assert_allclose(evaluate.densify_pixels(pixels, 1), pixels)


def test_build_gt_mask_matches_the_training_convention(tmp_path):
    """build_gt_mask must agree with create_train_test.create_mask_from_json."""
    try:
        from src.ptb_xl.create_train_test import create_mask_from_json
    except Exception as e:  # pragma: no cover - depends on the environment
        pytest.skip(f"create_train_test not importable here: {e}")

    rng = np.random.default_rng(3)
    leads = []
    for name in ["I", "II", "V3", "aVR"]:
        pixels = np.column_stack(
            (rng.uniform(0, 19.99, 40), rng.uniform(0, 29.99, 40))
        )
        leads.append(_lead(name, pixels.round(2)))
    leads.append(_lead("II", [[1.5, 1.5]], start_sample=0, end_sample=10))
    path = _write_json(tmp_path, leads)

    mask_path = str(tmp_path / "train_mask.png")
    create_mask_from_json(path, mask_path, rgb=False, multilabel=True)
    assert os.path.exists(mask_path), "create_mask_from_json swallowed an error"
    reference = np.asarray(Image.open(mask_path))

    ours, _ = evaluate.build_gt_mask(path, resample_factor=1)
    np.testing.assert_array_equal(ours, reference)


# ------------------------------------------------------------- dice_per_lead
def _mask_with(shape, label, rows, cols):
    mask = np.zeros(shape, dtype=np.uint8)
    mask[rows, cols] = label
    return mask


def test_dice_identical_masks(tmp_path):
    gt = np.zeros((20, 20), dtype=np.uint8)
    gt[5, 2:10] = LEAD_LABEL_MAPPING["I"]
    gt[9, 2:10] = LEAD_LABEL_MAPPING["V4"]

    rows = {r["lead"]: r for r in evaluate.dice_per_lead(gt, gt.copy())}
    assert [r for r in rows] == list(LEAD_LABEL_MAPPING)
    for lead in ("I", "V4"):
        assert rows[lead]["dice_raw"] == pytest.approx(1.0)
        assert rows[lead]["dice_tol"] == pytest.approx(1.0)
        assert rows[lead]["gt_pixels"] == 8 == rows[lead]["pred_pixels"]
    for lead in ("II", "aVL"):
        assert np.isnan(rows[lead]["dice_raw"])
        assert np.isnan(rows[lead]["dice_tol"])
        assert np.isnan(rows[lead]["precision_tol"])
        assert np.isnan(rows[lead]["recall_tol"])
        assert rows[lead]["gt_pixels"] == 0 == rows[lead]["pred_pixels"]


def test_dice_disjoint_masks_far_apart():
    label = LEAD_LABEL_MAPPING["I"]
    gt = _mask_with((20, 20), label, 2, slice(2, 8))
    pred = _mask_with((20, 20), label, 15, slice(2, 8))
    row = evaluate.dice_per_lead(gt, pred, tolerance=1)[0]
    assert row["lead"] == "I"
    assert row["dice_raw"] == 0.0
    assert row["dice_tol"] == 0.0
    assert row["precision_tol"] == 0.0
    assert row["recall_tol"] == 0.0


def test_dice_one_side_empty():
    label = LEAD_LABEL_MAPPING["I"]
    gt = _mask_with((20, 20), label, 2, slice(2, 8))
    pred = np.zeros((20, 20), dtype=np.uint8)

    row = evaluate.dice_per_lead(gt, pred)[0]
    assert row["dice_raw"] == 0.0 and row["dice_tol"] == 0.0
    assert np.isnan(row["precision_tol"])  # no predicted pixels
    assert row["recall_tol"] == 0.0

    row = evaluate.dice_per_lead(pred, gt)[0]
    assert row["dice_raw"] == 0.0 and row["dice_tol"] == 0.0
    assert row["precision_tol"] == 0.0
    assert np.isnan(row["recall_tol"])  # no ground-truth pixels


def test_dice_one_pixel_shift_is_zero_raw_but_one_tolerant():
    label = LEAD_LABEL_MAPPING["V2"]
    gt = _mask_with((20, 20), label, 5, slice(2, 15))
    pred = _mask_with((20, 20), label, 6, slice(2, 15))
    rows = {r["lead"]: r for r in evaluate.dice_per_lead(gt, pred, tolerance=1)}
    row = rows["V2"]
    assert row["dice_raw"] == 0.0
    assert row["dice_tol"] == pytest.approx(1.0)
    assert row["precision_tol"] == pytest.approx(1.0)
    assert row["recall_tol"] == pytest.approx(1.0)

    # With zero tolerance the shift is not forgiven any more.
    row0 = evaluate.dice_per_lead(gt, pred, tolerance=0)[LEAD_LABEL_MAPPING["V2"] - 1]
    assert row0["dice_tol"] == 0.0


def test_dice_shape_mismatch_raises():
    with pytest.raises(ValueError) as excinfo:
        evaluate.dice_per_lead(np.zeros((4, 5), np.uint8), np.zeros((4, 6), np.uint8))
    assert "(4, 5)" in str(excinfo.value) and "(4, 6)" in str(excinfo.value)


# ------------------------------------------------------------------ CLI / run
def test_parser_mask_defaults():
    args = evaluate.get_parser().parse_args(["-g", "a", "-p", "b"])
    assert args.gt_json_folder is None
    assert args.pred_mask_folder is None
    assert args.resample_factor == 3
    assert args.mask_tolerance == 1
    assert args.mask_output_file is None

    args = evaluate.get_parser().parse_args(
        ["-g", "a", "-p", "b", "--gt_json_folder", "j", "--pred_mask_folder", "m",
         "--resample_factor", "5", "--mask_tolerance", "2",
         "--mask_output_file", "c.csv"]
    )
    assert (args.gt_json_folder, args.pred_mask_folder) == ("j", "m")
    assert args.resample_factor == 5 and args.mask_tolerance == 2
    assert args.mask_output_file == "c.csv"


def test_run_mask_metrics_is_skipped_without_both_folders(tmp_path):
    args = argparse.Namespace(gt_json_folder=str(tmp_path), pred_mask_folder=None)
    assert evaluate.run_mask_metrics(args) is None
    args = argparse.Namespace(gt_json_folder=None, pred_mask_folder=str(tmp_path))
    assert evaluate.run_mask_metrics(args) is None
    # An old-style Namespace without the new attributes must not break.
    assert evaluate.run_mask_metrics(argparse.Namespace()) is None


SIGNAL_COLUMNS = [
    "record", "prediction", "lead", "placement", "n_samples", "window_start",
    "window_end", "snr_raw", "pearson_r", "snr_aligned", "shift", "offset",
    "missing",
]


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


def test_end_to_end_signal_and_mask_metrics(tmp_path):
    gt_dir = tmp_path / "gt"
    pred_dir = tmp_path / "pred"
    json_dir = tmp_path / "json"
    mask_dir = tmp_path / "masks"
    for d in (gt_dir, pred_dir, json_dir, mask_dir):
        d.mkdir()

    n = 200
    t = np.arange(n) / 500.0
    wave = np.sin(2 * np.pi * 4.0 * t)
    gt = np.column_stack((wave, wave))
    _write_record(gt_dir, "rec001", gt, ["I", "II"])
    _write_record(pred_dir, "rec001-0_0000", gt.copy(), ["I", "II"])

    # Ground-truth JSON: two horizontal lines, one per lead.
    leads = [
        _lead("I", [[5.0, c] for c in range(2, 18)]),
        _lead("II", [[9.0, c] for c in range(2, 18)], start_sample=0, end_sample=5000),
    ]
    json_path = _write_json(json_dir, leads, name="rec001-0_0000")
    assert os.path.exists(json_path)

    # Prediction: lead I exact, lead II shifted by one pixel.
    pred_mask = np.zeros((20, 30), dtype=np.uint8)
    pred_mask[5, 2:18] = LEAD_LABEL_MAPPING["I"]
    pred_mask[10, 2:18] = LEAD_LABEL_MAPPING["II"]
    Image.fromarray(pred_mask).save(mask_dir / "rec001-0_0000_mask.png")
    (mask_dir / "rec001-0_0000_mask.json").write_text(
        json.dumps({"rot_angle": 0.0, "height": 20, "width": 30})
    )

    out_csv = tmp_path / "signals.csv"
    mask_csv = tmp_path / "masks.csv"
    args = argparse.Namespace(
        gt_folder=str(gt_dir),
        pred_folder=str(pred_dir),
        output_file=str(out_csv),
        placement="auto",
        max_shift=5,
        verbose=False,
        gt_json_folder=str(json_dir),
        pred_mask_folder=str(mask_dir),
        resample_factor=3,
        mask_tolerance=1,
        mask_output_file=str(mask_csv),
    )
    df = evaluate.run(args)

    # The signal evaluation is untouched.
    assert list(df.columns) == SIGNAL_COLUMNS
    assert len(df) == 2 and not df["missing"].any()
    import pandas as pd
    assert list(pd.read_csv(out_csv).columns) == SIGNAL_COLUMNS

    assert os.path.exists(mask_csv)
    mask_df = pd.read_csv(mask_csv)
    assert len(mask_df) == len(LEAD_LABEL_MAPPING)
    assert set(mask_df["record"]) == {"rec001-0_0000"}
    rows = mask_df.set_index("lead")
    assert rows.loc["I", "dice_raw"] == pytest.approx(1.0)
    assert rows.loc["I", "gt_pixels"] == 16
    assert rows.loc["II", "dice_raw"] == 0.0
    assert rows.loc["II", "dice_tol"] == pytest.approx(1.0)
    assert np.isnan(rows.loc["V1", "dice_raw"])
    assert rows.loc["I", "rot_angle"] == 0.0


def test_run_mask_metrics_warns_on_rotation(tmp_path, capsys):
    json_dir = tmp_path / "json"
    mask_dir = tmp_path / "masks"
    json_dir.mkdir()
    mask_dir.mkdir()

    _write_json(json_dir, [_lead("I", [[5.0, 5.0]])], name="rec-0_0000", rotate=5)
    pred_mask = np.zeros((20, 30), dtype=np.uint8)
    pred_mask[5, 5] = LEAD_LABEL_MAPPING["I"]
    Image.fromarray(pred_mask).save(mask_dir / "rec-0_0000_mask.png")
    (mask_dir / "rec-0_0000_mask.json").write_text(json.dumps({"rot_angle": -2.5}))

    args = argparse.Namespace(
        gt_json_folder=str(json_dir),
        pred_mask_folder=str(mask_dir),
        resample_factor=1,
        mask_tolerance=1,
        mask_output_file=None,
        verbose=False,
    )
    mask_df = evaluate.run_mask_metrics(args)
    out = capsys.readouterr().out
    assert "may not match" in out
    # The numbers are still reported.
    assert mask_df is not None
    assert float(mask_df.set_index("lead").loc["I", "dice_raw"]) == pytest.approx(1.0)


def test_run_mask_metrics_skips_a_shape_mismatch(tmp_path, capsys):
    json_dir = tmp_path / "json"
    mask_dir = tmp_path / "masks"
    json_dir.mkdir()
    mask_dir.mkdir()

    _write_json(json_dir, [_lead("I", [[5.0, 5.0]])], name="rec-0_0000")
    Image.fromarray(np.zeros((21, 30), dtype=np.uint8)).save(
        mask_dir / "rec-0_0000_mask.png"
    )
    args = argparse.Namespace(
        gt_json_folder=str(json_dir),
        pred_mask_folder=str(mask_dir),
        resample_factor=1,
        mask_tolerance=1,
        mask_output_file=None,
        verbose=False,
    )
    assert evaluate.run_mask_metrics(args) is None
    assert "Mask shape mismatch" in capsys.readouterr().out
