"""Unit tests for the Einthoven WARNING line of src/run/digitize.py.

einthoven_warning() turns the Einthoven ratio of compute_consistency_qc() (the RMS
of I + III - II over the mean RMS of the three leads) into one WARNING line when it
reaches EINTHOVEN_RATIO_WARNING or is NaN, and run() prints that line after the QC
line, with or without --verbose, without changing the signals or qc.csv. No model:
the signals and the one page of the run() tests are synthesised with numpy. One test
reads the log parser of the evaluation scripts
(data/gen_eval/drivers/diag/real_qc_c3.py, not part of the repository) when it is
there and is skipped otherwise.
"""
import copy
import csv
import importlib.util
import inspect
import os
import re

import numpy as np
import pytest
import torch
from torchvision.io.image import write_png

from config import (
    FREQUENCY,
    LEAD_LABEL_MAPPING,
    LONG_SIGNAL_LENGTH_SEC,
    SHORT_SIGNAL_LENGTH_SEC,
    Y_SHIFT_RATIO,
)
from src.run import digitize

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
NUM_SAMPLES = int(LONG_SIGNAL_LENGTH_SEC * FREQUENCY)
SHORT_SAMPLES = int(SHORT_SIGNAL_LENGTH_SEC * FREQUENCY)
RECORD = "00022_hr-0_0000"
QC_KEYS = [
    "einthoven_rms",
    "einthoven_rms_demedian",
    "einthoven_ratio",
    "einthoven_n",
    "goldberger_rms",
    "goldberger_rms_demedian",
    "goldberger_ratio",
    "goldberger_n",
]
# The pattern the log parser of the evaluation scripts reads these lines with
# (RE_C4["w_einthoven"] of data/gen_eval/drivers/diag/real_qc_c3.py, which is not part
# of the repository): the record ends at whitespace or at one of : , ; . ( ). Compared
# with the parser's own below when it is there.
EINTHOVEN_WARNING_PATTERN = re.compile(
    r"WARNING: Einthoven check failed for record ([^\s:,;.()]+)"
)
LOG_PARSER = os.path.join(
    REPO_ROOT, "data", "gen_eval", "drivers", "diag", "real_qc_c3.py"
)


def qc_of(ratio, n=SHORT_SAMPLES):
    return {"einthoven_ratio": ratio, "einthoven_n": n}


# ------------------------------------------------------------ einthoven_warning
def test_the_threshold_is_the_one_fitted_on_the_development_scans():
    assert digitize.EINTHOVEN_RATIO_WARNING == 0.1797
    threshold = inspect.signature(digitize.einthoven_warning).parameters["threshold"]
    assert threshold.default == digitize.EINTHOVEN_RATIO_WARNING


@pytest.mark.parametrize("ratio", [0.0, 0.05, 0.1359, 0.1796, 0.17969999])
def test_a_ratio_below_the_threshold_gives_no_line(ratio):
    assert digitize.einthoven_warning(qc_of(ratio), RECORD) is None


@pytest.mark.parametrize("ratio", [0.1797, 0.17976, 0.25, 1.39, np.inf])
def test_a_ratio_at_or_above_the_threshold_gives_one_line(ratio):
    line = digitize.einthoven_warning(qc_of(ratio), RECORD)
    assert line is not None
    assert "\n" not in line
    assert line == (
        f"WARNING: Einthoven check failed for record {RECORD} "
        f"(ratio {ratio:.3f} >= 0.1797): leads I, II and III disagree, "
        "check this page."
    )


def test_the_line_of_one_page():
    assert digitize.einthoven_warning(qc_of(0.25), RECORD) == (
        "WARNING: Einthoven check failed for record 00022_hr-0_0000 "
        "(ratio 0.250 >= 0.1797): leads I, II and III disagree, check this page."
    )


def test_a_page_without_a_ratio_is_reported():
    assert digitize.einthoven_warning(qc_of(np.nan, n=0), RECORD) == (
        "WARNING: Einthoven check failed for record 00022_hr-0_0000 "
        "(no ratio: no sample with leads I, II and III all read), check this page."
    )
    assert digitize.einthoven_warning(qc_of(np.nan), RECORD) == (
        "WARNING: Einthoven check failed for record 00022_hr-0_0000 "
        "(no ratio: leads I, II and III are zero on their common samples), "
        "check this page."
    )


def test_another_threshold_is_named_in_the_line():
    assert digitize.einthoven_warning(qc_of(0.15), RECORD, threshold=0.2) is None
    line = digitize.einthoven_warning(qc_of(0.15), RECORD, threshold=0.1359)
    assert "(ratio 0.150 >= 0.1359)" in line


@pytest.mark.parametrize(
    "record", ["00022_hr-0_0000", "00013_lr-0", "05251_hr-0_0000", "page"]
)
@pytest.mark.parametrize("ratio", [0.5, np.nan])
def test_the_log_parser_reads_the_record_exactly(record, ratio):
    line = digitize.einthoven_warning(qc_of(ratio, n=0), record)
    match = EINTHOVEN_WARNING_PATTERN.search(line)
    assert match is not None
    assert match.group(1) == record
    assert line.startswith(match.group(0))


def test_the_copied_pattern_is_the_one_of_the_log_parser():
    if not os.path.exists(LOG_PARSER):
        pytest.skip("the log parser of the evaluation scripts is not here")
    spec = importlib.util.spec_from_file_location("real_qc_c3", LOG_PARSER)
    parser = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(parser)
    pattern = parser.RE_C4["w_einthoven"]
    assert pattern.pattern == EINTHOVEN_WARNING_PATTERN.pattern
    log = "\n".join(
        [
            f"QC {RECORD}: Einthoven RMS 0.1000 mV",
            digitize.einthoven_warning(qc_of(0.3), RECORD),
            "QC 00013_lr-0: Einthoven RMS 0.0100 mV",
            digitize.einthoven_warning(qc_of(np.nan, n=0), "00013_lr-0"),
        ]
    )
    assert [m.group(1) for m in pattern.finditer(log)] == [RECORD, "00013_lr-0"]
    # Every line that starts the WARNING for a record is one the pattern parsed (the
    # parser keys a page by its record without the image suffix -0_0000).
    keep = {parser.base_record(RECORD), parser.base_record("00013_lr-0")}
    assert keep == {"00022_hr", "00013_lr-0"}
    counts = parser.c4_line_counts([log], keep)
    assert counts["w_einthoven"] == (2, 2)


# ---------------------------------------------------- compute_consistency_qc
def limb_a(t):
    return 0.3 * np.sin(2 * np.pi * 1.1 * t) + 0.2 * np.sin(2 * np.pi * 3.3 * t + 0.5)


def limb_b(t):
    return 0.25 * np.cos(2 * np.pi * 0.8 * t) + 0.15 * np.sin(2 * np.pi * 5.0 * t)


COLUMNS = {
    "I": 0, "III": 0, "aVR": 1, "aVL": 1, "aVF": 1,
    "V1": 2, "V2": 2, "V3": 2, "V4": 3, "V5": 3, "V6": 3,
}


def twelve_leads(scale_i=1.0):
    """A 3x4 page with rhythm strip II = a + b, I = scale_i a and III = b."""
    signals, offsets = {}, {}
    for k, (lead, column) in enumerate(COLUMNS.items()):
        t = column * SHORT_SIGNAL_LENGTH_SEC + np.arange(SHORT_SAMPLES) / FREQUENCY
        if lead == "I":
            signals[lead] = scale_i * limb_a(t)
        elif lead == "III":
            signals[lead] = limb_b(t)
        else:
            signals[lead] = 0.2 * np.sin(2 * np.pi * (0.9 + 0.1 * k) * t)
        offsets[lead] = {"raw": column * 2.5, "snapped": column * 2.5}
    t = np.arange(NUM_SAMPLES) / FREQUENCY
    signals["II"] = limb_a(t) + limb_b(t)
    offsets["II"] = {"raw": 0.0, "snapped": 0.0}
    return digitize.assemble_signals(signals, offsets, NUM_SAMPLES, "column")


def test_consistent_limb_leads_give_no_line():
    qc = digitize.compute_consistency_qc(*twelve_leads())
    assert qc["einthoven_n"] == SHORT_SAMPLES
    assert qc["einthoven_ratio"] == pytest.approx(0.0, abs=1e-9)
    assert digitize.einthoven_warning(qc, RECORD) is None


def test_lead_i_twice_as_large_gives_a_line():
    qc = digitize.compute_consistency_qc(*twelve_leads(scale_i=2.0))
    assert qc["einthoven_ratio"] > 0.5
    line = digitize.einthoven_warning(qc, RECORD)
    assert line.startswith(f"WARNING: Einthoven check failed for record {RECORD} (ratio ")
    assert f"(ratio {qc['einthoven_ratio']:.3f} >= 0.1797)" in line


def test_a_missing_lead_and_flat_leads_give_a_line_without_a_ratio():
    signals, sig_names = twelve_leads()
    keep = [i for i, name in enumerate(sig_names) if name != "III"]
    qc = digitize.compute_consistency_qc(signals[:, keep], [sig_names[i] for i in keep])
    assert "(no ratio: no sample with leads I, II and III all read)" in (
        digitize.einthoven_warning(qc, RECORD)
    )
    flat = signals.copy()
    for lead in ["I", "II", "III"]:
        flat[:SHORT_SAMPLES, sig_names.index(lead)] = 0.0
    qc = digitize.compute_consistency_qc(flat, sig_names)
    assert qc["einthoven_n"] == SHORT_SAMPLES
    assert "(no ratio: leads I, II and III are zero on their common samples)" in (
        digitize.einthoven_warning(qc, RECORD)
    )


@pytest.mark.parametrize("scale_i", [1.0, 2.0])
def test_the_line_leaves_the_qc_dict_as_it_is(scale_i):
    signals, sig_names = twelve_leads(scale_i)
    qc = digitize.compute_consistency_qc(signals, sig_names)
    before = copy.deepcopy(qc)
    digitize.einthoven_warning(qc, RECORD)
    assert list(qc) == QC_KEYS
    assert qc == before
    assert digitize.compute_consistency_qc(signals, sig_names) == before


# ------------------------------------------------------------------- end to end
HEIGHT = 1700
WIDTH = 2200
P_PAGE = HEIGHT * digitize.PAGE_PITCH_RATIO
LINE0 = 118.62


def label_page(scale_i):
    """A 3x4 label mask with rhythm strip whose limb leads are twelve_leads()'."""
    label = np.zeros((HEIGHT, WIDTH), dtype=np.uint8)
    for k, lead in enumerate(list(COLUMNS) + ["II"]):
        is_long = lead == "II"
        column = 0 if is_long else COLUMNS[lead]
        n_columns = digitize.NUM_COLUMNS if is_long else 1
        x = np.arange(WIDTH) + 0.5
        start = LINE0 + column * P_PAGE
        columns = np.flatnonzero((x >= start) & (x < start + n_columns * P_PAGE))
        t = (x[columns] - LINE0) / P_PAGE * SHORT_SIGNAL_LENGTH_SEC
        if is_long:
            value = limb_a(t) + limb_b(t)
        elif lead == "I":
            value = scale_i * limb_a(t)
        elif lead == "III":
            value = limb_b(t)
        else:
            value = 0.2 * np.sin(2 * np.pi * (0.9 + 0.1 * k) * t)
        ratio = Y_SHIFT_RATIO["full" if is_long else lead]
        rows = np.floor(
            digitize.baseline_row(ratio, HEIGHT) - value / (6.25 / P_PAGE)
        ).astype(int)
        label[rows, columns] = LEAD_LABEL_MAPPING[lead]
    return label


def run_page(tmp_path, name, scale_i, verbose=False):
    data = tmp_path / f"{name}_data"
    masks = tmp_path / f"{name}_masks"
    out = tmp_path / f"{name}_out"
    for folder in (data, masks):
        folder.mkdir(exist_ok=True)
    image = torch.full((3, HEIGHT, WIDTH), 255, dtype=torch.uint8)
    write_png(image, str(data / f"{RECORD}.png"))
    label = label_page(scale_i)
    write_png(torch.from_numpy(label)[None], str(masks / f"{RECORD}_mask.png"))
    argv = [
        "-d", str(data), "-o", str(out), "--mask_folder", str(masks),
        "--rotation", "hough", "--perspective", "off", "--resolution", "keep",
    ]
    argv.append("--verbose" if verbose else "--no-verbose")
    digitize.run(digitize.get_parser().parse_args(argv))
    with open(out / "qc.csv", newline="") as f:
        row = next(csv.DictReader(f))
    files = {}
    for suffix in (".dat", ".hea"):
        with open(out / f"{RECORD}{suffix}", "rb") as f:
            files[suffix] = f.read()
    with open(out / "qc.csv", "rb") as f:
        files["qc.csv"] = f.read()
    return row, files


def test_run_prints_no_line_for_consistent_limb_leads(tmp_path, capsys):
    row, _ = run_page(tmp_path, "consistent", 1.0)
    out = capsys.readouterr().out
    assert f"QC {RECORD}: Einthoven RMS" in out
    assert float(row["einthoven_ratio"]) < digitize.EINTHOVEN_RATIO_WARNING
    assert "WARNING: Einthoven check failed" not in out


def test_run_prints_the_line_without_verbose_and_changes_nothing_else(
    tmp_path, capsys, monkeypatch
):
    row, files = run_page(tmp_path, "twice", 2.0)
    out = capsys.readouterr().out
    assert "Storing signals for record" not in out
    ratio = float(row["einthoven_ratio"])
    assert ratio >= digitize.EINTHOVEN_RATIO_WARNING
    lines = [line for line in out.splitlines() if "Einthoven check failed" in line]
    assert lines == [
        f"WARNING: Einthoven check failed for record {RECORD} "
        f"(ratio {ratio:.3f} >= 0.1797): leads I, II and III disagree, check this page."
    ]
    # The line comes right after the QC line of its record.
    out_lines = out.splitlines()
    assert out_lines[out_lines.index(lines[0]) - 1].startswith(f"QC {RECORD}: ")
    assert EINTHOVEN_WARNING_PATTERN.search(lines[0]).group(1) == RECORD
    assert list(row)[: 3 + len(QC_KEYS)] == ["record", "placement", "sharpen"] + QC_KEYS

    # The same page without the line: signals, header and qc.csv byte for byte.
    monkeypatch.setattr(digitize, "einthoven_warning", lambda qc, record: None)
    _, files_without = run_page(tmp_path, "twice_without", 2.0)
    assert "Einthoven check failed" not in capsys.readouterr().out
    assert files_without == files

    # --verbose prints it once as well, right after the QC line.
    monkeypatch.undo()
    _, files_verbose = run_page(tmp_path, "twice_verbose", 2.0, verbose=True)
    out = capsys.readouterr().out
    assert f"Storing signals for record {RECORD}" in out
    assert out.count("WARNING: Einthoven check failed") == 1
    out_lines = out.splitlines()
    assert out_lines[out_lines.index(lines[0]) - 1].startswith(f"QC {RECORD}: ")
    assert files_verbose == files
