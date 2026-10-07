"""Unit tests for the rescale WARNING line of src/run/digitize.py.

run() rescales an output with a sample beyond 10 mV on either side to -1..1 over all
its leads before it writes it, and has always said so with a 'Signal out of range'
line. rescale_warning() is the WARNING line that run() prints right after that one:
the written signals of such a page are not in millivolts. It changes neither the
rescale, nor the signals, nor qc.csv, which is written before the rescale. No model:
the pages of the run() tests are synthesised with numpy and read with saved masks.
"""
import csv
import os
import re
import warnings

import numpy as np
import pytest
import torch
import wfdb
from torchvision.io.image import write_png

from config import (
    ADC_GAIN,
    LEAD_LABEL_MAPPING,
    SHORT_SIGNAL_LENGTH_SEC,
    Y_SHIFT_RATIO,
)
from src.run import digitize

RECORD = "00022_hr-0_0000"
# How a log parser would read the record of the line, as the one of the evaluation
# scripts reads it off the Einthoven line: it ends at whitespace or at one of
# : , ; . ( ).
RESCALE_WARNING_PATTERN = re.compile(
    r"WARNING: output rescaled for record ([^\s:,;.()]+)"
)
# The first floats beyond the limit on either side.
JUST_ABOVE = float(np.nextafter(10.0, 11.0))
JUST_BELOW = float(np.nextafter(-10.0, -11.0))


def old_line(record):
    """The line run() has always printed for an output out of range."""
    return (
        f"Signal out of range for record {record}, "
        "normalizing to range between 1 and -1"
    )


OLD_LINE = old_line(RECORD)


# -------------------------------------------------------------- rescale_warning
def test_the_line_of_one_page():
    assert digitize.rescale_warning(RECORD, -0.53, 11.0042) == (
        "WARNING: output rescaled for record 00022_hr-0_0000 "
        "(range -0.530 to 11.004 mV, outside +-10 mV): the written signals are "
        "rescaled to -1..1 and are not in millivolts, check this page."
    )


@pytest.mark.parametrize(
    "lowest, highest, named",
    [
        (-0.53, 11.0042, "range -0.530 to 11.004 mV"),
        (-12.25, 0.875, "range -12.250 to 0.875 mV"),
        (-31.5, 2047.123456, "range -31.500 to 2047.123 mV"),
        (0.0, np.inf, "range 0.000 to inf mV"),
        # The range is named to three decimals: a sample less than 0.0005 mV beyond
        # the limit reads as the limit itself.
        (-0.699, JUST_ABOVE, "range -0.699 to 10.000 mV"),
        (JUST_BELOW, 0.802, "range -10.000 to 0.802 mV"),
        (-0.699, 10.0004, "range -0.699 to 10.000 mV"),
    ],
)
def test_the_line_names_the_record_and_the_range(lowest, highest, named):
    # The range comes as numpy floats, which is what run() has.
    line = digitize.rescale_warning(RECORD, np.float64(lowest), np.float64(highest))
    assert line == (
        f"WARNING: output rescaled for record {RECORD} ({named}, outside +-10 mV): "
        "the written signals are rescaled to -1..1 and are not in millivolts, "
        "check this page."
    )
    assert "\n" not in line
    assert line.isascii()


@pytest.mark.parametrize(
    "record", ["00022_hr-0_0000", "00013_lr-0", "05251_hr-0_0000", "page"]
)
def test_a_log_parser_reads_the_record_exactly(record):
    line = digitize.rescale_warning(record, -12.0, 0.5)
    assert line.startswith(f"WARNING: output rescaled for record {record} ")
    match = RESCALE_WARNING_PATTERN.search(line)
    assert match.group(1) == record
    assert line.startswith(match.group(0))


# ------------------------------------------------------------------- end to end
HEIGHT = 1700
WIDTH = 2200
P_PAGE = HEIGHT * digitize.PAGE_PITCH_RATIO
LINE0 = 118.62
MV_PER_PIXEL = 6.25 / P_PAGE
COLUMNS = {
    "I": 0, "III": 0, "aVR": 1, "aVL": 1, "aVF": 1,
    "V1": 2, "V2": 2, "V3": 2, "V4": 3, "V5": 3, "V6": 3,
}
# The outputs out of range, as (excursion of the drawn page, samples set in the
# assembled signals). Drawn: one lead of the last column leaves its row for half a
# second, V6, the lowest short lead of the page, by 12 mV upwards, or V4, the highest
# one, by 12 mV downwards. Set: one sample of the page in range is put at the first
# float beyond the limit. All four stay off the first column, where the Einthoven
# check reads, and off every other sample the QC reads.
OUT_OF_RANGE = {
    "high": (("V6", 12.0), None),
    "low": (("V4", -12.0), None),
    "just above 10 mV": (None, {(2600, "II"): JUST_ABOVE}),
    "just below -10 mV": (None, {(3900, "V5"): JUST_BELOW}),
}
CONSISTENCY_COLUMNS = [
    "einthoven_rms", "einthoven_rms_demedian", "einthoven_ratio", "einthoven_n",
    "goldberger_rms", "goldberger_rms_demedian", "goldberger_ratio", "goldberger_n",
]


def limb_a(t):
    return 0.3 * np.sin(2 * np.pi * 1.1 * t) + 0.2 * np.sin(2 * np.pi * 3.3 * t + 0.5)


def limb_b(t):
    return 0.25 * np.cos(2 * np.pi * 0.8 * t) + 0.15 * np.sin(2 * np.pi * 5.0 * t)


def label_page(excursion=None, strip=True):
    """A 3x4 label mask with rhythm strip II = a + b, I = a and III = b.

    excursion = (lead, mV) adds so many mV to that lead over the half second in the
    middle of its column. strip=False leaves the rhythm strip out, and lead II with it.
    """
    label = np.zeros((HEIGHT, WIDTH), dtype=np.uint8)
    for k, lead in enumerate(list(COLUMNS) + ["II"]):
        is_long = lead == "II"
        if is_long and not strip:
            continue
        column = 0 if is_long else COLUMNS[lead]
        n_columns = digitize.NUM_COLUMNS if is_long else 1
        x = np.arange(WIDTH) + 0.5
        start = LINE0 + column * P_PAGE
        columns = np.flatnonzero((x >= start) & (x < start + n_columns * P_PAGE))
        t = (x[columns] - LINE0) / P_PAGE * SHORT_SIGNAL_LENGTH_SEC
        if is_long:
            value = limb_a(t) + limb_b(t)
        elif lead == "I":
            value = limb_a(t)
        elif lead == "III":
            value = limb_b(t)
        else:
            value = 0.2 * np.sin(2 * np.pi * (0.9 + 0.1 * k) * t)
        if excursion is not None and lead == excursion[0]:
            t_column = t - column * SHORT_SIGNAL_LENGTH_SEC
            value = value + excursion[1] * ((t_column >= 1.0) & (t_column < 1.5))
        ratio = Y_SHIFT_RATIO["full" if is_long else lead]
        rows = np.floor(
            digitize.baseline_row(ratio, HEIGHT) - value / MV_PER_PIXEL
        ).astype(int)
        assert rows.min() >= 0 and rows.max() < HEIGHT
        label[rows, columns] = LEAD_LABEL_MAPPING[lead]
    return label


def watch(monkeypatch, put_in=None):
    """What run() assembles, in mV, and what it hands to write_record().

    put_in = {(sample, lead): mV} sets these samples of the assembled signals, the
    way to an output that is at 10 mV exactly or at the first float beyond it. The
    last page is under "assembled" and "written", every page under "pages" by its
    record, in the order run() wrote them.
    """
    seen = {"pages": {}}
    assemble, write = digitize.assemble_signals, digitize.write_record

    def assemble_and_keep(*args, **kwargs):
        signals, sig_names = assemble(*args, **kwargs)
        for (sample, lead), value in (put_in or {}).items():
            signals[sample, sig_names.index(lead)] = value
        seen["assembled"], seen["sig_names"] = signals.copy(), list(sig_names)
        return signals, sig_names

    def write_and_keep(record, signals, sig_names, output_folder, placement):
        seen["written"] = signals.copy()
        seen["pages"][record] = {
            "assembled": seen["assembled"], "written": seen["written"],
        }
        return write(record, signals, sig_names, output_folder, placement)

    monkeypatch.setattr(digitize, "assemble_signals", assemble_and_keep)
    monkeypatch.setattr(digitize, "write_record", write_and_keep)
    return seen


def run_folder(tmp_path, name, pages, verbose=False):
    """run() on a folder of pages {record: (excursion, strip)}; its output folder."""
    data = tmp_path / f"{name}_data"
    masks = tmp_path / f"{name}_masks"
    out = tmp_path / f"{name}_out"
    for folder in (data, masks):
        folder.mkdir(exist_ok=True)
    image = torch.full((3, HEIGHT, WIDTH), 255, dtype=torch.uint8)
    for record, (excursion, strip) in pages.items():
        write_png(image, str(data / f"{record}.png"))
        label = label_page(excursion, strip)
        write_png(torch.from_numpy(label)[None], str(masks / f"{record}_mask.png"))
    argv = [
        "-d", str(data), "-o", str(out), "--mask_folder", str(masks),
        "--rotation", "hough", "--perspective", "off", "--resolution", "keep",
    ]
    argv.append("--verbose" if verbose else "--no-verbose")
    digitize.run(digitize.get_parser().parse_args(argv))
    return out


def run_page(tmp_path, name, excursion=None, verbose=False, strip=True):
    out = run_folder(tmp_path, name, {RECORD: (excursion, strip)}, verbose)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        record = wfdb.rdrecord(str(out / RECORD))
    files = {}
    for suffix in (".dat", ".hea"):
        with open(out / f"{RECORD}{suffix}", "rb") as f:
            files[suffix] = f.read()
    with open(out / "qc.csv", "rb") as f:
        files["qc.csv"] = f.read()
    with open(out / "qc.csv", newline="") as f:
        rows = list(csv.DictReader(f))
    assert len(rows) == 1
    return record.p_signal, rows[0], files


def unpatched_rescale(signals):
    """What run() has always written for an output out of range."""
    max_val = np.nanmax(signals)
    min_val = np.nanmin(signals)
    return (signals - min_val) / (max_val - min_val) * 2 - 1


def assert_consistency_columns(row, seen):
    """The lead consistency columns of the QC row are those of the assembled signals,
    in mV, before the rescale."""
    qc = digitize.compute_consistency_qc(seen["assembled"], seen["sig_names"])
    for name in CONSISTENCY_COLUMNS:
        assert float(row[name]) == pytest.approx(qc[name], nan_ok=True), name
    return qc


@pytest.fixture(scope="module")
def qc_in_range(tmp_path_factory):
    """qc.csv of the page as drawn, in range, read by the code as it is."""
    _, _, files = run_page(tmp_path_factory.mktemp("reference"), "in_range")
    return files["qc.csv"]


@pytest.mark.parametrize(
    "put_in",
    [None, {(2600, "II"): 10.0, (3900, "V5"): -10.0}],
    ids=["as drawn", "at 10 mV exactly"],
)
def test_run_prints_neither_line_for_an_output_in_range(
    tmp_path, capsys, monkeypatch, put_in
):
    seen = watch(monkeypatch, put_in)
    on_disk, _, _ = run_page(tmp_path, "in_range")
    out = capsys.readouterr().out
    assert f"QC {RECORD}: Einthoven RMS" in out
    assert "Signal out of range" not in out
    assert "WARNING: output rescaled" not in out
    assert "WARNING: Einthoven check failed" not in out
    # The signals are written as they were assembled, in mV.
    assert np.array_equal(seen["written"], seen["assembled"], equal_nan=True)
    highest, lowest = np.nanmax(seen["assembled"]), np.nanmin(seen["assembled"])
    if put_in is None:
        assert -2.0 < lowest < highest < 2.0
    else:
        assert (lowest, highest) == (-10.0, 10.0)
    assert np.allclose(on_disk, seen["assembled"], atol=0.5 / ADC_GAIN, equal_nan=True)


@pytest.mark.parametrize("case", list(OUT_OF_RANGE))
def test_run_prints_both_lines_for_an_output_out_of_range(
    tmp_path, capsys, monkeypatch, qc_in_range, case
):
    excursion, put_in = OUT_OF_RANGE[case]
    seen = watch(monkeypatch, put_in)
    on_disk, row, files = run_page(tmp_path, "page", excursion)
    lines = capsys.readouterr().out.splitlines()
    highest, lowest = np.nanmax(seen["assembled"]), np.nanmin(seen["assembled"])
    # The limit is 10 mV on either side, and each side alone is enough: the first
    # float beyond it is out of range, as the samples at 10 mV exactly are not.
    if case == "high":
        assert highest > 12.0 and lowest > -2.0
    elif case == "low":
        assert lowest < -12.0 and highest < 2.0
    elif case == "just above 10 mV":
        assert highest == JUST_ABOVE and 10.0 < highest < 10.0005 and lowest > -2.0
    else:
        assert lowest == JUST_BELOW and -10.0005 < lowest < -10.0 and highest < 2.0

    # The line that was always there, and the WARNING right after it, once each.
    warning_line = digitize.rescale_warning(RECORD, lowest, highest)
    assert warning_line == (
        f"WARNING: output rescaled for record {RECORD} "
        f"(range {lowest:.3f} to {highest:.3f} mV, outside +-10 mV): the written "
        "signals are rescaled to -1..1 and are not in millivolts, check this page."
    )
    assert [line for line in lines if "out of range" in line] == [OLD_LINE]
    assert [line for line in lines if "output rescaled" in line] == [warning_line]
    assert lines.index(warning_line) == lines.index(OLD_LINE) + 1
    assert RESCALE_WARNING_PATTERN.search(warning_line).group(1) == RECORD

    # What is written is what the code without the line writes: the same rescale of
    # the same signals, which no longer are millivolts.
    expected = unpatched_rescale(seen["assembled"])
    assert np.array_equal(seen["written"], expected, equal_nan=True)
    assert not np.array_equal(seen["written"], seen["assembled"], equal_nan=True)
    assert np.nanmax(on_disk) == 1.0 and np.nanmin(on_disk) == -1.0
    assert np.array_equal(np.isnan(on_disk), np.isnan(seen["assembled"]))
    assert np.allclose(on_disk, expected, atol=0.5 / ADC_GAIN, equal_nan=True)

    # The sample out of range is off the first column, so the limb leads agree and the
    # Einthoven check, which reads the signals before the rescale, says nothing:
    # the page is reported by the new line alone.
    assert float(row["einthoven_ratio"]) < digitize.EINTHOVEN_RATIO_WARNING
    assert not any("WARNING: Einthoven check failed" in line for line in lines)
    assert_consistency_columns(row, seen)
    # It is off every other sample the QC reads as well, so qc.csv is the one of the
    # page in range byte for byte: the rescale leaves no trace in any of its columns.
    assert files["qc.csv"] == qc_in_range


@pytest.mark.parametrize("form", ["ratio", "no ratio"])
def test_a_page_that_fails_the_einthoven_check_as_well_prints_both_warnings(
    tmp_path, capsys, monkeypatch, form
):
    # ratio: III leaves its row in the first column, where the Einthoven check reads.
    # no ratio: the page has no rhythm strip, so no lead II to check and no column
    # grid (the --time_mapping bbox fallback), and V6 leaves its row.
    seen = watch(monkeypatch)
    if form == "ratio":
        on_disk, row, _ = run_page(tmp_path, "both", ("III", 12.0))
    else:
        on_disk, row, _ = run_page(tmp_path, "both", ("V6", 12.0), strip=False)
    lines = capsys.readouterr().out.splitlines()
    highest, lowest = np.nanmax(seen["assembled"]), np.nanmin(seen["assembled"])
    assert highest > 12.0 and lowest > -2.0

    qc = assert_consistency_columns(row, seen)
    if form == "ratio":
        assert qc["einthoven_ratio"] >= digitize.EINTHOVEN_RATIO_WARNING
        reason = f"(ratio {qc['einthoven_ratio']:.3f} >= 0.1797)"
    else:
        assert np.isnan(qc["einthoven_ratio"]) and qc["einthoven_n"] == 0
        reason = "(no ratio: no sample with leads I, II and III all read)"
        assert sum("falling back to --time_mapping bbox" in line for line in lines) == 1
    einthoven_line = digitize.einthoven_warning(qc, RECORD)
    assert einthoven_line.startswith(
        f"WARNING: Einthoven check failed for record {RECORD} {reason}"
    )

    # Each of the three lines once: the Einthoven WARNING where it always was, after
    # the QC line, and the rescale WARNING right after the line that was always there.
    warning_line = digitize.rescale_warning(RECORD, lowest, highest)
    told = [
        line
        for line in lines
        if re.search("WARNING: Einthoven|out of range|output rescaled", line)
    ]
    assert told == [einthoven_line, OLD_LINE, warning_line]
    assert lines.index(einthoven_line) == [
        k for k, line in enumerate(lines) if line.startswith(f"QC {RECORD}: ")
    ][0] + 1
    assert lines.index(warning_line) == lines.index(OLD_LINE) + 1

    expected = unpatched_rescale(seen["assembled"])
    assert np.array_equal(seen["written"], expected, equal_nan=True)
    assert np.nanmax(on_disk) == 1.0 and np.nanmin(on_disk) == -1.0


@pytest.mark.parametrize("verbose", [False, True], ids=["quiet", "verbose"])
def test_each_rescaled_page_of_a_folder_gets_its_own_line(
    tmp_path, capsys, monkeypatch, verbose
):
    excursions = {
        "p_a-0_0000": None,
        "p_b-0_0000": ("V6", 12.0),
        "p_c-0_0000": ("V4", -12.0),
        "p_d-0_0000": None,
    }
    seen = watch(monkeypatch)
    # run() takes the pages in the order the file system lists them. They are listed
    # by name here, so that on every file system a page in range comes before the
    # rescaled ones and another one after them.
    listdir = os.listdir
    monkeypatch.setattr(digitize.os, "listdir", lambda path: sorted(listdir(path)))
    pages = {record: (excursion, True) for record, excursion in excursions.items()}
    run_folder(tmp_path, "folder", pages, verbose)
    text = capsys.readouterr().out
    assert text.endswith("\n")
    lines = text.splitlines()
    assert list(seen["pages"]) == sorted(excursions)

    # One line for each page out of range, with its own record and its own range,
    # right after its own 'Signal out of range' line, and none for a page in range,
    # whether it comes before or after a rescaled one.
    rescaled = ["p_b-0_0000", "p_c-0_0000"]
    warning_lines = []
    for record in rescaled:
        page = seen["pages"][record]
        lowest, highest = np.nanmin(page["assembled"]), np.nanmax(page["assembled"])
        assert lowest < -10 or highest > 10
        warning_lines.append(digitize.rescale_warning(record, lowest, highest))
        expected = unpatched_rescale(page["assembled"])
        assert np.array_equal(page["written"], expected, equal_nan=True)
    assert len(set(warning_lines)) == 2
    old_lines = [old_line(record) for record in rescaled]
    assert [line for line in lines if "out of range" in line] == old_lines
    assert [line for line in lines if "output rescaled" in line] == warning_lines
    for old, new in zip(old_lines, warning_lines):
        assert lines.index(new) == lines.index(old) + 1
    for record in ("p_a-0_0000", "p_d-0_0000"):
        page = seen["pages"][record]
        assert np.array_equal(page["written"], page["assembled"], equal_nan=True)
        assert not any(f"rescaled for record {record} " in line for line in lines)
    if verbose:
        assert lines[-1] == "Done."


def test_the_line_changes_neither_the_signals_nor_the_qc_row(
    tmp_path, capsys, monkeypatch, qc_in_range
):
    excursion = OUT_OF_RANGE["high"][0]
    _, row, files = run_page(tmp_path, "with", excursion)
    lines = capsys.readouterr().out.splitlines()
    warning_lines = [line for line in lines if "WARNING: output rescaled" in line]
    assert len(warning_lines) == 1
    assert not any("Storing signals for record" in line for line in lines)

    # The same page without the line: signals, header and qc.csv byte for byte.
    with monkeypatch.context() as patch:
        patch.setattr(digitize, "rescale_warning", lambda record, lowest, highest: "")
        _, row_without, files_without = run_page(tmp_path, "without", excursion)
    out = capsys.readouterr().out
    assert OLD_LINE in out
    assert "WARNING: output rescaled" not in out
    assert files_without == files
    assert row_without == row

    # qc.csv is the one of the page in range, header and row: no column for the line.
    assert files["qc.csv"] == qc_in_range
    assert not any("rescale" in name for name in row)

    # --verbose prints both lines once as well, after it says what it stores and
    # before it says it is done.
    _, _, files_verbose = run_page(tmp_path, "verbose", excursion, verbose=True)
    lines = capsys.readouterr().out.splitlines()
    stored = [k for k, line in enumerate(lines) if "Storing signals for record" in line]
    old = [k for k, line in enumerate(lines) if line == OLD_LINE]
    new = [k for k, line in enumerate(lines) if line == warning_lines[0]]
    assert len(stored) == len(old) == len(new) == 1
    assert new[0] == old[0] + 1 == stored[0] + 2
    assert lines[new[0] + 1 :] == ["Done."]
    assert files_verbose == files
