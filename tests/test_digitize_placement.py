"""Unit tests for lead placement, QC and record writing in src/run/digitize.py.

No model, no data files: everything is synthesised with numpy.
"""
import warnings

import numpy as np
import pytest
import wfdb

from config import FREQUENCY, LONG_SIGNAL_LENGTH_SEC, SHORT_SIGNAL_LENGTH_SEC
from src.run import digitize


NUM_SAMPLES = int(LONG_SIGNAL_LENGTH_SEC * FREQUENCY)
SHORT_SAMPLES = int(SHORT_SIGNAL_LENGTH_SEC * FREQUENCY)
SEC_PER_PIXEL = 2.5 / 500.0  # 500 px per column


def _layout_positions(jitter=None):
    """Positions of a normal 3x4 layout plus a rhythm strip, x1 in pixels."""
    jitter = jitter or {}
    columns = {
        "I": 0, "II": 0, "III": 0,
        "aVR": 1, "aVL": 1, "aVF": 1,
        "V1": 2, "V2": 2, "V3": 2,
        "V4": 3, "V5": 3, "V6": 3,
    }
    positions = {}
    for lead, column in columns.items():
        positions[lead] = {"y1": 10, "x1": 100 + 500 * column + jitter.get(lead, 0)}
    return positions


def _layout_lengths(with_rhythm=True):
    lengths = {lead: SHORT_SAMPLES for lead in _layout_positions()}
    if with_rhythm:
        lengths["II"] = NUM_SAMPLES
    return lengths


# -------------------------------------------------------- compute_lead_offsets
def test_offsets_snap_to_column_grid_with_jitter():
    jitter = {"I": 3, "aVR": -4, "V1": 2, "V4": -3, "V6": 5}
    offsets = digitize.compute_lead_offsets(
        _layout_positions(jitter), _layout_lengths(), SEC_PER_PIXEL, "rec"
    )
    expected = {
        "I": 0.0, "III": 0.0,
        "aVR": 2.5, "aVL": 2.5, "aVF": 2.5,
        "V1": 5.0, "V2": 5.0, "V3": 5.0,
        "V4": 7.5, "V5": 7.5, "V6": 7.5,
    }
    for lead, value in expected.items():
        assert offsets[lead]["snapped"] == pytest.approx(value)
    # The rhythm lead itself is the reference.
    assert offsets["II"]["raw"] == 0.0
    assert offsets["II"]["snapped"] == 0.0


def test_offsets_use_rhythm_x1_as_reference():
    positions = _layout_positions()
    # Shift everything, including the rhythm strip, by a constant.
    for lead in positions:
        positions[lead]["x1"] += 777
    offsets = digitize.compute_lead_offsets(
        positions, _layout_lengths(), SEC_PER_PIXEL, "rec"
    )
    assert offsets["V1"]["raw"] == pytest.approx(5.0)
    assert offsets["V4"]["snapped"] == pytest.approx(7.5)


def test_offsets_pick_the_smallest_x1_among_several_long_leads():
    positions = _layout_positions()
    positions["II"]["x1"] = 100
    positions["V1"]["x1"] = 600
    lengths = _layout_lengths()
    lengths["V1"] = NUM_SAMPLES  # a second long lead, further right
    offsets = digitize.compute_lead_offsets(
        positions, lengths, SEC_PER_PIXEL, "rec"
    )
    assert offsets["aVR"]["raw"] == pytest.approx(2.5)


def test_offsets_fall_back_to_the_standard_table_without_rhythm_lead(capsys):
    offsets = digitize.compute_lead_offsets(
        _layout_positions(), _layout_lengths(with_rhythm=False), SEC_PER_PIXEL, "rec"
    )
    for lead, value in digitize.STANDARD_LEAD_OFFSETS_SEC.items():
        assert offsets[lead]["snapped"] == pytest.approx(value)
        assert np.isnan(offsets[lead]["raw"])
    out = capsys.readouterr().out
    assert "No rhythm lead" in out
    assert "rec" in out


def test_offsets_warn_when_far_from_a_column_boundary(capsys):
    # 0.9 s away from a multiple of 2.5 s => 180 px at this scale.
    offsets = digitize.compute_lead_offsets(
        _layout_positions({"V1": 180}), _layout_lengths(), SEC_PER_PIXEL, "rec"
    )
    assert offsets["V1"]["raw"] == pytest.approx(5.9)
    assert offsets["V1"]["snapped"] == pytest.approx(5.0)
    out = capsys.readouterr().out
    assert "V1" in out
    assert "rec" in out


def test_offsets_clamp_beyond_the_last_column_and_warn(capsys):
    # Two extra columns to the right => raw 12.5 s, clamped to 7.5 s.
    offsets = digitize.compute_lead_offsets(
        _layout_positions({"V6": 1000}), _layout_lengths(), SEC_PER_PIXEL, "rec"
    )
    assert offsets["V6"]["raw"] == pytest.approx(12.5)
    assert offsets["V6"]["snapped"] == pytest.approx(7.5)
    assert "V6" in capsys.readouterr().out


def test_offsets_clamp_negative_positions_to_zero():
    offsets = digitize.compute_lead_offsets(
        _layout_positions({"I": -600}), _layout_lengths(), SEC_PER_PIXEL, "rec"
    )
    assert offsets["I"]["snapped"] == 0.0


# ----------------------------------------------------------- assemble_signals
def _signal_dict():
    rng = np.random.default_rng(0)
    signals = {
        lead: rng.normal(size=SHORT_SAMPLES) for lead in _layout_positions()
    }
    signals["II"] = rng.normal(size=NUM_SAMPLES)
    return signals


def test_assemble_start_matches_the_legacy_loop():
    signals = _signal_dict()
    offsets = digitize.compute_lead_offsets(
        _layout_positions(), _layout_lengths(), SEC_PER_PIXEL, "rec"
    )

    # Inline re-implementation of the legacy assembling loop.
    signal_list = []
    for signal in signals.values():
        if len(signal) < NUM_SAMPLES:
            nan_signal = np.empty(NUM_SAMPLES)
            nan_signal[:] = np.nan
            nan_signal[: int(len(signal))] = signal
            signal_list.append(nan_signal)
        else:
            signal_list.append(signal)
    expected = np.array(signal_list).T

    assembled, sig_names = digitize.assemble_signals(
        signals, offsets, NUM_SAMPLES, "start"
    )
    assert sig_names == list(signals.keys())
    assert np.array_equal(assembled, expected, equal_nan=True)


def test_assemble_column_places_leads_in_their_window():
    signals = _signal_dict()
    offsets = digitize.compute_lead_offsets(
        _layout_positions(), _layout_lengths(), SEC_PER_PIXEL, "rec"
    )
    assembled, sig_names = digitize.assemble_signals(
        signals, offsets, NUM_SAMPLES, "column"
    )
    assert assembled.shape == (NUM_SAMPLES, len(signals))

    v1 = assembled[:, sig_names.index("V1")]
    assert np.allclose(v1[2500:3750], signals["V1"])
    assert np.all(np.isnan(v1[:2500]))
    assert np.all(np.isnan(v1[3750:]))

    # The rhythm lead covers everything and is untouched.
    ii = assembled[:, sig_names.index("II")]
    assert np.allclose(ii, signals["II"])

    i = assembled[:, sig_names.index("I")]
    assert np.allclose(i[:1250], signals["I"])
    assert np.all(np.isnan(i[1250:]))


def test_assemble_column_truncates_a_signal_that_does_not_fit():
    signals = {"V6": np.arange(2000.0)}
    offsets = {"V6": {"raw": 7.5, "snapped": 7.5}}
    assembled, _ = digitize.assemble_signals(signals, offsets, NUM_SAMPLES, "column")
    assert np.allclose(assembled[3750:, 0], np.arange(1250.0))
    assert np.all(np.isnan(assembled[:3750, 0]))


# --------------------------------------------------------------- write_record
def _small_record_signals():
    signals = np.full((NUM_SAMPLES, 2), np.nan)
    signals[:, 0] = np.linspace(-1.0, 1.0, NUM_SAMPLES)
    signals[2500:3750, 1] = np.linspace(-0.5, 0.5, 1250)
    return signals, ["II", "V1"]


def test_write_record_column_round_trips_nan(tmp_path):
    signals, sig_names = _small_record_signals()
    digitize.write_record("rec_col", signals, sig_names, str(tmp_path), "column")
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        read_back = wfdb.rdrecord(str(tmp_path / "rec_col")).p_signal

    assert np.array_equal(np.isnan(read_back), np.isnan(signals))
    finite = np.isfinite(signals)
    assert np.max(np.abs(read_back[finite] - signals[finite])) < 1e-3


def test_write_record_column_does_not_warn(tmp_path, recwarn):
    signals, sig_names = _small_record_signals()
    digitize.write_record("rec_quiet", signals, sig_names, str(tmp_path), "column")
    messages = [str(w.message) for w in recwarn.list if w.category is RuntimeWarning]
    assert messages == []


def test_write_record_start_writes_zeros_instead_of_nan(tmp_path):
    signals, sig_names = _small_record_signals()
    digitize.write_record("rec_start", signals, sig_names, str(tmp_path), "start")
    read_back = wfdb.rdrecord(str(tmp_path / "rec_start")).p_signal

    assert not np.any(np.isnan(read_back))
    assert np.allclose(read_back[:2500, 1], 0.0, atol=1e-3)


# -------------------------------------------------- the +-10 mV renormalisation
def test_renormalisation_expression_keeps_nan_and_maps_into_unit_range():
    signals = np.array([[-50.0, np.nan], [0.0, 30.0], [np.nan, 100.0]])
    max_val = np.nanmax(signals)
    min_val = np.nanmin(signals)
    normalised = (signals - min_val) / (max_val - min_val) * 2 - 1

    assert np.array_equal(np.isnan(normalised), np.isnan(signals))
    finite = np.isfinite(normalised)
    assert normalised[finite].min() == pytest.approx(-1.0)
    assert normalised[finite].max() == pytest.approx(1.0)


# ----------------------------------------------------- compute_consistency_qc
def _consistent_signals(num_samples):
    t = np.arange(num_samples) / FREQUENCY
    lead_i = np.sin(2 * np.pi * t)
    lead_ii = np.cos(2 * np.pi * t)
    return {
        "I": lead_i,
        "II": lead_ii,
        "III": lead_ii - lead_i,
        "aVR": -(lead_i + lead_ii) / 2,
        "aVL": lead_i - lead_ii / 2,
        "aVF": lead_ii - lead_i / 2,
    }


def _assembled_consistent(placement, extra_iii=0.0):
    short = _consistent_signals(SHORT_SAMPLES)
    if placement == "column":
        long = _consistent_signals(NUM_SAMPLES)
        short["II"] = long["II"]
    short["III"] = short["III"] + extra_iii
    offsets = {lead: {"raw": 0.0, "snapped": 0.0} for lead in short}
    if placement == "column":
        for lead in ["aVR", "aVL", "aVF"]:
            offsets[lead] = {"raw": 2.5, "snapped": 2.5}
    return digitize.assemble_signals(short, offsets, NUM_SAMPLES, placement)


@pytest.mark.parametrize("placement", ["start", "column"])
def test_qc_is_zero_for_consistent_leads(placement):
    signals, sig_names = _assembled_consistent(placement)
    qc = digitize.compute_consistency_qc(signals, sig_names)
    assert qc["einthoven_rms"] == pytest.approx(0.0, abs=1e-9)
    assert qc["goldberger_rms"] == pytest.approx(0.0, abs=1e-9)
    assert qc["einthoven_n"] == SHORT_SAMPLES
    assert qc["goldberger_n"] == SHORT_SAMPLES
    assert qc["einthoven_ratio"] == pytest.approx(0.0, abs=1e-9)


def test_qc_offset_in_iii_shows_up_as_rms_but_not_after_demedian():
    signals, sig_names = _assembled_consistent("column", extra_iii=0.1)
    qc = digitize.compute_consistency_qc(signals, sig_names)
    assert qc["einthoven_rms"] == pytest.approx(0.1, abs=1e-6)
    assert qc["einthoven_rms_demedian"] == pytest.approx(0.0, abs=1e-9)
    assert qc["goldberger_rms"] == pytest.approx(0.0, abs=1e-9)


def test_qc_is_nan_when_a_lead_is_missing():
    signals, sig_names = _assembled_consistent("column")
    keep = [i for i, name in enumerate(sig_names) if name != "aVF"]
    qc = digitize.compute_consistency_qc(
        signals[:, keep], [sig_names[i] for i in keep]
    )
    assert np.isnan(qc["goldberger_rms"])
    assert np.isnan(qc["goldberger_ratio"])
    assert qc["goldberger_n"] == 0
    assert qc["einthoven_n"] == SHORT_SAMPLES


def test_qc_is_nan_when_no_samples_overlap():
    signals = np.full((NUM_SAMPLES, 3), np.nan)
    signals[:1250, 0] = 1.0
    signals[2500:3750, 1] = 1.0
    signals[3750:, 2] = 1.0
    qc = digitize.compute_consistency_qc(signals, ["I", "III", "II"])
    assert np.isnan(qc["einthoven_rms"])
    assert qc["einthoven_n"] == 0


# ------------------------------------------------------------ parser defaults
def test_parser_defaults():
    args = digitize.get_parser().parse_args(["-d", "data", "-o", "out"])
    assert args.lead_placement == "column"
    assert args.save_mask is False
    assert args.mask_folder is None
