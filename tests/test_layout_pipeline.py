"""Unit tests for --layout and the layout path of src/run/digitize.py.

The first part pins the standard page: every function that takes a layout gives,
with layout=STANDARD and without it, bit for bit what it gave before it took one. The
expected digests below were computed on the code before the change (git fd17cc4) and
are SHA-1 digests of the outputs (floats by their hex form, arrays by their bytes, a
column map by its values on a fixed set of millimetres). The second part draws pages
of other layouts here, a 1 mm / 5 mm grid at 200 dpi with thin dark traces and their
label mask, and reads them on the layout path. No model and no data files.
"""
import contextlib
import csv
import hashlib
import io
import warnings
from functools import lru_cache

import numpy as np
import pytest
import torch
import wfdb
from torchvision.io.image import write_png

from config import FREQUENCY, LEAD_LABEL_MAPPING, Y_SHIFT_RATIO
from src.run import digitize
from src.run.layout import KNOWN_LAYOUTS, LEADS, STANDARD, layout_from_key

# ------------------------------------------------------------------ the drawn page
WIDTH, HEIGHT = 2200, 1700
PERIOD = 200 / 25.4  # px per printed millimetre at 200 dpi
ORIGIN_MM = 15  # the first column starts on the 15th line of the page
G0_DRAWN = ORIGIN_MM * PERIOD
SPAN = 250.0  # millimetres of 10 s at 25 mm/s
GAIN = {
    "I": 0.6, "II": 1.0, "III": 0.4, "aVR": -0.8, "aVL": 0.1, "aVF": 0.7,
    "V1": -0.5, "V2": 0.9, "V3": 1.1, "V4": 1.0, "V5": 0.8, "V6": 0.6,
}
# A fixed permutation of the labels, as a model that labels by position has them.
SHUFFLED = dict(zip(LEADS, np.random.RandomState(7).permutation(12) + 1))


def beats(t):
    """An ECG-like signal in mV at the times t in seconds."""
    value = np.zeros_like(t)
    for beat in np.arange(0.4, 10.5, 0.83):
        value += np.exp(-(((t - beat) / 0.018) ** 2))
        value += 0.25 * np.exp(-(((t - beat - 0.25) / 0.06) ** 2))
        value += 0.1 * np.exp(-(((t - beat + 0.16) / 0.03) ** 2))
    return value


def _strokes(positions, bold, length):
    """Darkness of lines at the positions along one axis, area sampled per pixel."""
    profile = np.zeros(length + 8)
    for x, is_bold in zip(positions, bold):
        half, dark = (1.0, 170.0) if is_bold else (0.5, 90.0)
        first = int(np.floor(x - half))
        for pixel in range(first, first + 4):
            cover = min(pixel + 1, x + half) - max(pixel, x - half)
            if cover > 0 and 0 <= pixel < profile.size:
                profile[pixel] = max(profile[pixel], dark * min(cover, 1.0))
    return profile[:length]


def pieces_of(layout, drop_strip_short=True):
    """(lead, first mm, last mm, baseline ratio) of every trace a page of layout has."""
    lines = layout.grid_lines_per_column
    pieces = []
    for r, row in enumerate(layout.lead_rows):
        for c, lead in enumerate(row):
            if drop_strip_short and lead in layout.strips:
                continue
            pieces.append((lead, c * lines, (c + 1) * lines, layout.row_ratio(r)))
    for k, lead in enumerate(layout.strips):
        pieces.append((lead, 0.0, SPAN, layout.strip_ratio(k)))
    return pieces


@lru_cache(maxsize=None)
def drawn(key="3x4+II", compress=0.0, early=0, labels="correct", gain=1.0,
          drop_strip_short=True):
    """A drawn page of a layout: (image [3, H, W] uint8, label mask [1, H, W] uint8).

    The last 62.5 mm of the 1 mm grid (every fifth line bold) are narrower by the
    share compress, lines and traces alike; early extends the masks of the top row's
    leads by that many pixels to the left. labels: "correct", "shuffled" (SHUFFLED)
    or "one" (every trace label 1). gain scales the drawn signals. A strip lead has
    no short trace when drop_strip_short, as the model leaves out the short II.
    """
    layout = layout_from_key(key)

    def trace_x(mm):
        mm = np.asarray(mm, dtype=float)
        return G0_DRAWN + mm * PERIOD - compress * PERIOD * np.clip(
            mm - SPAN + 62.5, 0, None
        )

    k = np.arange(-ORIGIN_MM - 2, int(WIDTH / PERIOD) + 3)
    vertical = _strokes(trace_x(k), k % 5 == 0, WIDTH)
    rows = np.arange(0, int(HEIGHT / PERIOD) + 1)
    horizontal = _strokes(rows * PERIOD + 0.5 * PERIOD, rows % 5 == 0, HEIGHT)
    darkness = np.maximum(vertical[None, :], horizontal[:, None])
    image = np.full((3, HEIGHT, WIDTH), 255, dtype=np.uint8)
    # Red lines, as on a printed page: darkest in the two lower channels.
    image[1] = image[2] = np.round(255 - darkness).astype(np.uint8)

    mask = np.zeros((1, HEIGHT, WIDTH), dtype=np.uint8)
    fine_mm = np.linspace(-5, SPAN + 5, 20 * int(SPAN + 10) + 1)
    fine_x = trace_x(fine_mm)
    for lead, first_mm, last_mm, ratio in pieces_of(layout, drop_strip_short):
        label = {
            "correct": LEAD_LABEL_MAPPING[lead],
            "shuffled": SHUFFLED[lead],
            "one": 1,
        }[labels]
        first = int(np.ceil(trace_x(first_mm) - 0.5))
        last = int(np.ceil(trace_x(last_mm) - 0.5)) - 1
        mm_at = np.interp(np.arange(first, last + 2, dtype=float), fine_x, fine_mm)
        y = (
            digitize.baseline_row(ratio, HEIGHT)
            - gain * GAIN[lead] * beats(mm_at / 25.0) * 10 * PERIOD
        )
        for index, pixel in enumerate(range(first, last + 1)):
            top = int(np.floor(min(y[index], y[index + 1]) - 0.5))
            bottom = int(np.floor(max(y[index], y[index + 1]) - 0.5)) + 1
            image[:, top : bottom + 1, pixel] = 0
            mask[0, top - 1 : bottom + 2, pixel] = label
        if early and ratio == layout.row_ratio(0) and first_mm == 0.0:
            top = int(np.floor(y[0] - 0.5))
            mask[0, top - 1 : top + 3, first - early : first] = label
    return torch.from_numpy(image), torch.from_numpy(mask)


# ------------------------------------------------------------------ digests
# The millimetres a column map is read at for its digest.
PROBE_MM = np.linspace(-10.0, 260.0, 541)


def digest(value):
    """SHA-1 (16 hex digits) of an output: floats by hex, arrays by their bytes."""
    sha = hashlib.sha1()

    def feed(v):
        if v is None:
            sha.update(b"N")
        elif isinstance(v, (bool, np.bool_)):
            sha.update(b"B1" if v else b"B0")
        elif isinstance(v, (int, np.integer)):
            sha.update(f"I{int(v)}".encode())
        elif isinstance(v, (float, np.floating)):
            sha.update(f"F{float(v).hex()}".encode())
        elif isinstance(v, str):
            sha.update(f"S{v}".encode())
        elif isinstance(v, torch.Tensor):
            feed(v.numpy())
        elif isinstance(v, np.ndarray):
            sha.update(f"A{v.dtype}{v.shape}".encode())
            sha.update(np.ascontiguousarray(v).tobytes())
        elif isinstance(v, dict):
            sha.update(b"D")
            for key in sorted(v, key=str):
                feed(str(key))
                feed(v[key])
        elif isinstance(v, (list, tuple, range)):
            sha.update(b"L")
            for item in v:
                feed(item)
        elif callable(v):
            feed(v(PROBE_MM))
        else:
            raise TypeError(f"no digest for {type(v)}")

    feed(value)
    return sha.hexdigest()[:16]


# The standard pages of the first part: name -> the arguments of drawn().
STANDARD_PAGES = {
    # A page the column map reads (its fourth column is a little narrow).
    "squeezed": {"compress": 0.015},
    # A fourth column 2.6 % narrower: the fit fails, the rescue of the fit reads it.
    "fit": {"compress": 0.026},
    # Masks that start 7 px early: the lines are refused, the rescue of the phase.
    "phase": {"compress": 0.015, "early": 7},
    # An even page: the column map is inside its dead band.
    "even": {},
}


def standard_outputs(name, explicit):
    """Every output of the functions that take a layout, on one standard page.

    explicit passes layout=STANDARD, otherwise the functions are called as before.
    """
    kw = {"layout": STANDARD} if explicit else {}
    image, mask = drawn(**STANDARD_PAGES[name])
    masks, positions, _ = digitize.cut_binary(mask, image)
    out = {}
    out["fit_page"] = digitize.fit_column_grid(
        masks, positions, HEIGHT, "page", "rec", quiet=True, **kw
    )
    out["fit_fit"] = digitize.fit_column_grid(
        masks, positions, HEIGHT, "fit", "rec", quiet=True, **kw
    )
    out["fit_tol"] = digitize.fit_column_grid(
        masks, positions, HEIGHT, "page", "rec", tolerance=0.04, quiet=True, **kw
    )
    g0, P, long_leads, reason = out["fit_page"]
    if g0 is None:
        out["rescue_fit"] = digitize.rescue_grid_fit(
            image, masks, positions, "page", "rec", reason, snap_offset=0.0, **kw
        )
        g0, P, long_leads, _ = out["fit_tol"]
    g0, P, lines = digitize.refine_grid_from_lines(image, g0, P, snap_offset=0.0, **kw)
    out["refine"] = (g0, P, lines)
    edges = digitize.column_mapping_edges(masks, positions, long_leads, **kw)
    out["edges"] = edges
    if lines["reason"]:
        out["rescue_phase"] = digitize.rescue_grid_phase(
            image, g0, P, lines, edges, 0.0, **kw
        )
    x_map, map_info = digitize.measure_column_mapping(
        image, g0, P, edges, snap_offset=0.0, **kw
    )
    out["map"] = (x_map, map_info)
    out["map_raw"] = digitize.measure_column_mapping(
        image, g0, P, edges, snap_offset=0.0, median_mm=0.0, **kw
    )
    row_maps = {}
    if map_info["knots_m"] is not None:
        rows = digitize.measure_row_mapping(
            image, g0, P, masks, positions, long_leads, map_info, snap_offset=0.0, **kw
        )
        out["rows"] = rows
        row_maps = {row["row"]: row["x_map"] for row in rows if row["accepted"]}
    out["lead_row"] = [digitize.lead_row(lead, long_leads, **kw) for lead in LEADS]
    ink_map = digitize._ink(image)
    signals, bbox, infos = {}, {}, {}
    widths = [m.shape[2] for m in masks.values() if m is not None]
    sec_per_pixel = 2.5 / np.mean([w for w in widths if w < 2 * np.median(widths)])
    mV_per_pixel = 25 * (2.5 / P) / 10
    for lead, lead_mask in masks.items():
        if lead_mask is None:
            continue
        column = 0 if lead in long_leads else int(
            digitize.STANDARD_LEAD_OFFSETS_SEC[lead] / 2.5
        )
        info = {}
        signals[lead] = digitize.vectorise_grid(
            image, lead_mask, positions[lead], g0, P, column, lead in long_leads,
            Y_SHIFT_RATIO, lead, 1.001, ink_map=ink_map, info=info, x_shift=0.3,
            sharpen="bandlimited",
            x_map=row_maps.get(digitize.lead_row(lead, long_leads), x_map), **kw
        )
        infos[lead] = info
        bbox[lead] = digitize.vectorise(
            image, lead_mask, positions[lead]["y1"], sec_per_pixel, mV_per_pixel,
            Y_SHIFT_RATIO, lead, **kw
        )
    out["signals"], out["ink"], out["bbox"] = signals, infos, bbox
    out["plain"] = {
        lead: digitize.vectorise_grid(
            image, lead_mask, positions[lead], g0, P,
            0 if lead in long_leads else int(
                digitize.STANDARD_LEAD_OFFSETS_SEC[lead] / 2.5
            ),
            lead in long_leads, Y_SHIFT_RATIO, lead, **kw
        )
        for lead, lead_mask in masks.items()
        if lead_mask is not None
    }
    out["baseline"] = digitize.estimate_baseline_shift(
        signals, long_leads, mV_per_pixel, P, "rec", **kw
    )
    # A shifted page, so that the shift is measured and not zero.
    shifted = {lead: s + 0.05 for lead, s in signals.items()}
    out["baseline_shifted"] = digitize.estimate_baseline_shift(
        shifted, long_leads, mV_per_pixel, P, "rec", **kw
    )
    lengths = {lead: len(s) for lead, s in signals.items()}
    out["grid_offsets"] = digitize.compute_grid_lead_offsets(
        positions, lengths, long_leads, g0, P, **kw
    )
    bbox_lengths = {lead: len(s) for lead, s in bbox.items()}
    out["offsets"] = digitize.compute_lead_offsets(
        positions, bbox_lengths, sec_per_pixel, "rec", **kw
    )
    # Without a rhythm strip the offsets come from the table.
    short_lengths = {lead: 1250 for lead in bbox_lengths}
    out["offsets_table"] = digitize.compute_lead_offsets(
        positions, short_lengths, sec_per_pixel, "rec", **kw
    )
    return {key: digest(value) for key, value in out.items()}


# Digests of standard_outputs() computed on the code before the change.
EXPECTED = {
    'squeezed': {
        'fit_page': 'a7ef23de208ad70e',
        'fit_fit': 'a7ef23de208ad70e',
        'fit_tol': 'a7ef23de208ad70e',
        'refine': '7945de7b1f14c874',
        'edges': 'c60f26e44091b260',
        'map': '17d3e4fc9dbce049',
        'map_raw': 'f8db3c5011354339',
        'rows': 'e7f19c083370ac2f',
        'lead_row': 'adb4560eb463823d',
        'signals': '3fc2bc3b02b20c41',
        'ink': '4fb6b9ca69dfff40',
        'bbox': '8124fca0e02dd00f',
        'plain': '39a6f2326dbcfafc',
        'baseline': 'bb044a2ae028e6bb',
        'baseline_shifted': '6be6dfb334c52abf',
        'grid_offsets': 'be6bb36fe460f368',
        'offsets': '42a2301a89f7b579',
        'offsets_table': '7ab091d1c2fe9eaf',
    },
    'fit': {
        'fit_page': 'a10a1891a9349c27',
        'fit_fit': 'a10a1891a9349c27',
        'fit_tol': '146cca38148c8226',
        'rescue_fit': 'bcd9252d78da49b6',
        'refine': 'b97fff77f04b6aaf',
        'edges': 'bc5481336f78cf5f',
        'map': '9061280c9aba3998',
        'map_raw': 'f8c065318abdd881',
        'rows': '06a11416e22c7fb8',
        'lead_row': 'adb4560eb463823d',
        'signals': 'cb2bd3e634135400',
        'ink': '4fb6b9ca69dfff40',
        'bbox': 'f85fc6a90075bf01',
        'plain': '09f2bbd18ffb7d93',
        'baseline': '618355eda0873f5c',
        'baseline_shifted': '8cb92e4d36dcde23',
        'grid_offsets': 'ac8802dfb8cd3c1f',
        'offsets': '899ab2ba18e7d3fa',
        'offsets_table': '7ab091d1c2fe9eaf',
    },
    'phase': {
        'fit_page': '3639a1d11c254d8a',
        'fit_fit': '679f9b04945475ae',
        'fit_tol': '3639a1d11c254d8a',
        'refine': 'fc50f37ac04a7ee2',
        'edges': '5cb27ec6f99bcb47',
        'rescue_phase': '62280a1311f61ce7',
        'map': 'a01edb2e0ddcb8dc',
        'map_raw': 'afca884d9ff4da4f',
        'rows': '6ff364af5c059395',
        'lead_row': 'adb4560eb463823d',
        'signals': 'a19e6923ee0f9c92',
        'ink': '4fb6b9ca69dfff40',
        'bbox': '47c12c258b5df3c0',
        'plain': '7d438f7f50d9a5c1',
        'baseline': 'afa13d98db688729',
        'baseline_shifted': '0a927520dbbe2768',
        'grid_offsets': '2b2806d337f68c69',
        'offsets': '0c7089b9ca1f71d9',
        'offsets_table': '7ab091d1c2fe9eaf',
    },
    'even': {
        'fit_page': 'c88b4b49d615d5b5',
        'fit_fit': 'a5ab14108f6195c1',
        'fit_tol': 'c88b4b49d615d5b5',
        'refine': '6132df701ca505d0',
        'edges': 'afebbf8792ac3bd5',
        'map': '51f86590938e2000',
        'map_raw': 'aadba81444849aa0',
        'rows': '090595a946c240ad',
        'lead_row': 'adb4560eb463823d',
        'signals': 'b1394ae64702439c',
        'ink': '4fb6b9ca69dfff40',
        'bbox': '68ebbc842753d64e',
        'plain': '6dc295d97fb04caa',
        'baseline': '70b5eb97b39df3b9',
        'baseline_shifted': '314c1b8cdb4e4c1a',
        'grid_offsets': 'fdb540211578fe06',
        'offsets': '2b98dea31beb336e',
        'offsets_table': '7ab091d1c2fe9eaf',
    },
}

# The stages of the end-to-end runs: the drawn pages are straight and at 200 dpi.
STAGES = (
    "--rotation", "hough", "--perspective", "off", "--resolution", "keep",
    "--grid_line_offset", "0", "--paper_normalisation", "off", "-f", "--no-verbose",
)


def run_pages(folder, pages, *flags):
    """run() over drawn pages {name: drawn() arguments} with the masks drawn for them.

    Returns (output folder, what run() printed).
    """
    data, masks, out = folder / "data", folder / "masks", folder / "out"
    data.mkdir(parents=True, exist_ok=True)
    masks.mkdir(parents=True, exist_ok=True)
    for name, arguments in pages.items():
        image, mask = drawn(**arguments)
        write_png(image, str(data / f"{name}.png"))
        write_png(mask, str(masks / f"{name}_mask.png"))
    argv = ["-d", str(data), "-o", str(out), "--mask_folder", str(masks)]
    printed = io.StringIO()
    with contextlib.redirect_stdout(printed):
        digitize.run(digitize.get_parser().parse_args(argv + [*STAGES, *flags]))
    return out, printed.getvalue()


def dat_digests(out, names):
    """SHA-1 (16 hex digits) of the .dat file of every page."""
    return {
        name: hashlib.sha1((out / f"{name}.dat").read_bytes()).hexdigest()[:16]
        for name in names
    }


# Digests of the .dat files of run_pages() on the standard pages, before the change.
EXPECTED_RUN = {
    'squeezed': '818baa30e6aa35fb',
    'fit': '19a21f23d8be60c2',
    'phase': '818baa30e6aa35fb',
    'even': 'bfd54ebdbcc0868b',
}


# ------------------------------------------------- the standard page is as it was
@pytest.mark.parametrize("explicit", [False, True])
@pytest.mark.parametrize("name", list(STANDARD_PAGES))
def test_every_function_reads_the_standard_page_as_before(name, explicit):
    assert standard_outputs(name, explicit) == EXPECTED[name]


@pytest.mark.parametrize(
    "flags",
    [(), ("--layout", "auto"), ("--layout", "standard"), ("--layout", "3x4+II")],
)
def test_run_on_standard_pages_writes_what_it_wrote_before(tmp_path, flags):
    out, printed = run_pages(tmp_path, STANDARD_PAGES, *flags)
    assert dat_digests(out, STANDARD_PAGES) == EXPECTED_RUN
    # The standard page says nothing about its layout and has no agreement.
    assert "Layout for record" not in printed
    rows = qc_rows(out)
    assert {row["layout"] for row in rows.values()} == {"standard"}
    assert all(np.isnan(float(row["layout_label_agreement"])) for row in rows.values())


def test_the_standard_layout_keeps_the_constants():
    assert digitize.column_scale(STANDARD) == 1.0
    assert digitize.column_scale(KNOWN_LAYOUTS["6x2+II"]) == 0.5
    assert digitize.column_scale(KNOWN_LAYOUTS["12x1+none"]) == 0.25
    assert STANDARD.y_shift_ratio() == Y_SHIFT_RATIO
    assert STANDARD.page_pitch_ratio == digitize.PAGE_PITCH_RATIO
    assert STANDARD.grid_lines_per_column == digitize.GRID_LINES_PER_COLUMN
    assert STANDARD.columns == digitize.NUM_COLUMNS
    assert STANDARD.num_rows == digitize.NUM_ROWS
    assert STANDARD.layout_row() == digitize.LAYOUT_ROW
    assert [digitize.lead_row(lead, ["II"]) for lead in LEADS] == [
        3 if lead == "II" else digitize.LAYOUT_ROW[lead] for lead in LEADS
    ]
    # Without "full:<lead>" keys a strip is at "full", as before.
    assert digitize._strip_ratio(Y_SHIFT_RATIO, "II") == Y_SHIFT_RATIO["full"]


# ------------------------------------------------------------------- the flag
def test_the_flag_takes_auto_standard_and_layout_keys(capsys):
    parse = digitize.get_parser().parse_args
    folders = ["-d", "in", "-o", "out"]
    assert parse(folders).layout == "auto"
    assert parse(folders + ["--layout", "standard"]).layout == "standard"
    for key in ("6x2+II", "3x4+II,V1,V5", "12x1+none", "3x4+V1", "6x2+none"):
        assert parse(folders + ["--layout", key]).layout == key
    for wrong in ("12x1", "4x3+II", "3x4+V7", "3x4+II,II", "nope", "3x4+II,V1,V5,V2"):
        with pytest.raises(SystemExit):
            parse(folders + ["--layout", wrong])
    assert "not auto, standard or a layout key" in capsys.readouterr().err


# ------------------------------------------------------- the cut by position
def qc_rows(folder):
    with open(folder / "qc.csv", newline="") as f:
        return {row["record"]: row for row in csv.DictReader(f)}


@pytest.mark.parametrize("labels", ["shuffled", "one"])
@pytest.mark.parametrize("key", ["6x2+II", "6x2+none", "12x1+none", "3x4+II,V1,V5"])
def test_cut_layout_gives_every_cell_to_the_lead_of_its_position(key, labels, capsys):
    image, mask = drawn(key, labels=labels, gain=0.5)
    _, right = drawn(key, gain=0.5)
    layout, info, qc = digitize.choose_layout(mask, "auto", "rec")
    assert qc == key, info["reason"]
    assert f"Layout for record rec: {key} (" in capsys.readouterr().out
    # The masks of the model were labelled otherwise.
    assert info["label_agreement"] < 1.0
    by_position = digitize.layout_label_mask(mask, layout, info)
    truth = right[0].numpy()
    # Every trace pixel goes to the lead drawn there, but for the pixel column at the
    # boundary of two cells, which the equal columns of the trace width may put one
    # pixel off the drawn boundary.
    wrong = (truth > 0) & (by_position != truth)
    columns = np.flatnonzero(wrong.any(axis=0))
    boundaries = [cell["span"][0] for cell in info["cells"] if cell["column"] > 0]
    assert all(min(abs(x - b) for b in boundaries) <= 1 for x in columns)
    masks, positions, _ = digitize.cut_layout(mask, image, layout, info)
    drawn_masks, drawn_positions, _ = digitize.cut_binary(right, image)
    for lead in LEADS:
        assert abs(masks[lead].shape[2] - drawn_masks[lead].shape[2]) <= 1, lead
        assert abs(positions[lead]["x1"] - drawn_positions[lead]["x1"]) <= 1, lead
        assert positions[lead]["y1"] == drawn_positions[lead]["y1"], lead


def test_the_strip_of_a_lead_with_a_short_cell_is_its_output():
    # The short V1 and V5 are drawn with the label of their strips, as the oracle of
    # D1 has them: the strip is kept, the short cell dropped, as the short II is.
    key = "3x4+II,V1,V5"
    image, mask = drawn(key, gain=0.5, drop_strip_short=False)
    layout, info, _ = digitize.choose_layout(mask, "auto", "rec")
    masks, positions, _ = digitize.cut_layout(mask, image, layout, info)
    label_mask = digitize.layout_label_mask(mask, layout, info)
    trace_width = info["x_extent"][1] - info["x_extent"][0]
    for k, lead in enumerate(("II", "V1", "V5")):
        assert masks[lead].shape[2] > 0.95 * trace_width
        y0, y1 = info["strip_cells"][k]["band"]
        assert positions[lead]["y1"] >= y0 and positions[lead]["y1"] < y1
    short_v1 = next(c for c in info["cells"] if c["lead_by_position"] == "V1")
    (y0, y1), (x0, x1) = short_v1["band"], short_v1["span"]
    assert mask[0, y0:y1, x0:x1].any()
    assert not label_mask[y0:y1, x0:x1].any()


def test_a_given_key_names_the_strip_the_mask_cannot(capsys):
    # A 3x4 page whose strip is V1 is the standard page to the mask.
    image, mask = drawn("3x4+V1", gain=0.5)
    layout, info, qc = digitize.choose_layout(mask, "auto", "rec")
    assert layout == STANDARD and qc == "standard"
    assert capsys.readouterr().out == ""
    layout, info, qc = digitize.choose_layout(mask, "3x4+V1", "rec")
    assert layout.key() == qc == "3x4+V1"
    assert [cell["lead_by_position"] for cell in info["strip_cells"]] == ["V1"]
    printed = capsys.readouterr().out
    assert "Layout for record rec: 3x4+V1 (3 rows x 4 columns" in printed
    masks, positions, _ = digitize.cut_layout(mask, image, layout, info)
    assert masks["V1"].shape[2] > 0.95 * (info["x_extent"][1] - info["x_extent"][0])
    # The short II of this page is a lead of its own.
    assert masks["II"].shape[2] < 0.3 * (info["x_extent"][1] - info["x_extent"][0])
    # The key of the standard page is the standard page.
    layout, info, qc = digitize.choose_layout(mask, "3x4+II", "rec")
    assert layout == STANDARD and qc == "standard"


def test_a_given_key_the_mask_does_not_show_reads_the_standard_page(capsys):
    _, mask = drawn("6x2+II", gain=0.5)
    layout, info, qc = digitize.choose_layout(mask, "12x1+none", "rec")
    assert layout == STANDARD
    assert qc == "standard(given 12x1+none, mask differs)"
    printed = capsys.readouterr().out
    assert printed.startswith(
        "WARNING: layout 12x1+none given for record rec but the mask shows 6 rows x 2 "
        "columns and 1 strip(s) (found 6x2+II); reading the page as the standard"
    )


def test_a_page_without_a_layout_says_so_in_one_line(capsys):
    mask = torch.zeros((1, HEIGHT, WIDTH), dtype=torch.uint8)
    mask[0, 800:820, 900:1000] = 3
    layout, info, qc = digitize.choose_layout(mask, "auto", "rec")
    assert layout == STANDARD and qc == "standard(not detected)"
    printed = capsys.readouterr().out.splitlines()
    assert printed == [
        f"Layout for record rec: not detected ({info['reason']}), read as the standard "
        "3x4 page with rhythm strip II."
    ]
    assert not printed[0].startswith("WARNING")
    # --layout standard does not look at all.
    chosen = digitize.choose_layout(mask, "standard", "rec")
    assert chosen == (STANDARD, None, "standard")


def band_cut(mask, layout, info):
    """The cut by bands alone, without following the rows (for the test below)."""
    labels = mask[0].numpy()
    out = np.zeros(labels.shape, np.uint8)
    for cell in info["cells"] + info["strip_cells"]:
        (y0, y1), (x0, x1) = cell["band"], cell["span"]
        inked = labels[y0:y1, x0:x1] > 0
        out[y0:y1, x0:x1][inked] = LEAD_LABEL_MAPPING[cell["lead_by_position"]]
    return out


def test_the_rows_are_followed_beyond_their_bands():
    # QRS complexes that reach past the middle between two rows but do not touch the
    # next trace: the bands give their tips to the next row, the following does not.
    key = "6x2+none"
    _, mask = drawn(key, gain=1.6)
    layout, info, _ = digitize.choose_layout(mask, "auto", "rec")
    assert layout.key() == key
    truth = mask[0].numpy()
    x0, x1 = info["x_extent"]
    inside = np.zeros(truth.shape, bool)
    inside[:, x0:x1] = truth[:, x0:x1] > 0
    bands = band_cut(mask, layout, info)
    assert np.count_nonzero(inside & (bands != truth)) > 500
    followed = digitize.layout_label_mask(mask, layout, info)
    assert np.count_nonzero(inside & (followed != truth)) == 0
    # The labels of the model are not read.
    _, shuffled = drawn(key, gain=1.6, labels="shuffled")
    assert np.array_equal(digitize.layout_label_mask(shuffled, layout, info), followed)


# -------------------------------------------- the functions on another layout
def layout_cut(key, gain=0.5):
    image, mask = drawn(key, labels="shuffled", gain=gain)
    layout, info, _ = digitize.choose_layout(mask, "auto", "rec")
    masks, positions, _ = digitize.cut_layout(mask, image, layout, info)
    return image, masks, positions, layout, info


def test_the_column_grid_of_6x2_and_12x1():
    for key, strips in (("6x2+II", ["II"]), ("12x1+none", [])):
        image, masks, positions, layout, info = layout_cut(key)
        g0, P, long_leads, reason = digitize.fit_column_grid(
            masks, positions, HEIGHT, "page", "rec", layout=layout, long_leads=strips
        )
        assert reason == "" and long_leads == strips
        assert g0 == pytest.approx(G0_DRAWN, abs=1.0)
        assert P == pytest.approx(layout.grid_lines_per_column * PERIOD, rel=0.002)
        g0, P, lines = digitize.refine_grid_from_lines(
            image, g0, P, snap_offset=0.0, layout=layout
        )
        assert lines["reason"] == ""
        assert g0 == pytest.approx(G0_DRAWN, abs=0.05)
        assert P == pytest.approx(layout.grid_lines_per_column * PERIOD, abs=0.05)
        edges = digitize.column_mapping_edges(masks, positions, long_leads, layout)
        assert max(mm for _, mm in edges) == pytest.approx(250.0)
        x_map, map_info = digitize.measure_column_mapping(
            image, g0, P, edges, snap_offset=0.0, layout=layout
        )
        # An even page: the map is inside its dead band, one value per column.
        assert map_info["reason"] == "" and map_info["dead_band"]
        assert len(map_info["column_shift"]) == layout.columns
        rows = digitize.measure_row_mapping(
            image, g0, P, masks, positions, long_leads, map_info, snap_offset=0.0,
            layout=layout,
        )
        assert [row["row"] for row in rows] == list(range(layout.num_rows))
        assert all(len(row["d"]) == layout.columns for row in rows)
    # The width rule of the standard page finds no strip on a 6x2 page.
    image, masks, positions, layout, info = layout_cut("6x2+none")
    assert digitize.fit_column_grid(masks, positions, HEIGHT, "page", "rec")[3] in (
        "no rhythm strip found",
        "rhythm strip does not span all columns",
        "not every column has a short lead",
    )


def test_a_lead_of_6x2_is_read_over_its_5_s_and_placed_in_its_window():
    layout = KNOWN_LAYOUTS["6x2+II"]
    image, masks, positions, _, _ = layout_cut("6x2+II")
    g0, P = G0_DRAWN, 125 * PERIOD
    ratios = layout.y_shift_ratio()
    v4 = digitize.vectorise_grid(
        image, masks["V4"], positions["V4"], g0, P, 1, False, ratios, "V4",
        layout=layout,
    )
    ii = digitize.vectorise_grid(
        image, masks["II"], positions["II"], g0, P, 0, True, ratios, "II", layout=layout
    )
    assert v4.shape == (2500,) and ii.shape == (5000,)
    t = np.arange(5000) / FREQUENCY
    assert np.corrcoef(v4.numpy(), 0.5 * GAIN["V4"] * beats(t[2500:]))[0, 1] > 0.99
    assert np.corrcoef(ii.numpy(), 0.5 * GAIN["II"] * beats(t))[0, 1] > 0.99
    lengths = {"V4": 2500, "II": 5000, "aVR": 2500}
    offsets = digitize.compute_grid_lead_offsets(
        positions, lengths, ["II"], g0, P, layout
    )
    assert {lead: o["snapped"] for lead, o in offsets.items()} == {
        "V4": 5.0, "II": 0.0, "aVR": 0.0
    }
    assert offsets["V4"]["raw"] == pytest.approx(5.0, abs=0.01)
    signals, names = digitize.assemble_signals(
        {"V4": v4.numpy(), "II": ii.numpy()}, offsets, 5000, "column"
    )
    assert np.flatnonzero(np.isfinite(signals[:, 0]))[[0, -1]].tolist() == [2500, 4999]
    assert np.isfinite(signals[:, 1]).all()
    # The bounding box fallback reads the kind of a lead off the strips.
    bbox = digitize.vectorise(
        image, masks["V4"], positions["V4"]["y1"], 5.0 / masks["V4"].shape[2],
        0.1, ratios, "V4", layout=layout, long_leads=["II"],
    )
    assert bbox.shape == (2500,)
    bbox_ii = digitize.vectorise(
        image, masks["II"], positions["II"]["y1"], 5.0 / masks["V4"].shape[2],
        0.1, ratios, "II", layout=layout, long_leads=["II"],
    )
    assert bbox_ii.shape == (5000,)


def test_the_offsets_and_rows_of_other_layouts():
    six = KNOWN_LAYOUTS["6x2+none"]
    lengths = {lead: 2500 for lead in LEADS}
    positions = {lead: {"x1": 0, "y1": 0} for lead in LEADS}
    offsets = digitize.compute_lead_offsets(positions, lengths, 0.01, "rec", six)
    assert {lead: o["snapped"] for lead, o in offsets.items()} == six.lead_offsets_sec()
    three = KNOWN_LAYOUTS["3x4+II,V1,V5"]
    strips = ["II", "V1", "V5"]
    assert [digitize.lead_row(lead, strips, three) for lead in strips] == [3, 4, 5]
    assert digitize.lead_row("V2", strips, three) == 1
    ratios = three.y_shift_ratio()
    assert [digitize._strip_ratio(ratios, lead) for lead in strips] == [
        three.strip_ratio(k) for k in range(3)
    ]
    # The rule of a 12x1 page lives in its one column of 5000 samples.
    twelve = KNOWN_LAYOUTS["12x1+none"]
    signals = {lead: np.full(5000, 0.0) for lead in LEADS}
    signals["II"] = np.full(5000, 0.2)
    shift, disagreement = digitize.estimate_baseline_shift(
        signals, [], 0.01, 1968.0, "rec", twelve
    )
    assert np.isfinite(disagreement)
    assert digitize.estimate_baseline_shift(signals, [], 0.01, 1968.0, "rec")[1] != (
        disagreement
    )


def test_append_qc_row_writes_the_layout_columns_last(tmp_path):
    qc = {"layout": "6x2+II", "layout_label_agreement": 0.25}
    digitize.append_qc_row(str(tmp_path), "rec", "column", qc, 0.0)
    with open(tmp_path / "qc.csv", newline="") as f:
        reader = csv.DictReader(f)
        row = next(reader)
        assert len(reader.fieldnames) == 39
        assert reader.fieldnames[-2:] == ["layout", "layout_label_agreement"]
        assert reader.fieldnames.index("sheet_curl_shift_px") == 36
    assert row["layout"] == "6x2+II" and float(row["layout_label_agreement"]) == 0.25


# ------------------------------------------------------------------- end to end
def read_well(snr):
    """Every lead above 25 dB, aVL above 10 dB: it is drawn at a tenth of a mV."""
    others = [value for lead, value in snr.items() if lead != "aVL"]
    return snr["aVL"] > 10.0 and min(others) > 25.0


def drawn_snr(out, name, gain):
    """SNR in dB and the window [first, last] of every lead of a page read by run()."""
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        record = wfdb.rdrecord(str(out / name))
    snr, window = {}, {}
    for index, lead in enumerate(record.sig_name):
        read = record.p_signal[:, index]
        valid = np.isfinite(read)
        truth = gain * GAIN[lead] * beats(np.arange(read.size) / FREQUENCY)
        signal = truth[valid] - truth[valid].mean()
        error = (read[valid] - read[valid].mean()) - signal
        snr[lead] = float(10 * np.log10(np.sum(signal**2) / np.sum(error**2)))
        window[lead] = np.flatnonzero(valid)[[0, -1]].tolist()
    return snr, window


# key -> (gain of the drawn page, the window every lead is written in)
LAYOUT_PAGES = {
    "6x2+II": (1.0, {lead: [0, 4999] if lead == "II" else
                     ([2500, 4999] if lead.startswith("V") else [0, 2499])
                     for lead in LEADS}),
    "12x1+none": (0.8, {lead: [0, 4999] for lead in LEADS}),
    "3x4+II,V1,V5": (1.0, {lead: [0, 4999] if lead in ("II", "V1", "V5") else
                           [1250 * STANDARD.column_of(lead),
                            1250 * STANDARD.column_of(lead) + 1249]
                           for lead in LEADS}),
}


@pytest.mark.parametrize("key", list(LAYOUT_PAGES))
def test_run_reads_a_page_of_another_layout(tmp_path, key):
    gain, windows = LAYOUT_PAGES[key]
    pages = {"page": {"key": key, "labels": "shuffled", "gain": gain}}
    out, printed = run_pages(tmp_path, pages)
    assert f"Layout for record page: {key} (" in printed
    assert "no column grid" not in printed
    assert "Einthoven check failed" not in printed
    row = qc_rows(out)["page"]
    assert row["layout"] == key
    assert float(row["layout_label_agreement"]) < 1.0
    assert row["column_mapping"] == "uniform: dead band"
    snr, window = drawn_snr(out, "page", gain)
    assert window == windows
    assert read_well(snr), snr


def test_run_reads_the_strip_a_key_names(tmp_path):
    # The strip of a 3x4 page labelled V1: auto reads it as II, the key as V1.
    pages = {"page": {"key": "3x4+V1", "gain": 0.8}}
    out, printed = run_pages(tmp_path / "auto", pages)
    assert "Layout for record" not in printed
    _, window = drawn_snr(out, "page", 0.8)
    assert window["V1"] == [0, 4999] and qc_rows(out)["page"]["layout"] == "standard"
    out, printed = run_pages(tmp_path / "key", pages, "--layout", "3x4+V1")
    assert "Layout for record page: 3x4+V1 (" in printed
    snr, window = drawn_snr(out, "page", 0.8)
    assert window["V1"] == [0, 4999] and window["II"] == [0, 1249]
    assert qc_rows(out)["page"]["layout"] == "3x4+V1"
    assert read_well(snr), snr
