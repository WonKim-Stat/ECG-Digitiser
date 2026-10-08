"""Unit tests for src/run/layout.py: the Layout tables and the layout detection.

The first part checks that STANDARD reproduces the constants the digitiser reads
today (config.Y_SHIFT_RATIO and the 3x4 constants of src/run/digitize.py) with the
same floats, and that the other layouts put their rows where the generator of D1
draws them. The second part draws label masks of whole pages here, traces of 3 px
at the generator's geometry, and runs detect_layout on them. No data files.
"""
import ast
import inspect

import numpy as np
import pytest

import config
from src.run import digitize
from src.run import layout as layout_module
from src.run.layout import (
    KNOWN_LAYOUTS,
    LEADS,
    STANDARD,
    Layout,
    detect_layout,
    layout_from_key,
    make_layout,
)

PAGE_HEIGHT = 1700
PAGE_WIDTH = 2200
# The generated pages are drawn at 200 dpi.
PX_PER_MM = 200 / 25.4


# ---------------------------------------------------------------------------------
# The Layout tables.


def pixel_rows(layout, height=PAGE_HEIGHT):
    """Baselines of the short rows and of the strips, pixel rows from the top."""
    short = [(1 - layout.row_ratio(r)) * height for r in range(layout.n_short_rows)]
    strips = [(1 - layout.strip_ratio(k)) * height for k in range(layout.n_strips)]
    return short, strips


def test_standard_reproduces_y_shift_ratio_exactly():
    ratios = STANDARD.y_shift_ratio()
    assert set(ratios) == set(config.Y_SHIFT_RATIO)
    for key, value in config.Y_SHIFT_RATIO.items():
        assert ratios[key] == value, key


def test_standard_reproduces_the_digitiser_constants():
    assert STANDARD.lead_offsets_sec() == digitize.STANDARD_LEAD_OFFSETS_SEC
    assert STANDARD.layout_row() == digitize.LAYOUT_ROW
    assert STANDARD.rhythm_row == digitize.RHYTHM_ROW
    assert STANDARD.num_rows == digitize.NUM_ROWS
    assert STANDARD.columns == digitize.NUM_COLUMNS
    assert STANDARD.page_pitch_ratio == digitize.PAGE_PITCH_RATIO
    assert STANDARD.grid_lines_per_column == digitize.GRID_LINES_PER_COLUMN
    assert STANDARD.seconds_per_column == config.SHORT_SIGNAL_LENGTH_SEC
    assert STANDARD.samples_per_column == int(
        config.SHORT_SIGNAL_LENGTH_SEC * config.FREQUENCY
    )
    for lead in LEADS:
        assert STANDARD.offset_sec(lead) == digitize.STANDARD_LEAD_OFFSETS_SEC[lead]


def test_standard_names_and_numbers():
    assert STANDARD.key() == "3x4+II"
    assert STANDARD.strips == ("II",)
    assert (STANDARD.n_short_rows, STANDARD.n_strips) == (3, 1)
    assert STANDARD.short_leads == (
        "I", "aVR", "V1", "V4", "II", "aVL", "V2", "V5", "III", "aVF", "V3", "V6",
    )
    assert STANDARD.strip_row(0) == 3
    assert STANDARD.column_of("V2") == 2
    assert STANDARD.row_of("aVF") == 2
    assert STANDARD.column_of("X") is None and STANDARD.row_of("X") is None
    with pytest.raises(IndexError):
        STANDARD.strip_row(1)


def test_columns_of_the_other_tables():
    six = KNOWN_LAYOUTS["6x2+II"]
    assert six.seconds_per_column == 5.0
    assert six.samples_per_column == 2500
    assert six.grid_lines_per_column == 125.0
    assert six.offset_sec("V4") == 5.0 and six.offset_sec("aVR") == 0.0
    assert six.row_of("aVR") == 3 and six.column_of("V6") == 1
    twelve = KNOWN_LAYOUTS["12x1+none"]
    assert twelve.seconds_per_column == 10.0
    assert twelve.samples_per_column == 5000
    assert all(twelve.offset_sec(lead) == 0.0 for lead in LEADS)
    assert twelve.layout_row()["V6"] == 11
    assert twelve.page_pitch_ratio == pytest.approx(4 * STANDARD.page_pitch_ratio)


def test_rows_of_6x2_with_strip_ii_are_where_the_generator_draws_them():
    short, strips = pixel_rows(KNOWN_LAYOUTS["6x2+II"])
    assert short == pytest.approx([472.2, 661.1, 850.0, 1038.9, 1227.8, 1416.7], abs=0.1)
    assert strips == pytest.approx([1581.9], abs=0.1)


def test_rows_of_12x1_without_strip_are_where_the_generator_draws_them():
    short, strips = pixel_rows(KNOWN_LAYOUTS["12x1+none"])
    expected = [182.1 + 121.4286 * r for r in range(12)]
    assert short == pytest.approx(expected, abs=0.1)
    assert short[-1] == pytest.approx(1517.9, abs=0.1)
    assert np.diff(short) == pytest.approx([121.4] * 11, abs=0.1)
    assert strips == []


def test_rows_of_3x4_with_three_strips_are_where_the_generator_draws_them():
    short, strips = pixel_rows(KNOWN_LAYOUTS["3x4+II,V1,V5"])
    assert short == pytest.approx([531.3, 743.7, 956.3], abs=0.1)
    assert strips == pytest.approx([1145.1, 1357.6, 1570.1], abs=0.1)


def test_y_shift_ratio_of_several_strips():
    layout = KNOWN_LAYOUTS["3x4+II,V1,V5"]
    ratios = layout.y_shift_ratio()
    assert ratios["full"] == layout.strip_ratio(0)
    assert [ratios[f"full:{lead}"] for lead in ("II", "V1", "V5")] == [
        layout.strip_ratio(k) for k in range(3)
    ]
    assert "full" not in KNOWN_LAYOUTS["6x2+none"].y_shift_ratio()
    assert "full:II" not in STANDARD.y_shift_ratio()


def test_known_layouts_table():
    names = ("3x4", "6x2", "12x1")
    strips = ("none", "II", "V1", "V5", "II,V1,V5")
    assert set(KNOWN_LAYOUTS) == {f"{n}+{s}" for n in names for s in strips}
    for key, layout in KNOWN_LAYOUTS.items():
        assert layout.key() == key
        assert layout_from_key(key) == layout
        assert sorted(layout.short_leads) == sorted(LEADS)
    assert KNOWN_LAYOUTS["3x4+II"] == STANDARD
    assert hash(KNOWN_LAYOUTS["3x4+II"]) == hash(STANDARD)


def test_make_layout_refusals():
    with pytest.raises(ValueError):
        make_layout("3x4", ["II", "II"])
    with pytest.raises(ValueError):
        make_layout("3x4", ["V7"])
    with pytest.raises(ValueError):
        make_layout("3x4", ["II", "V1", "V5", "V2"])
    with pytest.raises(ValueError):
        make_layout("4x3", ["II"])
    with pytest.raises(ValueError):
        Layout("3x4", 4, [["I", "aVR", "V1"]], [])
    with pytest.raises(ValueError):
        layout_from_key("3x4")
    assert make_layout("12x1", ["V6", "I"]).key() == "12x1+V6,I"


def test_the_module_imports_no_torch():
    tree = ast.parse(inspect.getsource(layout_module))
    imported = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            imported |= {alias.name.split(".")[0] for alias in node.names}
        elif isinstance(node, ast.ImportFrom):
            imported.add((node.module or "").split(".")[0])
    assert imported <= {"numpy", "config"}


# ---------------------------------------------------------------------------------
# Detection on drawn masks.


def draw_page(
    layout,
    labels="correct",
    drop_strip_short=True,
    skip=(),
    amplitude=15.0,
    step=12.0,
    bridge=None,
    height=PAGE_HEIGHT,
    width=PAGE_WIDTH,
    scale=1.0,
):
    """Label mask of a page of `layout`, traces 3 px thick at the generator's geometry.

    labels: "correct", "shuffled" (a fixed permutation of the 12 labels) or "one"
    (every trace label 1). drop_strip_short leaves out the short cells of the strip
    leads, as the model does on a standard page. skip: leads whose short cell is not
    drawn. A trace is a sine of `amplitude` px whose phase runs on along its row, and
    cell c of a row is moved up or down by `step` / 2 px in turns, so that the traces
    of two cells next to each other do not touch when step > 3 (a step of 0 and an
    amplitude of 0 give straight traces that all touch). bridge = (row, column) joins
    the trace of that cell to the next one with a vertical stroke at their boundary.
    scale < 1 draws the page smaller in the middle of the mask.
    """
    rng = np.random.RandomState(7)
    permutation = dict(zip(LEADS, rng.permutation(12) + 1))
    mask = np.zeros((height, width), np.uint8)
    page_height, page_width = scale * height, scale * width
    top, left = (height - page_height) / 2, (width - page_width) / 2
    unit = page_height / PAGE_HEIGHT
    x_start = left + 15 * PX_PER_MM * unit
    trace_width = 250 * PX_PER_MM * unit

    def label_of(lead):
        if labels == "correct":
            return layout_module.LEAD_LABEL_MAPPING[lead]
        if labels == "shuffled":
            return permutation[lead]
        return 1

    def draw(lead, baseline, xa, xb, offset, phase):
        xs = np.arange(int(np.floor(xa)), int(np.floor(xb)))
        ys = baseline + offset + amplitude * np.sin(2 * np.pi * xs / 150 + phase)
        for k, x in enumerate(xs):
            nxt = ys[k + 1] if k + 1 < len(xs) else ys[k]
            lo = int(np.floor(min(ys[k], nxt))) - 1
            hi = int(np.ceil(max(ys[k], nxt))) + 1
            mask[max(0, lo) : hi + 1, x] = label_of(lead)
        return ys

    strip_leads = set(layout.strips) if drop_strip_short else set()
    for r, row in enumerate(layout.lead_rows):
        baseline = top + (1 - layout.row_ratio(r)) * page_height
        phase = 0.7 * r
        ends = {}
        for c, lead in enumerate(row):
            xa = x_start + c * trace_width / layout.columns
            xb = x_start + (c + 1) * trace_width / layout.columns
            offset = step / 2 if c % 2 == 0 else -step / 2
            if lead in strip_leads or lead in skip:
                continue
            ys = draw(lead, baseline, xa, xb, offset, phase)
            ends[c] = (ys[0], ys[-1], int(np.floor(xb)))
        if bridge is not None and bridge[0] == r:
            c = bridge[1]
            y_left, y_right = ends[c][1], ends[c + 1][0]
            x = ends[c][2]
            lo, hi = sorted((y_left, y_right))
            mask[int(lo) - 1 : int(np.ceil(hi)) + 2, x - 1 : x + 1] = label_of(row[c])
    for k, lead in enumerate(layout.strips):
        baseline = top + (1 - layout.strip_ratio(k)) * page_height
        draw(lead, baseline, x_start, x_start + trace_width, 0.0, 1.3 * k)
    return mask


def cell_table(info):
    """(row, column) -> lead_by_position of the cells found."""
    return {
        (cell["row"], cell["column"]): cell["lead_by_position"] for cell in info["cells"]
    }


DETECTED = [
    ("3x4+II", True),
    ("3x4+V1", True),
    ("3x4+II,V1,V5", True),
    ("6x2+II", True),
    ("6x2+none", True),
    ("12x1+none", True),
    ("12x1+II", False),
    ("3x4+II", False),
    ("6x2+II", False),
    ("3x4+II,V1,V5", False),
]


@pytest.mark.parametrize("key, drop_strip_short", DETECTED)
def test_detects_the_layout_of_a_drawn_page(key, drop_strip_short):
    truth = layout_from_key(key)
    mask = draw_page(truth, drop_strip_short=drop_strip_short)
    layout, info = detect_layout(mask)
    assert layout is not None, info["reason"]
    assert (info["columns"], info["rows"], info["strips"]) == (
        truth.columns,
        truth.n_short_rows,
        truth.n_strips,
    )
    if (truth.columns, truth.n_short_rows, truth.n_strips) == (4, 3, 1):
        # The standard page keeps the strip II whatever it is labelled.
        assert layout == STANDARD and info["standard"]
    else:
        assert layout == truth and not info["standard"]
    # Every drawn cell is found where the table of the truth has it.
    drawn = {
        (r, c): lead
        for r, row in enumerate(truth.lead_rows)
        for c, lead in enumerate(row)
        if not (drop_strip_short and lead in truth.strips)
    }
    assert cell_table(info) == drawn
    assert info["n_cells"] == len(drawn)
    assert info["label_agreement"] == 1.0
    for k, cell in enumerate(info["strip_cells"]):
        assert cell["majority_label"] == truth.strips[k]
        assert cell["lead_by_position"] == layout.strips[k]
    assert info["strip_labels"] == list(truth.strips)
    assert info["merged"] == 0
    assert info["reason"] == ""


def test_a_3x4_page_with_one_strip_is_standard_whatever_its_strip_label():
    # The strip of a 3x4+V1 page labelled V1 (oracle) or II (the model): STANDARD.
    mask = draw_page(KNOWN_LAYOUTS["3x4+V1"])
    layout, info = detect_layout(mask)
    assert layout == STANDARD and info["standard"]
    assert info["strip_cells"][0]["majority_label"] == "V1"
    assert info["strip_cells"][0]["lead_by_position"] == "II"
    # Without its short cell every V1 pixel is the strip.
    label = layout_module.LEAD_LABEL_MAPPING
    mask[mask == label["V1"]] = label["II"]
    layout, info = detect_layout(mask)
    assert layout == STANDARD and info["standard"]
    assert info["strip_cells"][0]["majority_label"] == "II"


@pytest.mark.parametrize(
    "key", ["3x4+II", "3x4+II,V1,V5", "6x2+II", "6x2+none", "12x1+none", "12x1+II"]
)
@pytest.mark.parametrize("labels", ["shuffled", "one"])
def test_labels_change_nothing_but_the_agreement(key, labels):
    truth = layout_from_key(key)
    drop = not key.startswith("12x1")
    layout_right, info_right = detect_layout(draw_page(truth, drop_strip_short=drop))
    mask = draw_page(truth, labels=labels, drop_strip_short=drop)
    layout_wrong, info_wrong = detect_layout(mask)
    assert layout_wrong is not None, info_wrong["reason"]
    # The layout, strips included, is the same whatever the labels.
    assert layout_wrong == layout_right
    assert layout_wrong.key() == (STANDARD.key() if key == "3x4+II" else key)
    assert cell_table(info_wrong) == cell_table(info_right)
    strip_names = [cell["lead_by_position"] for cell in info_wrong["strip_cells"]]
    assert strip_names == list(layout_right.strips)
    for key_ in ("rows_y", "x_extent", "n_cells", "merged", "strip_rule"):
        assert info_wrong[key_] == info_right[key_]
    assert info_right["label_agreement"] == 1.0
    cells = info_wrong["cells"]
    agreement = np.mean([c["majority_label"] == c["lead_by_position"] for c in cells])
    assert info_wrong["label_agreement"] == agreement < 1.0
    if labels == "one":
        assert all(cell["majority_label"] == "I" for cell in info_wrong["cells"])
        assert info_wrong["strip_labels"] == ["I"] * truth.n_strips


def test_a_strip_labelled_by_its_position_keeps_the_name_of_the_table():
    # D1: the model labels the strip V5 of a 3x4 page with three strips V4.
    truth = KNOWN_LAYOUTS["3x4+II,V1,V5"]
    mask = draw_page(truth)
    label = layout_module.LEAD_LABEL_MAPPING
    bottom = int(round((1 - truth.strip_ratio(1)) * PAGE_HEIGHT + 100))
    strip = mask[bottom:] == label["V5"]
    mask[bottom:][strip] = label["V4"]
    layout, info = detect_layout(mask)
    assert layout == truth, info["reason"]
    assert info["strip_labels"] == ["II", "V1", "V4"]
    assert info["strip_rule"] == "default table II, V1, V5"


def test_strip_names_given_by_the_caller():
    six_v5 = make_layout("6x2", ["V5"])
    mask = draw_page(six_v5)
    # Without names the strip is the first of the default table, whatever its label.
    layout, info = detect_layout(mask)
    assert layout == KNOWN_LAYOUTS["6x2+II"], info["reason"]
    assert info["strip_labels"] == ["V5"]
    assert info["strip_cells"][0]["lead_by_position"] == "II"
    # Given names are used from the top, a longer list is cut.
    layout, info = detect_layout(mask, strip_names=["V5", "I", "aVF"])
    assert layout == six_v5 and info["strip_rule"] == "given"
    assert info["strip_cells"][0]["lead_by_position"] == "V5"
    # A strip may be a lead with a short cell of its own; only strips must differ.
    mask = draw_page(KNOWN_LAYOUTS["3x4+II,V1,V5"], drop_strip_short=False)
    layout, info = detect_layout(mask, strip_names=["V2", "V1", "V5"])
    assert layout.key() == "3x4+V2,V1,V5"
    with pytest.raises(ValueError):
        detect_layout(mask, strip_names=["II", "II", "V1"])
    with pytest.raises(ValueError):
        detect_layout(mask, strip_names=["II", "V7", "V1"])
    with pytest.raises(ValueError):
        detect_layout(mask, strip_names=["II", "V1"])
    # The standard page stays STANDARD whatever names are given.
    layout, info = detect_layout(draw_page(STANDARD), strip_names=["V1"])
    assert layout == STANDARD and info["standard"]
    # A page without strips takes no name.
    layout, info = detect_layout(draw_page(KNOWN_LAYOUTS["6x2+none"]), strip_names=[])
    assert layout == KNOWN_LAYOUTS["6x2+none"] and info["strip_rule"] == "no strip"


def test_a_page_missing_one_lead_is_detected():
    truth = KNOWN_LAYOUTS["6x2+II"]
    layout, info = detect_layout(draw_page(truth, skip=("V3",)))
    assert layout == truth, info["reason"]
    assert info["n_cells"] == 10
    missing = {cell["lead_by_position"] for cell in info["missing_cells"]}
    assert missing == {"II", "V3"}
    layout, info = detect_layout(draw_page(STANDARD, skip=("aVF",)))
    assert layout == STANDARD and info["n_cells"] == 10


def test_two_merged_cells_are_reported():
    truth = KNOWN_LAYOUTS["3x4+II,V1,V5"]
    layout, info = detect_layout(draw_page(truth, bridge=(2, 1)))
    assert layout == truth, info["reason"]
    merged = {(cell["row"], cell["column"]) for cell in info["cells"] if cell["merged"]}
    assert merged == {(2, 1), (2, 2)}
    assert info["merged"] == 2


def test_straight_traces_touch_and_still_detect():
    truth = KNOWN_LAYOUTS["6x2+none"]
    layout, info = detect_layout(draw_page(truth, amplitude=0.0, step=0.0))
    assert layout == truth, info["reason"]
    # Every cell touches its neighbour: a cut by pieces of the mask would merge them.
    assert info["merged"] == info["n_cells"] == 12


def test_a_page_of_another_size_is_detected():
    for key in ("3x4+II", "6x2+II", "12x1+none"):
        truth = layout_from_key(key)
        mask = draw_page(truth, height=850, width=1100, amplitude=7.0, step=6.0)
        layout, info = detect_layout(mask)
        expected = STANDARD if key == "3x4+II" else truth
        assert layout == expected, (key, info["reason"])


def test_a_smaller_page_inside_the_mask():
    # The standard page is not held to the page frame; another layout is.
    layout, info = detect_layout(draw_page(STANDARD, scale=0.8))
    assert layout == STANDARD, info["reason"]
    layout, info = detect_layout(draw_page(KNOWN_LAYOUTS["6x2+II"], scale=0.8))
    assert layout is None
    assert "off the mask" in info["reason"]


def test_12x1_with_strip_ii_without_its_short_row_is_refused():
    # Without its short entry the row of II is empty: twelve rows that are not 12x1.
    mask = draw_page(KNOWN_LAYOUTS["12x1+II"], drop_strip_short=True)
    layout, info = detect_layout(mask)
    assert layout is None
    assert info["reason"].startswith("12 rows")


def test_the_standard_page_without_its_strip_is_not_3x4_without_strip():
    mask = draw_page(STANDARD)
    strip_row = int(round((1 - STANDARD.strip_ratio(0)) * PAGE_HEIGHT))
    mask[strip_row - 60 :, :] = 0
    layout, info = detect_layout(mask)
    assert layout is None
    assert "off the mask" in info["reason"]


def test_an_empty_mask_is_refused():
    layout, info = detect_layout(np.zeros((PAGE_HEIGHT, PAGE_WIDTH), np.uint8))
    assert layout is None and info["reason"] == "empty mask"


def test_random_blobs_are_refused():
    rng = np.random.RandomState(3)
    mask = np.zeros((PAGE_HEIGHT, PAGE_WIDTH), np.uint8)
    for _ in range(5):
        y, x = rng.randint(100, PAGE_HEIGHT - 300), rng.randint(100, PAGE_WIDTH - 300)
        h, w = rng.randint(20, 200), rng.randint(20, 200)
        mask[y : y + h, x : x + w] = rng.randint(1, 13)
    layout, info = detect_layout(mask)
    assert layout is None
    assert info["reason"]


def test_a_mask_with_a_leading_axis_is_read():
    mask = draw_page(KNOWN_LAYOUTS["6x2+II"])
    layout, info = detect_layout(mask[None])
    assert layout == KNOWN_LAYOUTS["6x2+II"], info["reason"]
    layout, info = detect_layout(np.zeros((2, 3, 4), np.uint8))
    assert layout is None and "two axes" in info["reason"]


def test_info_of_a_detected_page():
    layout, info = detect_layout(draw_page(KNOWN_LAYOUTS["6x2+II"]))
    assert info["row_pitch_px"] == pytest.approx(PAGE_HEIGHT / 9, rel=0.02)
    assert info["column_pitch_px"] == pytest.approx(125 * PX_PER_MM, rel=0.01)
    assert info["x_extent"][0] == pytest.approx(15 * PX_PER_MM, abs=2)
    assert abs(info["page_top"]) < 0.01 and abs(info["page_bottom"]) < 0.01
    assert info["strip_rule"] == "default table II, V1, V5"
    assert info["strip_labels"] == ["II"]
    cell = info["cells"][0]
    for key in ("row", "column", "x0", "x1", "y0", "y1", "area", "lead_by_position",
                "majority_label", "merged"):
        assert key in cell
    assert set(info["candidates"]) == {"6x2+1"}


def test_cells_keep_their_band_and_span():
    # The cut of the digitiser reads them: the bands of a row and the equal columns of
    # the trace width, which the ink box of a cell lies in.
    truth = KNOWN_LAYOUTS["6x2+II"]
    layout, info = detect_layout(draw_page(truth))
    x0, x1 = info["x_extent"]
    for cell in info["cells"] + info["missing_cells"]:
        (ya, yb), (xa, xb) = cell["band"], cell["span"]
        assert x0 <= xa < xb <= x1 and ya < yb
        if cell["area"]:
            assert xa <= cell["x0"] < cell["x1"] <= xb
            assert ya <= cell["y0"] < cell["y1"] <= yb
    spans = sorted({tuple(cell["span"]) for cell in info["cells"]})
    assert spans[0][0] == x0 and spans[-1][1] == x1 and spans[0][1] == spans[1][0]
    assert info["strip_cells"][0]["span"] == [x0, x1]
    assert layout_module.strips_with_trace(info) == ["II"]
    # A strip without ink is no strip with a trace.
    strip = info["strip_cells"][0]
    strip["coverage"] = 0.0
    assert layout_module.strips_with_trace(info) == []
