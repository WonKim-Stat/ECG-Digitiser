"""Page layouts of a 12-lead ECG print: the table of leads and its row geometry.

A printed 12-lead ECG puts short leads in a grid of R rows and C columns, every
column a window of L = 10 / C seconds, and S rhythm strips of 10 s below them. The
digitiser was written for one of them, the standard 3x4 page with the strip II, and
keeps its numbers in constants (config.Y_SHIFT_RATIO, STANDARD_LEAD_OFFSETS_SEC,
LAYOUT_ROW, ... in src/run/digitize.py). Layout holds the same numbers for any table
of leads, and STANDARD reproduces the constants exactly (tests/test_layout.py).

The row geometry is that of the image generator (ecg-image-kit, ecg_plot.py, and
its extension for other layouts): with rows = R + S rows of leads the page is cut
into rows + 2 row heights, short row r (0 = top) has its baseline
R + max(S, 1) - 0.5 - r row heights above the bottom edge of the page, and strip k
(0 = top strip) half a row height plus S - 1 - k row heights above it, plus 0.3 cm
of the 21.59 cm page height. Without a strip the lowest short row stays at 1.5 row
heights. Ratios are fractions of the page height counted from the bottom, as in
config.Y_SHIFT_RATIO: pixel row = (1 - ratio) * page height.

detect_layout() finds the layout of a page from its label mask, by the geometry of
the traces only (see the comment above it). Nothing here imports torch or reads a
file.
"""
import numpy as np

from config import FREQUENCY, LEAD_LABEL_MAPPING, LONG_SIGNAL_LENGTH_SEC

# The twelve leads in the order of the label numbers of the segmentation model.
LEADS = tuple(LEAD_LABEL_MAPPING)
# Lead name of every label number of the mask.
LEAD_OF_LABEL = {value: lead for lead, value in LEAD_LABEL_MAPPING.items()}
# Paper speed of the print in millimetres per second.
PAPER_SPEED_MM_PER_SEC = 25
# Height of the printed page in centimetres (US Letter, landscape).
PAGE_HEIGHT_CM = 21.59
# The generator draws a rhythm strip this many centimetres above half a row height.
STRIP_OFFSET_CM = 0.3

# Tables of the short leads, rows from the top.
LEAD_TABLES = {
    "3x4": (
        ("I", "aVR", "V1", "V4"),
        ("II", "aVL", "V2", "V5"),
        ("III", "aVF", "V3", "V6"),
    ),
    "6x2": (
        ("I", "V1"),
        ("II", "V2"),
        ("III", "V3"),
        ("aVR", "V4"),
        ("aVL", "V5"),
        ("aVF", "V6"),
    ),
    "12x1": tuple((lead,) for lead in LEADS),
}
# Most rhythm strips make_layout() puts on a page.
MAX_STRIPS = 3


class Layout:
    """A table of short leads in rows and columns and the rhythm strips below it.

    name is the name of the table ("3x4", "6x2", "12x1"), columns the number C of
    columns, lead_rows the rows of the table from the top (every row C lead names),
    strips the leads of the rhythm strips from the top (may be empty).
    """

    def __init__(self, name, columns, lead_rows, strips):
        self.name = str(name)
        self.columns = int(columns)
        self.lead_rows = tuple(tuple(row) for row in lead_rows)
        self.strips = tuple(strips)
        if self.columns < 1:
            raise ValueError(f"a layout needs at least one column, not {columns}")
        names = [lead for row in self.lead_rows for lead in row]
        for row in self.lead_rows:
            if len(row) != self.columns:
                raise ValueError(
                    f"row {list(row)} of layout {self.name} has {len(row)} leads, "
                    f"not {self.columns}"
                )
        for lead in names + list(self.strips):
            if lead not in LEAD_LABEL_MAPPING:
                raise ValueError(f"unknown lead {lead!r} in layout {self.name}")
        if len(set(names)) != len(names):
            raise ValueError(f"a lead appears twice in the table of layout {self.name}")
        if len(set(self.strips)) != len(self.strips):
            raise ValueError(f"a rhythm strip appears twice in layout {self.name}")

    # The numbers of the table.
    @property
    def n_short_rows(self):
        """R, the number of rows of short leads."""
        return len(self.lead_rows)

    @property
    def n_strips(self):
        """S, the number of rhythm strips."""
        return len(self.strips)

    @property
    def num_rows(self):
        """All rows of leads, short rows and strips (NUM_ROWS of the digitiser)."""
        return self.n_short_rows + self.n_strips

    @property
    def rhythm_row(self):
        """Row index of the first strip, the row below the short rows (RHYTHM_ROW)."""
        return self.n_short_rows

    @property
    def short_leads(self):
        """The leads of the short cells, row by row from the top."""
        return tuple(lead for row in self.lead_rows for lead in row)

    @property
    def seconds_per_column(self):
        """L, the window of one column in seconds."""
        return LONG_SIGNAL_LENGTH_SEC / self.columns

    @property
    def samples_per_column(self):
        """Samples of one column window at the sampling frequency."""
        return int(round(self.seconds_per_column * FREQUENCY))

    @property
    def page_pitch_ratio(self):
        """Width of one column relative to the page height (PAGE_PITCH_RATIO)."""
        return PAPER_SPEED_MM_PER_SEC * self.seconds_per_column / 10 / PAGE_HEIGHT_CM

    @property
    def grid_lines_per_column(self):
        """Printed 1 mm grid lines in one column (GRID_LINES_PER_COLUMN)."""
        return PAPER_SPEED_MM_PER_SEC * self.seconds_per_column

    # Where a lead is.
    def column_of(self, lead):
        """Column of the short cell of a lead, None if it has none."""
        for row in self.lead_rows:
            if lead in row:
                return row.index(lead)
        return None

    def row_of(self, lead):
        """Row of the short cell of a lead (top = 0), None if it has none."""
        for r, row in enumerate(self.lead_rows):
            if lead in row:
                return r
        return None

    def strip_row(self, k):
        """Row index of strip k (0 = top strip)."""
        if not 0 <= k < self.n_strips:
            raise IndexError(f"layout {self.key()} has no strip {k}")
        return self.n_short_rows + k

    def offset_sec(self, lead):
        """Start of the short cell of a lead in seconds, None if it has none."""
        column = self.column_of(lead)
        if column is None:
            return None
        return column * self.seconds_per_column

    # Row geometry of the generator, as fractions of the page height from the bottom.
    def row_ratio(self, r):
        """Baseline of short row r (top = 0) above the bottom edge of the page."""
        R, S = self.n_short_rows, self.n_strips
        if not 0 <= r < R:
            raise IndexError(f"layout {self.key()} has no short row {r}")
        return (R + max(S, 1) - 0.5 - r) / (R + S + 2)

    def strip_ratio(self, k):
        """Baseline of strip k (0 = top strip) above the bottom edge of the page."""
        R, S = self.n_short_rows, self.n_strips
        if not 0 <= k < S:
            raise IndexError(f"layout {self.key()} has no strip {k}")
        return (0.5 + (S - 1 - k)) / (R + S + 2) + STRIP_OFFSET_CM / PAGE_HEIGHT_CM

    def row_ratios(self):
        """Baseline ratios of all rows, short rows then strips, from the top."""
        ratios = [self.row_ratio(r) for r in range(self.n_short_rows)]
        ratios += [self.strip_ratio(k) for k in range(self.n_strips)]
        return ratios

    # The dicts of the digitiser.
    def y_shift_ratio(self):
        """Baseline ratio of every lead, in the shape of config.Y_SHIFT_RATIO.

        A lead has the ratio of its short row; "full" is the first strip. With more
        than one strip every strip also has its own key "full:<lead>".
        """
        ratios = {lead: self.row_ratio(self.row_of(lead)) for lead in LEADS}
        if self.n_strips >= 1:
            ratios["full"] = self.strip_ratio(0)
        if self.n_strips >= 2:
            for k, lead in enumerate(self.strips):
                ratios[f"full:{lead}"] = self.strip_ratio(k)
        return ratios

    def lead_offsets_sec(self):
        """Start of every short cell in seconds (STANDARD_LEAD_OFFSETS_SEC)."""
        return {lead: self.offset_sec(lead) for lead in LEADS}

    def layout_row(self):
        """Row of every short lead from the top (LAYOUT_ROW)."""
        return {lead: r for r, row in enumerate(self.lead_rows) for lead in row}

    def key(self):
        """Short name, e.g. 3x4+II, 6x2+none, 3x4+II,V1,V5."""
        strips = ",".join(self.strips) if self.strips else "none"
        return f"{self.name}+{strips}"

    def __eq__(self, other):
        if not isinstance(other, Layout):
            return NotImplemented
        return (self.name, self.columns, self.lead_rows, self.strips) == (
            other.name,
            other.columns,
            other.lead_rows,
            other.strips,
        )

    def __hash__(self):
        return hash((self.name, self.columns, self.lead_rows, self.strips))

    def __repr__(self):
        return f"Layout({self.key()})"


def make_layout(name, strips):
    """Layout of one of the tables of LEAD_TABLES with the given rhythm strips.

    strips is a list of 0 to MAX_STRIPS lead names, top to bottom; repeats and
    unknown leads are refused (ValueError).
    """
    if name not in LEAD_TABLES:
        raise ValueError(f"unknown layout {name!r}, not one of {sorted(LEAD_TABLES)}")
    strips = list(strips)
    if len(strips) > MAX_STRIPS:
        raise ValueError(f"at most {MAX_STRIPS} rhythm strips, not {len(strips)}")
    for lead in strips:
        if lead not in LEAD_LABEL_MAPPING:
            raise ValueError(f"unknown rhythm strip lead {lead!r}")
    if len(set(strips)) != len(strips):
        raise ValueError(f"a rhythm strip lead is repeated: {strips}")
    rows = LEAD_TABLES[name]
    return Layout(name, len(rows[0]), rows, strips)


def layout_from_key(key):
    """Layout of a key as Layout.key() writes it, e.g. 3x4+II or 6x2+none."""
    if "+" not in key:
        raise ValueError(f"a layout key is <table>+<strips>, not {key!r}")
    name, strips = key.split("+", 1)
    strips = [] if strips == "none" else strips.split(",")
    return make_layout(name, strips)


# The standard page of the digitiser: 3x4 with the rhythm strip II.
STANDARD = make_layout("3x4", ["II"])
# The layouts the digitiser knows by name, key -> Layout.
KNOWN_STRIPS = ([], ["II"], ["V1"], ["V5"], ["II", "V1", "V5"])
KNOWN_LAYOUTS = {}
for _name in ("3x4", "6x2", "12x1"):
    for _strips in KNOWN_STRIPS:
        _layout = make_layout(_name, _strips)
        KNOWN_LAYOUTS[_layout.key()] = _layout
del _name, _strips, _layout


# LAYOUT DETECTION FROM THE MASK
#
# The label numbers of a mask do not decide the layout: the segmentation model labels
# a trace by its position on a 3x4 page, so on a 6x2 or 12x1 page most labels are
# wrong (D1: 56 % and 86 % of the short traces). What the model gets right is where
# the traces are, so the layout is read off mask > 0 alone.
#
# The cells of a row are not separate pieces of the mask: one lead ends where the
# next one starts, and where the two traces are at about the same height they touch
# (on the D1 oracle masks a third to a half of the column boundaries, and after
# closing gaps of 1 % of the page width about 70 %). A trace also falls apart at its
# steep QRS strokes. So the rows are found, and the number of rows names the table:
#  1. Rows (_find_rows): the number of trace pixels in every pixel row, smoothed over
#     ROW_SMOOTH of the page height, has a peak at the baseline of every row of leads,
#     because a trace spends most of its time near its baseline. A peak is a row when
#     its band has ink over enough of the trace width; the trace width is where the
#     long rows start and end.
#  2. Candidates: 3 to 6 rows are 3x4 with 0 to 3 strips, 6 to 9 rows are 6x2 with 0
#     to 3 strips, 12 to 15 rows are 12x1 with 0 to 3 strips, the strips at the bottom
#     (six rows are 3x4 with three strips or 6x2 without one). A strip is not told
#     from a short row by its look: in the mask a strip and a row of short leads that
#     touch each other are alike. It is told by the number of rows and where they are.
#  3. Checks of every candidate (_check_candidate):
#     - the baselines fit the row geometry of the layout (Layout.row_ratios) up to an
#       offset and a scale within ROW_FIT_TOLERANCE row heights: the rows are evenly
#       spaced, and a strip sits 0.3 cm closer to the row above it;
#     - the trace width is 250 mm on a page of 215.9 mm: the width over the page
#       height that the fit gives is within WIDTH_RATIO_TOLERANCE of that;
#     - a layout other than the standard one must fill the mask as the generator
#       fills the page: the page top and bottom that the fit gives are within
#       PAGE_FRAME_TOLERANCE of the edges of the mask. The two layouts of six rows put
#       their rows 0.13 page heights apart, and a page that lost its bottom row ends
#       short by more than a row height. The standard page is not held to it: it is
#       what the digitiser reads anyway, also on a photograph whose page is a part of
#       the picture.
#     The page gets the one candidate that passes, and None when none or two do.
#  4. The cells are the row bands cut into C equal columns of the trace width; at
#     least MIN_CELL_FRACTION of the short cells must have a trace. A cell is named
#     by the table of the layout (lead_by_position). Only now are the labels read, to
#     report them: the most frequent label of a cell (majority_label) and the share
#     of the cells where it is the lead of the table (label_agreement).
#  5. The strips are named by the table too, never by their labels: the geometry
#     cannot tell which lead a strip is, and the model labels a strip by its position
#     (D1: the strip V5 of a 3x4 page with three strips labelled V4 or II). They are
#     DEFAULT_STRIP_NAMES from the top, or the names the caller gives (strip_names).
#     Their labels are reported in info["strip_labels"].
# A page of three rows of four columns and one strip is the standard page of the
# digitiser and gets STANDARD, whatever its strip is labelled (info["standard"]).

# Smoothing window of the row profile, relative to the page height.
ROW_SMOOTH = 0.01
# Two peaks of the row profile closer than this, relative to the page height, are one
# peak. The closest rows of the known layouts are 0.045 apart (12x1 with 3 strips).
ROW_MIN_DISTANCE = 0.025
# Peaks lower than this share of the highest peak are not looked at (the strokes of
# the QRS complexes make peaks of about 0.01 to 0.1 of it).
ROW_MIN_PEAK = 0.1
# Peaks closer than this share of the page height to its top or bottom edge are not
# rows: no layout puts a trace there.
ROW_EDGE = 0.01
# Half height of the narrow band around a peak, relative to the page height (about
# 25 px on a page of 1700 px).
ROW_BAND = 0.015
# Share of the trace width the narrow band of a peak has ink in (the QRS strokes
# between two rows about 0.03 to 0.1, rows 0.6 to 1).
ROW_MIN_PEAK_COVERAGE = 0.3
# Peaks closer than this share of their median spacing are one row (a trace whose
# baseline wanders; the closest rows of the known layouts are 0.76 of it apart).
ROW_MERGE = 0.45
# Share of the trace width a row has ink in, in its band halfway to the rows next to
# it. A row of the standard page without its short II has 0.75 of it.
ROW_MIN_COVERAGE = 0.4
# A run of inked columns starts or ends a row when at least half of the columns of
# this window, relative to the page width, are inked.
EXTENT_WINDOW = 0.01
# Rows with at least this share of the inked columns of the most inked row set the
# trace width (the median of their starts and of their ends).
EXTENT_MIN_SHARE = 0.5
# Least trace width, relative to the width of the mask. The traces of a page span 250
# of its 279.4 mm; a photograph may show the page smaller.
MIN_TRACE_WIDTH = 0.3
# Largest distance of a baseline from the fitted row geometry, in row heights. A lead
# drawn with an offset moves its row: on the 12x1 oracle pages up to 0.14 (17 px).
ROW_FIT_TOLERANCE = 0.15
# Trace width over page height of the generator: 10 s at 25 mm/s on 215.9 mm.
WIDTH_RATIO = LONG_SIGNAL_LENGTH_SEC * PAPER_SPEED_MM_PER_SEC / (10 * PAGE_HEIGHT_CM)
# Largest relative deviation from WIDTH_RATIO (scanned pages up to 0.06; a page read
# with one row too few is off by 1 / (rows found + 2), 0.14 with five rows found).
WIDTH_RATIO_TOLERANCE = 0.1
# Largest distance of the fitted page top and bottom from the edges of the mask,
# relative to its height, for a layout other than the standard one (generated pages
# 0.015, scans 0.05).
PAGE_FRAME_TOLERANCE = 0.06
# Share of the short cells that must have a trace (11 of 12 on a standard page, whose
# short II the model leaves out).
MIN_CELL_FRACTION = 2 / 3
# A cell has a trace when this share of its columns has ink.
CELL_MIN_COVERAGE = 0.2
# Half width of the window across a column boundary in which a trace is followed to
# tell whether two cells touch, relative to the page height (5 px of 1700 px).
BOUNDARY_HALF_WIDTH = 0.003
# The tables and their rows of short leads, for the candidates.
TABLES_BY_ROWS = (("3x4", 3), ("6x2", 6), ("12x1", 12))
# Names of the strips from the top when the caller gives none: the table of the
# known layouts (KNOWN_STRIPS).
DEFAULT_STRIP_NAMES = ("II", "V1", "V5")


def _running_mean(values, window):
    """Centred running mean of a 1-D array (window >= 1, the edges padded with 0)."""
    window = max(1, int(window))
    return np.convolve(values, np.ones(window) / window, mode="same")


def _row_peaks(binary):
    """Peaks of the row profile: [(y, height)], top to bottom."""
    height = binary.shape[0]
    profile = _running_mean(binary.sum(axis=1).astype(float), round(ROW_SMOOTH * height))
    top = profile.max()
    if top <= 0:
        return []
    distance = max(1, int(round(ROW_MIN_DISTANCE * height)))
    edge = ROW_EDGE * height
    peaks = []
    for y in np.flatnonzero(profile >= ROW_MIN_PEAK * top):
        if y < edge or y > height - 1 - edge:
            continue
        # A peak is the highest point within +-distance; on a plateau the first one.
        lo, hi = max(0, y - distance), min(height, y + distance + 1)
        if profile[y] < profile[lo:hi].max():
            continue
        if peaks and y - peaks[-1][0] <= distance:
            continue
        peaks.append((int(y), float(profile[y])))
    return peaks


def _band(y, half, height):
    """Pixel rows [y0, y1) of the band of half height `half` around row y."""
    return max(0, int(round(y - half))), min(height, int(round(y + half)) + 1)


def _ink_ends(inked, window):
    """First and last column of the ink of a row, specks left out.

    A column starts the row when at least half of the `window` columns from it on are
    inked, and ends it when at least half of the `window` columns up to it are.
    None if there is no such column.
    """
    window = max(1, int(window))
    n = len(inked)
    if n < window:
        return None
    sums = np.concatenate([[0], np.cumsum(inked.astype(int))])
    # counts[i] = inked columns in [i, i + window).
    counts = sums[window:] - sums[:-window]
    starts = np.flatnonzero(inked[: n - window + 1] & (counts * 2 >= window))
    ends = np.flatnonzero(inked[window - 1 :] & (counts * 2 >= window)) + window - 1
    if len(starts) == 0 or len(ends) == 0:
        return None
    return int(starts[0]), int(ends[-1])


def _extent(rows):
    """Trace width [x0, x1) of rows: the median start and end of the long rows.

    A long row has at least EXTENT_MIN_SHARE of the inked columns of the most inked
    row; the median leaves out a row that misses its first or last cell. A row
    without a run of ink (ends None) does not count. (0, 0) when no row does.
    """
    rows = [row for row in rows if row["ends"] is not None]
    if not rows:
        return 0, 0
    most = max(int(row["inked"].sum()) for row in rows)
    long_rows = [row for row in rows if row["inked"].sum() >= EXTENT_MIN_SHARE * most]
    x0 = int(np.median([row["ends"][0] for row in long_rows]))
    x1 = int(np.median([row["ends"][1] for row in long_rows])) + 1
    return x0, x1


def _find_rows(binary):
    """Rows of traces of a page: (rows, (x0, x1)), or ([], reason).

    Every row is a dict with its baseline y (a peak of the row profile), its band
    [y0, y1) (halfway to the rows next to it) and the share of the trace width
    [x0, x1) its band has ink in (coverage). Three passes:
      1. the peaks of the row profile whose narrow band (+-ROW_BAND) has ink over
         ROW_MIN_PEAK_COVERAGE of the trace width; the peaks of the QRS strokes
         between two rows do not;
      2. a trace whose baseline wanders can make two peaks: peaks closer than
         ROW_MERGE of the median spacing are one row, at the higher peak;
      3. the band of every row reaches halfway to the next row; the ink of these
         bands sets the trace width, and the row with the least coverage is dropped
         while it is below ROW_MIN_COVERAGE.
    """
    height, page_width = binary.shape
    window = max(1, int(round(EXTENT_WINDOW * page_width)))
    half_band = max(1.0, ROW_BAND * height)

    # Pass 1.
    peaks = []
    for y, peak_height in _row_peaks(binary):
        y0, y1 = _band(y, half_band, height)
        inked = binary[y0:y1].any(axis=0)
        ends = _ink_ends(inked, window)
        if ends is not None:
            peaks.append({"y": y, "height": peak_height, "inked": inked, "ends": ends})
    if not peaks:
        return [], "no row of traces"
    x0, x1 = _extent(peaks)
    if x1 - x0 < MIN_TRACE_WIDTH * page_width:
        return [], f"traces only {x1 - x0} of {page_width} px wide"
    peaks = [p for p in peaks if p["inked"][x0:x1].mean() >= ROW_MIN_PEAK_COVERAGE]
    if not peaks:
        return [], "no row of traces"

    # Pass 2.
    if len(peaks) > 1:
        spacing = float(np.median(np.diff([p["y"] for p in peaks])))
        kept = []
        for p in sorted(peaks, key=lambda p: -p["height"]):
            if all(abs(p["y"] - q["y"]) >= ROW_MERGE * spacing for q in kept):
                kept.append(p)
        peaks = sorted(kept, key=lambda p: p["y"])

    # Pass 3.
    rows = [{"y": p["y"]} for p in peaks]
    while rows:
        ys = [row["y"] for row in rows]
        for k, row in enumerate(rows):
            above = (ys[k] - ys[k - 1]) / 2 if k > 0 else None
            below = (ys[k + 1] - ys[k]) / 2 if k < len(rows) - 1 else None
            if above is None:
                above = below if below is not None else half_band
            if below is None:
                below = above
            y0 = max(0, int(round(ys[k] - above)))
            y1 = min(height, int(round(ys[k] + below)))
            row["band"] = (y0, y1)
            row["inked"] = binary[y0:y1].any(axis=0)
            row["ends"] = _ink_ends(row["inked"], window)
        x0, x1 = _extent(rows)
        if x1 - x0 < MIN_TRACE_WIDTH * page_width:
            return [], f"traces only {x1 - x0} of {page_width} px wide"
        for row in rows:
            row["coverage"] = float(row["inked"][x0:x1].mean())
        weakest = min(rows, key=lambda row: row["coverage"])
        if weakest["coverage"] >= ROW_MIN_COVERAGE:
            break
        rows.remove(weakest)
    if not rows:
        return [], "no row of traces"
    for row in rows:
        del row["inked"], row["ends"]
    return rows, (x0, x1)


def _candidates(n_rows):
    """Layouts (strips named by DEFAULT_STRIP_NAMES for now) with n_rows rows."""
    found = []
    for name, short_rows in TABLES_BY_ROWS:
        strips = n_rows - short_rows
        if 0 <= strips <= MAX_STRIPS:
            found.append(make_layout(name, DEFAULT_STRIP_NAMES[:strips]))
    return found


def _fit_rows(rows_y, layout):
    """Fit the baselines to the row geometry of a layout: (residuals, row height, top).

    The expected baseline of row k is e_k = (1 - ratio_k) * (rows + 2) row heights
    below the top of the page; the fit is rows_y = top + row_height * e_k by least
    squares. residuals are the distances from the fit in row heights.
    """
    total = layout.num_rows + 2
    expected = np.array([(1 - ratio) * total for ratio in layout.row_ratios()])
    observed = np.asarray(rows_y, float)
    row_height, top = np.polyfit(expected, observed, 1)
    if row_height <= 0:
        return np.full(len(observed), np.inf), float(row_height), float(top)
    residuals = np.abs(observed - (top + row_height * expected)) / row_height
    return residuals, float(row_height), float(top)


def _check_candidate(layout, rows, extent, height):
    """Problems of a candidate layout for the rows of a page: (problems, numbers)."""
    ys = [row["y"] for row in rows]
    residuals, row_height, top = _fit_rows(ys, layout)
    width = extent[1] - extent[0]
    page_height = row_height * (layout.num_rows + 2)
    numbers = {
        "fit_residual": float(residuals.max()),
        "row_height": row_height,
        "page_top": top / height,
        "page_bottom": (top + page_height - height) / height,
        "width_ratio": width / page_height if page_height > 0 else float("inf"),
    }
    problems = []
    if numbers["fit_residual"] > ROW_FIT_TOLERANCE:
        problems.append(
            f"rows off the row geometry by {numbers['fit_residual']:.2f} row heights"
        )
    if abs(numbers["width_ratio"] / WIDTH_RATIO - 1) > WIDTH_RATIO_TOLERANCE:
        problems.append(
            f"trace width {numbers['width_ratio']:.3f} page heights, "
            f"not {WIDTH_RATIO:.3f}"
        )
    is_standard = (layout.name, layout.n_strips) == (STANDARD.name, STANDARD.n_strips)
    if not is_standard:
        off = max(abs(numbers["page_top"]), abs(numbers["page_bottom"]))
        if off > PAGE_FRAME_TOLERANCE:
            problems.append(
                f"page top {numbers['page_top']:+.3f} / bottom "
                f"{numbers['page_bottom']:+.3f} off the mask"
            )
    return problems, numbers


def _connected_across(binary, y0, y1, xa, xb):
    """Whether an 8-connected path of ink runs from column xa to column xb in [y0, y1).

    The ink is followed column by column: a run of ink in a column is reached when
    one of its pixels touches a reached pixel of the column before (8-neighbours).
    """
    if xa < 0 or xb >= binary.shape[1] or y1 <= y0:
        return False
    reached = binary[y0:y1, xa].copy()
    for x in range(xa + 1, xb + 1):
        column = binary[y0:y1, x]
        near = reached.copy()
        near[1:] |= reached[:-1]
        near[:-1] |= reached[1:]
        # Number the runs of ink of this column and keep the runs that touch.
        starts = column & ~np.concatenate([[False], column[:-1]])
        run = np.cumsum(starts) * column
        touched = np.unique(run[column & near])
        reached = column & np.isin(run, touched[touched > 0])
        if not reached.any():
            return False
    return True


def _majority_label(labels):
    """Most frequent nonzero label of an array of labels as a lead name, else None."""
    values = labels[labels > 0]
    if values.size == 0:
        return None
    return LEAD_OF_LABEL.get(int(np.argmax(np.bincount(values.astype(np.int64)))))


def _cell(binary, labels, y0, y1, xa, xb):
    """Ink of the region [y0, y1) x [xa, xb): bounding box, area, coverage, label.

    band and span keep the region itself, which is what the digitiser cuts the mask
    by (src/run/digitize.py, layout_label_mask()); x0, x1, y0, y1 are the box of the
    ink inside it.
    """
    region = binary[y0:y1, xa:xb]
    cell = {
        "band": [int(y0), int(y1)],
        "span": [int(xa), int(xb)],
        "x0": None,
        "x1": None,
        "y0": None,
        "y1": None,
        "area": int(region.sum()),
        "coverage": float(region.any(axis=0).mean()) if xb > xa else 0.0,
        "majority_label": None,
    }
    if cell["area"]:
        rows_inked = np.flatnonzero(region.any(axis=1))
        columns_inked = np.flatnonzero(region.any(axis=0))
        cell["x0"] = xa + int(columns_inked[0])
        cell["x1"] = xa + int(columns_inked[-1]) + 1
        cell["y0"] = y0 + int(rows_inked[0])
        cell["y1"] = y0 + int(rows_inked[-1]) + 1
        cell["majority_label"] = _majority_label(labels[y0:y1, xa:xb])
    return cell


def _refuse(info, reason):
    info["reason"] = reason
    return None, info


def detect_layout(mask, strip_names=None):
    """Layout of a page from its label mask, by the geometry of the traces only.

    mask is the label mask of the page (H x W, 0 = background, 1..12 = leads; a
    leading axis of length 1 is dropped). strip_names names the strips of a layout
    other than the standard one, from the top: a list of lead names without repeats
    and at least as long as the strips found (ValueError otherwise; a strip may be a
    lead that also has a short cell). None names them DEFAULT_STRIP_NAMES. Returns
    (layout, info): layout is a Layout, or None when the page does not show one of
    the known tables clearly, and info says what was found and, for None, why. The
    rules are in the comment above ROW_SMOOTH. The keys of info:
      reason          why the page has no layout, "" when it has one
      standard        True for three rows of four columns and one strip (STANDARD)
      columns, rows, strips   C, R and S of the layout found (None if refused early)
      rows_y          baselines of all rows found, pixel rows from the top
      x_extent        [x0, x1), the columns of the traces
      candidates      per candidate "<table>+<strips>": its problems and numbers
      fit_residual, width_ratio, page_top, page_bottom   numbers of the layout found
      row_pitch_px    the row height of the fit; column_pitch_px the column width
      cells           short cells with a trace: dicts of row, column, band and span
                      (the rows [y0, y1) and columns [x0, x1) of the cell), x0, x1,
                      y0, y1 (the box of the ink), area (pixels), coverage (share of the
                      columns with ink), lead_by_position (from the table),
                      majority_label (the most frequent label, as a lead name) and
                      merged (the trace runs on into a cell next to it)
      missing_cells   the short cells without a trace, the same dicts
      n_cells, merged the number of cells and of merged cells
      strip_cells     one dict per strip, lead_by_position = its name in the layout
      strip_labels    the majority_label of every strip, top to bottom
      strip_rule      how the strips were named
      label_agreement share of the cells whose majority_label is lead_by_position
    """
    if strip_names is not None:
        strip_names = list(strip_names)
        for lead in strip_names:
            if lead not in LEAD_LABEL_MAPPING:
                raise ValueError(f"unknown strip lead {lead!r}")
        if len(set(strip_names)) != len(strip_names):
            raise ValueError(f"a strip lead is repeated: {strip_names}")
    labels = np.asarray(mask)
    if labels.ndim == 3 and labels.shape[0] == 1:
        labels = labels[0]
    info = {
        "reason": "",
        "standard": False,
        "columns": None,
        "rows": None,
        "strips": None,
        "n_cells": 0,
        "cells": [],
        "missing_cells": [],
        "strip_cells": [],
        "strip_labels": [],
        "strip_rule": "",
        "row_pitch_px": None,
        "column_pitch_px": None,
        "label_agreement": float("nan"),
        "merged": 0,
        "rows_y": [],
        "x_extent": None,
        "candidates": {},
        "fit_residual": None,
        "width_ratio": None,
        "page_top": None,
        "page_bottom": None,
    }
    if labels.ndim != 2:
        return _refuse(info, f"a mask has two axes, not the shape {labels.shape}")
    binary = labels > 0
    height = binary.shape[0]
    if not binary.any():
        return _refuse(info, "empty mask")

    # 1. Rows of traces and the trace width.
    rows, extent = _find_rows(binary)
    if not rows:
        return _refuse(info, extent)
    x0, x1 = extent
    width = x1 - x0
    info["x_extent"] = [x0, x1]
    info["rows_y"] = [row["y"] for row in rows]

    # 2. and 3. The candidates for this number of rows and their checks.
    candidates = _candidates(len(rows))
    if not candidates:
        return _refuse(
            info, f"{len(rows)} rows of traces, no known layout has that many"
        )
    passed = []
    for candidate in candidates:
        problems, numbers = _check_candidate(candidate, rows, extent, height)
        key = f"{candidate.name}+{candidate.n_strips}"
        info["candidates"][key] = {"problems": problems, **numbers}
        if not problems:
            passed.append((candidate, numbers))
    if len(passed) != 1:
        if passed:
            reason = "both " + " and ".join(info["candidates"]) + " fit"
        else:
            reason = "; ".join(
                f"{key}: {', '.join(checked['problems'])}"
                for key, checked in info["candidates"].items()
            )
        return _refuse(info, f"{len(rows)} rows: {reason}")
    candidate, numbers = passed[0]
    columns = candidate.columns
    short_rows, strips = candidate.n_short_rows, candidate.n_strips
    info["columns"], info["rows"], info["strips"] = columns, short_rows, strips
    info["fit_residual"] = numbers["fit_residual"]
    info["width_ratio"] = numbers["width_ratio"]
    info["page_top"], info["page_bottom"] = numbers["page_top"], numbers["page_bottom"]
    info["row_pitch_px"] = numbers["row_height"]
    info["column_pitch_px"] = width / columns

    # 4. Cells: the row bands cut into equal columns of the trace width.
    bounds = [int(round(x0 + width * c / columns)) for c in range(columns + 1)]
    half_width = max(2, int(round(BOUNDARY_HALF_WIDTH * height)))
    cells, missing = [], []
    for r in range(short_rows):
        y0, y1 = rows[r]["band"]
        for c in range(columns):
            cell = _cell(binary, labels, y0, y1, bounds[c], bounds[c + 1])
            cell["row"], cell["column"] = r, c
            cell["lead_by_position"] = candidate.lead_rows[r][c]
            # Merged: the trace runs on across a boundary into the cell next to it.
            inner = [bounds[c]] if c > 0 else []
            inner += [bounds[c + 1]] if c < columns - 1 else []
            cell["merged"] = any(
                _connected_across(binary, y0, y1, b - half_width, b + half_width)
                for b in inner
            )
            (cells if cell["coverage"] >= CELL_MIN_COVERAGE else missing).append(cell)
    strip_cells = []
    for k in range(strips):
        y0, y1 = rows[short_rows + k]["band"]
        cell = _cell(binary, labels, y0, y1, bounds[0], bounds[-1])
        cell["row"], cell["column"], cell["merged"] = short_rows + k, 0, False
        strip_cells.append(cell)
    info["cells"], info["missing_cells"] = cells, missing
    info["strip_cells"] = strip_cells
    info["strip_labels"] = [cell["majority_label"] for cell in strip_cells]
    info["n_cells"] = len(cells)
    info["merged"] = sum(cell["merged"] for cell in cells)
    if len(cells) < MIN_CELL_FRACTION * columns * short_rows:
        return _refuse(
            info,
            f"{len(cells)} of {columns * short_rows} short cells have a trace "
            f"(< {MIN_CELL_FRACTION:.2f})",
        )

    # 5. The strips are named by the table or by the caller, never by their labels.
    # The standard page keeps its strip II, whatever it is labelled.
    standard = (STANDARD.columns, STANDARD.n_short_rows, STANDARD.n_strips)
    if (columns, short_rows, strips) == standard:
        layout = STANDARD
        info["standard"] = True
        info["strip_rule"] = "standard: a 3x4 page with one strip has the strip II"
    elif strip_names is None:
        layout = make_layout(candidate.name, DEFAULT_STRIP_NAMES[:strips])
        info["strip_rule"] = "default table II, V1, V5" if strips else "no strip"
    else:
        if len(strip_names) < strips:
            raise ValueError(
                f"{len(strip_names)} strip names {strip_names} for a page with "
                f"{strips} strips"
            )
        layout = make_layout(candidate.name, strip_names[:strips])
        info["strip_rule"] = "given" if strips else "no strip"
    for k, cell in enumerate(strip_cells):
        cell["lead_by_position"] = layout.strips[k]
    agreeing = [cell["majority_label"] == cell["lead_by_position"] for cell in cells]
    info["label_agreement"] = float(np.mean(agreeing)) if agreeing else float("nan")
    return layout, info


def strips_with_trace(info):
    """The strips of a page that have a trace, top to bottom, as lead names.

    info is that of detect_layout(); a strip has a trace when CELL_MIN_COVERAGE of
    the trace width has ink in its band, as a short cell has.
    """
    return [
        cell["lead_by_position"]
        for cell in info["strip_cells"]
        if cell["coverage"] >= CELL_MIN_COVERAGE
    ]
