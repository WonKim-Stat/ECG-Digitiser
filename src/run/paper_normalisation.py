"""Paper normalisation: put a photographed ECG page into the frame the model reads.

The segmentation model was trained on pages that fill the image: US Letter landscape at
200 dpi, 2200 x 1700 px. A photograph shows the sheet somewhere on a table, turned,
tilted and at another scale, and the grid line stages of the digitiser cannot read it.
This module finds the paper in such an image and warps it onto the page frame BEFORE the
resolution stage, and leaves every page alone that the digitiser reads as it is. It
holds the image code only (numpy and cv2): the digitiser's own stages come in as two
callables, see normalise_page(), and nothing here reads or writes a file.

One page, decided on image signals only (normalise_page):

  pass_chain      the digitiser's pre-segmentation chain (resolution, rotation,
                  perspective, each with the grid lines) reads the page as it is: the
                  page is left as it is.
  pass_fullframe  otherwise, when there is no background to find a paper boundary in:
                  the image reaches at most FULLFRAME_EXTRA_MM beyond a Letter sheet in
                  both directions (mm from the robust extent of the printed grid region,
                  which spans PRINT_SCALE of the sheet), or no paper edge is visible
                  while the grid is within TOUCH_GAP_MM of the image border on all four
                  sides, or the paper quadrilateral found covers >= FULLFRAME_AREA of
                  the image. The page is left as it is (or scaled, below).
  normalised      otherwise, when all four paper edges are found and the frame check
                  passes: the page is warped by the homography that takes the paper
                  corners onto the corners of the 2200 x 1700 page, long side
                  horizontal, in the orientation chosen below, scaled about the page
                  centre so that the 1 mm grid has FRAME_PERIOD px along x and y (frame
                  "corrected"; "unmeasured" = a period NaN, kept as warped).
  failed          otherwise (a paper edge not found, the paper running off the image on
                  a side, an implausible quadrilateral: not convex, mirrored, long /
                  short side ratio outside ASPECT_RANGE, holding less than CONTAIN_MIN
                  of the grid region; or a frame failure: the warped grid anisotropic
                  beyond FRAME_ANISO_MAX or an axis period outside FRAME_AXIS_RANGE of
                  FRAME_PERIOD): the page is left as it is, the reason says why (or
                  scaled, below).
  scaled          a failed or pass_fullframe page whose resolution stage kept its scale
                  (no usable grid period) but whose printed grid says it is more than
                  SCALE_DEAD_BAND below 200 dpi of paper: shrunk (INTER_AREA) to 200 dpi
                  of paper, not rectified.

Paper edges (find_paper). (a) At a work scale (long side WORK_LONG px, INTER_AREA) the
printed grid region = fine texture (log10 band-pass energy, Gaussian ENERGY_SIGMAS at
full resolution) of the darkest channel above its centre median - ENERGY_DROP AND of the
red chroma R - min(G, B) above its centre median - CHROMA_DROP (the grid is red, traces
black; a textured background has little red texture; a page without red at the centre
uses the first test only), opened, closed, the component at the centre with its holes
filled. (b) Per side, NBANDS bands across it; each band profile along the outward normal
(Lab of the median blurred work image + TEXTURE_WEIGHT x both energies) is split at the
change points with the least squared deviation from the segment means, either grid |
background (the paper ends with the print) or grid | white margin | background (margin
MARGIN_MM wide, taken when it explains MODEL_GAIN of the two-segment residual; a
"margin" nearer in colour to the background beyond it than to the band's own paper white
(LOCAL_WHITE_MM) is a strip of background and its inner end is the edge); the edge must
lie at most EDGE_INSIDE_MM inside the grid region, step by >= EDGE_MIN_STEP and leave >=
EDGE_BORDER_MM of background before the image border; a deterministic RANSAC line (all
band pairs, LINE_TOL_MM) through the band edges needs LINE_MIN_INLIERS of the bands.
(c) Each side again at full resolution between the coarse corners: the strongest colour
step within REFINE_SEARCH_MM of the coarse line at REFINE_SAMPLES places (strip mean of
2 REFINE_STRIP_PX + 1 samples, sub-pixel by a parabola), a trimmed total least squares
line; a side whose steps pile up at the window's end is refined again around its new
line (REFINE_CLIP_SHARE, REFINE_MAX_PASSES); the corners are the intersections.

Homography (paper_homography, warp_to_page): paper corners -> page corners (-0.5, -0.5),
(2199.5, -0.5), (2199.5, 1699.5), (-0.5, 1699.5) in cv2 pixel coordinates (centre of
pixel (0, 0) at (0, 0)), then the frame scaling; the image first shrunk by INTER_AREA to
about the target scale (geometric mean of the smallest and largest local scale of the
homography, skipped above 0.95) and then warped once with INTER_CUBIC
(BORDER_REPLICATE). The pre-resize size and the warp matrix actually applied are enough
to build the same view again (apply_warp, replay_page).

Orientation (choose_orientation): long side horizontal; of the two landscape
orientations (0 / 180 deg, or 90 / 270 deg for a portrait photo) the one whose trace ink
lies lower on the page: the layout puts the three lead rows and the rhythm strip at
about 80-200 mm of the 216 mm sheet, the band above them is grid only. Ink = pixels
whose brightest channel is below ORIENT_INK of the local white (the red grid, shadows
and uneven light do not count), rows within ORIENT_HALF_MM above vs below the middle of
the half-size normalised page; margin = (lower - upper) / (lower + upper) of the chosen
orientation (the other one has -margin); a margin below ORIENT_LOW_MARGIN is flagged
(orient_low).

Known limits. The frame fixes the scale, not the position: an edge found at the print
boundary (no white margin found) or beyond the paper (a background like paper) leaves
the print off the page centre; the curl of the sheet is left as it is.

Where the thresholds come from. The code below is the image code of the view builder
that made the normalised photographs of the real-page results (its version 2, see
VERSION), function for function, and every threshold is the one chosen there on the
development photographs of the ECG-Image-Database (v2, part D0; "dev" in the comments:
115 PTB-XL records split by patient, 3 photo conditions, 345 pages; see the README) and
on synthetic photos of generator pages. The comments on the constants are the builder's
and say what each one was chosen on.

Deterministic: fixed parameters, no random numbers, CPU numpy / cv2 only.
"""
import cv2
import numpy as np

# The decision rules and the pixel semantics (decode, pre-resize, warp) a record of this
# module stands for, and the versions whose records replay_page() rebuilds.
VERSION = "paper_normalisation 1 (real_norm_view 2)"
REPLAYABLE = (VERSION,)

# Target page: US Letter landscape at 200 dpi, the generator's page and nnU-Net's
# training frame.
PAGE_W, PAGE_H = 2200, 1700
PAPER_MM = (279.4, 215.9)
# The ECG image is printed at about this fraction of the sheet (C2: the scans' grid is
# 0.954 x 200 dpi), so the printed grid spans about this much of the paper; only used to
# turn the grid region into a rough px/mm for the search windows, the full-frame test
# and the nominal margin of an inferred side (find_paper's infer, not used here).
PRINT_SCALE = 0.954

# Work scale of the paper search: the long side of the image in px.
WORK_LONG = 1024
# Band-pass of the texture energy (full-resolution px): the 1 mm grid of a page at 7.5
# to 13 px/mm.
ENERGY_SIGMAS = (1.0, 4.0)
# The printed grid region: energy above the centre median minus this many decades.
ENERGY_DROP = 0.6
ENERGY_SMOOTH = 2.0         # work px
# The red chroma texture (the grid is red; traces black) must be present at the centre
# (decades) to be used, and above its centre median minus CHROMA_DROP.
CHROMA_MIN = 0.3
CHROMA_DROP = 0.5
# Weight of one decade of texture energy in the edge profile, in Lab units.
TEXTURE_WEIGHT = 20.0
# The printed grid fills the frame (scans, renders, synthetic pages): the image reaches
# at most this many mm beyond a Letter sheet in either direction, the mm from the robust
# extent of the grid region (dev survey, data/real_results/c4/norm/paper_survey.txt:
# renders and scans <= 6.1 mm except one mould scan at 9.7 mm, which then fails the edge
# search and is left as it is too; photos 5th percentile 10.6-17.9 mm) ...
FULLFRAME_EXTRA_MM = 8.0
# ... or the paper quadrilateral found covers at least this fraction of the image.
FULLFRAME_AREA = 0.95
# Paper edge search per side: bands across the side, window around the grid region
# boundary.
NBANDS = 24
EDGE_OUT_MM = 25.0          # profile window: this far either side of the grid region edge
MARGIN_SEARCH_MM = 5.0      # the grid boundary within this of the grid region edge
MARGIN_MM = (2.0, 15.0)     # width range of the white print margin
# margin vs background, Lab units (texture: TEXTURE_WEIGHT / decade)
EDGE_MIN_STEP = 6.0
# the background must show at least this much before the border
EDGE_BORDER_MM = 1.0
# largest fraction of grid-textured columns in the margin segment
MARGIN_MAX_TEXTURE = 0.9
# share of the grid | background residual a margin segment must explain
MODEL_GAIN = 0.2
# the paper edge at most this far inside the grid region edge (bleed)
EDGE_INSIDE_MM = 1.5
# share of the grid region the paper quadrilateral must contain
CONTAIN_MIN = 0.97
EDGE_SIDE_MM = 3.0          # profile length either side of an edge for its contrast
LINE_TOL_MM = 1.0           # RANSAC inlier tolerance of the band steps
LINE_MIN_INLIERS = 0.5      # fraction of the bands with a step that has to be on the line
# A side without an edge whose grid region is this close (mm) to the image border: the
# paper reaches the border there (print margin <= ~7 mm plus a few mm of background at
# most).
TOUCH_GAP_MM = 12.0
# The paper colour an edge profile's "margin" is compared with is that of its own band:
# the grid region pixels within LOCAL_WHITE_MM inside the band's grid edge (the global
# paper white takes a shadowed white margin for background under uneven light); the
# global one when the band has fewer than LOCAL_WHITE_MIN_PX such pixels.
LOCAL_WHITE_MM = 30.0
LOCAL_WHITE_MIN_PX = 50
# Full-resolution refinement.
REFINE_SAMPLES = 48
REFINE_ENDS = 0.06          # fraction of the side left out at either corner
REFINE_SEARCH_MM = 1.2      # the step is looked for within this of the work-scale line
REFINE_STRIP_PX = 8         # half width of the strip averaged along the side
REFINE_STEP_PX = 6          # half window of the step detector
# A side whose steps sit at the end of the search window on REFINE_CLIP_SHARE of its
# samples was cut short by the window: it is refined again around its new line (the
# corners re-centred), at most REFINE_MAX_PASSES passes in all; the re-refined line is
# kept whatever its clipping.
REFINE_CLIP_SHARE = 0.5
REFINE_MAX_PASSES = 4
# Plausibility of the quadrilateral: long / short side ratio of the paper is 1.294 (dev
# quads 1.22-1.39 but one wrong one at 1.086, review of 2026-09-29).
ASPECT_RANGE = (1.18, 1.45)
# Orientation cue: trace ink in the ORIENT_HALF_MM above against below the middle of the
# paper height (the outer ~24 mm, header text and page border, are left out).
ORIENT_HALF_MM = 84.0
# a trace pixel is darker than this fraction of the local white
ORIENT_INK = 0.5
ORIENT_LOW_MARGIN = 0.1     # a margin below this is flagged (orient_low) in the meta
# Frame of a normalised page. The page warped from the paper corners is measured: the
# 1 mm grid period along x and along y (the digitiser's measure_grid_period on the page
# and on the page transposed, each NaN below its comb contrast 5). A found edge is
# sometimes the print boundary rather than the paper edge (no white margin found), which
# leaves the grid 4.8 % larger on that axis. The page is failed when the grid is
# anisotropic (|py / px - 1| > FRAME_ANISO_MAX: a side far off, e.g. a dark shadow over
# the header) or an axis period lies outside FRAME_AXIS_RANGE of FRAME_PERIOD;
# otherwise, both periods measured, it is scaled about its centre so that both are
# FRAME_PERIOD (composed into the homography, still one interpolation); with a period
# NaN it is kept as warped and flagged (frame "unmeasured"). FRAME_PERIOD is the paper
# frame (paper corners on the page corners, the print at PRINT_SCALE: the scans' 0.954 x
# 200 dpi); the render and nnU-Net training frame would be 200 / 25.4 = 7.874 px (a user
# decision; both are inside the digitiser's resolution dead band). Thresholds chosen on
# dev (data/real_results/c4/norm/frame_survey.txt, all 345 dev photo pages): on the 254
# quadrilaterals with both periods the correction lowers the largest corner error
# (placement oracle) from a median 5.68 to 2.59 mm and makes it worse by > 1 mm on 1;
# |py / px - 1| reaches 0.065 and the axis periods 0.945-1.073 of FRAME_PERIOD there, so
# the bars sit just outside dev (0 dev pages fail) and catch gross failures only
# (synthetic dark shadows: 0.099 / 0.161). Periods below the digitiser's contrast 5 are
# not used: corrected with them (contrast >= 3, 49 pages) the median moved only 5.90 ->
# 5.29 mm.
FRAME_PERIOD = PRINT_SCALE * 200 / 25.4
FRAME_ANISO_MAX = 0.08
FRAME_AXIS_RANGE = (0.92, 1.10)
# Scale-only fallback: a page the chain cannot read (failed, pass_fullframe or a frame
# failure) whose digitiser resolution stage kept its scale (no usable period) but whose
# printed grid region says it is at photo scale (px per paper mm from the grid extent,
# PRINT_SCALE of the sheet) more than SCALE_DEAD_BAND below 200 dpi of paper is shrunk
# (INTER_AREA) to 200 dpi of paper: decision "scaled". Dev renders and scans lie at
# 7.4-8.2 px per paper mm, inside the band.
SCALE_DEAD_BAND = 0.12      # the digitiser's RESOLUTION_DEAD_BAND
# an image axis counts for that estimate when the grid is this clear of both borders
FALLBACK_GAP_MM = 1.0
PAPER_PMM = 200 / 25.4      # target px per paper mm


# ------------------------------------------------------------------------- paper search
def work_scale(shape):
    h, w = shape[:2]
    s = WORK_LONG / max(h, w)
    return int(round(w * s)), int(round(h * s))


def band_energy(channel, size):
    """log10 band-pass energy of one full-resolution channel, at the work size (w, h)."""
    bp = cv2.GaussianBlur(channel, (0, 0), ENERGY_SIGMAS[0]) - cv2.GaussianBlur(
        channel, (0, 0), ENERGY_SIGMAS[1])
    e = cv2.resize(bp * bp, size, interpolation=cv2.INTER_AREA)
    return np.log10(cv2.GaussianBlur(e, (0, 0), ENERGY_SMOOTH) + 1.0)


def texture_energy(rgb, size):
    """Fine texture of the darkest channel and of the red chroma, at the work size.

    The darkest channel shows grid, traces, text and textured backgrounds; the red
    chroma R - min(G, B) the red grid only: traces are black, most backgrounds grey or
    brown. Both as log10 band-pass energy.
    """
    a = rgb.astype(np.float32)
    lum = band_energy(255.0 - a.min(axis=2), size)
    chroma = band_energy(a[..., 0] - np.minimum(a[..., 1], a[..., 2]), size)
    return lum, chroma


def centre_box(a, lo=0.3, hi=0.7):
    h, w = a.shape[:2]
    return a[int(lo * h):int(hi * h), int(lo * w):int(hi * w)]


def fill_component(mask):
    """The component of mask that holds most of the centre box, closed, holes filled."""
    m = mask.astype(np.uint8)
    m = cv2.morphologyEx(m, cv2.MORPH_OPEN, np.ones((3, 3), np.uint8))
    m = cv2.morphologyEx(m, cv2.MORPH_CLOSE, np.ones((9, 9), np.uint8))
    n, lab = cv2.connectedComponents(m, connectivity=4)
    counts = np.bincount(centre_box(lab).ravel(), minlength=n)
    counts[0] = 0
    if counts.max() == 0:
        return np.zeros_like(m, bool)
    comp = lab == int(np.argmax(counts))
    inv = (~comp).astype(np.uint8)
    n2, lab2 = cv2.connectedComponents(inv, connectivity=4)
    border = np.unique(np.concatenate([lab2[0], lab2[-1], lab2[:, 0], lab2[:, -1]]))
    holes = inv.astype(bool) & ~np.isin(lab2, border)
    return comp | holes


def grid_region(rgb):
    """Work-scale region of the printed grid.

    Fine texture in the darkest channel AND in the red chroma (when the page has a red
    grid at all), each above its centre median minus ENERGY_DROP / CHROMA_DROP.
    Returns (region, profile features, lum energy, lum threshold, info).
    """
    size = work_scale(rgb.shape)
    work = cv2.resize(rgb, size, interpolation=cv2.INTER_AREA)
    lum, chroma = texture_energy(rgb, size)
    c_lum = float(np.median(centre_box(lum)))
    c_chroma = float(np.median(centre_box(chroma)))
    tex = lum > c_lum - ENERGY_DROP
    red = c_chroma >= CHROMA_MIN
    if red:
        tex &= chroma > c_chroma - CHROMA_DROP
    region = fill_component(tex)
    lab = cv2.cvtColor(cv2.medianBlur(work, 5), cv2.COLOR_RGB2LAB).astype(np.float32)
    feats = np.concatenate([lab, TEXTURE_WEIGHT * lum[..., None],
                            TEXTURE_WEIGHT * chroma[..., None]], axis=2)
    info = {"energy_centre": c_lum, "chroma_centre": c_chroma, "red_grid": int(red)}
    return region, feats, lum, c_lum - ENERGY_DROP, info


SIDES = ("top", "bottom", "left", "right")


def side_view(a, side):
    """a (H x W [x C]) seen from one side: rows = along the side, columns = outward."""
    if side in ("top", "bottom"):
        a = np.swapaxes(a, 0, 1)
    if side in ("top", "left"):
        a = a[:, ::-1]
    return a


def _sse_table(csum, csq, starts, stops):
    """Sum of squared deviations from the mean of the rows [start, stop) of a profile."""
    n = (stops - starts).astype(float)
    tot = csum[stops] - csum[starts]
    return (csq[stops] - csq[starts]) - (tot * tot).sum(axis=-1) / np.maximum(n, 1)


def band_model(f, texf, b, pmm, paper):
    """Change points of one band profile f (n x features, inner side first).

    b is the grid region edge. Two models of the profile: grid | background (one change
    point x within MARGIN_SEARCH_MM of b) and grid | print margin | background (x1
    within MARGIN_SEARCH_MM of b, margin MARGIN_MM wide), each the split with the least
    squared deviation from the segment means. The margin model is taken when it explains
    MODEL_GAIN more of the two-segment residual (a margin is a segment of its own); its
    margin must not be grid textured through, and when its colour is nearer the
    background beyond it than the paper colour (Lab of the grid region, paper) it is a
    strip of background and the edge is its inner end (model 31). The edge must step by
    >= EDGE_MIN_STEP between EDGE_SIDE_MM of profile either side. Returns a dict with
    edge (the profile index the paper edge lies before, or None), margin_mm, contrast,
    gain, model.
    """
    n = len(f)
    m_lo = max(2, int(round(MARGIN_MM[0] * pmm)))
    m_hi = int(round(MARGIN_MM[1] * pmm))
    search = int(round(MARGIN_SEARCH_MM * pmm))
    side_px = max(2, int(round(EDGE_SIDE_MM * pmm)))
    min_bg = max(2, int(round(1.0 * pmm)))
    out = {"edge": None}
    if n < 3 * m_lo + min_bg:
        return out
    csum = np.concatenate([np.zeros((1, f.shape[1])), np.cumsum(f, axis=0)])
    csq = np.concatenate([[0.0], np.cumsum((f * f).sum(axis=1))])
    tsum = np.concatenate([[0.0], np.cumsum(texf)])
    inside = int(round(EDGE_INSIDE_MM * pmm))
    # two segments: the edge at most EDGE_INSIDE_MM inside the grid region edge
    xs = np.arange(max(1, b - inside), min(n - min_bg, b + search) + 1)
    if xs.size == 0:
        return out
    zeros, full = np.zeros_like(xs), np.full_like(xs, n)
    sse2 = _sse_table(csum, csq, zeros, xs) + _sse_table(csum, csq, xs, full)
    x = int(xs[int(np.argmin(sse2))])
    best2 = float(sse2.min())
    # three segments: the grid boundary within MARGIN_SEARCH_MM of the grid region edge
    x1 = np.arange(max(1, b - search), min(n - m_lo - min_bg, b + search) + 1)
    best3, i1, i2 = np.inf, None, None
    if x1.size:
        X1, WD = np.meshgrid(x1, np.arange(m_lo, m_hi + 1), indexing="ij")
        X2 = X1 + WD
        ok = X2 <= n - min_bg
        if ok.any():
            X1, X2 = X1[ok], X2[ok]
            zeros, full = np.zeros_like(X1), np.full_like(X1, n)
            sse3 = (_sse_table(csum, csq, zeros, X1) + _sse_table(csum, csq, X1, X2)
                    + _sse_table(csum, csq, X2, full))
            # the paper edge itself stays outside the grid as in the two segment model
            sse3 = np.where(X2 >= b - inside, sse3, np.inf)
            k = int(np.argmin(sse3))
            if np.isfinite(sse3[k]):
                best3, i1, i2 = float(sse3[k]), int(X1[k]), int(X2[k])

    def mean(lo, hi):
        lo, hi = max(0, lo), min(n, hi)
        return (csum[hi] - csum[lo]) / max(hi - lo, 1)

    gain = 1.0 - best3 / best2 if best2 > 0 and np.isfinite(best3) else 0.0
    out["gain"] = float(gain)
    if gain >= MODEL_GAIN:
        out["model"] = 3
        margin = mean(i1, i2)
        outer = mean(i2, i2 + max(side_px, i2 - i1))
        # a white print margin looks like the paper; a strip of background beyond the
        # paper edge (a shadow, a fold of the cloth) looks like the background beyond it
        to_paper = float(np.linalg.norm(margin[:3] - paper[:3]))
        to_bg = float(np.linalg.norm(margin[:3] - outer[:3]))
        out["margin_paper"] = to_paper
        out["margin_bg"] = to_bg
        if to_paper <= to_bg or i1 < b - inside:
            out["margin_mm"] = (i2 - i1) / pmm
            out["margin_tex"] = float((tsum[i2] - tsum[i1]) / (i2 - i1))
            out["contrast"] = float(np.linalg.norm(mean(i2, i2 + side_px) - margin))
            if (out["margin_tex"] <= MARGIN_MAX_TEXTURE
                    and out["contrast"] >= EDGE_MIN_STEP):
                out["edge"] = i2
        else:
            out["model"] = 31
            out["margin_mm"] = 0.0
            out["contrast"] = float(np.linalg.norm(mean(i1, i1 + side_px)
                                                   - mean(i1 - side_px, i1)))
            if out["contrast"] >= EDGE_MIN_STEP:
                out["edge"] = i1
    else:
        out["model"] = 2
        out["margin_mm"] = 0.0
        out["contrast"] = float(np.linalg.norm(mean(x, x + side_px)
                                               - mean(x - side_px, x)))
        if out["contrast"] >= EDGE_MIN_STEP:
            out["edge"] = x
    return out


def side_bands(region, feats, energy, thr, side, pmm, paper):
    """Grid boundary and paper edge in NBANDS bands across one side (work scale).

    Along the outward normal each band profile (Lab + texture) is split into three
    segments, grid | print margin | background, at the two change points that leave the
    least squared deviation from the segment means (the grid boundary x1 within
    MARGIN_SEARCH_MM of the grid region's edge, the margin MARGIN_MM wide, the window
    ending EDGE_OUT_MM beyond the region or EDGE_BORDER_MM before the image border). A
    band counts when its margin is not grid textured through and differs from the
    background by >= EDGE_MIN_STEP and by more than it differs from the grid (the paper
    edge is the stronger step). Returns a list of dicts per band: along (work px), b
    (grid region edge, px from the inner side), gap_mm (region edge to image border)
    and, when found, edge = (x, y) of the paper edge in work coordinates, margin_mm,
    contrast.
    """
    reg = side_view(region, side)
    img = side_view(feats, side).astype(np.float64)
    tex = (side_view(energy, side) > thr).astype(np.float64)
    n_along, n_across = reg.shape
    rows = np.flatnonzero(reg.any(axis=1))
    if rows.size < NBANDS:
        return []
    lo, hi = rows[0], rows[-1]
    span = hi - lo
    edges = np.linspace(lo + 0.08 * span, hi - 0.08 * span, NBANDS + 1).astype(int)
    border = int(np.ceil(EDGE_BORDER_MM * pmm))
    grid_len = int(round(EDGE_OUT_MM * pmm))
    out = []
    for a0, a1 in zip(edges[:-1], edges[1:]):
        if a1 <= a0:
            continue
        sub = reg[a0:a1]
        has = sub.any(axis=1)
        if not has.any():
            continue
        last = n_across - 1 - np.argmax(sub[:, ::-1], axis=1)
        b = int(round(float(np.median(last[has]))))
        rec = {"along": (a0 + a1 - 1) / 2.0, "b": b, "gap_mm": (n_across - 1 - b) / pmm}
        g = b + 0.5 if side in ("bottom", "right") else n_across - 1 - b - 0.5
        rec["grid_edge"] = ((rec["along"], g) if side in ("top", "bottom")
                            else (g, rec["along"]))
        out.append(rec)
        prof = img[a0:a1].mean(axis=0)                  # (n_across, features)
        texf = tex[a0:a1].mean(axis=0)
        g0 = max(0, b - grid_len)
        end = min(n_across - border, b + grid_len)
        # the paper colour of this band: its grid region pixels within LOCAL_WHITE_MM of
        # its edge
        w0 = max(0, b - int(round(LOCAL_WHITE_MM * pmm)))
        vals = img[a0:a1, w0:b + 1][sub[:, w0:b + 1]]
        ref = paper_white(vals) if len(vals) >= LOCAL_WHITE_MIN_PX else paper
        res = band_model(prof[g0:end], texf[g0:end], b - g0, pmm, ref)
        rec.update({k: v for k, v in res.items() if k != "edge"})
        if res.get("edge") is None:
            continue
        across = g0 + res["edge"] - 0.5     # between profile pixels edge - 1 and edge
        c = across if side in ("bottom", "right") else (n_across - 1 - across)
        rec["edge"] = ((rec["along"], c) if side in ("top", "bottom")
                       else (c, rec["along"]))
    return out


def ransac_line(pts, vertical, tol):
    """Deterministic RANSAC over all point pairs, then least squares on the inliers.

    vertical: fit x = m y + c (left / right sides), else y = m x + c.
    Returns (m, c, inliers).
    """
    # (t, v)
    P = np.array([(p[1], p[0]) if vertical else (p[0], p[1]) for p in pts], float)
    n = len(P)
    best = None
    for i in range(n):
        for j in range(i + 1, n):
            dt = P[j, 0] - P[i, 0]
            if abs(dt) < 1e-9:
                continue
            m = (P[j, 1] - P[i, 1]) / dt
            c = P[i, 1] - m * P[i, 0]
            res = np.abs(P[:, 1] - (m * P[:, 0] + c)) / np.sqrt(1 + m * m)
            inl = res <= tol
            key = (int(inl.sum()), -float(res[inl].sum()))
            if best is None or key > best[0]:
                best = (key, inl)
    if best is None:
        return None
    inl = best[1]
    m, c = np.polyfit(P[inl, 0], P[inl, 1], 1)
    return float(m), float(c), inl


def line_points(m, c, vertical):
    """Homogeneous line through x = m y + c (vertical) or y = m x + c."""
    if vertical:   # x - m y - c = 0
        return np.array([1.0, -m, -c])
    return np.array([-m, 1.0, -c])   # y - m x - c = 0


def intersect(l1, l2):
    p = np.cross(l1, l2)
    return p[:2] / p[2]


def edge_profiles(rgbf, p0, p1, outward, half_len, fractions):
    """Colour profiles along the outward normal at points of the segment p0 -> p1.

    Returns (points (n x 2), offsets (m), profiles (n x m x 3)); each profile is the
    mean over a strip of 2 REFINE_STRIP_PX + 1 samples along the segment, bilinear.
    """
    direction = (p1 - p0) / np.linalg.norm(p1 - p0)
    offs = np.arange(-half_len, half_len + 1, dtype=np.float32)
    strip = np.arange(-REFINE_STRIP_PX, REFINE_STRIP_PX + 1, dtype=np.float32)
    pts = p0[None, :] + fractions[:, None] * (p1 - p0)[None, :]
    X = (pts[:, 0, None, None] + offs[None, None, :] * outward[0]
         + strip[None, :, None] * direction[0]).astype(np.float32)
    Y = (pts[:, 1, None, None] + offs[None, None, :] * outward[1]
         + strip[None, :, None] * direction[1]).astype(np.float32)
    n, s_, m = X.shape
    prof = cv2.remap(rgbf, X.reshape(n * s_, m), Y.reshape(n * s_, m), cv2.INTER_LINEAR,
                     borderMode=cv2.BORDER_REPLICATE).reshape(n, s_, m, 3).mean(axis=1)
    return pts, offs, prof


def step_positions(offs, prof, k):
    """Sub-pixel position (in offs units) of the strongest colour step of each profile."""
    csum = np.concatenate([np.zeros((prof.shape[0], 1, 3)),
                           np.cumsum(prof, axis=1, dtype=np.float64)], axis=1)
    idx = np.arange(k, prof.shape[1] - k + 1)
    inner = (csum[:, idx] - csum[:, idx - k]) / k
    outer = (csum[:, idx + k] - csum[:, idx]) / k
    step = np.linalg.norm(outer - inner, axis=2)
    i = np.argmax(step, axis=1)
    pos = idx[i].astype(float)
    ok = (i > 0) & (i < step.shape[1] - 1)
    rows = np.arange(len(i))
    l = np.where(ok, step[rows, np.clip(i - 1, 0, None)], 0)
    m = step[rows, i]
    r = np.where(ok, step[rows, np.clip(i + 1, None, step.shape[1] - 1)], 0)
    curv = l - 2 * m + r
    adj = np.where(ok & (curv < 0), 0.5 * (l - r) / np.where(curv < 0, curv, -1), 0.0)
    # boundary index p lies between samples p - 1 and p
    return offs[0] + pos + adj - 0.5, m


def fit_line_trimmed(P):
    """Total least squares line through points, trimmed at 3 robust sigmas (twice)."""
    keep = np.ones(len(P), bool)
    for _ in range(3):
        mean = P[keep].mean(axis=0)
        n2 = np.linalg.svd(P[keep] - mean)[2][1]
        res = (P - mean) @ n2
        sigma = max(1.4826 * np.median(np.abs(res[keep] - np.median(res[keep]))), 0.5)
        keep = np.abs(res - np.median(res[keep])) <= 3 * sigma
    mean = P[keep].mean(axis=0)
    n2 = np.linalg.svd(P[keep] - mean)[2][1]
    res = (P[keep] - mean) @ n2
    return np.array([n2[0], n2[1], -n2 @ mean]), float(np.sqrt(np.mean(res ** 2))), keep


def refine_side(rgbf, p0, p1, outward, pmm_full):
    """The paper edge between two coarse corners, again at full resolution.

    At REFINE_SAMPLES points between REFINE_ENDS and 1 - REFINE_ENDS of the side the
    strongest colour step within REFINE_SEARCH_MM of the coarse line (sub-pixel by a
    parabola) and a trimmed total least squares line through them. Returns (line, rms
    px, points kept, median |shift| px, share of the samples whose step sits at the end
    of the search window).
    """
    half_len = int(round(REFINE_SEARCH_MM * pmm_full)) + REFINE_STEP_PX
    fr = np.linspace(REFINE_ENDS, 1 - REFINE_ENDS, REFINE_SAMPLES)
    pts, offs, prof = edge_profiles(rgbf, p0, p1, outward, half_len, fr)
    off, strength = step_positions(offs, prof, REFINE_STEP_PX)
    # the steps can lie within +-(half_len - REFINE_STEP_PX + 0.5) of the line; at the
    # end = clipped
    lim = half_len - REFINE_STEP_PX + 0.5
    clip = float(np.mean(np.abs(off) >= lim - 0.25))
    P = pts + off[:, None] * outward[None, :]
    line, rms, keep = fit_line_trimmed(P)
    return line, rms, int(keep.sum()), float(np.median(np.abs(off))), clip


def quad_corners(lines):
    """TL, TR, BR, BL from the four side lines."""
    return np.array([intersect(lines["top"], lines["left"]),
                     intersect(lines["top"], lines["right"]),
                     intersect(lines["bottom"], lines["right"]),
                     intersect(lines["bottom"], lines["left"])])


def refine_lines(rgbf, lines, sides, pmm_full, info):
    """The side lines again at full resolution.

    Pass 1 refines every one of sides between the corners of lines; a side cut short by
    the search window (REFINE_CLIP_SHARE of its steps at the window's end) is refined
    again around its new line with the corners re-centred, at most REFINE_MAX_PASSES
    passes; the last line of a side is kept whatever its clipping. Per side the rms,
    points kept, summed median shift, clipped share and passes go to info.
    Returns lines.
    """
    lines = dict(lines)
    ends = {"top": (0, 1), "right": (1, 2), "bottom": (2, 3), "left": (3, 0)}
    corners = quad_corners(lines)
    todo = list(sides)
    npass = 0
    for npass in range(1, REFINE_MAX_PASSES + 1):
        centre = corners.mean(axis=0)
        again = []
        for side in todo:
            p0, p1 = corners[ends[side][0]], corners[ends[side][1]]
            d = (p1 - p0) / np.linalg.norm(p1 - p0)
            outward = np.array([d[1], -d[0]])
            if outward @ ((p0 + p1) / 2 - centre) < 0:
                outward = -outward
            new, rms, n, shift, clip = refine_side(rgbf, p0, p1, outward, pmm_full)
            info[f"{side}_refine_rms"] = rms
            info[f"{side}_refine_n"] = n
            info[f"{side}_refine_shift"] = (
                shift if npass == 1 else info[f"{side}_refine_shift"] + shift)
            info[f"{side}_refine_clip"] = clip
            info[f"{side}_refine_passes"] = npass
            lines[side] = new
            if clip >= REFINE_CLIP_SHARE:
                again.append(side)
        corners = quad_corners(lines)
        todo = again
        if not todo:
            break
    info["refine_passes"] = npass
    info["refine_clipped_sides"] = len(todo)    # still clipped after the last pass
    return lines


def paper_white(values):
    """Reference colour of the paper itself (features of the grid region pixels).

    The 90th percentile of L (the white between the grid lines and traces), a and b of
    the brightest quarter; the texture features as their median.
    """
    L = values[:, 0]
    bright = values[L >= np.percentile(L, 75)]
    ref = np.median(values, axis=0)
    ref[0] = np.percentile(L, 90)
    ref[1:3] = np.median(bright[:, 1:3], axis=0)
    return ref


def grid_pmm(ext_x, ext_y):
    """Work px per paper mm from the extent of the printed grid (PRINT_SCALE of it)."""
    long_px, short_px = max(ext_x, ext_y), min(ext_x, ext_y)
    return float(np.mean([long_px / (PRINT_SCALE * PAPER_MM[0]),
                          short_px / (PRINT_SCALE * PAPER_MM[1])]))


def find_paper(rgb, infer=False):
    """Paper quadrilateral of a page image (H x W x 3 uint8).

    Returns a dict; kind is one of fullframe (no paper boundary to find), quad (corners
    found), partial / bad (not usable). infer is a candidate of the view builder (the
    paper edge of a side that runs off the image, inferred from the grid): the digitiser
    never passes True, the parameter and its branch are kept so that this function stays
    the builder's, statement for statement.
    """
    h, w = rgb.shape[:2]
    region, feats, energy, thr, info = grid_region(rgb)
    wh, ww = region.shape
    sx, sy = ww / w, wh / h
    info["work_sx"] = sx
    info["region_frac"] = float(region.mean())
    if not region.any():
        info.update(kind="partial", reason="no grid region at the image centre")
        return info
    ys, xs = np.nonzero(region)
    ext_x, ext_y = xs.max() - xs.min() + 1, ys.max() - ys.min() + 1
    pmm0 = grid_pmm(ext_x, ext_y)
    paper = paper_white(feats[region])
    bands = {side: side_bands(region, feats, energy, thr, side, pmm0, paper)
             for side in SIDES}
    if not all(bands.values()):
        info.update(kind="partial", reason="grid region too small for the side bands")
        return info
    # robust extent of the grid: median band edge per side (a protrusion or leak moves
    # one band)
    gap_px = {side: float(np.median([r["gap_mm"] for r in bands[side]])) * pmm0
              for side in SIDES}
    ext_x = ww - gap_px["left"] - gap_px["right"]
    ext_y = wh - gap_px["top"] - gap_px["bottom"]
    pmm = grid_pmm(ext_x, ext_y)
    info["pmm_work"] = pmm
    gaps = {side: gap_px[side] / pmm for side in SIDES}
    info.update({f"gap_{side}_mm": v for side, v in gaps.items()})
    # px per paper mm for the scale-only fallback: from the image axes whose grid extent
    # is whole (the grid region clear of the border on both sides; a sheet cut by the
    # frame is short on that axis), x the paper's long axis when the image is landscape;
    # else as pmm
    dims = (PAPER_MM[0], PAPER_MM[1]) if ww >= wh else (PAPER_MM[1], PAPER_MM[0])
    est = []
    if min(gaps["left"], gaps["right"]) >= FALLBACK_GAP_MM:
        est.append(ext_x / (PRINT_SCALE * dims[0]))
    if min(gaps["top"], gaps["bottom"]) >= FALLBACK_GAP_MM:
        est.append(ext_y / (PRINT_SCALE * dims[1]))
    info["pmm_fallback"] = float(np.mean(est)) if est else pmm
    info["pmm_fallback_axes"] = len(est)
    extra = sorted([ww / pmm, wh / pmm])[::-1]
    info["extra_long_mm"] = float(extra[0] - PAPER_MM[0])
    info["extra_short_mm"] = float(extra[1] - PAPER_MM[1])
    if max(info["extra_long_mm"], info["extra_short_mm"]) <= FULLFRAME_EXTRA_MM:
        info.update(kind="fullframe", reason=(
            f"printed grid fills the frame (image {info['extra_long_mm']:+.1f} / "
            f"{info['extra_short_mm']:+.1f} mm beyond Letter)"))
        return info
    lines = {}
    T = np.array([[sx, 0, 0.5 * sx - 0.5], [0, sy, 0.5 * sy - 0.5], [0, 0, 1.0]])
    need = max(6, int(np.ceil(LINE_MIN_INLIERS * NBANDS)))
    for side in SIDES:
        pts = [r for r in bands[side] if "edge" in r]
        info[f"{side}_bands"] = len(pts)
        info[f"{side}_inliers"] = 0
        if len(pts) < need:
            continue
        vertical = side in ("left", "right")
        fit = ransac_line([r["edge"] for r in pts], vertical, LINE_TOL_MM * pmm)
        if fit is None:
            continue
        m, c, inl = fit
        used = [r for r, k in zip(pts, inl) if k]
        info[f"{side}_inliers"] = len(used)
        info[f"{side}_contrast"] = float(np.median([r["contrast"] for r in used]))
        info[f"{side}_margin_mm"] = float(np.median([r["margin_mm"] for r in used]))
        if len(used) < need:
            continue
        # work -> full resolution: x_work = T x_full, so l_full = T^T l_work
        lines[side] = T.T @ line_points(m, c, vertical)
    missing = [s for s in SIDES if s not in lines]
    info["sides_found"] = len(lines)
    inferred = []
    if missing:
        touching = [s for s in missing if gaps[s] <= TOUCH_GAP_MM]
        if len(touching) == 4:
            info.update(kind="fullframe",
                        reason="no paper edge visible, grid near the border on all sides")
            return info
        if infer and len(missing) == 1 and len(touching) == 1:
            # candidate (off by default, never taken from the digitiser): the paper edge
            # of a side that runs off the image as the grid edge there plus the nominal
            # print margin
            side = missing[0]
            vertical = side in ("left", "right")
            # only bands whose grid region stops short of the image border show the grid
            # edge
            seen = [r["grid_edge"] for r in bands[side] if r["gap_mm"] >= 1.0]
            fit = ransac_line(seen, vertical,
                              LINE_TOL_MM * pmm) if len(seen) >= 2 else None
            if fit is not None and fit[2].sum() >= need:
                m, c, _ = fit
                # the margin across this side: the paper dimension along its normal (x
                # for left / right), the long one when that image axis is the paper's
                # long axis
                along_long = (ext_x >= ext_y) if vertical else (ext_y > ext_x)
                margin = (1 - PRINT_SCALE) / 2 * PAPER_MM[0 if along_long else 1] * pmm
                sign = 1.0 if side in ("bottom", "right") else -1.0
                c = c + sign * margin * np.sqrt(1 + m * m)
                lines[side] = T.T @ line_points(m, c, vertical)
                inferred = [side]
                info["inferred_side"] = side
        if not inferred:
            if len(touching) == len(missing):
                info.update(kind="partial", reason="paper reaches the image border on "
                            + "/".join(missing))
            else:
                info.update(kind="partial",
                            reason="no paper edge on " + "/".join(missing))
            return info
    rgbf = rgb.astype(np.float32)
    pmm_full = pmm / sx
    coarse = quad_corners(lines)
    info["coarse_corners"] = coarse
    lines = refine_lines(rgbf, lines, [s for s in SIDES if s not in inferred],
                         pmm_full, info)
    corners = quad_corners(lines)
    tl, tr, br, bl = corners
    info["corners"] = corners
    area = abs(cv2.contourArea(corners.astype(np.float32))) / (w * h)
    info["paper_frac"] = float(area)
    top_len = (np.linalg.norm(tr - tl) + np.linalg.norm(br - bl)) / 2
    side_len = (np.linalg.norm(bl - tl) + np.linalg.norm(br - tr)) / 2
    info["aspect"] = float(max(top_len, side_len) / min(top_len, side_len))
    info["portrait"] = int(side_len > top_len)
    # the paper holds the print: the share of the grid region inside the quadrilateral
    quad_work = np.array([[sx * (x + 0.5) - 0.5, sy * (y + 0.5) - 0.5]
                          for x, y in corners])
    inside_mask = np.zeros(region.shape, np.uint8)
    cv2.fillConvexPoly(inside_mask, np.round(quad_work * 16).astype(np.int32),
                       1, cv2.LINE_8, 4)
    info["contain"] = float((region & (inside_mask > 0)).sum() / max(region.sum(), 1))
    # TL, TR, BR, BL run clockwise on the screen (y down): positive shoelace area; a
    # negative one would be a mirrored page
    nxt = np.roll(corners, -1, axis=0)
    info["signed_area"] = float(0.5 * np.sum(corners[:, 0] * nxt[:, 1]
                                             - nxt[:, 0] * corners[:, 1]) / (w * h))
    if not cv2.isContourConvex(corners.astype(np.float32).reshape(-1, 1, 2)):
        info.update(kind="bad", reason="paper quadrilateral not convex")
    elif info["signed_area"] <= 0:
        info.update(kind="bad",
                    reason="paper quadrilateral mirrored (corners not clockwise)")
    elif info["contain"] < CONTAIN_MIN:
        info.update(
            kind="bad",
            reason=f"paper quadrilateral holds only {info['contain']:.3f} of the grid")
    elif not ASPECT_RANGE[0] <= info["aspect"] <= ASPECT_RANGE[1]:
        info.update(kind="bad", reason=f"paper aspect {info['aspect']:.2f}")
    elif area >= FULLFRAME_AREA:
        info.update(kind="fullframe", reason=f"paper covers {area:.3f} of the image")
    else:
        info.update(
            kind="quad_inferred" if inferred else "quad",
            reason=f"{inferred[0]} edge inferred from the grid" if inferred else "")
    return info


# -------------------------------------------------------------- homography and the warp
PAGE_CORNERS = np.array([[-0.5, -0.5], [PAGE_W - 0.5, -0.5], [PAGE_W - 0.5, PAGE_H - 0.5],
                         [-0.5, PAGE_H - 0.5]], float)


def paper_homography(corners, orientation):
    """Image (cv2 px) -> page (cv2 px) for image corners TL, TR, BR, BL and a turn.

    orientation (degrees) 0: image TL -> page TL; 180: image BR -> page TL; 90: image TR
    -> page TL (the page is the image turned 90 deg counter-clockwise); 270: image BL ->
    page TL.
    """
    shift = {0: 0, 90: 1, 180: 2, 270: 3}[orientation]
    src = np.roll(corners, -shift, axis=0)
    return cv2.getPerspectiveTransform(
        src.astype(np.float32), PAGE_CORNERS.astype(np.float32)).astype(np.float64)


def local_scales(H):
    """Page px per image px (sqrt |det J|) of H at a 5 x 5 grid of page points."""
    Hi = np.linalg.inv(H)
    out = []
    for x in np.linspace(0, PAGE_W - 1, 5):
        for y in np.linspace(0, PAGE_H - 1, 5):
            p = Hi @ np.array([x, y, 1.0])
            q = p[:2] / p[2]
            d = H @ np.array([q[0], q[1], 1.0])
            J = (H[:2, :2] * d[2] - np.outer(d[:2], H[2, :2])) / d[2] ** 2
            out.append(np.sqrt(abs(np.linalg.det(J))))
    return np.array(out)


def warp_to_page(rgb, H, size=(PAGE_W, PAGE_H)):
    """INTER_AREA pre-resize to about the target scale, then one INTER_CUBIC warp.

    Returns (page, meta) where meta holds the pre-resize size and the warp matrix used.
    """
    h, w = rgb.shape[:2]
    sc = local_scales(H)
    f = float(np.sqrt(sc.min() * sc.max()))
    if size != (PAGE_W, PAGE_H):
        f *= size[0] / PAGE_W
    if f < 0.95:
        pw, ph = int(round(w * f)), int(round(h * f))
        sx, sy = pw / w, ph / h
    else:
        pw, ph, sx, sy = w, h, 1.0, 1.0
    S = np.array([[sx, 0, 0.5 * sx - 0.5], [0, sy, 0.5 * sy - 0.5], [0, 0, 1.0]])
    G = np.diag([size[0] / PAGE_W, size[1] / PAGE_H, 1.0])
    G[0, 2] = 0.5 * G[0, 0] - 0.5
    G[1, 2] = 0.5 * G[1, 1] - 0.5
    M = G @ H @ np.linalg.inv(S)
    page = apply_warp(rgb, [pw, ph], M, size)
    return page, {"pre_size": [pw, ph], "warp": M, "scale_min": float(sc.min()),
                  "scale_max": float(sc.max())}


def apply_warp(rgb, pre_size, M, size=(PAGE_W, PAGE_H)):
    """The view of a normalised page from its stored pre-resize size and warp matrix.

    INTER_AREA to pre_size (when it differs from the image), then one INTER_CUBIC
    warpPerspective. warp_to_page and replay_page both build the page here, so a replay
    is the same code path.
    """
    h, w = rgb.shape[:2]
    pw, ph = int(pre_size[0]), int(pre_size[1])
    small = rgb if (pw, ph) == (w, h) else cv2.resize(
        rgb, (pw, ph), interpolation=cv2.INTER_AREA)
    return cv2.warpPerspective(small, np.asarray(M, np.float64), tuple(size),
                               flags=cv2.INTER_CUBIC, borderMode=cv2.BORDER_REPLICATE)


def apply_resize(rgb, size):
    """The view of a scaled page: INTER_AREA to size (w, h)."""
    return cv2.resize(rgb, (int(size[0]), int(size[1])), interpolation=cv2.INTER_AREA)


def frame_decision(fr):
    """(status, scale x, scale y, reason) of a page warped from the paper corners.

    fr holds the 1 mm grid periods px and py of that page. bad (a side far off),
    corrected (both periods measured: scale them to FRAME_PERIOD) or unmeasured.
    """
    px, py = fr["px"], fr["py"]
    for name, p in (("x", px), ("y", py)):
        if (np.isfinite(p)
                and not FRAME_AXIS_RANGE[0] <= p / FRAME_PERIOD <= FRAME_AXIS_RANGE[1]):
            return "bad", 1.0, 1.0, (f"normalised grid period along {name} {p:.3f} px = "
                                     f"{p / FRAME_PERIOD:.3f} of the frame's")
    if np.isfinite(px) and np.isfinite(py):
        if abs(py / px - 1) > FRAME_ANISO_MAX:
            return "bad", 1.0, 1.0, f"normalised grid anisotropic (py / px {py / px:.3f})"
        return "corrected", FRAME_PERIOD / px, FRAME_PERIOD / py, ""
    return "unmeasured", 1.0, 1.0, ""


def frame_scale(H, sx, sy):
    """H followed by a scaling of the page about its centre (cv2 px)."""
    cx, cy = (PAGE_W - 1) / 2, (PAGE_H - 1) / 2
    S = np.array([[sx, 0, cx - sx * cx], [0, sy, cy - sy * cy], [0, 0, 1.0]])
    return S @ H


def trace_ink(page):
    """Dark marks (traces, text) of a page.

    Pixels whose brightest channel is below ORIENT_INK of the local white of the page,
    so that the red grid (bright in its red channel), shadows and uneven light do not
    count.
    """
    v = page.max(axis=2).astype(np.float32)
    white = cv2.GaussianBlur(cv2.dilate(v, np.ones((9, 9), np.uint8)), (0, 0), 5)
    return (v < ORIENT_INK * white).astype(np.float32)


def orientation_margin(page):
    """(lower - upper) / (lower + upper) trace ink of a page.

    Lower and upper are the ORIENT_HALF_MM of paper height below and above its middle.
    """
    rows = trace_ink(page).sum(axis=1)
    mm = ((np.arange(page.shape[0]) + 0.5) / page.shape[0] - 0.5) * PAPER_MM[1]
    up = rows[(mm >= -ORIENT_HALF_MM) & (mm < 0)].sum()
    lo = rows[(mm >= 0) & (mm < ORIENT_HALF_MM)].sum()
    return float((lo - up) / max(lo + up, 1e-9))


def choose_orientation(rgb, corners, portrait):
    """Orientation (deg) of the two landscape candidates whose ink lies lower.

    Returns (orientation, its margin).
    """
    cands = (90, 270) if portrait else (0, 180)
    H = paper_homography(corners, cands[0])
    small, _ = warp_to_page(rgb, H, size=(PAGE_W // 2, PAGE_H // 2))
    m = orientation_margin(small)
    # the other candidate is the same page turned by 180 deg: its margin is -m
    return (cands[0], m) if m >= 0 else (cands[1], -m)


# -------------------------------------------------------------------- one page, decided
def fallback_factor(rgb, c, paper):
    """Resize factor of the scale-only fallback, or None (see SCALE_DEAD_BAND)."""
    if c is not None and (c["res_scale"] != 1.0
                          or (np.isfinite(c["res_period"]) and not c["res_reason"])):
        # the digitiser's resolution stage reads the grid: it rescales, or keeps it
        return None
    if paper.get("pmm_fallback") is None or not paper.get("work_sx"):
        return None
    pmm_full = paper["pmm_fallback"] / paper["work_sx"]
    f = PAPER_PMM / pmm_full
    return float(f) if f < 1 - SCALE_DEAD_BAND else None


def normalise_page(rgb, chain, frame_periods):
    """Decide and build the view of one page.

    Returns (view pixels or None = as it is, meta).

    chain(rgb) -> the dict of the digitiser's pre-segmentation chain (ok, res_scale,
    res_period, res_reason, ...); frame_periods(page) -> px, py, px_raw, py_raw, cx, cy
    of a warped page.
    """
    meta = {"decision": "", "reason": "", "size": f"{rgb.shape[1]}x{rgb.shape[0]}"}
    c = chain(rgb)
    meta["chain"] = c
    if c["ok"]:
        meta.update(decision="pass_chain", reason="")
        return None, meta
    paper = find_paper(rgb)
    meta["paper"] = paper
    kind = paper["kind"]
    if kind == "fullframe":
        meta.update(decision="pass_fullframe", reason=paper["reason"])
    elif kind in ("partial", "bad"):
        meta.update(decision="failed", reason=paper["reason"])
    else:
        orient, margin = choose_orientation(rgb, paper["corners"], paper["portrait"])
        H0 = paper_homography(paper["corners"], orient)
        page0, w0 = warp_to_page(rgb, H0)
        fr = frame_periods(page0)
        status, fsx, fsy, freason = frame_decision(fr)
        meta["frame"] = {**fr, "status": status, "scale_x": fsx, "scale_y": fsy}
        meta.update(orientation=orient, orient_margin=margin,
                    orient_low=int(margin < ORIENT_LOW_MARGIN), homography_paper=H0)
        if status == "bad":
            meta.update(decision="failed", reason=freason)
        else:
            if status == "corrected":
                H = frame_scale(H0, fsx, fsy)
                page, wmeta = warp_to_page(rgb, H)
            else:
                H, page, wmeta = H0, page0, w0
            reason = paper["reason"]
            if status == "unmeasured":
                reason = ((reason + "; " if reason else "")
                          + "frame unmeasured (grid period NaN), kept as warped")
            meta.update(decision="normalised", reason=reason, homography=H, **wmeta)
            return page, meta
    f = fallback_factor(rgb, c, paper)
    if f is not None:
        h, w = rgb.shape[:2]
        size = [int(round(w * f)), int(round(h * f))]
        meta.update(scaled_from=meta["decision"], scale_factor=f, resize=size,
                    decision="scaled",
                    reason=f"{meta['reason']}; shrunk by {f:.3f} to 200 dpi of paper")
        return apply_resize(rgb, size), meta
    return None, meta


def replay_page(rgb, block):
    """The view of a page from the numbers of its record alone, None = as it is.

    block is the record of the page (the paper_normalisation block of a mask's JSON):
    a normalised page is built from the stored pre-resize size and warp matrix, a scaled
    one from the stored size, with no chain and no paper search, so that a mask can be
    laid over the very view it was predicted on whatever the decision rules of the day.
    """
    decision = block["decision"]
    if decision in ("pass_chain", "pass_fullframe", "failed"):
        return None
    if decision == "normalised":
        return apply_warp(rgb, block["pre_size"], block["warp"])
    if decision == "scaled":
        return apply_resize(rgb, block["resize"])
    raise ValueError(f"decision {decision!r} cannot be replayed")


# ------------------------------------------------------------------------- meta records
def _num(v):
    """v as plain Python values: numpy scalars and arrays unpacked, NaN and inf None."""
    if isinstance(v, (np.floating, float)):
        return None if not np.isfinite(v) else float(v)
    if isinstance(v, (np.integer,)):
        return int(v)
    if isinstance(v, np.ndarray):
        return [_num(x) for x in v.tolist()] if v.ndim else _num(v.item())
    if isinstance(v, (list, tuple)):
        return [_num(x) for x in v]
    if isinstance(v, dict):
        return {k: _num(x) for k, x in v.items()}
    return v


def page_record(meta):
    """The JSON-able record of one page from the meta of normalise_page().

    Decision, reason, homographies and the numbers behind them: what the paper search
    found, what the chain said and the frame of the warped page.
    """
    keep = {k: meta[k] for k in ("decision", "reason", "size", "orientation",
                                 "orient_margin", "orient_low", "homography_paper",
                                 "homography", "pre_size", "warp", "scale_min",
                                 "scale_max", "frame", "scaled_from", "scale_factor",
                                 "resize") if k in meta}
    if "paper" in meta:
        p = meta["paper"]
        keep["paper"] = {k: p[k] for k in ("kind", "reason", "corners", "paper_frac",
                                           "aspect", "portrait", "region_frac",
                                           "pmm_work", "pmm_fallback",
                                           "pmm_fallback_axes", "work_sx",
                                           "refine_passes", "refine_clipped_sides")
                         if k in p}
        keep["paper"]["refine_clip"] = [p.get(f"{s}_refine_clip") for s in SIDES]
    if "chain" in meta:
        keep["chain"] = {k: meta["chain"][k] for k in ("res_reason", "rot_reason",
                                                        "persp_reason", "res_scale",
                                                        "res_period", "rot_angle")}
    return _num(keep)
