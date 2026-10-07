"""Sheet curl: straighten a page whose sheet is bent, by its printed grid lines.

A photographed sheet is seldom flat. The paper normalisation puts its four corners on
the page frame and the perspective stage fits one homography to its grid, and both
leave the bend of the sheet in the page: its grid lines are curves, the column grid and
the column map do not stand on them, and the traces are read off a time axis that is
not theirs. This module measures a smooth map from the pixels of such a page to the
coordinates of its grid lines and warps the page, and a label mask that was predicted
on it, into the frame in which the lines are straight. The digitiser calls it between
its perspective stage and the segmentation (--sheet_curl lines), on the page as that
stage left it.

It holds the estimator and the record of what it did, and no import of the digitiser:
the grid line helpers of the perspective stage (the band profiles, the page-wide
period, the gradient search, the unwrapping, the joint fit of the projective part) and
the constants that go with them come in as one object, D, that names them (HELPERS;
src.run.digitize.sheet_curl_helpers() builds it). Nothing here reads or writes a file.

THE ESTIMATOR, measure_field(image, options, D) -> (field, info).
  Coordinates are those of the digitiser: pixel c covers [c, c+1).
  1. The carrier, from two candidates that are both measured with the windows of step
     2. 'page': the strongest comb of the page-wide profile, as the perspective stage
     finds it. 'windows': the strongest comb of the windows themselves where the
     resolution stage has put the 1 mm lines, GRID_LINE_PERIOD_200DPI / scale over the
     scales of its dead band (the amplitudes summed window by window without their
     phases at CARRIER_STEPS line frequencies, the top refined by a parabola). The
     candidate whose windows have the larger contrast is the carrier; when the two are
     the same comb for a window (their periods differ by less than one line over its
     length) the page-wide one stays. Why: on a blurred sheet that is bent the lines
     are not coherent over the page, the 1 mm peak of the page-wide spectrum smears,
     and what is left on top is a harmonic of the bold 5 mm lines, where the windows
     have no comb. The page-wide contrast test of the perspective stage is not used,
     for the same reason: the contrast here is that of the windows (MIN_CONTRAST).
  2. Phase windows of both line families: WINDOW_LINES lines long, one every
     1 / WINDOW_OVERLAP of that length, on the median profiles of bands of BAND_ROWS
     rows, plus one window that ends on the last pixel. A window says where the 1 mm
     lines are modulo one line.
  3. The integer ambiguity. Start: the page is cut into START_BLOCKS x START_BLOCKS
     blocks, the local line frequency of each is found by the gradient search of the
     perspective stage (a coherent sum, nothing is unwrapped), and a quadratic through
     the block gradients integrates to a cubic start. Then rounds: every window is
     unwrapped against the current MODEL, one family at a time, the inliers picked
     from scratch in every round; the projective part is fitted to both families
     jointly (affine in the first round, and affine as well when the projective terms
     exceed MAX_PROJECTIVE_TERM); and the smooth part, a polynomial of total degree 2,
     then 3, then DEGREE (Legendre products in centred, scaled coordinates), is fitted
     per family to what the projective part leaves. The rounds end when no window
     changes its line any more, after MAX_ROUNDS at the latest.
  4. The map lives on a lattice of LATTICE px (bilinear in between). The projective
     part is evaluated everywhere; the smooth part only inside the union of the
     supports of the inlier windows of its family, and a node outside holds the smooth
     part of the nearest node inside: a polynomial is not extrapolated.
  5. The gauge. The lines give (u, v) up to their unit and their origin: T(x, y) = a +
     s (u, v)(x, y), with the translation a and the ONE scale s for which the squared
     displacement T - (x, y), summed over the measured nodes of the lattice, is
     smallest. No rotation: the grid gives the orientation.
  6. Decisions. refused: no carrier, no contrast, too few windows of a family (before
     and after the fit), a residual of the fit above the tolerance, a family that
     spans less than PERSPECTIVE_MIN_SPAN of the page, a cover below MIN_COVER, a map
     that folds or that stretches or squeezes the page by more than a factor of two
     somewhere (JACOBIAN_FLOOR, per cell of the lattice), a correction above the
     largest one a page may ask for, an inverse that has not converged, a straightened
     page that fails the check of step 8. dead band: the map moves no measured node by
     the dead band, or its smooth part explains the lines no better than the
     projective part while that part moves the page by less than the dead band.
     applied: the rest. The Jacobian and the largest correction are taken over the
     WHOLE lattice: where the smooth part is held from the nearest measured node, two
     parts of the border of a hole can disagree and tear the map, so the floor is in
     practice a refusal of pages whose cover has holes the polynomial does not bridge.
  7. The inverse, for the warp: a fixed point iteration p <- q - d(p) on the lattice,
     accepted when no node is more than INVERSE_TOLERANCE px off its target after at
     most INVERSE_ROUNDS rounds. The page is resampled by cv2.remap, bicubic with a
     black constant border as warp_page() of the digitiser, a label mask nearest with
     border 0, which holds no value that was not in the mask.
  8. The check. The straightened page is measured again. What it still asks for is
     composed into the map while it is above the dead band and the passes are not used
     up, and the last measurement decides: the lines have to be straight, within
     min(tolerance, AFTER_MARGIN x the residual before + AFTER_SLACK) of a projective
     fit, and the page must not ask for more than max(dead band, AFTER_SHARE x what it
     asked for).

THE KEYS OF info.
  decision     applied | dead band | refused; out of scope is said by out_of_scope()
  reason       why a page is refused
  carrier, period  how the carrier was found (page | windows | none) and its period, px
  windows_u, windows_v  inlier windows of the vertical lines (u, along x) and of the
               horizontal ones (v)
  share        inlier windows / windows with a comb
  contrast     comb of the windows relative to their amplitude between the harmonics
               of the 5 mm lines (the smaller of the two families)
  res_proj     RMS residual in periods of the projective part alone, over the inliers
               of the final model
  res_field    RMS residual in periods of the final model of the first measurement,
               over the inliers that model picks itself: what the tolerance is for
  res_all      the same over ALL windows with a comb, each on the line of the model
               nearest to it (0.289 = no relation at all). A sheet that is bent locally
               leaves its windows here and not in res_field
  res_after, shift_after  res_proj and shift_max measured again on the straightened
               page; share_after, cover_after: share and cover of that measurement,
               told and never judged
  shift_max, shift_rms  displacement in px of the map against the identity over the
               lattice nodes both line families cover, in the gauge of step 5;
               shift_all the largest over the whole lattice, measured or not
  cover        share of the page inside the supports of the inlier windows of both
               families
  proj_shift   what the rectified projective part alone moves the page by, px
  jac_min      smallest Jacobian determinant of the map in a cell of the lattice
  scale, gauge_dx, gauge_dy  the scale of the map at the page centre and the px it
               moves the centre by
  rounds, passes  rounds of unwrap-and-fit of the first measurement; measurements
               composed into the map
  degree, inverse, seconds  the degree of the smooth part, the error of the inverse in
               px, the time the page took

WHAT 'applied' MEANS. The map was accepted and the page was warped with it. It does
not mean the sheet is flat now. A whole-sheet smooth bend is taken out; a fold or a
local dent is not (the model has no term for it): the bulk of such a page is
straightened, the bent part stays, and can end further from the truth than it was.
share_after, cover_after and res_all are the keys that show it.

WHERE THE NUMBERS COME FROM. The estimator is the one of the driver the stage was
developed with (c5_curl_digitize.py), function for function; its constants were fixed
on synthetic pages, generator pages shown through known maps, before a real page was
read, and the comments on them say what each was chosen on. Whole-sheet smooth bends
of 10 and 25 px are recovered there to 0.03 to 0.05 px RMS, a twist to 0.18 px; folds
and local dents are not recovered. The options of the command line (TOLERANCE, PASSES,
MIN_SHIFT_PX) and the scope were chosen later, on the development pages of the
ECG-Image-Database (v2, part D0: 115 PTB-XL records split by patient; 345 photographs
in 3 conditions, 230 clean scans, 230 mould-damaged scans, 115 rendered pages; see the
README), and their comments say so.

THE DEAD BAND OF THE STAGE (straighten()). The estimator has a threshold of its own,
min_shift, with three roles in measure_field(): the dead band of the first measurement
(step 6), the end of the passes and the floor of the check on the straightened page
(step 8). The stage always runs the estimator at ESTIMATOR_MIN_SHIFT_PX, 1 px, the
threshold every number of the development pages was read with, and takes its own dead
band, --sheet_curl_min_shift, on the map it would apply: a page whose accepted map (the
final one, all passes composed) moves no measured node by the dead band is left as it
is. So the dead band of the command line decides whether an accepted map is used, and
never which map is measured. A dead band at or below the threshold of the estimator is
that threshold: the stage is then measure_field() at that value and nothing else.

THE RECORD (mask_record(), mask_frame()). A mask that is saved in a run with the stage on
gets a record of what the stage did to its page: the decision, the dead band, the
options and constants that define the map, and for an applied page the map itself at a
lattice of CHECK_POINTS points. A mask whose record says applied was predicted on the
straightened page: it is never warped again, it only fits a page that this run
straightens by the same map, to MASK_SHIFT_TOLERANCE px at those points, and it keeps
that record when a run saves it again, whatever that run does with the page.

Deterministic: fixed parameters, no random numbers, CPU numpy / scipy / cv2 only (torch
is here to take a page that comes as a tensor, and for nothing else).
"""
import math
import time

import cv2
import numpy as np
import torch
from scipy import ndimage

# The record this module writes next to a mask, and the only one mask_frame() reads.
VERSION = 1

# What this module takes from the digitiser, by name: the grid line helpers of its
# perspective stage and the constants that go with them. measure_field() and what it
# calls reach them as D.<name>, D being any object that has these names. A name that is
# missing raises in measure_field(); through straighten(), which the run calls, it does
# NOT end the run: every exception inside the measurement is a refused page there, with
# the WARNING 'error: AttributeError: ...' on every page. The tests hold the list
# against the code, which is what keeps that from happening.
HELPERS = (
    "_darkness",
    "_band_profiles",
    "_grid_line_period",
    "_phase_gradient",
    "_unwrap_windows",
    "_grid_map_least_squares",
    "_apply_homography",
    "_rectify_homography",
    "_homography_shift",
    "GRID_LINE_PERIOD_200DPI",
    "GRID_LINE_MIN_BAND_AMPLITUDE",
    "RESOLUTION_DEAD_BAND",
    "PERSPECTIVE_MIN_BLOCK_WINDOWS",
    "PERSPECTIVE_MIN_BLOCKS",
    "PERSPECTIVE_MIN_WINDOWS",
    "PERSPECTIVE_MIN_SPAN",
    "PERSPECTIVE_MAX_SHIFT",
)

# --- the options of the command line -------------------------------------------------
# All four were CHOSEN ON THE DEVELOPMENT PAGES of the ECG-Image-Database (345
# photographs, 230 clean scans, 230 mould-damaged scans and 115 rendered pages of the
# same 115 records, each read from the mask saved for it, the mask warped with the
# page): they are what read those pages best, not what an untouched set has confirmed.
# Every count below is from the driver the stage was developed with, with every page in
# scope and the threshold of the estimator at 1 px (ESTIMATOR_MIN_SHIFT_PX).
#
# Largest RMS residual of the fit that is accepted (res_field, and res_after on the
# straightened page), as a share of the grid line period: --sheet_curl_tolerance. On
# clean synthetic pages the fit of a bent sheet ends below 0.1, the tolerance of the
# grid line stages of the digitiser, and with 0.1 the stage accepts 11 of the 345
# photographs: the fit of a photograph leaves 0.10 to 0.25 of a period on nine pages in
# ten (median 0.18). With 0.25 it accepts 135 of them at 2 passes and 188 at 3.
# WHAT 0.25 LETS THROUGH. On a drawn page with a known map (the grid of the tests, bent
# by 38 px and blurred) a fit that leaves 0.17 of a period is accepted and is a WRONG
# map, 9 px RMS and 25 px at its worst off the truth: the check on the straightened
# page does not catch it, 0.1 refuses it (tests). On the photographs the pages accepted
# between the two tolerances still gain more often than they lose. The 188 straightened
# photographs (tolerance 0.25, 3 passes, threshold 1 px) by what the fit leaves, with
# the pages that gain / lose more than 1 dB of page SNR: below 0.10 of a period 11
# pages, 7 / 1; 0.10 to 0.15: 56 pages, 28 / 9; 0.15 to 0.20: 80 pages, 47 / 5; 0.20
# and above: 41 pages, 17 / 7, their median page SNR going from 0.46 to 1.88 dB. On the
# mould-damaged scans 202 of the 207 straightened pages leave less than 0.10. So a fit
# above 0.1 is no proof of a right map: it is accepted because, on these pages, it paid.
TOLERANCE = 0.25
# No residual is above half a period: every window is unwrapped onto the line of the
# model it is nearest to, so a tolerance beyond it means nothing.
MAX_TOLERANCE = 0.5
# How many measurements may be composed into the map: --sheet_curl_passes. The
# straightened page is measured again, and while it still asks for the threshold of the
# estimator or more and this number is not reached, what it asks for is added to the
# map; the last measurement is the check. On synthetic pages the second measurement
# only ever was the check; on the photographs 3 let 188 pages through where 2 let 135
# (tolerance 0.25, threshold 1 px, every page in scope).
PASSES = 3
MAX_PASSES = 5
# The threshold of the estimator in px, min_shift of measure_field(), which has three
# roles there: a first map that moves no measured node by this much is the dead band, a
# further pass is composed only while the straightened page asks for this much or more,
# and the check lets pass what a straightened page still asks for below it. Flat
# synthetic pages ask for at most 0.64 px. It is the value of the driver, and every
# number of the development pages was read with it: the stage runs the estimator with
# it whatever the dead band of the command line is (straighten()), unless that dead
# band is smaller.
ESTIMATOR_MIN_SHIFT_PX = 1.0
# The dead band of the stage in px: --sheet_curl_min_shift. A page whose accepted map,
# all passes composed, moves no measured node by this much is left as it is (shift_max
# of the final map; exactly this much is applied). Why 15, with every page in scope: a
# small bend costs a page that reads well, since the page is resampled once more and
# its mask is warped nearest, which moves the edge of a trace by up to half a pixel.
# With the threshold of 1 px alone the stage straightens 193 of the 230 clean scans (the
# map of a clean scan moves it by 6.5 px in the median), and 67 of them lose more than
# 1 dB of page SNR where 17 gain;
# with the pages that move less than 10 px left alone it still straightens 53, and 24
# lose where 7 gain. With 15 px it straightens 2 clean scans (none loses, 1 gains), 139
# of the 345 photographs (11 lose, 84 gain; 12 / 94 of 158 pages at 10 px, 22 / 99 of
# 188 at 1 px), 151 of the 230 mould-damaged scans (2 lose, 135 gain; 207 pages, 9 /
# 159 at 1 px) and none of the 115 rendered pages. These counts leave the pages below
# the dead band as they are without the stage, which is what straighten() does.
MIN_SHIFT_PX = 15.0
# Which pages the stage looks at: --sheet_curl_scope. all = every page; normalised =
# only a page that the paper normalisation warped onto the Letter page, a photographed
# sheet. Why all: the mould-damaged scans, which the paper normalisation does not warp,
# gain most from the stage (leads without a usable signal 32.5 -> 11.1 % at a dead band
# of 15 px), and the dead band, not the scope, is what keeps the clean scans and the
# rendered pages out.
SCOPE = "all"
SCOPES = ("normalised", "all")

# --- the other options of the estimator ----------------------------------------------
# Fixed on synthetic pages before a real page was read, and not on the command line.
# Total degree of the smooth part. 3: a twist of 25 px is left with 1.5 px RMS and a
# half sine across each side with 0.66 px (it needs the fourth order). 4: every
# whole-sheet bend tried at or below 0.27 px up to 25 px. 6 follows a twist better and
# lets the map of a flat augmented page grow (shift_max 0.20 -> 0.29 px).
DEGREE = 4
# The phase windows: their length in grid lines, the rows of a band of the line
# profile, and the step of the windows as a share of their length (the perspective
# stage: 32 lines, 100 rows, half steps). With the windows of that stage the gradient
# search is held inside grating lobes at a 6.2 % period mismatch and a 4.5 degree
# tilt, and the windows in the corners of a sheet bent by 40 px wash out; with 16
# lines, quarter steps and 50 rows both families still cover 0.95 of a page bent by
# 60 px. Longer windows lose windows from 40 px on.
WINDOW_LINES = 16
BAND_ROWS = 50
WINDOW_OVERLAP = 4
# Blocks per axis of the start of the phase model (the perspective stage: 3).
START_BLOCKS = 4
# Smallest share of the page that the inlier windows of both families have to cover.
MIN_COVER = 0.5
# Largest correction in px a page may ask for anywhere on the lattice (shift_all),
# above which it is refused. None = PERSPECTIVE_MAX_SHIFT of the digitiser x the page
# width, the limit of its perspective stage: more than that is a misread grid.
MAX_SHIFT_PX = None

# Step in px of the lattice the straightening map lives on: its displacement is
# bilinear between the nodes. The displacement of a sheet has second derivatives of
# about 1e-4 px / px^2 (25 px over half a page, squared), which a bilinear cell of 8 px
# follows to 1e-3 px.
LATTICE = 8
# The comb of a window relative to its amplitude off the carrier, at these frequencies
# relative to the carrier: half way between the harmonics of the bold 5 mm lines (0.6,
# 0.8, 1.2, 1.4).
BACKGROUND = (0.7, 0.9, 1.1, 1.3)
# Smallest such contrast, summed over the windows of a family. With the default windows
# the 96 flat generator pages have 4.2 to 9.0 (a 16 line window still sees the 5 mm
# harmonics next to its carrier, which is what keeps the number low), a page curled by
# 25 px 4.5, a page of noise 1.0.
MIN_CONTRAST = 2.0
# The second candidate of the carrier: the strongest comb of the windows where the
# resolution stage has put the 1 mm lines, GRID_LINE_PERIOD_200DPI / scale for the
# scales of its dead band (1 +- RESOLUTION_DEAD_BAND: inside it that stage leaves a
# page as it is, outside it resamples the page to the middle). This many steps over the
# band: 1 % each, a window of 16 lines has a main lobe of +-12.5 %, so a parabola
# through three steps finds its top.
CARRIER_STEPS = 25
# Smallest Jacobian determinant of the straightening map in a cell of the lattice:
# below it the map squeezes that cell to less than half its area (the warp then
# stretches it by more than two), which no bent sheet does. Over 72 first maps of
# synthetic pages (ten bends at 10, 25 and 40 px on two pages where a map was reached,
# and 21 pages with a part of the grid erased) no cell whose four nodes are measured is
# below 0.897, and the 8 maps with a cell below the floor (-0.58 to 0.46) have it where
# the smooth part is held from the nearest measured node; the whole-sheet bends up to
# 25 px on a clean page have 0.897 and more over the whole lattice.
JACOBIAN_FLOOR = 0.5
# Half range of the block gradient search, relative to the carrier: a period mismatch
# and a tilt of the lines (0.079 = 4.5 degrees). Both are cut to 0.45 of the first
# grating lobe of the window lattice (one lobe per window step along the lines' normal,
# one per band across).
GRADIENT_RANGE = 0.08
GRADIENT_TILT = 0.079
# Fewest blocks with a gradient for the cubic start (nine unknowns, two equations per
# block).
CUBIC_START_BLOCKS = 9
# Degree of the smooth part in the first rounds; from the third round on DEGREE.
DEGREE_SCHEDULE = (2, 3)
MAX_ROUNDS = 8
# Largest projective terms (g, h in centred coordinates scaled to +-1) the projective
# part may have; a fit beyond it has bent a curl into a perspective with its pole near
# the page, and the projective part is then fitted affine.
MAX_PROJECTIVE_TERM = 0.2
# The fixed point iteration of the inverse map: accepted when no node is further than
# this many px from its target, after at most this many rounds.
INVERSE_TOLERANCE = 1e-3
INVERSE_ROUNDS = 40
# The straightened page is measured again. What it still asks for has to be inside the
# dead band or below this share of what the page asked for, or the page is refused: a
# map that left a third of its correction undone (27 of 72 px on the one wrong map of
# the truth tests, a sheet bent by 110 px with the contrast test lifted) has not
# straightened the page.
AFTER_SHARE = 0.25
# The noise margin of that check. The lines of the straightened page have to be as
# straight as the measurement can tell: res_after <= min(tolerance, AFTER_MARGIN x
# max(res_proj, res_field) + AFTER_SLACK periods). Without a margin a page whose
# distortion is purely projective (a rotation, a perspective) was refused on a coin
# flip, its res_proj being at the noise floor before the warp: over 68 such generator
# pages res_after - res_proj ranged from -0.00153 to +0.00073 periods, the largest
# excess being 6 % of res_proj, and 12 of the 68 were refused. 15 % and 0.002 periods
# are about twice that excess; a wrong map is still caught by the tolerance and by
# AFTER_SHARE.
AFTER_MARGIN = 1.15
AFTER_SLACK = 0.002

# Option of measure_field() -> its value unless the caller gives another one. min_shift
# is the threshold of the estimator; the dead band of the stage is no option of
# measure_field() but an argument of straighten().
DEFAULTS = {
    "degree": DEGREE,
    "tol": TOLERANCE,
    "min_shift": ESTIMATOR_MIN_SHIFT_PX,
    "max_shift": MAX_SHIFT_PX,
    "lines": WINDOW_LINES,
    "band": BAND_ROWS,
    "overlap": WINDOW_OVERLAP,
    "blocks": START_BLOCKS,
    "passes": PASSES,
    "cover": MIN_COVER,
}
# How a carrier was found -> what a refusal calls it.
CARRIER_NAMES = {
    "page": "the page-wide period",
    "windows": "the window comb near 1 mm at 200 dpi",
}

# --- the record next to a mask --------------------------------------------------------
# The record of an applied page holds its map at a lattice of this many points along x
# and along y, the borders of the page included: the corners and the edges, where the
# map moves the page most, and its inside.
CHECK_POINTS = (9, 7)
# Largest disagreement in pixels, at those points, between the map a mask's page was
# straightened by and the map used now (PERSPECTIVE_MASK_SHIFT_TOLERANCE of the
# digitiser, which is for the corners of a homography).
MASK_SHIFT_TOLERANCE = 0.1


def _resolved(options):
    """The options with every default filled in; a key that is no option is refused."""
    unknown = sorted(set(options or {}) - set(DEFAULTS))
    if unknown:
        raise ValueError(
            f"{', '.join(unknown)}: no option of the sheet curl stage, which has "
            f"{', '.join(DEFAULTS)}"
        )
    out = dict(DEFAULTS)
    out.update(options or {})
    return out


# ----------------------------------------------------------------------------
# The field estimator
# ----------------------------------------------------------------------------
def _window_combs(profiles, period, lines, overlap, edge):
    """The windows along band profiles [bands, n] and their comb amplitude at any period.

    Derived from _grid_phase_field() of the digitiser, which has the window length
    (PERSPECTIVE_WINDOW_LINES lines), the step (half a window) and the band height
    fixed: here the length is lines x period, a window starts every length // overlap
    px, and with edge one more window ends on the last pixel, so that the windows
    reach the far edge of the page. The profiles are those of _band_profiles(darkness,
    band). With lines 32, overlap 2 and no edge window the amplitudes at the carrier
    are those of _grid_phase_field() (tests).
    Returns (starts, length, comb) with comb(p) the complex amplitudes [bands, windows]
    of the comb of period p, or None when the profiles are too short for one window.
    """
    count = profiles.shape[1]
    length = int(round(lines * period))
    if profiles.shape[0] == 0 or length < 2 or count < length:
        return None
    starts = np.arange(0, count - length + 1, max(length // overlap, 1))
    if edge and starts[-1] != count - length:
        starts = np.r_[starts, count - length]
    windows = profiles[:, starts[:, None] + np.arange(length)]
    # The mean of the single window, not of the page: uneven light is a slow ramp.
    centred = windows - windows.mean(axis=2, keepdims=True)
    # Pixel c covers [c, c+1). The Hann window leaves no side lobes, as in the comb of
    # the digitiser.
    coordinates = starts[:, None] + np.arange(length) + 0.5
    hann = np.hanning(length)

    def comb(comb_period):
        kernel = hann * np.exp(-2j * np.pi * coordinates / comb_period)
        return np.einsum("bwl,wl->bw", centred, kernel)

    return starts, length, comb


def _families(band_profiles, period, lines, overlap):
    """The phase windows of both line families at a carrier, and their contrast.

    band_profiles is [(profiles, band centres)] of the page (the vertical lines, u) and
    of the page transposed (the horizontal lines, v). Returns (families, contrast), or
    None when the page is too small for one window. The contrast is the comb of the
    windows at the carrier, summed over the windows, relative to their mean amplitude
    at the BACKGROUND frequencies; the smaller one of the two families.
    """
    families, contrast = [], []
    for axis, (profiles, centres) in enumerate(band_profiles):
        combs = _window_combs(profiles, period, lines, overlap, True)
        if combs is None:
            return None
        starts, length, comb = combs
        amplitudes = comb(period)
        background = np.mean(
            [np.abs(comb(period / factor)).sum() for factor in BACKGROUND]
        )
        contrast.append(float(np.abs(amplitudes).sum() / max(background, 1e-9)))
        along, across = np.meshgrid(starts + length / 2, centres)
        x, y = (along, across) if axis == 0 else (across, along)
        families.append(
            {
                "amplitudes": amplitudes.ravel(),
                "x": x.ravel(),
                "y": y.ravel(),
                "length": length,
            }
        )
    return families, min(contrast)


def _window_period(D, band_profiles, lines, overlap):
    """Period in px of the strongest comb of the windows near the 1 mm pitch at 200 dpi.

    The amplitudes of the windows (their length that of the 200 dpi pitch) are summed
    over both line families, window by window without their phases, at CARRIER_STEPS
    line frequencies over the dead band of the resolution stage, and the top is refined
    by a parabola. A window does not ask the lines to be coherent over the page, which
    is what the page-wide spectrum of _grid_line_period() needs. NaN when the page is
    too small or shows nothing there.
    """
    centre = float(D.GRID_LINE_PERIOD_200DPI)
    half = float(D.RESOLUTION_DEAD_BAND)
    scales = 1 + np.linspace(-half, half, CARRIER_STEPS)
    total = np.zeros(len(scales))
    for profiles, _ in band_profiles:
        combs = _window_combs(profiles, centre, lines, overlap, True)
        if combs is None:
            return float("nan")
        total += np.array([np.abs(combs[2](centre / scale)).sum() for scale in scales])
    best = int(np.argmax(total))
    if not total[best] > 0:
        return float("nan")
    scale = scales[best]
    if 0 < best < len(scales) - 1:
        left, peak, right = total[best - 1 : best + 2]
        curvature = left - 2 * peak + right
        if curvature < 0:
            scale += 0.5 * (left - right) / curvature * (scales[1] - scales[0])
    return float(centre / scale)


def _carrier(D, page_period, band_profiles, lines, overlap):
    """The carrier of the phase windows: a dict (name, period, families, contrast,
    others), or the reason why the page has none.

    Two candidates: 'page', the strongest comb of the page-wide profile
    (_grid_line_period(), as the perspective stage takes it), and 'windows',
    _window_period(). On a blurred sheet that is bent the lines are not coherent over
    the page, the 1 mm peak of the page-wide spectrum smears in proportion to its
    frequency, and what is left on top is a harmonic of the bold 5 mm lines (about
    13 px, a third of 5 mm, on synthetic photo and blur pages from 15 px of curl): the
    windows have no comb there, and the page would be refused as 'no grid lines'. So
    both candidates are measured with the windows themselves, and the one with the
    larger contrast is the carrier. When the two are the same comb for a window (their
    periods differ by less than one line over its length) the page-wide one stays: it
    is the sharper number of the two on a flat page.
    """
    periods = (
        ("page", page_period),
        ("windows", _window_period(D, band_profiles, lines, overlap)),
    )
    if not any(np.isfinite(period) for _, period in periods):
        return "no grid line period found"
    found = []
    for name, period in periods:
        measured = (
            _families(band_profiles, float(period), lines, overlap)
            if np.isfinite(period)
            else None
        )
        if measured is not None:
            found.append(
                {
                    "name": name,
                    "period": float(period),
                    "families": measured[0],
                    "contrast": measured[1],
                }
            )
    if not found:
        return "image too small for the grid line windows"
    chosen = found[0]
    if len(found) == 2:
        page, windows = found
        another_comb = abs(windows["period"] / page["period"] - 1) > 1 / lines
        if another_comb and windows["contrast"] > page["contrast"]:
            chosen = windows
    chosen["others"] = [candidate for candidate in found if candidate is not chosen]
    return chosen


def _least_squares_gauge(u, v, x, y):
    """(s, ax, ay) of T = (ax, ay) + s (u, v): the translation and the one scale for
    which the sum of |T - (x, y)|^2 over the points given is smallest. No rotation: the
    grid lines give the orientation of the page. (nan, nan, nan) when (u, v) does not
    vary."""
    u, v, x, y = (np.asarray(values, float).ravel() for values in (u, v, x, y))
    um, vm, xm, ym = u.mean(), v.mean(), x.mean(), y.mean()
    power = float(np.sum((u - um) ** 2 + (v - vm) ** 2))
    if not power > 0:
        return float("nan"), float("nan"), float("nan")
    scale = float(np.sum((u - um) * (x - xm) + (v - vm) * (y - ym))) / power
    return scale, float(xm - scale * um), float(ym - scale * vm)


def _terms(degree):
    """Number of products x^i y^j with i + j <= degree."""
    return (degree + 1) * (degree + 2) // 2 if degree > 0 else 0


def _basis(xn, yn, degree):
    """Legendre products L_i(x) L_j(y), i + j <= degree, ordered by total degree:
    [n, terms].

    In centred coordinates scaled to +-1 on the longer side. Ordered by total degree,
    so that the first _terms(d) columns are the basis of degree d."""
    lx = np.polynomial.legendre.legvander(xn, degree)
    ly = np.polynomial.legendre.legvander(yn, degree)
    return np.stack(
        [
            lx[:, i] * ly[:, total - i]
            for total in range(degree + 1)
            for i in range(total + 1)
        ],
        axis=1,
    )


def _model_start(D, amplitudes, x, y, ranges, blocks, width, height):
    """Start of the phase model of one line family, in cycles relative to the carrier.

    Derived from _phase_model_start() of the digitiser, which lays a plane through the
    gradients of 3 x 3 blocks of windows and so integrates them to a quadratic: the
    lines of a curled sheet move by several periods against any quadratic. Here the
    page is cut into blocks x blocks blocks, the gradient of each is found by
    _phase_gradient() (a coherent sum, nothing is unwrapped), and a quadratic is laid
    through the gradients, which integrates to a cubic. A block whose gradient is more
    than four robust sigmas off that quadratic is left out once (text, a border). With
    fewer than CUBIC_START_BLOCKS blocks the start is the quadratic of the perspective
    stage, with fewer than PERSPECTIVE_MIN_BLOCKS + 1 a plane through the gradient of
    the whole page.
    Returns the model at every window, the constant included.
    """
    scale = max(width, height) / 2
    column = np.clip((x / width * blocks).astype(int), 0, blocks - 1)
    row = np.clip((y / height * blocks).astype(int), 0, blocks - 1)
    index = column * blocks + row
    parts = [
        index == block
        for block in np.unique(index)
        if np.count_nonzero(index == block) >= D.PERSPECTIVE_MIN_BLOCK_WINDOWS
    ]
    if len(parts) <= D.PERSPECTIVE_MIN_BLOCKS:
        # The whole page, so that there is one gradient to fall back on.
        parts = [np.ones(len(x), bool)] + parts
    rows, gradients, weights = [], [], []
    for inside in parts:
        fx, fy, peak = D._phase_gradient(amplitudes[inside], x[inside], y[inside], ranges)
        cx = (x[inside].mean() - width / 2) / scale
        cy = (y[inside].mean() - height / 2) / scale
        # The gradient of a x + b y + c x^2 + d x y + e y^2 + f x^3 + g x^2 y + h x y^2
        # + k y^3.
        rows += [
            [1, 0, 2 * cx, cy, 0, 3 * cx * cx, 2 * cx * cy, cy * cy, 0],
            [0, 1, 0, cx, 2 * cy, 0, cx * cx, 2 * cx * cy, 3 * cy * cy],
        ]
        gradients += [fx * scale, fy * scale]
        weights += [np.abs(peak)] * 2
    terms = (
        9
        if len(parts) >= CUBIC_START_BLOCKS
        else 5
        if len(parts) > D.PERSPECTIVE_MIN_BLOCKS
        else 2
    )
    rows, gradients, weights = (
        np.array(rows, float)[:, :terms],
        np.array(gradients),
        np.array(weights),
    )
    keep = np.ones(len(gradients), bool)
    for _ in range(2):
        root = np.sqrt(weights[keep])[:, None]
        solution = np.linalg.lstsq(
            rows[keep] * root, gradients[keep] * root[:, 0], rcond=None
        )[0]
        off = gradients - rows @ solution
        off = np.hypot(off[0::2], off[1::2])
        good = off <= 4 * max(1.4826 * np.median(off), 1e-9)
        if good.all() or 2 * np.count_nonzero(good) < terms + 4:
            break
        keep = np.repeat(good, 2)
    xn, yn = (x - width / 2) / scale, (y - height / 2) / scale
    model = (
        np.stack(
            [xn, yn, xn * xn, xn * yn, yn * yn, xn**3, xn * xn * yn, xn * yn * yn, yn**3][
                :terms
            ],
            axis=1,
        )
        @ solution
    )
    # The constant the gradients say nothing about, from the coherent sum again.
    return model + np.angle(np.sum(amplitudes * np.exp(-2j * np.pi * model))) / (
        2 * np.pi
    )


class Field:
    """A straightening map T(x, y) -> (X, Y) of a page, as its displacement on a lattice.

    Coordinates are those of the digitiser: pixel c covers [c, c+1). Node (i, j) of the
    lattice sits at (j * step, i * step), dx and dy are X - x and Y - y at the nodes,
    bilinear in between and held beyond the outer nodes. straighten() is the map,
    warp_array() and warp_labels() resample a page and its label mask into the straight
    frame through the inverse, which invert() finds on the same lattice.
    """

    def __init__(self, width, height, step, dx, dy, measured=None):
        self.width, self.height, self.step = int(width), int(height), int(step)
        self.dx, self.dy = np.asarray(dx, float), np.asarray(dy, float)
        self.measured = measured  # nodes inside the area both line families cover
        self.ex = self.ey = None  # the displacement of the inverse at the target nodes
        self.inverse_error = float("nan")
        self.page = None  # the page measure_field() was given, straightened: [C, H, W]

    def _at(self, grid, x, y):
        coordinates = np.stack(
            [np.asarray(y, float) / self.step, np.asarray(x, float) / self.step]
        )
        return ndimage.map_coordinates(grid, coordinates, order=1, mode="nearest")

    def straighten(self, x, y):
        """(X, Y) of the points (x, y) of the page as it is."""
        x, y = np.asarray(x, float), np.asarray(y, float)
        return x + self._at(self.dx, x, y), y + self._at(self.dy, x, y)

    def compose(self, later):
        """The map of this field followed by another one, measured on the page this one
        made."""
        ny, nx = self.dx.shape
        x, y = np.meshgrid(
            np.arange(nx) * float(self.step), np.arange(ny) * float(self.step)
        )
        X, Y = x + self.dx, y + self.dy
        return Field(
            self.width,
            self.height,
            self.step,
            self.dx + later._at(later.dx, X, Y),
            self.dy + later._at(later.dy, X, Y),
            self.measured,
        )

    def jacobian(self):
        """Smallest determinant of the Jacobian of the map in every cell of the
        lattice, [ny - 1, nx - 1].

        The map is bilinear inside a cell, so its determinant there is linear in x and
        in y and smallest at one of the four corners, taken with the edges of that cell
        itself. (Central differences over two cells read 0.465 where one cell had
        0.277.)"""
        ny, nx = self.dx.shape
        x, y = np.meshgrid(
            np.arange(nx) * float(self.step), np.arange(ny) * float(self.step)
        )
        X, Y = x + self.dx, y + self.dy
        # The edges along x, and those along y.
        X_x, Y_x = np.diff(X, axis=1) / self.step, np.diff(Y, axis=1) / self.step
        X_y, Y_y = np.diff(X, axis=0) / self.step, np.diff(Y, axis=0) / self.step
        upper, lower = slice(None, -1), slice(1, None)
        corners = [
            X_x[row] * Y_y[:, column] - X_y[:, column] * Y_x[row]
            for row in (upper, lower)
            for column in (upper, lower)
        ]
        return np.min(corners, axis=0)

    def invert(self):
        """The source of every node of the straight frame, by a fixed point iteration.

        p <- q - d(p) from p = q - d(q). The iteration converges while the displacement
        d is a contraction: a map that folds is not one, and neither is a map that
        stretches the page by more than a factor of two somewhere without folding.
        Returns the largest |T(p) - q| over the nodes in px, which is the convergence
        check: the caller refuses the page above INVERSE_TOLERANCE.
        """
        ny, nx = self.dx.shape
        qx, qy = np.meshgrid(
            np.arange(nx) * float(self.step), np.arange(ny) * float(self.step)
        )
        px, py = qx - self.dx, qy - self.dy
        error = float("inf")
        for _ in range(INVERSE_ROUNDS):
            fx, fy = self._at(self.dx, px, py), self._at(self.dy, px, py)
            error = float(np.max(np.hypot(px + fx - qx, py + fy - qy)))
            if error <= INVERSE_TOLERANCE:
                break
            px, py = qx - fx, qy - fy
        self.ex, self.ey, self.inverse_error = px - qx, py - qy, error
        return error

    def _full(self, grid):
        """A lattice displacement at every pixel centre (bilinear), [H, W]."""
        xs = (np.arange(self.width) + 0.5) / self.step
        ys = (np.arange(self.height) + 0.5) / self.step
        i0 = np.minimum(xs.astype(int), grid.shape[1] - 2)
        tx = xs - i0
        rows = grid[:, i0] * (1 - tx) + grid[:, i0 + 1] * tx
        j0 = np.minimum(ys.astype(int), grid.shape[0] - 2)
        ty = (ys - j0)[:, None]
        return rows[j0] * (1 - ty) + rows[j0 + 1] * ty

    def maps(self):
        """The cv2.remap maps of the warp: for every pixel of the straight frame its
        source.

        cv2 has the pixel centres at integers and the pipeline at i + 0.5 (see
        warp_page() of the digitiser), so the source of target pixel j, whose centre is
        j + 0.5, is j + 0.5 + e - 0.5 in cv2's terms."""
        if self.ex is None:
            self.invert()
        map_x = (np.arange(self.width)[None, :] + self._full(self.ex)).astype(np.float32)
        map_y = (np.arange(self.height)[:, None] + self._full(self.ey)).astype(np.float32)
        return map_x, map_y

    def warp_array(self, array, nearest=False, maps=None):
        """A [C, H, W] array in the straight frame: bicubic with a black border as
        warp_page() of the digitiser, or nearest with border 0 for a label mask (no
        value that was not in the mask)."""
        map_x, map_y = self.maps() if maps is None else maps
        channels = np.ascontiguousarray(np.moveaxis(np.asarray(array), 0, 2))
        warped = cv2.remap(
            channels,
            map_x,
            map_y,
            cv2.INTER_NEAREST if nearest else cv2.INTER_CUBIC,
            borderMode=cv2.BORDER_CONSTANT,
            borderValue=(0, 0, 0, 0),
        )
        if warped.ndim == 2:  # cv2 drops a single channel
            warped = warped[:, :, None]
        return np.ascontiguousarray(np.moveaxis(warped, 2, 0))

    def warp_labels(self, array, maps=None):
        """A label mask [1, H, W] in the straight frame, with no value that was not in
        it: nearest with border 0.

        A binary mask warped nearest moves its edge by up to half a pixel against the
        ink of the bicubically warped page, and the digitiser reads the time of a steep
        stroke off the columns its mask covers: on pixel-perfect generator pages that
        costs as much as the error of the map, which is one reason for the dead band.
        """
        labels = np.asarray(array)
        return self.warp_array(labels, True, maps)


def _refusal(info, reason, started):
    info["decision"], info["reason"] = "refused", reason
    info["seconds"] = time.time() - started
    return None, info


def _blank_info(options):
    nan = float("nan")
    return {
        "decision": "refused",
        "reason": "",
        "period": nan,
        "windows_u": 0,
        "windows_v": 0,
        "share": nan,
        "res_proj": nan,
        "res_field": nan,
        "shift_max": nan,
        "shift_rms": nan,
        "degree": int(options["degree"]),
        "contrast": nan,
        "cover": nan,
        "jac_min": nan,
        "proj_shift": nan,
        "rounds": 0,
        "passes": 0,
        "res_after": nan,
        "shift_after": nan,
        "inverse": nan,
        "seconds": 0.0,
        "carrier": "none",
        "scale": nan,
        "gauge_dx": nan,
        "gauge_dy": nan,
        "shift_all": nan,
        "share_after": nan,
        "cover_after": nan,
        "res_all": nan,
    }


def _measure_once(D, array, opt):
    """One measurement of the map of a page [C, H, W]: (Field or None, info).

    The field is None when info["reason"] says why the page gives none. Nothing is
    decided here about the dead band or about applying it; measure_field() does that.
    """
    started = time.time()
    info = _blank_info(opt)
    height, width = int(array.shape[1]), int(array.shape[2])
    darkness = D._darkness(array)

    # The carrier: the page-wide period as the perspective stage finds it, or the
    # strongest comb of the windows near the 1 mm pitch of a 200 dpi page, whichever
    # the windows carry with the larger contrast (_carrier()).
    profiles, _ = D._band_profiles(darkness)
    if profiles.shape[0] == 0:
        return _refusal(info, "image too small for the grid line profile", started)
    band, lines, overlap = int(opt["band"]), int(opt["lines"]), int(opt["overlap"])
    # axis 0: bands of rows cut along x, the vertical lines, u; axis 1: the page
    # transposed.
    band_profiles = [
        D._band_profiles(darkness.T if axis == 1 else darkness, band) for axis in (0, 1)
    ]
    if any(family_profiles.shape[0] == 0 for family_profiles, _ in band_profiles):
        return _refusal(info, "image too small for the grid line windows", started)
    carrier = _carrier(D, D._grid_line_period(profiles), band_profiles, lines, overlap)
    if isinstance(carrier, str):
        return _refusal(info, carrier, started)
    period, families = carrier["period"], carrier["families"]
    info["period"], info["carrier"], info["contrast"] = (
        period,
        carrier["name"],
        carrier["contrast"],
    )
    # The page-wide contrast of the perspective stage is no test here: the comb of a
    # curled sheet is not coherent over the page, which is one of the reasons the
    # stages refuse it. This one is local.
    if not info["contrast"] >= MIN_CONTRAST:
        seen = "; ".join(
            f"{candidate['contrast']:.1f} at {candidate['period']:.2f} px, "
            f"{CARRIER_NAMES[candidate['name']]}"
            for candidate in [carrier] + carrier["others"]
        )
        return _refusal(info, f"no grid lines found (contrast {seen})", started)

    step = max(families[0]["length"] // overlap, 1)
    along_range = min(GRADIENT_RANGE / period, 0.45 / step)
    across_range = min(GRADIENT_TILT / period, 0.45 / band)
    points, values, weights, axes, model = [], [], [], [], []
    for axis, family in enumerate(families):
        amplitude = np.abs(family["amplitudes"])
        # Strict, as in the perspective stage: a family without lines, all amplitudes
        # zero, drops out.
        keep = amplitude > D.GRID_LINE_MIN_BAND_AMPLITUDE * np.median(amplitude)
        if np.count_nonzero(keep) < D.PERSPECTIVE_MIN_WINDOWS:
            return _refusal(
                info,
                f"only {np.count_nonzero(keep)} windows with "
                f"{'vertical' if axis == 0 else 'horizontal'} grid lines",
                started,
            )
        complex_amplitude, x, y = (
            family["amplitudes"][keep],
            family["x"][keep],
            family["y"][keep],
        )
        # The lines of a window sit at (psi + k) * period, so the grid coordinate of
        # its centre is x / period - psi for the vertical family, modulo one line.
        psi = -np.angle(complex_amplitude) / (2 * np.pi)
        carrier = (x if axis == 0 else y) / period
        start = _model_start(
            D,
            complex_amplitude,
            x,
            y,
            (along_range, across_range) if axis == 0 else (across_range, along_range),
            int(opt["blocks"]),
            width,
            height,
        )
        points.append(np.stack([x, y], axis=1))
        values.append(carrier - psi)
        weights.append(amplitude[keep])
        axes.append(np.full(x.shape, axis))
        model.append(carrier + start)
    points, values, weights = (
        np.concatenate(points),
        np.concatenate(values),
        np.concatenate(weights),
    )
    axes, model = np.concatenate(axes), np.concatenate(model)
    scale = max(width, height) / 2
    normalised = (points - [width / 2, height / 2]) / scale
    degree = int(opt["degree"])
    basis = (
        _basis(normalised[:, 0], normalised[:, 1], degree)
        if degree > 0
        else np.zeros((len(values), 0))
    )

    def unwrap(against):
        # Against the model, never against a neighbour, and the inliers picked from
        # scratch: _unwrap_windows(), one family at a time, each with its own sigma.
        unwrapped, keep = np.empty_like(values), np.zeros(len(values), bool)
        for axis in (0, 1):
            family = axes == axis
            unwrapped[family], _, keep[family] = D._unwrap_windows(
                values[family], against[family], period
            )
        return unwrapped, keep

    def fit(unwrapped, keep, smooth_degree, projective):
        # The projective part: both families jointly, as the perspective stage fits it
        # ...
        matrix = D._grid_map_least_squares(
            normalised[keep], unwrapped[keep], weights[keep], axes[keep], projective
        )
        if projective and max(abs(matrix[2, 0]), abs(matrix[2, 1])) > MAX_PROJECTIVE_TERM:
            matrix = D._grid_map_least_squares(
                normalised[keep], unwrapped[keep], weights[keep], axes[keep], False
            )
        projected = D._apply_homography(matrix, normalised)[np.arange(len(axes)), axes]
        # ... and the smooth part of each family on what the projective part leaves.
        count = _terms(smooth_degree)
        coefficients, smooth = [], np.zeros(len(values))
        for axis in (0, 1):
            family = axes == axis
            inside = keep & family
            if count == 0:
                coefficients.append(np.zeros(0))
                continue
            root = np.sqrt(weights[inside])[:, None]
            solution = np.linalg.lstsq(
                basis[inside][:, :count] * root,
                (unwrapped[inside] - projected[inside]) * root[:, 0],
                rcond=None,
            )[0]
            coefficients.append(solution)
            smooth[family] = basis[family][:, :count] @ solution
        return matrix, coefficients, projected, projected + smooth

    def short(keep):
        for axis in (0, 1):
            count = int(np.count_nonzero(keep & (axes == axis)))
            if count < D.PERSPECTIVE_MIN_WINDOWS:
                return (
                    f"only {count} windows agree on the "
                    f"{'vertical' if axis == 0 else 'horizontal'} grid lines"
                )
        return ""

    schedule = [min(low, degree) for low in DEGREE_SCHEDULE] + [degree]
    turns = None
    for round_ in range(MAX_ROUNDS):
        unwrapped, keep = unwrap(model)
        if short(keep):
            return _refusal(info, short(keep), started)
        # The first round fits the projective part affine, as the perspective stage
        # does, and the degree is raised round by round: a low degree is carried by the
        # bulk of the windows, and the windows a start put a line off come back as
        # inliers once the model is right.
        _, _, _, model = fit(
            unwrapped, keep, schedule[min(round_, len(schedule) - 1)], round_ > 0
        )
        info["rounds"] = round_ + 1
        now = np.round(values - model)
        if (
            round_ >= len(schedule) - 1
            and turns is not None
            and np.array_equal(now, turns)
        ):
            break  # no window changes its line any more
        turns = now
    unwrapped, keep = unwrap(model)
    if short(keep):
        return _refusal(info, short(keep), started)
    matrix, coefficients, projected, model = fit(unwrapped, keep, degree, True)
    info["windows_u"] = int(np.count_nonzero(keep & (axes == 0)))
    info["windows_v"] = int(np.count_nonzero(keep & (axes == 1)))
    info["share"] = float(np.count_nonzero(keep) / len(keep))
    info["res_field"] = float(np.sqrt(np.mean((unwrapped - model)[keep] ** 2)))
    info["res_proj"] = float(np.sqrt(np.mean((unwrapped - projected)[keep] ** 2)))
    # The same model over every window with a comb, the outliers included, each on the
    # line of the model it is nearest to: a sheet that is bent locally leaves its
    # windows here, while res_field, over the inliers the model picks itself, does not
    # see them. 0.289 = no relation.
    off_model = values - model
    info["res_all"] = float(np.sqrt(np.mean((off_model - np.round(off_model)) ** 2)))
    if not info["res_field"] <= opt["tol"]:
        return _refusal(
            info,
            f"grid line phase is {info['res_field']:.3f} periods off the field",
            started,
        )
    for axis in (0, 1):
        span = np.ptp(points[keep & (axes == axis)], axis=0)
        if (
            span[0] < D.PERSPECTIVE_MIN_SPAN * width
            or span[1] < D.PERSPECTIVE_MIN_SPAN * height
        ):
            return _refusal(
                info,
                f"{'vertical' if axis == 0 else 'horizontal'} grid lines only span "
                f"{span[0]:.0f} x {span[1]:.0f} px",
                started,
            )

    # The map on the lattice. The projective part is evaluated everywhere, the smooth
    # part only where the inlier windows of its family lie (the union of their
    # supports); a node outside holds the smooth part of the nearest node inside: a
    # polynomial is not extrapolated.
    nx, ny = int(math.ceil(width / LATTICE)) + 1, int(math.ceil(height / LATTICE)) + 1
    gx, gy = np.meshgrid(np.arange(nx) * float(LATTICE), np.arange(ny) * float(LATTICE))
    nodes = np.stack(
        [(gx.ravel() - width / 2) / scale, (gy.ravel() - height / 2) / scale], axis=1
    )
    node_projected = D._apply_homography(matrix, nodes)
    node_basis = _basis(nodes[:, 0], nodes[:, 1], degree) if degree > 0 else None
    grid, covered = [], []
    for axis in (0, 1):
        cover = np.zeros((ny, nx), bool)
        half_x = (families[axis]["length"] if axis == 0 else band) / 2
        half_y = (band if axis == 0 else families[axis]["length"]) / 2
        # The rows left over below the last band are counted to it.
        slack_x = width % band if axis == 1 else 0
        slack_y = height % band if axis == 0 else 0
        for px, py in points[keep & (axes == axis)]:
            x0, x1 = (
                math.ceil((px - half_x) / LATTICE),
                math.floor((px + half_x + slack_x) / LATTICE),
            )
            y0, y1 = (
                math.ceil((py - half_y) / LATTICE),
                math.floor((py + half_y + slack_y) / LATTICE),
            )
            cover[max(y0, 0) : y1 + 1, max(x0, 0) : x1 + 1] = True
        smooth = np.zeros((ny, nx))
        if degree > 0:
            smooth = (node_basis @ coefficients[axis]).reshape(ny, nx)
            if not cover.all():
                nearest = ndimage.distance_transform_edt(
                    ~cover, return_distances=False, return_indices=True
                )
                smooth = smooth[nearest[0], nearest[1]]
        grid.append(node_projected[:, axis].reshape(ny, nx) + smooth)
        covered.append(cover)
    measured = covered[0] & covered[1]
    info["cover"] = float(measured.mean())
    if not info["cover"] >= opt["cover"] or not measured.any():
        return _refusal(
            info, f"grid lines cover {100 * info['cover']:.0f} % of the page", started
        )

    # The gauge. The grid lines give (u, v) up to their unit and their origin: T(x, y)
    # = a + s (u, v)(x, y), with the translation a and the one scale s for which the
    # squared displacement T - (x, y) over the measured nodes is smallest. No rotation:
    # the lines give the orientation. (Keeping the page centre and the area there, the
    # rule of _rectify_homography(), makes whatever the sheet did at that one point a
    # shift and a zoom of the whole page.)
    u, v = grid
    unit, ax, ay = _least_squares_gauge(
        u[measured], v[measured], gx[measured], gy[measured]
    )
    if not unit > 0:
        return _refusal(info, "the grid lines give no scale for the page", started)
    field = Field(
        width, height, LATTICE, ax + unit * u - gx, ay + unit * v - gy, measured
    )
    pixel = matrix @ np.array(
        [
            [1 / scale, 0, -width / (2 * scale)],
            [0, 1 / scale, -height / (2 * scale)],
            [0, 0, 1],
        ]
    )
    info["proj_shift"] = float(
        D._homography_shift(D._rectify_homography(pixel, width, height), width, height)
    )
    reason = _judge(D, field, info, opt)
    if reason:
        return _refusal(info, reason, started)
    info["seconds"] = time.time() - started
    return field, info


def _judge(D, field, info, opt):
    """Shift, gauge and Jacobian of a map into info; '' or why the map is refused."""
    shift = np.hypot(field.dx, field.dy)
    measured = shift[field.measured]
    info["shift_max"], info["shift_rms"] = (
        float(measured.max()),
        float(np.sqrt(np.mean(measured**2))),
    )
    info["shift_all"] = float(shift.max())
    # The gauge against the one that keeps the page centre and the area there: where
    # the centre goes, and the scale of the map at the centre.
    cx, cy, h = field.width / 2, field.height / 2, float(field.step)
    X, Y = field.straighten(
        np.array([cx, cx + h, cx - h, cx, cx]), np.array([cy, cy, cy, cy + h, cy - h])
    )
    determinant = float((X[1] - X[2]) * (Y[3] - Y[4]) - (X[3] - X[4]) * (Y[1] - Y[2])) / (
        4 * h * h
    )
    info["gauge_dx"], info["gauge_dy"] = float(X[0] - cx), float(Y[0] - cy)
    info["scale"] = math.sqrt(determinant) if determinant > 0 else float("nan")
    info["jac_min"] = float(field.jacobian().min())
    if not info["jac_min"] > 0:
        return (
            f"the map folds (Jacobian determinant {info['jac_min']:.2f} in a cell of "
            f"the lattice)"
        )
    if info["jac_min"] < JACOBIAN_FLOOR:
        return (
            "the map stretches or squeezes the page by more than a factor of two "
            f"somewhere (Jacobian determinant {info['jac_min']:.2f} in a cell of the "
            "lattice)"
        )
    limit = (
        opt["max_shift"]
        if opt["max_shift"] is not None
        else D.PERSPECTIVE_MAX_SHIFT * field.width
    )
    if info["shift_all"] > limit:
        return (
            f"grid lines ask for a {info['shift_all']:.0f} px correction somewhere on "
            f"the page ({info['shift_max']:.0f} px where they were measured)"
        )
    return ""


def measure_field(image, options, D):
    """The straightening map of a page from its grid lines: (field, info).

    image is the page as the perspective stage of the digitiser left it, a [C, H, W]
    uint8 tensor or array; options a dict of the keys of DEFAULTS, None for the
    defaults; D the helpers of the digitiser (HELPERS). info["decision"] is 'applied'
    (the map is accepted and outside the dead band: field is the map and field.page the
    straightened page, a new array), 'dead band' (field is the map, nothing is to be
    touched) or 'refused' (field is None, info["reason"] says why). See the header for
    the method and the keys.
    """
    opt = _resolved(options)
    started = time.time()
    array = image.numpy() if torch.is_tensor(image) else np.asarray(image)
    if array.ndim != 3 or array.dtype != np.uint8:
        return _refusal(
            _blank_info(opt),
            f"page is {array.dtype} {list(array.shape)}, not uint8 [C, H, W]",
            started,
        )
    field, info = _measure_once(D, array, opt)
    if field is None:
        return _refusal(info, info["reason"], started)
    info["passes"] = 1
    better = info["res_field"] < info["res_proj"]
    # Inside the dead band nothing is touched. A smooth part that explains the lines no
    # better than the projective part alone is nothing to apply either, unless that
    # part itself moves the page by more than the dead band.
    if info["shift_max"] < opt["min_shift"] or (
        not better and info["proj_shift"] < opt["min_shift"]
    ):
        info["decision"], info["seconds"] = "dead band", time.time() - started
        return field, info

    first = dict(info)
    passes = int(opt["passes"])
    while True:
        info["inverse"] = field.invert()
        if not info["inverse"] <= INVERSE_TOLERANCE:
            return _refusal(
                info,
                "the map stretches the page too much for the inverse to converge "
                f"({info['inverse']:.3f} px off after {INVERSE_ROUNDS} rounds)",
                started,
            )
        straight = field.warp_array(array)
        # The straightened page is measured again: what is left on it is composed into
        # the map (as the perspective stage measures a second time on the warped page),
        # and the last measurement is the check that the warp did what the map says.
        again, check = _measure_once(D, straight, opt)
        if again is None:
            return _refusal(
                info,
                f"the straightened page gives no field ({check['reason']})",
                started,
            )
        if info["passes"] >= passes or check["shift_max"] < opt["min_shift"]:
            break
        field = field.compose(again)
        info["passes"] += 1
        reason = _judge(D, field, info, opt)
        if reason:
            return _refusal(info, reason, started)
    info["res_after"], info["shift_after"] = check["res_proj"], check["shift_max"]
    # Told, not judged: how much of the straightened page the check measurement stands
    # on.
    info["share_after"], info["cover_after"] = check["share"], check["cover"]
    # A field that is applied leaves the lines straight: within the tolerance of a
    # projective fit and no less straight than the measurement can tell (the noise
    # margin, see AFTER_MARGIN), and asking for no more than the dead band or
    # AFTER_SHARE of what the page asked for.
    limit = min(
        opt["tol"],
        AFTER_MARGIN * max(first["res_proj"], first["res_field"]) + AFTER_SLACK,
    )
    if not info["res_after"] <= limit:
        return _refusal(
            info,
            f"lines are not straight after the warp ({info['res_after']:.4f} periods "
            f"off a projective fit, {first['res_proj']:.4f} before, {limit:.4f} "
            f"allowed)",
            started,
        )
    if not info["shift_after"] < max(opt["min_shift"], AFTER_SHARE * info["shift_max"]):
        return _refusal(
            info,
            f"the straightened page asks for {info['shift_after']:.1f} px more, the "
            f"page as it was for {info['shift_max']:.1f} px",
            started,
        )
    field.page = straight
    info["decision"], info["seconds"] = "applied", time.time() - started
    return field, info


# ----------------------------------------------------------------------------
# The stage of the digitiser: one page, its lines, its QC values and its record
# ----------------------------------------------------------------------------
def estimator_options(options, dead_band=None):
    """The options the stage measures a page with, for a dead band of the stage.

    options are options of measure_field(), None for its defaults; dead_band is the
    dead band of the stage in px, None for MIN_SHIFT_PX. The threshold of the
    estimator stays what the options say, ESTIMATOR_MIN_SHIFT_PX unless they say
    otherwise, whatever the dead band is: only a dead band below it takes its place,
    so that no page is measured with a threshold above the dead band it is held to.
    """
    opt = _resolved(options)
    band = MIN_SHIFT_PX if dead_band is None else float(dead_band)
    opt["min_shift"] = min(opt["min_shift"], band)
    return opt


def straighten(image, options, D, dead_band=None):
    """measure_field() as the stage of the digitiser calls it: (field, info).

    options are those of measure_field(), and dead_band is the dead band of the stage
    in px, --sheet_curl_min_shift, None for MIN_SHIFT_PX. The page is measured with
    estimator_options(options, dead_band): the threshold of the estimator is not the
    dead band of the stage. A page whose map is accepted ('applied' for the estimator)
    and moves no measured node by the dead band, shift_max of the final map with all
    its passes composed, is a page in the dead band: its decision becomes 'dead band',
    info["accepted"] says that its map was measured, checked on the straightened page
    and accepted, and shift_max stays that of the map. A map that moves a node by
    exactly the dead band is applied. A dead band at or below the threshold of the
    estimator changes nothing: the stage is measure_field() with that threshold.

    field is the map of a page that is to be straightened, with field.page the
    straightened page, and None for every other page, the one in the dead band
    included: a page that is not 'applied' is left exactly as it is. An exception
    inside the measurement is a refusal as well, whatever was raised, a programming
    error included; its reason starts with 'error: ', so that one page whose numbers
    break a fit does not end the run of a folder: the page is kept, and the WARNING of
    a refused page names what was raised.
    The stage decides on the page alone. The warp of a mask that came with the page is
    the caller's, outside this function: a mask that cannot be warped (Field.warp_labels
    raises) is an error of the run, not a refusal of the page.
    """
    started = time.time()
    band = MIN_SHIFT_PX if dead_band is None else float(dead_band)
    try:
        opt = estimator_options(options, band)
        field, info = measure_field(image, opt, D)
    except Exception as error:
        info = _blank_info(_resolved(options))
        info["reason"] = f"error: {type(error).__name__}: {' '.join(str(error).split())}"
        info["seconds"] = time.time() - started
        return None, info
    if (
        info["decision"] == "applied"
        and band > opt["min_shift"]
        and info["shift_max"] < band
    ):
        # The map is a good one and a small one: inside the dead band of the stage
        # nothing is touched, as inside that of the estimator.
        info["decision"], info["accepted"] = "dead band", True
    return (field if info["decision"] == "applied" else None), info


def out_of_scope(options):
    """The info of a page the stage does not look at: nothing is measured."""
    info = _blank_info(_resolved(options))
    info["decision"] = "out of scope"
    return info


def warning_line(record, info):
    """The WARNING of a page that is refused: the page is kept as it is."""
    return (
        f"WARNING: sheet curl not straightened for record {record} "
        f"({info['reason']}), keeping the page as it is."
    )


def verbose_line(record, info):
    """The line of a page under --verbose: the decision and the numbers of the fit.

    A page that was not measured as far as a number shows nan there, and a page out of
    scope, which was not measured at all, has no numbers. 'dead band' is a page whose
    first map is inside the threshold of the estimator: nothing was warped, and 'after
    the warp' is nan. 'dead band of the accepted map' is a page whose map was composed,
    checked on the straightened page and accepted, and moves the page by less than the
    dead band of the stage: its numbers are those of that map, and the page is kept.
    """
    if info["decision"] == "out of scope":
        return (
            f"Sheet curl for record {record}: out of scope, not a page the paper "
            f"normalisation warped onto the page frame"
        )
    decision = info["decision"]
    if decision == "dead band" and info.get("accepted"):
        # Not the dead band of the first measurement: the map was composed, checked
        # and accepted, and is below the dead band of the stage.
        decision = "dead band of the accepted map"
    return (
        f"Sheet curl for record {record}: {decision}, carrier "
        f"{info['carrier']} {info['period']:.3f} px, windows {info['windows_u']} + "
        f"{info['windows_v']}, residual {info['res_proj']:.4f} periods off the "
        f"projective part, {info['res_field']:.4f} off the field, "
        f"{info['res_after']:.4f} after the warp, {info['res_all']:.4f} over all "
        f"windows, shift max {info['shift_max']:.2f} px, rms {info['shift_rms']:.2f} "
        f"px, whole page {info['shift_all']:.2f} px, scale {info['scale']:.5f}, "
        f"jac_min {info['jac_min']:.3f}, passes {info['passes']}, cover "
        f"{info['cover']:.3f}, share {info['share']:.3f}"
    )


def qc_values(info):
    """(sheet_curl, sheet_curl_shift_px) of a page for its QC row.

    The decision, with the reason of a refused page, and shift_max of a page that was
    applied or left in the dead band, NaN for every other one.
    """
    decision = info["decision"]
    text = f"refused: {info['reason']}" if decision == "refused" else decision
    moved = decision in ("applied", "dead band")
    return text, float(info["shift_max"]) if moved else float("nan")


def check_points(field):
    """The map of a field at the lattice of CHECK_POINTS: rows [x, y, X, Y]."""
    x, y = np.meshgrid(
        np.linspace(0.0, field.width, CHECK_POINTS[0]),
        np.linspace(0.0, field.height, CHECK_POINTS[1]),
    )
    x, y = x.ravel(), y.ravel()
    X, Y = field.straighten(x, y)
    return [[float(value) for value in row] for row in zip(x, y, X, Y)]


def mask_record(info, field, options, scope, dead_band=None):
    """The record of what the stage did to a page, in plain values for the JSON next to
    its mask.

    info and field are those of straighten() (out_of_scope() for a page that was not
    looked at), options and dead_band the ones straighten() was called with and scope
    the scope of the run. dead_band is the dead band of the stage; the options, as the
    page was measured with them (estimator_options(): min_shift is the threshold of the
    estimator), and the constants are what defines the map; points is the map itself,
    of an applied page only, see mask_frame().
    """
    opt = estimator_options(options, dead_band)
    shift = info["shift_max"]
    block = {
        "version": VERSION,
        "decision": info["decision"],
        "reason": info["reason"],
        "scope": scope,
        "dead_band": MIN_SHIFT_PX if dead_band is None else float(dead_band),
        "options": {key: opt[key] for key in DEFAULTS},
        "constants": {
            "lattice": LATTICE,
            "background": list(BACKGROUND),
            "min_contrast": MIN_CONTRAST,
            "carrier_steps": CARRIER_STEPS,
            "jacobian_floor": JACOBIAN_FLOOR,
            "gradient_range": GRADIENT_RANGE,
            "gradient_tilt": GRADIENT_TILT,
            "cubic_start_blocks": CUBIC_START_BLOCKS,
            "degree_schedule": list(DEGREE_SCHEDULE),
            "max_rounds": MAX_ROUNDS,
            "max_projective_term": MAX_PROJECTIVE_TERM,
            "inverse_tolerance": INVERSE_TOLERANCE,
            "inverse_rounds": INVERSE_ROUNDS,
            "after_share": AFTER_SHARE,
            "after_margin": AFTER_MARGIN,
            "after_slack": AFTER_SLACK,
        },
        # NaN is no JSON: a page without a map has none.
        "shift_max": float(shift) if np.isfinite(shift) else None,
    }
    if field is not None:
        block["page_size"] = [field.width, field.height]
        block["points"] = check_points(field)
    return block


def mask_frame(block, field, kept=""):
    """Whether a saved mask lives in the straightened frame, and whether it fits.

    block is the sheet curl record next to the mask, None for a mask without one (a
    JSON null is no record either); field is the map this run straightens the page by,
    None when it keeps the page as it is, and kept then says why it does, for the words
    of the problem.
    Returns (straightened, problem). straightened says that the mask is used as it is
    and never warped: a mask that was predicted on a straightened page, which only a
    record with the decision 'applied' says. A mask without a record, or with one of a
    page that was not straightened, was predicted on the page as the perspective stage
    left it, and is warped with the page when this run straightens it. problem is ''
    or what is wrong between the mask and the page of this run, in words that go after
    'mask of record <record> ' and before '; the mask does not fit.'.
    A record that is no object says nothing that can be read: it is a problem, and the
    mask is used as it is, since a mask that may live on a straightened page must not
    be warped on a guess. An applied record has to hold its whole map, every one of the
    CHECK_POINTS points and the size of the page they are on, or it is a record
    without the points of its map: a part of the points is no check of the map.
    """
    if block is None:
        return False, ""
    if not isinstance(block, dict):
        return True, "has a sheet curl record that is no object"
    if block.get("decision") != "applied":
        return False, ""
    if block.get("version") != VERSION:
        return True, (
            f"has a sheet curl record of version {block.get('version')!r}, this code "
            f"reads {VERSION!r}"
        )
    if field is None:
        return True, (
            "was predicted on a page whose sheet curl was straightened, the page is "
            f"now kept as it is{f' ({kept})' if kept else ''}"
        )
    try:
        points = np.array(block["points"], float).reshape(-1, 4)
        width, height = (float(value) for value in block["page_size"])
    except (KeyError, TypeError, ValueError):
        points, width, height = np.zeros((0, 4)), float("nan"), float("nan")
    if (
        len(points) != CHECK_POINTS[0] * CHECK_POINTS[1]
        or not np.isfinite(points).all()
        or (width, height) != (field.width, field.height)
    ):
        return True, "has a sheet curl record without the points of its map"
    X, Y = field.straighten(points[:, 0], points[:, 1])
    distance = float(np.max(np.hypot(X - points[:, 2], Y - points[:, 3])))
    if not distance <= MASK_SHIFT_TOLERANCE:
        # Two decimals: the tolerance is a tenth of a pixel, and a distance just above
        # it is not to read as the tolerance itself.
        return True, (
            "was predicted on a page straightened by a sheet curl map that is "
            f"{distance:.2f} px off the one used now"
        )
    return True, ""
