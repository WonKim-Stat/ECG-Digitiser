"""Unit tests for --paper_normalisation: src/run/paper_normalisation.py and the stage of
src/run/digitize.py that decides with it.

No model and no data files. A page is drawn with numpy and cv2 as the generator prints
one, a red 1 mm grid under the traces of a 3x4 layout with its rhythm strip, and it is
photographed by a known homography: printed at PRINT_SCALE on a Letter sheet that lies
on a darker table, under uneven light, blurred and with noise. In place of the model
run() is handed the label of that page, put by the same homographies into the frame the
page should be in by then: the stand-in takes no page of another size, and it notes
the digest of every page it is asked about. A photograph takes a moment to build and
to decide on, so the one most tests look at is built, digitised and saved once for
the module.
"""
import ast
import contextlib
import csv
import hashlib
import inspect
import io
import json
import os
import re
import shutil
import warnings
from functools import lru_cache
from types import SimpleNamespace

import cv2
import numpy as np
import pytest
import torch
import wfdb
from torchvision.io.image import write_png

from config import (
    FREQUENCY,
    LONG_SIGNAL_LENGTH_SEC,
    SHORT_SIGNAL_LENGTH_SEC,
    Y_SHIFT_RATIO,
)
from src.run import digitize, paper_normalisation

WIDTH, HEIGHT = 2200, 1700
# 1 mm at 200 dpi, the pitch of the printed grid lines of a generator page.
PERIOD = 200 / 25.4
# The columns of the layout on that page: their pitch and the line the first starts on.
PITCH = HEIGHT * digitize.PAGE_PITCH_RATIO
LINE0 = 118.62
SHORT_LEADS = {
    "I": 0, "III": 0, "aVR": 1, "aVL": 1, "aVF": 1,
    "V1": 2, "V2": 2, "V3": 2, "V4": 3, "V5": 3, "V6": 3,
}
LEADS = list(SHORT_LEADS) + ["II"]
# A quarter of a page with the grid of a whole one: at the 200 dpi scale, and cheap.
SMALL = (1100, 850)
# The photographs: size of the image, px per mm of the sheet and its corners TL, TR,
# BR, BL. A keystone, because that is what the grid line stages cannot read: a sheet
# seen from the front, turned or not, they read, and such a page is left to them. The
# large one is shrunk before it is warped, as every photograph of a camera is, the
# fast one is not.
KEYSTONE = (
    (3360, 2520), 10.0, ((525, 287.5), (2837.5, 287.5), (3100, 2287.5), (262.5, 2287.5))
)
FAST = ((2688, 2016), 8.0, ((420, 230), (2270, 230), (2480, 1830), (210, 1830)))
# The stages after the paper stage at their cheapest, for a page they are not asked
# about. The paper stage itself always measures with the grid lines.
CHEAP = ("--rotation", "hough", "--perspective", "off", "--resolution", "keep")
STAGES = (
    "normalise_resolution", "estimate_rotation", "estimate_perspective", "warp_page"
)
ONCE = {
    "normalise_resolution": 1, "estimate_rotation": 1, "estimate_perspective": 1,
    "warp_page": 0,
}
# What the record of every page holds, and what a normalised and a scaled one add.
RECORD_KEYS = [
    "version", "libraries", "decision", "reason", "input_size", "input_sha1",
    "output_size", "view_sha1",
]
NORMALISED_KEYS = ["pre_size", "warp", "homography", "orientation"]
SCALED_KEYS = ["resize", "scaled_from", "scale_factor"]
FITS_NOT = "the mask does not fit"
ANOTHER_VIEW = "the mask does not fit: the mask was predicted on another view."
# What every refusal ends on: the two ways out of it.
WAY_OUT = (
    " Give the page the mask was predicted on, or remove the paper_normalisation key "
    "from the mask's JSON to take the page as given."
)


# ------------------------------------------------------------------------ the pages
def analytic(t, k):
    """A slow sine and one narrow bump per second, another one for every lead."""
    return 0.3 * np.sin(2 * np.pi * (0.7 * t + k / 13.0)) + 0.6 * np.exp(
        -0.5 * (((t % 1.0) - 0.4 - 0.03 * k) / 0.03) ** 2
    )


@lru_cache(maxsize=None)
def _label():
    """The label mask of a 3x4 page with rhythm strip, traces three pixels wide.

    Every lead is its analytic signal at 25 mm/s and 10 mm/mV about the baseline row
    of its place in the layout. Cached, so no caller may write into it.
    """
    label = np.zeros((HEIGHT, WIDTH), dtype=np.uint8)
    x = np.arange(WIDTH) + 0.5
    for k, lead in enumerate(LEADS):
        is_long = lead == "II"
        column = 0 if is_long else SHORT_LEADS[lead]
        n_columns = digitize.NUM_COLUMNS if is_long else 1
        start = LINE0 - 0.5 + column * PITCH
        columns = np.flatnonzero((x >= start) & (x < start + n_columns * PITCH))
        # Page time: a lead of column c shows 2.5 c to 2.5 (c + 1) s.
        t = (x[columns] - (LINE0 - 0.5)) / PITCH * SHORT_SIGNAL_LENGTH_SEC
        ratio = Y_SHIFT_RATIO["full" if is_long else lead]
        rows = np.floor(
            digitize.baseline_row(ratio, HEIGHT) - analytic(t, k) / (6.25 / PITCH)
        ).astype(int)
        label[rows, columns] = digitize.LEAD_LABEL_MAPPING[lead]
    return cv2.dilate(label, np.ones((3, 3), np.uint8))


@lru_cache(maxsize=None)
def _page(size=(WIDTH, HEIGHT), traces=True):
    """A printed page: a red 1 mm grid, with the traces of _label() in black.

    The grid is red as on paper, which is what the paper search tells it from all
    else by, and the traces lie in the lower part of the page, which is what the
    page is put upright by: the layout leaves the top of the sheet to the grid.
    traces=False is the grid alone, for a page of another size. Cached.
    """
    width, height = size
    image = np.full((height, width, 3), 255, dtype=np.uint8)
    # Light red grid lines, every fifth one darker and thicker as on a printed page.
    for k in range(int(width / PERIOD) + 1):
        x = int(round(k * PERIOD))
        if x < width:
            heavy = k % 5 == 0
            cv2.line(image, (x, 0), (x, height - 1),
                     (255, 120, 120) if heavy else (255, 190, 190), 2 if heavy else 1)
    for k in range(int(height / PERIOD) + 1):
        y = int(round(k * PERIOD))
        if y < height:
            heavy = k % 5 == 0
            cv2.line(image, (0, y), (width - 1, y),
                     (255, 120, 120) if heavy else (255, 190, 190), 2 if heavy else 1)
    if traces:
        image[_label() > 0] = 0
    return image


def _tensor(np_image):
    """The page as run() reads it: [3, H, W] uint8."""
    return torch.from_numpy(np_image.transpose(2, 0, 1).copy())


def _tilted(image, angle_deg):
    """The content of a page turned by angle_deg about its centre, onto white."""
    height, width = image.shape[:2]
    matrix = cv2.getRotationMatrix2D((width / 2, height / 2), angle_deg, 1.0)
    return cv2.warpAffine(
        image,
        matrix,
        (width, height),
        flags=cv2.INTER_LINEAR,
        borderMode=cv2.BORDER_CONSTANT,
        borderValue=(255, 255, 255),
    )


def _flat_label(size=SMALL):
    """A flat trace per lead, in the column of the standard layout."""
    width, height = size
    label = np.zeros((height, width), dtype=np.uint8)
    pitch = width / 5
    for lead, value in digitize.LEAD_LABEL_MAPPING.items():
        column = int(digitize.STANDARD_LEAD_OFFSETS_SEC[lead] / 2.5)
        row = height // 6 + height // 17 * (value % 12)
        start = int(pitch / 2 + column * pitch)
        label[row, start : start + int(pitch)] = value
    return label


def _photograph(page, canvas, pmm, corners):
    """page printed on a Letter sheet that lies on a table, as a camera sees it.

    The sheet has pmm pixels per millimetre and the page is printed in its middle at
    PRINT_SCALE of its width, as the scanned pages are. corners is where the corners
    of the sheet come to lie in the photograph of canvas (width, height) px. The
    table is darker than the paper, blotched and grainy, the light falls off towards
    one corner, and the camera blurs and adds noise, the same every time.
    Returns (photo, homography from the pixels of page to those of the photo), with
    the centre of pixel (0, 0) at (0, 0) as cv2 has it.
    """
    rng = np.random.default_rng(0)
    paper_mm = paper_normalisation.PAPER_MM
    sheet_w, sheet_h = int(round(paper_mm[0] * pmm)), int(round(paper_mm[1] * pmm))
    k = paper_normalisation.PRINT_SCALE * sheet_w / page.shape[1]
    to_sheet = np.array(
        [
            [k, 0, sheet_w / 2 - k * page.shape[1] / 2 + 0.5 * k - 0.5],
            [0, k, sheet_h / 2 - k * page.shape[0] / 2 + 0.5 * k - 0.5],
            [0, 0, 1.0],
        ]
    )
    sheet = cv2.warpAffine(
        page, to_sheet[:2], (sheet_w, sheet_h), flags=cv2.INTER_LINEAR,
        borderMode=cv2.BORDER_CONSTANT, borderValue=(250, 250, 248),
    )
    width, height = canvas
    edges = np.float32(
        [[-0.5, -0.5], [sheet_w - 0.5, -0.5], [sheet_w - 0.5, sheet_h - 0.5],
         [-0.5, sheet_h - 0.5]]
    )
    to_photo = cv2.getPerspectiveTransform(edges, np.float32(corners)).astype(float)
    lying = cv2.warpPerspective(
        sheet, to_photo, canvas, flags=cv2.INTER_LINEAR, borderMode=cv2.BORDER_REPLICATE
    )
    # How much of each pixel the sheet covers: a sheet of ones with a rim of zeros.
    ones = np.zeros((sheet_h + 2, sheet_w + 2), dtype=np.float32)
    ones[1:-1, 1:-1] = 1.0
    rim = np.array([[1, 0, -1.0], [0, 1, -1.0], [0, 0, 1]])
    cover = cv2.warpPerspective(ones, to_photo @ rim, canvas, flags=cv2.INTER_LINEAR)
    low = cv2.GaussianBlur(
        rng.standard_normal((height // 8, width // 8)).astype(np.float32), (0, 0), 6
    )
    low = cv2.resize(low / (low.std() + 1e-9), canvas, interpolation=cv2.INTER_CUBIC)
    grain = cv2.GaussianBlur(
        rng.standard_normal((height, width)).astype(np.float32), (0, 0), 1.2
    )
    grain /= grain.std() + 1e-9
    table = np.float32([96, 80, 64])[None, None] + (14 * low + 6 * grain)[..., None]
    photo = cover[..., None] * lying.astype(np.float32) + (1 - cover[..., None]) * table
    y = np.arange(height, dtype=np.float32)[:, None]
    x = np.arange(width, dtype=np.float32)[None, :]
    light = 0.82 + 0.18 * (1 - y / height) * (0.7 + 0.3 * x / width)
    photo = cv2.GaussianBlur(photo * light[..., None], (0, 0), 0.8)
    # A few rows at a time: the noise of a whole photograph is 200 MB of doubles.
    for top in range(0, height, 256):
        rows = photo[top : top + 256]
        rows += rng.normal(0, 2.0, rows.shape).astype(np.float32)
    return np.clip(photo + 0.5, 0, 255).astype(np.uint8), to_photo @ to_sheet


def _shot(layout, move=(0.0, 0.0)):
    """The page photographed in a layout, its sheet pushed by move pixels."""
    canvas, pmm, corners = layout
    corners = np.array(corners, dtype=float) + move
    image, to_photo = _photograph(_page(), canvas, pmm, corners)
    return SimpleNamespace(image=image, corners=corners, to_photo=to_photo)


def _label_in(homography, size):
    """The label of the page in another frame: page pixels to those of that frame."""
    return cv2.warpPerspective(_label(), homography, size, flags=cv2.INTER_NEAREST)


def _onto_page(corners):
    """The frame of a normalised page: the corners of the sheet on those of the page."""
    return cv2.getPerspectiveTransform(
        np.float32(corners), np.float32(paper_normalisation.PAGE_CORNERS)
    ).astype(float)


def _pixels(view):
    """The H x W x 3 array of a page tensor."""
    return np.ascontiguousarray(view.permute(1, 2, 0).numpy())


# ------------------------------------------------------------------------- the runs
def _folder(path, **pages):
    """A data folder with every page written as <record>.png."""
    path.mkdir(parents=True, exist_ok=True)
    for record, page in pages.items():
        write_png(page, str(path / f"{record}.png"), compression_level=1)
    return path


def _run(data, out, *flags, label=None, pages=None):
    """run() over the pages of a folder, the label in place of the model.

    label is the mask the model would give for the page it is asked about, an array
    or a function of that page. None is a run that brings its masks and may not ask.
    pages are {record: tensor} that read_image hands out in place of the files of
    that name, for a photograph that is not worth a PNG of its own: the folder then
    only holds the names.
    Returns what was printed (log), how often each stage was called (calls), every
    page normalise_paper() was given next to what it returned (decided), and the
    shape and the digest of every page the model was asked about (asked).
    """
    calls = {name: 0 for name in STAGES}
    decided, asked = [], []
    real_paper, real_read = digitize.normalise_paper, digitize.read_image
    if pages is not None:
        data.mkdir(parents=True, exist_ok=True)
        for record in pages:
            (data / f"{record}.png").touch()

    real = {name: getattr(digitize, name) for name in STAGES}

    # Every wrapper takes the arguments of the function it stands for and no others,
    # as the wrappers of the drivers do: a call they could not take fails here.
    def normalise_resolution(image):
        calls["normalise_resolution"] += 1
        return real["normalise_resolution"](image)

    def estimate_rotation(image, method="hough"):
        calls["estimate_rotation"] += 1
        return real["estimate_rotation"](image, method)

    def estimate_perspective(image):
        calls["estimate_perspective"] += 1
        return real["estimate_perspective"](image)

    def warp_page(image, homography):
        calls["warp_page"] += 1
        return real["warp_page"](image, homography)

    def paper(image):
        result = real_paper(image)
        decided.append((image, result))
        return result

    def read(path):
        record = os.path.basename(path)[: -len(".png")]
        if pages is not None and record in pages:
            return pages[record]
        return real_read(path)

    def model(
        image, dataset_name, model_folder, device="auto", disable_tta=False, fold="all"
    ):
        assert label is not None, "the model was asked for a mask"
        mask = label(image) if callable(label) else label
        assert image.dtype == torch.uint8 and mask.shape == tuple(image.shape[1:])
        asked.append((tuple(image.shape), digitize.page_sha1(image)))
        # A copy: the label may be a cached one, and run() owns the mask it is given.
        return torch.from_numpy(mask.copy())[None]

    log = io.StringIO()
    with pytest.MonkeyPatch.context() as patch:
        for stage in (
            normalise_resolution, estimate_rotation, estimate_perspective, warp_page
        ):
            patch.setattr(digitize, stage.__name__, stage)
        patch.setattr(digitize, "normalise_paper", paper)
        patch.setattr(digitize, "read_image", read)
        patch.setattr(digitize, "predict_mask_nnunet", model)
        with contextlib.redirect_stdout(log):
            digitize.run(
                digitize.get_parser().parse_args(
                    ["-d", str(data), "-o", str(out), *flags]
                )
            )
    return SimpleNamespace(
        out=out, log=log.getvalue(), calls=calls, decided=decided, asked=asked
    )


def _file(folder, name):
    with open(os.path.join(folder, name), "rb") as f:
        return f.read()


def _qc(folder):
    with open(os.path.join(folder, "qc.csv"), newline="") as f:
        return next(csv.DictReader(f))


def _other_columns(folder):
    """The QC row of a run without the column of the paper stage, numbers as numbers."""
    row = _qc(folder)
    del row["paper_normalisation"]
    for column, value in row.items():
        try:
            row[column] = float(value)
        except ValueError:
            pass
    return row


def _same_columns(folder):
    """What the QC row of another run of the same page has to be equal to.

    A text column has to be the same text, a number the same to nine digits: the
    perspective stage does not repeat its last ones from one call on a page to the
    next (seen once, at 5e-12 of the shift and without the flag), so two runs are
    compared as numbers and not as the text of the file.
    """
    return pytest.approx(_other_columns(folder), rel=1e-9, abs=1e-12, nan_ok=True)


def _mask_meta(folder):
    with open(os.path.join(folder, "rec_mask.json")) as f:
        return json.load(f)


def _mask_json(folder):
    """The JSON next to the mask as it stands in the file, byte for byte."""
    return _file(folder, "rec_mask.json")


def _small_masks(path, page=None, decision="pass_chain", record="rec", **changes):
    """A folder with the flat label as the mask of a small page, written by hand.

    With a page the JSON holds the record of that page left as it is, changes laid
    over it; without one it holds the frame of the mask alone. Returns the record.
    """
    path.mkdir(exist_ok=True)
    write_png(torch.from_numpy(_flat_label())[None], str(path / f"{record}_mask.png"))
    meta = {"rot_angle": 0.0, "height": SMALL[1], "width": SMALL[0]}
    block = None
    if page is not None:
        digest = digitize.page_sha1(page)
        block = {
            "version": paper_normalisation.VERSION, "decision": decision,
            "input_size": list(SMALL), "input_sha1": digest,
            "output_size": list(SMALL), "view_sha1": digest,
        }
        block.update(changes)
        meta["paper_normalisation"] = block
    with open(path / f"{record}_mask.json", "w") as f:
        json.dump(meta, f)
    return block


def _stage_lines(log):
    """Everything a run printed but the lines of the paper stage."""
    return [
        line for line in log.splitlines()
        if not line.startswith("Paper normalisation for record")
    ]


def _error(folder):
    """Largest distance in mV of the digitised leads from the signals on the page."""
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        record = wfdb.rdrecord(os.path.join(folder, "rec"))
    samples = int(LONG_SIGNAL_LENGTH_SEC * FREQUENCY)
    short = int(SHORT_SIGNAL_LENGTH_SEC * FREQUENCY)
    t = np.arange(samples) / FREQUENCY
    errors = []
    for k, lead in enumerate(LEADS):
        window = slice(0, samples)
        if lead != "II":
            window = slice(SHORT_LEADS[lead] * short, (SHORT_LEADS[lead] + 1) * short)
        got = record.p_signal[window, record.sig_name.index(lead)]
        errors.append(np.max(np.abs(got - analytic(t, k)[window])))
    # A lead that is missing is NaN, and so then is this.
    return float(np.max(errors))


def _never(name):
    def called(*args, **kwargs):
        raise AssertionError(f"{name} was called")

    return called


def _spy(monkeypatch, name, module=digitize):
    """Count the calls of module.<name> and keep their arguments, behaviour as is."""
    calls = []
    real = getattr(module, name)

    def spy(*args, **kwargs):
        calls.append(args)
        return real(*args, **kwargs)

    monkeypatch.setattr(module, name, spy)
    return calls


def _plain_record(block, keys):
    """A record holds exactly keys, in values JSON gives back as they went in."""
    assert sorted(block) == sorted(keys)
    assert json.loads(json.dumps(block, allow_nan=False)) == block
    assert block["version"] == paper_normalisation.VERSION
    assert block["libraries"] == f"numpy {np.__version__} cv2 {cv2.__version__}"
    for size in (block["input_size"], block["output_size"]):
        assert [type(side) for side in size] == [int, int]


@pytest.fixture(scope="module")
def photo():
    """The keystone photograph of the page, of a size that is shrunk before the warp."""
    return _shot(KEYSTONE)


@pytest.fixture(scope="module")
def digitised(photo, tmp_path_factory):
    """The photograph digitised with the flag and its mask saved, once for the module.

    All stage flags at their defaults. The model is handed the label in the frame a
    normalised page has, whatever the page it is asked about: the sheet on the page.
    """
    folder = tmp_path_factory.mktemp("photo")
    data = _folder(folder / "data", rec=_tensor(photo.image))
    label = _label_in(_onto_page(photo.corners) @ photo.to_photo, (WIDTH, HEIGHT))
    flags = ("--paper_normalisation", "auto", "--save_mask")
    run = _run(data, folder / "out", *flags, label=label)
    assert len(run.decided) == 1
    _, (view, block, meta, stages) = run.decided[0]
    return SimpleNamespace(
        **vars(run), data=data, view=view, block=block, meta=meta, stages=stages
    )


@pytest.fixture(scope="module")
def replayed(digitised, tmp_path_factory):
    """The photograph once more, on the mask saved for it, and that mask saved again."""
    return _run(
        digitised.data,
        tmp_path_factory.mktemp("replayed"),
        "--paper_normalisation", "auto",
        "--mask_folder", str(digitised.out),
        "--save_mask",
    )


@pytest.fixture(scope="module")
def view_data(digitised, tmp_path_factory):
    """A data folder whose page is the normalised view of the photograph."""
    return _folder(tmp_path_factory.mktemp("view"), rec=digitised.view)


@pytest.fixture(scope="module")
def shrunk(tmp_path_factory):
    """A large photograph whose sheet is cut, digitised with the flag, its mask saved.

    The sheet runs off the right border, so its paper is not found, and the chain
    reads no grid that would say how large the page is. The model is handed the
    label of the photograph at the size of the page it is asked about.
    """
    folder = tmp_path_factory.mktemp("shrunk")
    cut = _shot(KEYSTONE, move=(580.0, 0.0))
    on_photo = _label_in(cut.to_photo, KEYSTONE[0])

    def label(image):
        size = (image.shape[2], image.shape[1])
        return cv2.resize(on_photo, size, interpolation=cv2.INTER_NEAREST)

    page = _tensor(cut.image)
    run = _run(
        folder / "data", folder / "out", "--paper_normalisation", "auto",
        "--save_mask", *CHEAP, label=label, pages={"rec": page},
    )
    return SimpleNamespace(
        **vars(run), data=folder / "data", cut=cut, page=page, label=label
    )


# ------------------------------------------------------- the flag and the constants
def test_the_flag_is_off_unless_auto_is_asked_for(capsys):
    parse = digitize.get_parser().parse_args
    folders = ["-d", "in", "-o", "out"]
    assert parse(folders).paper_normalisation == "off"
    for value in ("off", "auto"):
        args = parse(folders + ["--paper_normalisation", value])
        assert args.paper_normalisation == value
    with pytest.raises(SystemExit):
        parse(folders + ["--paper_normalisation", "on"])
    assert "--paper_normalisation: invalid choice: 'on'" in capsys.readouterr().err


def test_the_port_keeps_the_scales_of_the_digitiser():
    # The port may not import the digitiser, so it holds the two numbers again.
    assert paper_normalisation.SCALE_DEAD_BAND == digitize.RESOLUTION_DEAD_BAND
    assert paper_normalisation.PAPER_PMM == digitize.GRID_LINE_PERIOD_200DPI
    assert (paper_normalisation.PAGE_W, paper_normalisation.PAGE_H) == (WIDTH, HEIGHT)
    assert paper_normalisation.VERSION in paper_normalisation.REPLAYABLE


def test_the_port_is_image_code_on_numpy_and_cv2_alone():
    tree = ast.parse(inspect.getsource(paper_normalisation))
    imported = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            imported.update(alias.name.split(".")[0] for alias in node.names)
        elif isinstance(node, ast.ImportFrom):
            imported.add(node.module.split(".")[0])
    assert imported == {"cv2", "numpy"}


# -------------------------------------------------------------------- the flag off
def test_run_without_the_flag_never_looks_for_paper(tmp_path, monkeypatch):
    for name in ("normalise_paper", "paper_chain", "check_paper_block"):
        monkeypatch.setattr(digitize, name, _never(name))
    data = _folder(tmp_path / "data", rec=_tensor(_page(SMALL, traces=False)))
    flags = ("--time_mapping", "bbox", *CHEAP)
    # No flag at all is off, on a page the model is asked about ...
    first = _run(data, tmp_path / "first", *flags, "--save_mask", label=_flat_label())
    # ... and on one that brings its mask, which then has no record next to it.
    second = _run(
        data, tmp_path / "second", *flags, "--paper_normalisation", "off",
        "--mask_folder", str(first.out),
    )
    for run in (first, second):
        assert _qc(run.out)["paper_normalisation"] == "off"
        assert "aper normalisation" not in run.log
    assert sorted(_mask_meta(first.out)) == ["height", "rot_angle", "width"]
    assert _file(second.out, "rec.dat") == _file(first.out, "rec.dat")


def test_save_mask_files_without_a_record_writes_the_json_it_always_wrote(tmp_path):
    mask = torch.zeros((1, 6, 8), dtype=torch.uint8)
    plain = '{"rot_angle": 0.5, "height": 6, "width": 8}'
    digitize.save_mask_files(mask, "plain", str(tmp_path), 0.5)
    assert (tmp_path / "plain_mask.json").read_text() == plain
    digitize.save_mask_files(mask, "none", str(tmp_path), 0.5, paper=None)
    assert (tmp_path / "none_mask.json").read_text() == plain

    homography = np.array([[1.0, 0.25, -3.0], [0.0, 1.5, 2.0], [0.0, 0.0, 1.0]])
    digitize.save_mask_files(mask, "warped", str(tmp_path), -1.25, homography, 0.8)
    assert (tmp_path / "warped_mask.json").read_text() == (
        '{"rot_angle": -1.25, "height": 6, "width": 8, "homography": '
        '[[1.0, 0.25, -3.0], [0.0, 1.5, 2.0], [0.0, 0.0, 1.0]], "scale": 0.8}'
    )


def test_save_mask_files_writes_a_record_after_the_frame(tmp_path):
    mask = torch.zeros((1, 6, 8), dtype=torch.uint8)
    block = {"version": "v", "decision": "pass_chain", "input_size": [8, 6]}
    digitize.save_mask_files(mask, "rec", str(tmp_path), 0.5, paper=block)
    assert (tmp_path / "rec_mask.json").read_text() == (
        '{"rot_angle": 0.5, "height": 6, "width": 8, "paper_normalisation": '
        '{"version": "v", "decision": "pass_chain", "input_size": [8, 6]}}'
    )
    # Drivers call it with up to six arguments in this order: the record comes last.
    parameters = inspect.signature(digitize.save_mask_files).parameters
    assert list(parameters)[-2:] == ["scale", "paper"]
    assert parameters["paper"].default is None


# ------------------------------------------------------------- the keystone photo
def test_the_chain_cannot_read_the_photo_and_the_paper_is_found(photo, digitised):
    chain, paper = digitised.meta["chain"], digitised.meta["paper"]
    # What everything below stands on: only a page that the grid line stages do not
    # read is looked at by the paper search at all, and a keystone is such a page.
    assert chain["ok"] is False
    assert chain["res_reason"] and chain["rot_reason"] and chain["persp_reason"]
    assert digitised.block["decision"] == "normalised"
    assert digitised.block["reason"] == ""
    # Found by its red grid, each corner where the sheet was put. Measured: 0.06 px.
    assert paper["kind"] == "quad" and paper["red_grid"] == 1
    assert np.linalg.norm(paper["corners"] - photo.corners, axis=1).max() < 2.0
    assert digitised.meta["frame"]["status"] == "corrected"


def test_the_photo_is_warped_onto_the_page_the_right_way_up(photo, digitised):
    block, meta = digitised.block, digitised.meta
    # The traces lie in the lower part of the page. Measured: a margin of 0.42.
    assert block["orientation"] == 0
    assert meta["orient_margin"] > 0.3 and meta["orient_low"] == 0
    assert tuple(digitised.view.shape) == (3, HEIGHT, WIDTH)
    assert block["output_size"] == [WIDTH, HEIGHT]
    # The corners of the sheet on the corners of the page, top left on top left ...
    onto = cv2.perspectiveTransform(
        photo.corners.reshape(-1, 1, 2), np.array(block["homography"])
    ).reshape(-1, 2)
    # Measured: 0.16 px.
    assert np.linalg.norm(onto - paper_normalisation.PAGE_CORNERS, axis=1).max() < 1.0
    # ... and the printed grid at the period of the frame along both axes. Warped
    # by the corners of the paper alone it is 1.000112 and 1.000105 of that period,
    # which is what the frame check measured and took out: 0.9999993 and 0.9999995
    # are left, and the bound lies between the two.
    before = meta["frame"]
    periods = digitize.paper_frame_periods(_pixels(digitised.view))
    frame = paper_normalisation.FRAME_PERIOD
    for axis in ("px", "py"):
        assert abs(before[axis] / frame - 1) > 5e-5
        assert periods[axis] == pytest.approx(frame, rel=2e-5)
    assert before["scale_x"] == pytest.approx(frame / before["px"], rel=1e-12)
    assert before["scale_y"] == pytest.approx(frame / before["py"], rel=1e-12)


def test_paper_frame_periods_measures_each_axis_of_a_page():
    # A page stretched by 4 % along x alone: the period along x is the one that is
    # 4 % longer, and the page that is handed in has its rows first, as an image.
    small = _page(SMALL, traces=False)
    wide = cv2.resize(
        small, (int(round(SMALL[0] * 1.04)), SMALL[1]), interpolation=cv2.INTER_AREA
    )
    periods = digitize.paper_frame_periods(wide)
    assert sorted(periods) == ["cx", "cy", "px", "px_raw", "py", "py_raw"]
    assert all(type(value) is float for value in periods.values())
    # Measured: 1.0403 and 1.0003 of the period of the page.
    assert periods["px"] == pytest.approx(1.04 * PERIOD, rel=0.002)
    assert periods["py"] == pytest.approx(PERIOD, rel=0.002)
    assert periods["px_raw"] == periods["px"] and periods["py_raw"] == periods["py"]
    assert periods["cx"] > 3 and periods["cy"] > 3


def test_the_stages_run_on_the_view_and_leave_it_alone(digitised):
    # Nothing the chain measured on the photograph is used for its view: every stage
    # ran a second time, on the view, and none of them resampled it. The model was
    # asked about the very pixels the record has the digest of.
    assert digitised.stages is None
    assert digitised.calls == {
        "normalise_resolution": 2, "estimate_rotation": 2, "estimate_perspective": 2,
        "warp_page": 0,
    }
    assert digitised.asked == [((3, HEIGHT, WIDTH), digitised.block["view_sha1"])]
    assert _qc(digitised.out)["paper_normalisation"] == "normalised"
    assert digitised.log.count("Paper normalisation for record rec: normalised\n") == 1
    assert "WARNING: paper normalisation" not in digitised.log
    assert "WARNING: grid lines not used" not in digitised.log


def test_the_photo_gives_the_signals_drawn_on_the_page(digitised):
    # Measured: 0.023 mV at most, against 0.038 mV on the page itself. The traces
    # are read off the ink of the page under the mask, and off the mask alone where
    # there is none: the same label over a view that is upside down gives 0.091 mV.
    assert _error(digitised.out) < 0.05


def test_the_view_of_the_photo_is_built_the_same_twice(photo, digitised):
    # A second build of its own, on the answer the chain gave the first time.
    view, meta = paper_normalisation.normalise_page(
        photo.image, lambda rgb: digitised.meta["chain"], digitize.paper_frame_periods
    )
    assert hashlib.sha1(view.tobytes()).hexdigest() == digitised.block["view_sha1"]
    record = paper_normalisation.page_record(meta)
    assert record == paper_normalisation.page_record(digitised.meta)
    assert record["decision"] == "normalised"
    # The record of a page is JSON as it stands, and has nothing of the run in it.
    json.dumps(record, allow_nan=False)
    assert not [key for key in digitised.meta if key.startswith("t_")]
    assert "stages" not in digitised.meta


def test_the_record_of_the_photo_builds_its_view_again(photo, digitised):
    block = digitised.block
    _plain_record(block, RECORD_KEYS + NORMALISED_KEYS)
    # Sizes are width, height, and the digests those of the two pages.
    assert block["input_size"] == [3360, 2520]
    assert block["input_sha1"] == digitize.page_sha1(_tensor(photo.image))
    assert block["view_sha1"] == digitize.page_sha1(digitised.view)
    assert block["view_sha1"] != block["input_sha1"]
    # The image was shrunk before the warp, so the replay has to shrink it too.
    assert block["pre_size"] != block["input_size"]
    # --save_mask wrote the record next to the mask, after the frame of the mask.
    saved = _mask_meta(digitised.out)
    assert list(saved) == ["rot_angle", "height", "width", "paper_normalisation"]
    assert saved["paper_normalisation"] == block

    again = paper_normalisation.replay_page(photo.image, saved["paper_normalisation"])
    assert np.array_equal(again, _pixels(digitised.view))


def test_the_frame_check_scales_the_page_keeps_it_or_gives_it_up(photo, digitised):
    # The decision on the photograph again, with other answers of the frame check.
    frame = paper_normalisation.FRAME_PERIOD

    def decide(px, py):
        periods = {"px": px, "py": py, "px_raw": px, "py_raw": py, "cx": 9.0, "cy": 9.0}
        return paper_normalisation.normalise_page(
            photo.image, lambda rgb: digitised.meta["chain"], lambda page: periods
        )

    # No period along y: the page is warped by the corners of its paper alone, and
    # the reason says that the frame was not checked.
    view, meta = decide(frame, np.nan)
    assert meta["frame"]["status"] == "unmeasured"
    assert meta["decision"] == "normalised"
    assert meta["reason"] == "frame unmeasured (grid period NaN), kept as warped"
    assert np.array_equal(meta["homography"], meta["homography_paper"])
    assert view.shape == (HEIGHT, WIDTH, 3)

    # Both periods, a few per cent wide: an edge was found at the print and not at
    # the paper, and the page is scaled about its centre to the period of the frame.
    view, meta = decide(1.05 * frame, 1.04 * frame)
    assert meta["frame"]["status"] == "corrected"
    assert (meta["decision"], meta["reason"]) == ("normalised", "")
    scaled = paper_normalisation.frame_scale(
        meta["homography_paper"], 1 / 1.05, 1 / 1.04
    )
    assert np.allclose(meta["homography"], scaled, rtol=1e-12, atol=0)
    assert not np.allclose(meta["homography"], meta["homography_paper"], rtol=1e-3)

    # A grid a fifth too wide along x: a side of the sheet is far off, so the page is
    # not warped at all, and goes on as one whose paper was not found, only shrunk.
    view, meta = decide(1.2 * frame, frame)
    assert meta["frame"]["status"] == "bad"
    assert meta["decision"] == "scaled" and meta["scaled_from"] == "failed"
    assert meta["reason"].startswith("normalised grid period along x ")
    assert "homography" not in meta and "warp" not in meta
    assert view.shape == (meta["resize"][1], meta["resize"][0], 3)
    assert meta["resize"][0] < photo.image.shape[1]


def _turned(points, turns, shape):
    """Points of an image of shape (H, W) in np.rot90(image, turns)."""
    width = shape[1]
    height = shape[0]
    points = np.array(points, dtype=float)
    for _ in range(turns):
        # Counter-clockwise: (x, y) goes to (y, W - 1 - x), in an image W high.
        points = np.stack([points[:, 1], width - 1 - points[:, 0]], axis=1)
        width, height = height, width
    return points


@pytest.mark.parametrize("turns, orientation", [(1, 270), (2, 180), (3, 90)])
def test_a_turned_photo_is_found_and_put_upright(photo, turns, orientation):
    # The paper search and the orientation alone, with no chain before them.
    image = np.ascontiguousarray(np.rot90(photo.image, turns))
    corners = _turned(photo.corners, turns, photo.image.shape)
    paper = paper_normalisation.find_paper(image)
    assert paper["kind"] == "quad" and paper["red_grid"] == 1
    assert paper["portrait"] == turns % 2
    # The corners come as top left, top right, ... of the turned image, so each one
    # of the sheet is one of them, but not the one of its name.
    distance = np.linalg.norm(paper["corners"][:, None] - corners[None], axis=2)
    assert distance.min(axis=0).max() < 2.0

    found, margin = paper_normalisation.choose_orientation(
        image, paper["corners"], paper["portrait"]
    )
    assert found == orientation and margin > 0.3
    # The top left corner of the sheet comes to lie top left on the page again.
    onto = cv2.perspectiveTransform(
        corners.reshape(-1, 1, 2),
        paper_normalisation.paper_homography(paper["corners"], found),
    ).reshape(-1, 2)
    # Measured: 0.06 px.
    assert np.linalg.norm(onto - paper_normalisation.PAGE_CORNERS, axis=1).max() < 1.0


def test_a_sheet_photographed_from_the_front_is_left_to_the_chain(monkeypatch):
    # The gate: the same sheet on the same table, seen from the front, is a page the
    # grid line stages read, so it is not normalised and its paper never looked for.
    monkeypatch.setattr(paper_normalisation, "find_paper", _never("find_paper"))
    canvas, pmm, _ = FAST
    width = paper_normalisation.PAPER_MM[0] * pmm
    height = paper_normalisation.PAPER_MM[1] * pmm
    left, top = (canvas[0] - width) / 2, (canvas[1] - height) / 2
    corners = (
        (left, top), (left + width, top), (left + width, top + height),
        (left, top + height),
    )
    image, _ = _photograph(_page(), canvas, pmm, corners)
    page = _tensor(image)
    view, block, meta, stages = digitize.normalise_paper(page)
    assert meta["chain"]["ok"] is True
    assert block["decision"] == "pass_chain"
    assert view is page and stages is not None


@pytest.mark.parametrize(
    "stage", ["normalise_resolution", "estimate_rotation", "estimate_perspective"]
)
def test_a_page_one_stage_does_not_read_is_not_left_to_the_chain(monkeypatch, stage):
    # The gate is all three stages: a page of grid lines that one of them, whichever,
    # says it did not use is looked at by the paper search.
    page = _tensor(_page(SMALL, traces=False))
    real = getattr(digitize, stage)

    def unread(*args):
        result, info = real(*args)
        # The stage itself reads the page, as the other two do below.
        assert info["reason"] == ""
        if stage == "estimate_perspective":
            result = None
        return result, {**info, "reason": "said so by the test"}

    monkeypatch.setattr(digitize, stage, unread)
    searched = _spy(monkeypatch, "find_paper", module=paper_normalisation)
    view, block, meta, stages = digitize.normalise_paper(page)
    chain = meta["chain"]
    reasons = [chain["res_reason"], chain["rot_reason"], chain["persp_reason"]]
    assert sorted(reasons) == ["", "", "said so by the test"]
    assert chain["ok"] is False
    assert len(searched) == 1
    # The grid fills the image, so the page is still left as it is, but not as a
    # page the chain reads.
    assert block["decision"] == "pass_fullframe" and "paper" in meta
    assert view is page and stages is not None


# -------------------------------------------------------------- the other decisions
def test_a_large_photo_whose_sheet_is_cut_is_only_shrunk(shrunk):
    # No grid says how large the page is: the extent of the print does, and the page
    # is shrunk to 200 dpi of paper, no more.
    canvas = KEYSTONE[0]
    run, cut = shrunk, shrunk.cut
    page, (view, block, meta, stages) = run.decided[0]
    _plain_record(block, RECORD_KEYS + SCALED_KEYS)
    assert block["decision"] == "scaled" and block["scaled_from"] == "failed"
    # The sheet lies at 9.2 px per mm, from the area of its corners.
    area = cv2.contourArea(np.float32(cut.corners))
    paper_mm = paper_normalisation.PAPER_MM
    exact = paper_normalisation.PAPER_PMM / np.sqrt(area / (paper_mm[0] * paper_mm[1]))
    # Measured: 0.9 % below it. The page is shrunk beyond 12 % only.
    factor = block["scale_factor"]
    assert factor == pytest.approx(exact, rel=0.02)
    assert factor < 1 - paper_normalisation.SCALE_DEAD_BAND
    size = [int(round(canvas[0] * factor)), int(round(canvas[1] * factor))]
    assert block["input_size"] == list(canvas)
    assert block["resize"] == block["output_size"] == size
    assert view is not page and stages is None
    assert tuple(view.shape) == (3, size[1], size[0])
    assert run.asked == [(tuple(view.shape), block["view_sha1"])]
    assert block["view_sha1"] != block["input_sha1"]

    assert block["reason"] == (
        f"paper reaches the image border on right; shrunk by {factor:.3f} to 200 dpi "
        f"of paper"
    )
    warning = (
        f"WARNING: paper normalisation not applied to record rec ({block['reason']}), "
        f"page rescaled only."
    )
    assert run.log.count(warning) == 1
    assert run.log.count("WARNING: paper normalisation") == 1
    line = f"Paper normalisation for record rec: scaled ({block['reason']})\n"
    assert run.log.count(line) == 1
    assert run.log.count("Paper normalisation for record") == 1
    assert _qc(run.out)["paper_normalisation"] == "scaled"

    # --save_mask wrote the record next to the mask, and the view is built from it.
    record = _mask_meta(run.out)["paper_normalisation"]
    assert record == block
    again = paper_normalisation.replay_page(cut.image, record)
    assert np.array_equal(again, _pixels(view))


def test_a_saved_mask_replays_the_view_of_its_shrunk_photo(tmp_path, shrunk):
    # A scaled record goes through the frame check as a normalised one does.
    page, (view, block, meta, stages) = shrunk.decided[0]
    record = _mask_meta(shrunk.out)["paper_normalisation"]
    again, how = digitize.check_paper_block(record, shrunk.page, "rec", "auto")
    assert how == "replayed" and torch.equal(again, view)
    same, how = digitize.check_paper_block(record, view, "rec", "off")
    assert how == "view" and same is view

    run = _run(
        shrunk.data, tmp_path / "out", "--paper_normalisation", "auto", *CHEAP,
        "--mask_folder", str(shrunk.out), "--save_mask", pages={"rec": shrunk.page},
    )
    assert run.decided == [] and run.asked == []
    line = "Paper normalisation for record rec: scaled from the mask's record"
    assert run.log.count(f"{line} (replayed)\n") == 1
    assert "WARNING: paper normalisation" not in run.log and FITS_NOT not in run.log
    assert _qc(run.out)["paper_normalisation"] == "scaled"
    for name in ("rec.dat", "rec.hea"):
        assert _file(run.out, name) == _file(shrunk.out, name)
    assert _mask_json(run.out) == _mask_json(shrunk.out)


def test_a_shrunk_page_is_warned_about_once_without_verbose(
    tmp_path, monkeypatch, shrunk
):
    # The decision the verbose run took, handed to a run that is not verbose: the
    # warning is not a line of the verbose ones, the decision is.
    page, decided = shrunk.decided[0]
    monkeypatch.setattr(digitize, "normalise_paper", lambda image: decided)
    run = _run(
        shrunk.data, tmp_path / "out", "--paper_normalisation", "auto",
        "--no-verbose", *CHEAP, label=shrunk.label, pages={"rec": shrunk.page},
    )
    warning = (
        f"WARNING: paper normalisation not applied to record rec "
        f"({decided[1]['reason']}), page rescaled only."
    )
    assert run.log.count(warning) == 1
    assert run.log.count("aper normalisation") == 1
    assert _file(run.out, "rec.dat") == _file(shrunk.out, "rec.dat")


def test_a_photo_whose_paper_is_not_found_is_kept_with_a_warning(tmp_path):
    # The cut sheet at 7.4 px per mm: smaller than 200 dpi of paper, so not shrunk.
    canvas = FAST[0]
    cut = _shot(FAST, move=(470.0, 0.0))
    page = _tensor(cut.image)
    run = _run(
        tmp_path / "data", tmp_path / "out", "--paper_normalisation", "auto",
        "--save_mask", label=_label_in(cut.to_photo, canvas), pages={"rec": page},
    )
    given, (view, block, meta, stages) = run.decided[0]
    _plain_record(block, RECORD_KEYS)
    assert block["decision"] == "failed"
    assert block["reason"] == "paper reaches the image border on right"
    assert run.log.count(
        "WARNING: paper normalisation not applied to record rec (paper reaches the "
        "image border on right), page kept.\n"
    ) == 1
    assert run.log.count("WARNING: paper normalisation") == 1
    assert run.log.count(
        "Paper normalisation for record rec: failed (paper reaches the image border "
        "on right)\n"
    ) == 1
    assert run.log.count("Paper normalisation for record") == 1
    assert _qc(run.out)["paper_normalisation"] == "failed"
    # A page that was kept has its record next to its mask as every other.
    saved = _mask_meta(run.out)
    assert list(saved) == ["rot_angle", "height", "width", "paper_normalisation"]
    assert saved["paper_normalisation"] == block

    # The page goes on as it came: the same tensor, the digest of the photograph in
    # both places of the record, and those pixels in front of the model.
    digest = digitize.page_sha1(page)
    assert view is given and stages is not None
    assert block["input_size"] == block["output_size"] == list(canvas)
    assert block["input_sha1"] == block["view_sha1"] == digest
    assert run.asked == [((3, canvas[1], canvas[0]), digest)]
    # What the chain measured is what run() goes on with, each warning printed once.
    assert run.calls == ONCE
    for stage in ("resolution", "rotation", "perspective"):
        line = f"WARNING: grid lines not used for the {stage} of record rec"
        assert run.log.count(line) == 1


def test_fallback_factor_shrinks_a_page_beyond_the_dead_band_only():
    blind = {"res_scale": 1.0, "res_period": np.nan, "res_reason": "no grid lines"}
    dpi200 = paper_normalisation.PAPER_PMM

    def factor(pmm, chain=blind):
        # The extent of the print says pmm px per mm, measured at half the size.
        paper = {"pmm_fallback": pmm / 2, "work_sx": 0.5}
        return paper_normalisation.fallback_factor(None, chain, paper)

    assert factor(dpi200 / 0.8) == pytest.approx(0.8)
    assert factor(dpi200 / 0.87) == pytest.approx(0.87)
    # Within 12 % of 200 dpi the page keeps its pixels, and it is never enlarged.
    assert factor(dpi200 / 0.89) is None
    assert factor(dpi200) is None
    assert factor(dpi200 / 1.3) is None
    # A page whose grid the resolution stage reads is left to that stage, whether it
    # resamples the page or keeps it, and so is one whose print was not found.
    assert factor(dpi200 / 0.8, {**blind, "res_scale": 0.8}) is None
    assert factor(dpi200 / 0.8, {**blind, "res_period": 9.8, "res_reason": ""}) is None
    assert paper_normalisation.fallback_factor(None, blind, {"work_sx": 0.5}) is None


def test_a_blank_page_fills_the_frame_and_is_warned_about_once(tmp_path):
    blank = torch.full((3, SMALL[1], SMALL[0]), 255, dtype=torch.uint8)
    data = _folder(tmp_path / "data", rec=blank)
    flags = ("--time_mapping", "bbox", "--save_mask")
    off = _run(data, tmp_path / "off", *flags, label=_flat_label())
    auto = _run(
        data, tmp_path / "auto", "--paper_normalisation", "auto", *flags,
        label=_flat_label(),
    )
    page, (view, block, meta, stages) = auto.decided[0]
    _plain_record(block, RECORD_KEYS)
    # No stage reads a grid, and there is no background a sheet could end against.
    assert meta["chain"]["ok"] is False
    assert block["decision"] == "pass_fullframe"
    assert block["reason"].startswith("printed grid fills the frame")
    assert view is page and block["view_sha1"] == block["input_sha1"]
    assert _qc(auto.out)["paper_normalisation"] == "pass_fullframe"
    assert "WARNING: paper normalisation" not in auto.log
    line = f"Paper normalisation for record rec: pass_fullframe ({block['reason']})\n"
    assert auto.log.count(line) == 1
    assert auto.log.count("aper normalisation") == 1
    # The record of a page that was left as it is stands next to its mask, after
    # the frame of the mask, which is the JSON of the run without the flag.
    saved = _mask_meta(auto.out)
    assert list(saved) == ["rot_angle", "height", "width", "paper_normalisation"]
    assert saved.pop("paper_normalisation") == block
    assert saved == _mask_meta(off.out)

    # Every stage measured once and said so once, as without the flag.
    for run in (off, auto):
        assert run.calls == ONCE
        for stage in ("resolution", "rotation", "perspective"):
            line = f"WARNING: grid lines not used for the {stage} of record rec"
            assert run.log.count(line) == 1
        assert run.log.count("No rotation angle found for record rec") == 1
    assert _stage_lines(auto.log) == _stage_lines(off.log)
    assert _other_columns(auto.out) == _same_columns(off.out)
    assert _file(auto.out, "rec.dat") == _file(off.out, "rec.dat")


def test_frame_decision_scales_a_measured_page_to_the_period_of_the_frame():
    period = paper_normalisation.FRAME_PERIOD
    status, scale_x, scale_y, reason = paper_normalisation.frame_decision(
        {"px": 1.05 * period, "py": 1.04 * period}
    )
    assert (status, reason) == ("corrected", "")
    assert scale_x == pytest.approx(1 / 1.05) and scale_y == pytest.approx(1 / 1.04)


@pytest.mark.parametrize("periods", [(np.nan, 1.0), (1.0, np.nan), (np.nan, np.nan)])
def test_frame_decision_keeps_a_page_it_cannot_measure(periods):
    period = paper_normalisation.FRAME_PERIOD
    decision = paper_normalisation.frame_decision(
        {"px": periods[0] * period, "py": periods[1] * period}
    )
    assert decision == ("unmeasured", 1.0, 1.0, "")


@pytest.mark.parametrize(
    "periods, axis",
    [((1.11, 1.0), "x"), ((1.0, 0.91), "y"), ((0.91, 0.91), "x"), ((np.nan, 1.2), "y")],
)
def test_frame_decision_fails_a_period_far_from_the_frame(periods, axis):
    period = paper_normalisation.FRAME_PERIOD
    status, scale_x, scale_y, reason = paper_normalisation.frame_decision(
        {"px": periods[0] * period, "py": periods[1] * period}
    )
    assert (status, scale_x, scale_y) == ("bad", 1.0, 1.0)
    assert reason.startswith(f"normalised grid period along {axis} ")
    assert reason.endswith(" of the frame's")


def test_frame_decision_fails_an_anisotropic_page():
    # Both periods inside the range of an axis, and 9.5 % apart.
    period = paper_normalisation.FRAME_PERIOD
    status, scale_x, scale_y, reason = paper_normalisation.frame_decision(
        {"px": 0.95 * period, "py": 1.04 * period}
    )
    assert (status, scale_x, scale_y) == ("bad", 1.0, 1.0)
    assert reason == "normalised grid anisotropic (py / px 1.095)"
    # 7.4 % apart is a page to correct.
    status = paper_normalisation.frame_decision(
        {"px": 0.95 * period, "py": 1.02 * period}
    )[0]
    assert status == "corrected"


def test_frame_scale_scales_the_page_about_its_centre():
    homography = np.array([[1.1, 0.02, 30.0], [-0.01, 0.9, -20.0], [1e-5, 2e-5, 1.0]])
    scaled = paper_normalisation.frame_scale(homography, 0.9, 1.1)
    centre = np.array([(WIDTH - 1) / 2, (HEIGHT - 1) / 2])
    points = np.array([[[100.0, 200.0]], [[1500.0, 900.0]], [[2100.0, 50.0]]])
    before = cv2.perspectiveTransform(points, homography).reshape(-1, 2)
    after = cv2.perspectiveTransform(points, scaled).reshape(-1, 2)
    assert np.allclose(after - centre, (before - centre) * [0.9, 1.1])


def test_replay_page_builds_nothing_for_a_page_that_was_left_alone():
    page = np.zeros((4, 6, 3), dtype=np.uint8)
    for decision in ("pass_chain", "pass_fullframe", "failed"):
        assert paper_normalisation.replay_page(page, {"decision": decision}) is None
    with pytest.raises(ValueError, match="'normalised_inferred' cannot be replayed"):
        paper_normalisation.replay_page(page, {"decision": "normalised_inferred"})


# ------------------------------------------------------ a page that needs nothing
def test_a_page_the_chain_reads_is_digitised_as_without_the_flag(tmp_path):
    page = _tensor(_page())
    data = _folder(tmp_path / "data", rec=page)
    flag = "--paper_normalisation"
    off = _run(data, tmp_path / "off", flag, "off", "--save_mask", label=_label())
    auto = _run(data, tmp_path / "auto", flag, "auto", "--save_mask", label=_label())
    assert off.decided == [] and len(auto.decided) == 1
    given, (view, block, meta, stages) = auto.decided[0]
    # The page is not copied, and the record says that it is its own view.
    assert view is given and stages is not None
    _plain_record(block, RECORD_KEYS)
    assert (block["decision"], block["reason"]) == ("pass_chain", "")
    assert block["input_size"] == block["output_size"] == [WIDTH, HEIGHT]
    assert block["input_sha1"] == block["view_sha1"] == digitize.page_sha1(page)
    assert "paper" not in meta

    # The stages measured the page once with either flag, the model was asked about
    # the same pixels, and nothing else was printed or written but the decision.
    assert off.calls == auto.calls == ONCE
    assert off.asked == auto.asked == [((3, HEIGHT, WIDTH), block["view_sha1"])]
    for name in ("rec.dat", "rec.hea"):
        assert _file(auto.out, name) == _file(off.out, name)
    assert _qc(off.out)["paper_normalisation"] == "off"
    assert _qc(auto.out)["paper_normalisation"] == "pass_chain"
    assert _other_columns(auto.out) == _same_columns(off.out)
    assert auto.log.count("Paper normalisation for record rec: pass_chain\n") == 1
    assert _stage_lines(auto.log) == off.log.splitlines()
    # The mask and its frame are the ones of the run without the flag, which writes
    # no record, and the record of the page comes after them.
    assert _file(auto.out, "rec_mask.png") == _file(off.out, "rec_mask.png")
    saved = _mask_meta(auto.out)
    assert list(_mask_meta(off.out)) == ["rot_angle", "height", "width"]
    assert list(saved) == ["rot_angle", "height", "width", "paper_normalisation"]
    assert saved.pop("paper_normalisation") == block
    assert saved == _mask_meta(off.out)
    assert _mask_json(auto.out).startswith(_mask_json(off.out)[:-1] + b", ")


def test_a_page_the_resolution_stage_resamples_goes_on_resampled(tmp_path):
    # A quarter page scanned at 250 dpi. The chain reads it, so it is left as it is,
    # and what run() goes on with is what the resolution stage made of it in the
    # chain: the page shrunk to 200 dpi, which is the page the model is asked about.
    large = cv2.resize(
        _page(SMALL, traces=False), (1375, 1062), interpolation=cv2.INTER_AREA
    )
    page = _tensor(large)
    data = _folder(tmp_path / "data", rec=page)

    def label(image):
        return _flat_label((image.shape[2], image.shape[1]))

    flags = ("--time_mapping", "bbox", "--save_mask")
    off = _run(data, tmp_path / "off", *flags, label=label)
    auto = _run(
        data, tmp_path / "auto", "--paper_normalisation", "auto", *flags, label=label
    )
    given, (view, block, meta, stages) = auto.decided[0]
    assert _qc(auto.out)["paper_normalisation"] == "pass_chain"
    # The paper stage left the page at the size it came in: its record is of that
    # page, and the resampling is the resolution stage's, noted next to the mask.
    assert view is given and tuple(view.shape) == (3, 1062, 1375)
    assert block["input_size"] == block["output_size"] == [1375, 1062]
    assert block["input_sha1"] == block["view_sha1"] == digitize.page_sha1(page)
    assert meta["chain"]["res_scale"] == pytest.approx(0.8, abs=0.002)
    assert tuple(stages["image"].shape) == (3, 849, 1100)
    assert float(_qc(auto.out)["resolution_scale"]) == meta["chain"]["res_scale"]

    assert off.calls == auto.calls == ONCE
    assert len(auto.asked) == 1 and auto.asked[0][0] == (3, 849, 1100)
    assert auto.asked == off.asked
    assert auto.asked[0][1] != block["view_sha1"]
    for name in ("rec.dat", "rec.hea", "rec_mask.png"):
        assert _file(auto.out, name) == _file(off.out, name)
    assert _other_columns(auto.out) == _same_columns(off.out)
    assert _stage_lines(auto.log) == off.log.splitlines()
    saved = _mask_meta(auto.out)
    assert saved.pop("paper_normalisation") == block
    assert list(saved) == ["rot_angle", "height", "width", "scale"]
    assert saved == _mask_meta(off.out)
    assert (saved["width"], saved["height"]) == (1100, 849)


def _tilted_runs(tmp_path, *flags):
    """A small page tilted by 0.37 degrees, without the flag and with it."""
    page = _tensor(_tilted(_page(SMALL, traces=False), 0.37))
    data = _folder(tmp_path / "data", rec=page)
    flags = ("--time_mapping", "bbox", *flags)
    off = _run(data, tmp_path / "off", *flags, label=_flat_label())
    auto = _run(
        data, tmp_path / "auto", "--paper_normalisation", "auto", *flags,
        label=_flat_label(),
    )
    assert _qc(auto.out)["paper_normalisation"] == "pass_chain"
    # The same page in front of the model and the same numbers in every QC column.
    assert len(off.asked) == 1 and auto.asked == off.asked
    assert _other_columns(auto.out) == _same_columns(off.out)
    assert _stage_lines(auto.log) == off.log.splitlines()
    for name in ("rec.dat", "rec.hea"):
        assert _file(auto.out, name) == _file(off.out, name)
    return off, auto


def test_a_tilted_page_is_warped_once_under_either_flag(tmp_path):
    # Every flag at its default: the page is turned by picking pixels, and warped
    # once, to measure its perspective on. That warp is the chain's with the flag,
    # and run() takes what was measured on it instead of warping the page again.
    off, auto = _tilted_runs(tmp_path)
    assert float(_qc(off.out)["rotation_angle"]) == pytest.approx(-0.37, abs=0.02)
    assert off.calls == auto.calls == {**ONCE, "warp_page": 1}


def test_a_tilted_page_is_warped_once_with_bicubic_under_either_flag(tmp_path):
    # The page the chain measured the perspective on is the bicubic page of run().
    off, auto = _tilted_runs(tmp_path, "--interpolation", "bicubic")
    assert float(_qc(off.out)["rotation_angle"]) == pytest.approx(-0.37, abs=0.02)
    assert off.calls == auto.calls == {**ONCE, "warp_page": 1}


def test_the_bicubic_page_is_the_warp_of_the_chain_without_a_perspective_stage(
    tmp_path
):
    # --perspective off: run() measures no perspective, and the chain, which always
    # does, has warped the page for it. That warp is the bicubic page of run(), of
    # the same page by the same angle, so the page is not warped a second time.
    off, auto = _tilted_runs(
        tmp_path, "--perspective", "off", "--interpolation", "bicubic"
    )
    assert float(_qc(off.out)["rotation_angle"]) == pytest.approx(-0.37, abs=0.02)
    assert off.calls == {**ONCE, "estimate_perspective": 0, "warp_page": 1}
    assert auto.calls == {**ONCE, "warp_page": 1}


@pytest.mark.parametrize(
    "flags, again",
    [
        (("--resolution", "keep"), STAGES[:3]),
        (("--rotation", "hough"), STAGES[1:3]),
        (("--perspective", "off"), STAGES[2:3]),
    ],
)
def test_a_stage_with_another_flag_is_computed_as_without_the_paper_stage(
    tmp_path, flags, again
):
    # The chain always measures with the grid lines. What it found is taken up to
    # the first stage that is asked for something else: that one and the ones after
    # it are computed by run() as ever, or left out as ever, on top of the once the
    # chain called them.
    off, auto = _tilted_runs(tmp_path, *flags)
    for stage in STAGES[:3]:
        assert auto.calls[stage] == off.calls[stage] + (stage in again), stage
    # The chain warped the tilted page once, to measure its perspective on.
    assert auto.calls["warp_page"] == off.calls["warp_page"] + 1


# ------------------------------------------------------------------- saved masks
def test_a_saved_mask_replays_the_view_of_its_photo(digitised, replayed):
    # Nothing is decided a second time: the view is built from the record.
    assert replayed.decided == [] and replayed.asked == []
    line = "Paper normalisation for record rec: normalised from the mask's record"
    assert replayed.log.count(f"{line} (replayed)\n") == 1
    assert FITS_NOT not in replayed.log
    assert _qc(replayed.out)["paper_normalisation"] == "normalised"
    assert replayed.calls == ONCE
    for name in ("rec.dat", "rec.hea"):
        assert _file(replayed.out, name) == _file(digitised.out, name)


def test_a_mask_saved_again_keeps_its_record(tmp_path, digitised, replayed):
    # --mask_folder A --save_mask -o B: the record goes along as it stands, with the
    # mask and its frame ...
    assert _mask_meta(replayed.out) == _mask_meta(digitised.out)
    assert torch.equal(
        digitize.read_image(str(replayed.out / "rec_mask.png")),
        digitize.read_image(str(digitised.out / "rec_mask.png")),
    )
    # ... and B replays the photograph to the signals of A.
    again = _run(
        digitised.data, tmp_path / "out", "--paper_normalisation", "auto",
        "--mask_folder", str(replayed.out),
    )
    assert "(replayed)\n" in again.log and FITS_NOT not in again.log
    assert _file(again.out, "rec.dat") == _file(digitised.out, "rec.dat")


def test_a_mask_without_a_record_is_laid_over_the_page_as_it_is_given(
    tmp_path, monkeypatch
):
    for name in ("normalise_paper", "paper_chain", "check_paper_block"):
        monkeypatch.setattr(digitize, name, _never(name))
    data = _folder(tmp_path / "data", rec=_tensor(_page(SMALL, traces=False)))
    masks = tmp_path / "masks"
    masks.mkdir()
    write_png(torch.from_numpy(_flat_label())[None], str(masks / "rec_mask.png"))
    flags = (
        "--paper_normalisation", "auto", "--time_mapping", "bbox", *CHEAP,
        "--mask_folder", str(masks), "--save_mask",
    )
    # A mask with no JSON next to it, and one saved before there were records.
    bare = _run(data, tmp_path / "bare", *flags)
    _small_masks(masks)
    old = _run(data, tmp_path / "old", *flags)
    line = (
        "Paper normalisation for record rec: mask without a record, page used as "
        "given\n"
    )
    # Nothing looked at the page: the functions above would have said so. And a
    # mask that came without a record is saved again without one.
    for run in (bare, old):
        assert _qc(run.out)["paper_normalisation"] == "as given"
        assert run.log.count(line) == 1
        assert run.log.count("aper normalisation") == 1
        assert list(_mask_meta(run.out)) == ["rot_angle", "height", "width"]
    assert _mask_json(old.out) == _mask_json(bare.out)
    assert _file(old.out, "rec.dat") == _file(bare.out, "rec.dat")


def test_without_verbose_no_line_of_a_decision_is_printed(tmp_path):
    # Only the warnings of the paper stage are printed whatever --verbose says.
    flags = (
        "--time_mapping", "bbox", "--no-verbose", "--paper_normalisation", "auto",
    )
    page = _tensor(_page(SMALL, traces=False))
    data = _folder(tmp_path / "data", rec=page)
    # A page that comes with the record of its mask, and one whose mask has none.
    _small_masks(tmp_path / "with", page)
    _small_masks(tmp_path / "without")
    for masks, value in (("with", "pass_chain"), ("without", "as given")):
        run = _run(
            data, tmp_path / f"out_{masks}", *flags, *CHEAP,
            "--mask_folder", str(tmp_path / masks),
        )
        assert _qc(run.out)["paper_normalisation"] == value
        assert "aper normalisation" not in run.log

    # A page that is decided now: one of one channel that no stage reads is kept,
    # with the warning alone.
    grey = torch.full((1, SMALL[1], SMALL[0]), 255, dtype=torch.uint8)
    kept = _run(
        _folder(tmp_path / "grey", rec=grey), tmp_path / "kept", *flags,
        label=_flat_label(),
    )
    assert _qc(kept.out)["paper_normalisation"] == "failed"
    assert kept.log.count(
        "WARNING: paper normalisation not applied to record rec (page has 1 "
        "channels), page kept.\n"
    ) == 1
    assert kept.log.count("aper normalisation") == 1


@pytest.mark.parametrize("flag, saved", [("off", "once"), ("auto", "again")])
def test_a_page_that_is_the_view_of_its_mask_runs_with_either_flag(
    tmp_path, monkeypatch, digitised, replayed, view_data, flag, saved
):
    # The mask as the model's run saved it, and as the run on that mask saved it again.
    masks = digitised.out if saved == "once" else replayed.out
    monkeypatch.setattr(digitize, "normalise_paper", _never("normalise_paper"))
    monkeypatch.setattr(paper_normalisation, "replay_page", _never("replay_page"))
    run = _run(
        view_data, tmp_path / "out", "--paper_normalisation", flag,
        "--mask_folder", str(masks), "--save_mask",
    )
    line = "Paper normalisation for record rec: normalised from the mask's record"
    assert run.log.count(f"{line} (page is the view)\n") == 1
    assert run.log.count("aper normalisation") == 1
    assert FITS_NOT not in run.log
    # The QC column is the decision of the record, whatever the flag.
    assert _qc(run.out)["paper_normalisation"] == "normalised"
    assert _file(run.out, "rec.dat") == _file(digitised.out, "rec.dat")
    # And the mask saved again has the record it came with, byte for byte, whatever
    # the flag: it is the record of the page the mask was predicted on.
    assert _mask_json(run.out) == _mask_json(digitised.out)
    assert _mask_meta(run.out)["paper_normalisation"] == digitised.block


def test_check_paper_block_gives_the_page_back_or_the_view_built_again(
    photo, digitised
):
    block = digitised.block
    view, how = digitize.check_paper_block(block, digitised.view, "rec", "off")
    assert view is digitised.view and how == "view"
    # From the keys a record has to have, and from those alone.
    needed = (
        "version", "decision", "input_size", "input_sha1", "output_size", "view_sha1",
        "pre_size", "warp",
    )
    view, how = digitize.check_paper_block(
        {key: block[key] for key in needed}, _tensor(photo.image), "rec", "auto"
    )
    assert how == "replayed" and torch.equal(view, digitised.view)


def _masks(path, source, edit):
    """The mask of the folder source at path, the JSON next to it changed by edit."""
    path.mkdir()
    shutil.copy(os.path.join(source, "rec_mask.png"), path)
    meta = _mask_meta(source)
    edit(meta)
    with open(path / "rec_mask.json", "w") as f:
        json.dump(meta, f)
    return path


def _refused(tmp_path, monkeypatch, data, masks, flag, message, pages=None, extra=()):
    """The run is refused with message, before any stage has looked at the page."""
    resolution = _spy(monkeypatch, "normalise_resolution")
    rotation = _spy(monkeypatch, "estimate_rotation")
    with pytest.raises(ValueError, match=re.escape(message)) as refusal:
        _run(
            data, tmp_path / "out", "--paper_normalisation", flag,
            "--mask_folder", str(masks), *extra, pages=pages,
        )
    text = str(refusal.value)
    assert text.startswith("Mask of record rec ") and text.count(FITS_NOT) == 1
    assert text.endswith(WAY_OUT) and "\n" not in text
    assert resolution == [] and rotation == []
    assert not (tmp_path / "out" / "qc.csv").exists()
    return text


@pytest.mark.parametrize("flag", ["off", "auto"])
def test_a_mask_of_another_view_is_refused(
    tmp_path, monkeypatch, digitised, view_data, flag
):
    def another(meta):
        meta["paper_normalisation"]["view_sha1"] = "0" * 40

    masks = _masks(tmp_path / "masks", digitised.out, another)
    text = _refused(tmp_path, monkeypatch, view_data, masks, flag, ANOTHER_VIEW)
    # The refusal says what it compared: the decision, both sizes and the digests.
    block = digitised.block
    for part in ("normalised", "[2200, 1700]", "[3360, 2520]", "0" * 40,
                 block["input_sha1"], block["view_sha1"]):
        assert part in text


@pytest.mark.parametrize("flag", ["off", "auto"])
def test_a_page_of_the_size_of_the_view_with_other_pixels_is_refused(
    tmp_path, monkeypatch, digitised, view_data, flag
):
    # What the size check of the mask cannot see: one pixel of the view changed.
    other = digitised.view.clone()
    other[0, 0, 0] ^= 1
    _refused(
        tmp_path, monkeypatch, view_data, digitised.out, flag, ANOTHER_VIEW,
        pages={"rec": other},
    )


def test_a_photo_with_one_pixel_changed_is_refused_as_another_input_page(
    tmp_path, monkeypatch, photo, digitised
):
    # Not replayed at all: the view of another page would only fail its digest.
    monkeypatch.setattr(paper_normalisation, "replay_page", _never("replay_page"))
    other = _tensor(photo.image)
    other[0, 0, 0] ^= 1
    _refused(
        tmp_path, monkeypatch, digitised.data, digitised.out, "auto",
        "another input page; the mask does not fit.", pages={"rec": other},
    )


def test_without_the_flag_the_photo_is_refused_with_the_way_out(
    tmp_path, monkeypatch, digitised
):
    # -f does not let it through either: the signals would look like signals.
    _refused(
        tmp_path, monkeypatch, digitised.data, digitised.out, "off",
        "the mask does not fit: pass --paper_normalisation auto to replay the mask's "
        "record.",
        extra=("-f",),
    )


def test_without_the_flag_a_changed_photo_is_not_told_to_pass_the_flag(
    tmp_path, monkeypatch, photo, digitised
):
    # A page of the size of the photograph with other pixels: the flag would not
    # replay it either, it would refuse it as another input page.
    other = _tensor(photo.image)
    other[0, 0, 0] ^= 1
    text = _refused(
        tmp_path, monkeypatch, digitised.data, digitised.out, "off",
        "the mask does not fit: the mask was predicted on the view of another input "
        "page.",
        pages={"rec": other},
    )
    assert "--paper_normalisation auto" not in text
    for part in (digitised.block["input_sha1"], digitize.page_sha1(other)):
        assert text.count(part) == 1


@pytest.mark.parametrize(
    "flag, decision",
    [
        ("off", "pass_chain"), ("auto", "pass_chain"), ("auto", "pass_fullframe"),
        ("off", "failed"), ("auto", "failed"),
    ],
)
def test_a_changed_page_is_refused_by_the_record_of_a_page_left_as_it_is(
    tmp_path, monkeypatch, flag, decision
):
    # Such a record has nothing to replay, so the page has to be the very page.
    page = _tensor(_page(SMALL, traces=False))
    block = _small_masks(tmp_path / "masks", page, decision)
    other = page.clone()
    other[0, 0, 0] ^= 1
    data = _folder(tmp_path / "data", rec=other)
    text = _refused(tmp_path, monkeypatch, data, tmp_path / "masks", flag, FITS_NOT)
    # The view of that record is the page it was made from: one size and one
    # digest, each said once.
    assert text == (
        f"Mask of record rec was predicted on a {decision} page of [1100, 850] px "
        f"(width, height) with SHA-1 {block['view_sha1']}; the page given now is "
        f"[1100, 850] px with SHA-1 {digitize.page_sha1(other)}; the mask does not "
        f"fit: the mask was predicted on another page.{WAY_OUT}"
    )


def test_a_refused_page_stops_the_run_after_the_pages_before_it(tmp_path, monkeypatch):
    # Two pages in the order of their names, the mask of the second of another view.
    page = _tensor(_page(SMALL, traces=False))
    data = _folder(tmp_path / "data", a=page, b=page)
    masks = tmp_path / "masks"
    _small_masks(masks, page, record="a")
    _small_masks(masks, page, record="b", view_sha1="0" * 40)
    listdir = os.listdir
    monkeypatch.setattr(os, "listdir", lambda path: sorted(listdir(path)))
    with pytest.raises(ValueError, match="^Mask of record b was predicted on a "):
        _run(
            data, tmp_path / "out", "--paper_normalisation", "auto", "--time_mapping",
            "bbox", *CHEAP, "--mask_folder", str(masks),
        )
    # What is on disk is the first page, whole, and nothing of the second.
    assert sorted(listdir(tmp_path / "out")) == ["a.dat", "a.hea", "qc.csv"]
    with open(tmp_path / "out" / "qc.csv", newline="") as f:
        rows = list(csv.DictReader(f))
    assert [(row["record"], row["paper_normalisation"]) for row in rows] == [
        ("a", "pass_chain")
    ]


def test_the_digest_of_a_page_with_an_alpha_channel_is_that_of_its_colours(tmp_path):
    # run() drops the fourth channel before the paper stage, when the record is
    # written and when a page is checked against it.
    small = _page(SMALL, traces=False)
    alpha = np.full(small.shape[:2] + (1,), 255, dtype=np.uint8)
    alpha[::2] = 128
    data = tmp_path / "data"
    data.mkdir()
    bgra = np.concatenate([small[:, :, ::-1], alpha], axis=2)
    assert cv2.imwrite(str(data / "rec.png"), bgra)
    rgba = digitize.read_image(str(data / "rec.png"))
    assert rgba.shape[0] == 4 and torch.equal(rgba[:3], _tensor(small))
    digest = digitize.page_sha1(rgba[:3])
    assert digest != digitize.page_sha1(rgba)

    flags = ("--time_mapping", "bbox", *CHEAP, "--paper_normalisation")
    first = _run(
        data, tmp_path / "first", *flags, "auto", "--save_mask", label=_flat_label()
    )
    page, (view, block, meta, stages) = first.decided[0]
    assert page.shape[0] == 3 and block["decision"] == "pass_chain"
    assert first.asked == [((3, SMALL[1], SMALL[0]), digest)]
    assert block["input_sha1"] == block["view_sha1"] == digest
    assert _mask_meta(first.out)["paper_normalisation"] == block
    line = (
        "Paper normalisation for record rec: pass_chain from the mask's record (page "
        "is the view)\n"
    )
    for flag in ("off", "auto"):
        again = _run(
            data, tmp_path / flag, *flags, flag, "--mask_folder", str(first.out)
        )
        assert again.log.count(line) == 1 and FITS_NOT not in again.log
        assert _file(again.out, "rec.dat") == _file(first.out, "rec.dat")


@pytest.mark.parametrize(
    "flag, page", [("off", "view"), ("auto", "view"), ("auto", "photo")]
)
def test_a_record_of_an_unknown_version_is_refused_whatever_the_page(
    tmp_path, monkeypatch, digitised, view_data, flag, page
):
    # Also on the very view of the mask, whose digest and size are those of the record.
    def unknown(meta):
        meta["paper_normalisation"]["version"] = "paper_normalisation 0"

    masks = _masks(tmp_path / "masks", digitised.out, unknown)
    data = view_data if page == "view" else digitised.data
    _refused(
        tmp_path, monkeypatch, data, masks, flag,
        "has a paper normalisation record of version 'paper_normalisation 0', this "
        f"code reads {paper_normalisation.VERSION!r}; the mask does not fit.",
    )


NO_RECORDS = [
    (None, "that is not an object"),
    ("normalised", "that is not an object"),
    ([], "that is not an object"),
    (3, "that is not an object"),
    ({}, "without a version, this code reads {known}"),
    ({"decision": "normalised"}, "without a version, this code reads {known}"),
    ({"version": None}, "of version None, this code reads {known}"),
    ({"version": ["x"]}, "of version ['x'], this code reads {known}"),
]


def _no_record(what):
    """The refusal of a record that is none: only one with a version is told of one."""
    what = what.format(known=repr(paper_normalisation.VERSION))
    return (
        f"Mask of record rec has a paper normalisation record {what}; the mask does "
        f"not fit.{WAY_OUT}"
    )


@pytest.mark.parametrize("record, what", NO_RECORDS[:3] + NO_RECORDS[4:5])
def test_a_record_that_is_no_record_is_refused(
    tmp_path, monkeypatch, digitised, view_data, record, what
):
    def empty(meta):
        meta["paper_normalisation"] = record

    masks = _masks(tmp_path / "masks", digitised.out, empty)
    text = _refused(tmp_path, monkeypatch, view_data, masks, "auto", FITS_NOT)
    assert text == _no_record(what)


@pytest.mark.parametrize("flag", ["off", "auto"])
@pytest.mark.parametrize("record, what", NO_RECORDS)
def test_check_paper_block_says_what_a_record_that_is_none_lacks(flag, record, what):
    page = torch.zeros((3, 4, 6), dtype=torch.uint8)
    with pytest.raises(ValueError) as refusal:
        digitize.check_paper_block(record, page, "rec", flag)
    assert str(refusal.value) == _no_record(what)


def test_a_record_with_an_unknown_decision_is_refused(
    tmp_path, monkeypatch, digitised, view_data
):
    def inferred(meta):
        meta["paper_normalisation"]["decision"] = "normalised_inferred"

    masks = _masks(tmp_path / "masks", digitised.out, inferred)
    _refused(
        tmp_path, monkeypatch, view_data, masks, "auto",
        "with the decision 'normalised_inferred', known are pass_chain, "
        "pass_fullframe, failed, normalised, scaled; the mask does not fit.",
    )


@pytest.mark.parametrize(
    "key", ["input_size", "input_sha1", "output_size", "view_sha1", "pre_size", "warp"]
)
def test_a_record_without_a_key_it_needs_is_refused(
    tmp_path, monkeypatch, digitised, view_data, key
):
    def without(meta):
        del meta["paper_normalisation"][key]

    masks = _masks(tmp_path / "masks", digitised.out, without)
    _refused(
        tmp_path, monkeypatch, view_data, masks, "auto",
        f"record of a normalised page without {key}; the mask does not fit.",
    )


def test_a_scaled_record_needs_the_size_it_was_scaled_to(
    tmp_path, monkeypatch, digitised, view_data
):
    def scaled(meta):
        meta["paper_normalisation"]["decision"] = "scaled"

    masks = _masks(tmp_path / "masks", digitised.out, scaled)
    _refused(
        tmp_path, monkeypatch, view_data, masks, "off",
        "record of a scaled page without resize; the mask does not fit.",
    )


def test_a_view_that_is_built_again_as_another_is_refused_with_what_differs(
    photo, digitised
):
    # The record of another library version: its warp gives other pixels now. What
    # differs is said first, the libraries after it.
    page = _tensor(photo.image)
    block = json.loads(json.dumps(digitised.block))
    block["warp"][0][2] += 1.0
    block["libraries"] = "numpy 1.0 cv2 3.0"
    now = f"numpy {np.__version__} cv2 {cv2.__version__}"
    with pytest.raises(ValueError) as refusal:
        digitize.check_paper_block(block, page, "rec", "auto")
    start = (
        f"Mask of record rec was predicted on a normalised view of [2200, 1700] px "
        f"(width, height) with SHA-1 {block['view_sha1']}, the view built again from "
        f"its record has that size and SHA-1 "
    )
    end = (
        f": other pixels (libraries then: numpy 1.0 cv2 3.0, now: {now}); the mask "
        f"does not fit.{WAY_OUT}"
    )
    text = str(refusal.value)
    assert text.startswith(start) and text.endswith(end)
    again = text[len(start) : -len(end)]
    assert re.fullmatch("[0-9a-f]{40}", again) and again != block["view_sha1"]

    # A record that says another size than its numbers give, and no libraries.
    block = json.loads(json.dumps(digitised.block))
    block["output_size"] = [2200, 1701]
    del block["libraries"]
    with pytest.raises(ValueError) as refusal:
        digitize.check_paper_block(block, page, "rec", "auto")
    assert str(refusal.value) == (
        f"Mask of record rec was predicted on a normalised view of [2200, 1701] px "
        f"(width, height), the view built again from its record is [2200, 1700] px: "
        f"another size (libraries then: not recorded, now: {now}); the mask does not "
        f"fit.{WAY_OUT}"
    )


@lru_cache(maxsize=None)
def _forged_page(decision):
    """A small page of noise and the record of its view, as normalise_paper has it."""
    rng = np.random.default_rng(1)
    page = rng.integers(0, 255, (100, 200, 3), dtype=np.uint8)
    if decision == "normalised":
        warp = [[8.0, 0.1, 3.0], [0.05, 9.0, -2.0], [1e-5, 0.0, 1.0]]
        numbers = {"pre_size": [200, 100], "warp": warp}
        view = paper_normalisation.apply_warp(page, [200, 100], warp)
    else:
        numbers = {"resize": [100, 50]}
        view = paper_normalisation.apply_resize(page, [100, 50])
    block = {
        "version": paper_normalisation.VERSION, "decision": decision,
        "input_size": [200, 100], "input_sha1": digitize.page_sha1(_tensor(page)),
        "output_size": [view.shape[1], view.shape[0]],
        "view_sha1": digitize.page_sha1(_tensor(view)), **numbers,
    }
    return _tensor(page), block, _tensor(view)


def _forged(decision):
    """That page and a record of its own for the caller to change."""
    page, block, _ = _forged_page(decision)
    return page, json.loads(json.dumps(block))


@pytest.mark.parametrize("decision", ["normalised", "scaled"])
def test_a_record_written_by_hand_replays_its_page(decision):
    # What the tests below change one number of.
    page, block, view = _forged_page(decision)
    again, how = digitize.check_paper_block(block, page, "rec", "auto")
    assert how == "replayed" and torch.equal(again, view)


@pytest.mark.parametrize(
    "decision, key, value",
    [
        ("normalised", "warp", [[1.0, 0.0, 0.0], [0.0, 1.0, 0.0]]),
        ("normalised", "warp", None),
        ("normalised", "warp", "abc"),
        ("normalised", "warp", [[1, 0, 0], [0, 1], [0, 0, 1]]),
        ("normalised", "pre_size", None),
        ("normalised", "pre_size", [0, 0]),
        ("normalised", "pre_size", "ab"),
        ("scaled", "resize", None),
        ("scaled", "resize", [0, 0]),
        ("scaled", "resize", "ab"),
        ("scaled", "resize", [100]),
    ],
)
def test_a_record_whose_numbers_build_no_view_is_refused(decision, key, value):
    # Whatever numpy or cv2 make of the number, the run is told which record it is.
    page, block = _forged(decision)
    block[key] = value
    with pytest.raises(ValueError) as refusal:
        digitize.check_paper_block(block, page, "rec", "auto")
    text = str(refusal.value)
    start = (
        f"Mask of record rec has the paper normalisation record of a {decision} page: "
        f"the record cannot be replayed ("
    )
    assert text.startswith(start) and "\n" not in text
    assert re.match(r"\w+: \S", text[len(start) :])
    assert text.endswith(f"); the mask does not fit.{WAY_OUT}")
    # What the library said is kept behind the refusal.
    assert isinstance(refusal.value.__cause__, Exception)


@pytest.mark.parametrize("decision", ["normalised", "scaled"])
def test_a_record_of_a_colour_page_is_not_replayed_on_a_page_of_one_channel(decision):
    page, block = _forged(decision)
    grey = page[:1].clone()
    block["input_sha1"] = digitize.page_sha1(grey)
    with pytest.raises(ValueError, match=re.escape(FITS_NOT)) as refusal:
        digitize.check_paper_block(block, grey, "rec", "auto")
    text = str(refusal.value)
    assert text.startswith("Mask of record rec ") and text.endswith(WAY_OUT)
    assert "the record cannot be replayed (" in text


def test_a_record_that_cannot_be_replayed_is_refused_before_the_stages(
    tmp_path, monkeypatch, digitised
):
    def cut(meta):
        meta["paper_normalisation"]["warp"] = None

    masks = _masks(tmp_path / "masks", digitised.out, cut)
    _refused(
        tmp_path, monkeypatch, digitised.data, masks, "auto",
        "has the paper normalisation record of a normalised page: the record cannot "
        "be replayed (",
    )


@pytest.mark.parametrize("decision", [[], {}, ["normalised"], 3, None, "Scaled"])
def test_a_decision_that_is_not_one_of_the_five_is_refused_as_unknown(decision):
    page, block = _forged("scaled")
    block["decision"] = decision
    for flag in ("off", "auto"):
        with pytest.raises(ValueError) as refusal:
            digitize.check_paper_block(block, page, "rec", flag)
        assert str(refusal.value) == (
            f"Mask of record rec has a paper normalisation record with the decision "
            f"{decision!r}, known are pass_chain, pass_fullframe, failed, normalised, "
            f"scaled; the mask does not fit.{WAY_OUT}"
        )


@pytest.mark.parametrize(
    "flag, raw",
    [
        ("off", "null"), ("off", "3"), ("off", "[]"), ("off", '"paper_normalisation"'),
        ("off", '["paper_normalisation"]'), ("auto", "null"),
        ("auto", '["paper_normalisation"]'),
    ],
)
def test_a_mask_json_that_holds_no_object_fails_where_it_always_did(
    tmp_path, monkeypatch, flag, raw
):
    # Not in the paper stage, which finds no record in it: in the check of the frame
    # of the mask, after the stages, as before there was a paper stage.
    monkeypatch.setattr(digitize, "check_paper_block", _never("check_paper_block"))
    data = _folder(tmp_path / "data", rec=_tensor(_page(SMALL, traces=False)))
    masks = tmp_path / "masks"
    _small_masks(masks)
    (masks / "rec_mask.json").write_text(raw)
    rotation = _spy(monkeypatch, "estimate_rotation")
    with pytest.raises(AttributeError, match="object has no attribute 'get'"):
        _run(
            data, tmp_path / "out", "--paper_normalisation", flag, "--time_mapping",
            "bbox", *CHEAP, "--mask_folder", str(masks),
        )
    assert len(rotation) == 1


@pytest.mark.parametrize("flag", ["off", "auto"])
def test_a_mask_json_that_cannot_be_parsed_stops_the_run_before_the_stages(
    tmp_path, monkeypatch, flag
):
    # The JSON is read for the record before anything is measured on the page.
    data = _folder(tmp_path / "data", rec=_tensor(_page(SMALL, traces=False)))
    masks = tmp_path / "masks"
    _small_masks(masks)
    (masks / "rec_mask.json").write_text('{"rot_angle": 0.0, ')
    rotation = _spy(monkeypatch, "estimate_rotation")
    with pytest.raises(json.JSONDecodeError):
        _run(
            data, tmp_path / "out", "--paper_normalisation", flag, "--time_mapping",
            "bbox", *CHEAP, "--mask_folder", str(masks),
        )
    assert rotation == []


def test_a_mask_of_another_size_is_told_when_its_page_was_used_as_given(tmp_path):
    page = _tensor(_page(SMALL, traces=False))
    data = _folder(tmp_path / "data", rec=page)
    masks = tmp_path / "masks"
    half = torch.zeros((1, SMALL[1] // 2, SMALL[0] // 2), dtype=torch.uint8)
    runs = []

    def message(flag):
        write_png(half, str(masks / "rec_mask.png"))
        runs.append(flag)
        with pytest.raises(ValueError) as refusal:
            _run(
                data, tmp_path / f"out{len(runs)}", "--paper_normalisation", flag,
                *CHEAP, "--mask_folder", str(masks),
            )
        return str(refusal.value)

    # The message of the size check as it always was, to the letter ...
    always = (
        "Mask of record rec is 550 x 425 px, the page is 1100 x 850 px; the mask was "
        "predicted for another --resolution."
    )
    # ... for a mask without a record, as one of a view that was built elsewhere:
    # only the flag that asks for the normalisation is told that there was none.
    _small_masks(masks)
    assert message("off") == always
    assert message("auto") == always + (
        " The mask has no paper normalisation record, so the page was used as given."
    )
    # A mask with a record has had its page checked, with either flag.
    _small_masks(masks, page)
    assert message("off") == always
    assert message("auto") == always


def test_a_mask_without_a_record_is_saved_by_the_call_it_always_was(
    tmp_path, monkeypatch
):
    # Drivers put a wrapper with the six parameters save_mask_files had in its place.
    real = digitize.save_mask_files
    saved = []

    def save_mask_files(
        mask, record, output_folder, rot_angle, homography=None, scale=1.0
    ):
        saved.append(output_folder)
        return real(mask, record, output_folder, rot_angle, homography, scale)

    monkeypatch.setattr(digitize, "save_mask_files", save_mask_files)
    data = _folder(tmp_path / "data", rec=_tensor(_page(SMALL, traces=False)))
    flags = ("--time_mapping", "bbox", *CHEAP, "--save_mask", "--paper_normalisation")
    # Without the flag, and with a mask that has no record under either flag.
    first = _run(data, tmp_path / "first", *flags, "off", label=_flat_label())
    masks = ("--mask_folder", str(first.out))
    second = _run(data, tmp_path / "second", *flags, "off", *masks)
    third = _run(data, tmp_path / "third", *flags, "auto", *masks)
    assert saved == [str(run.out) for run in (first, second, third)]
    for run in (second, third):
        assert _mask_json(run.out) == _mask_json(first.out)
    assert list(_mask_meta(first.out)) == ["rot_angle", "height", "width"]
    # A record is one more thing to save, and goes in by its name.
    _small_masks(tmp_path / "masks", _tensor(_page(SMALL, traces=False)))
    with pytest.raises(TypeError, match="unexpected keyword argument 'paper'"):
        _run(
            data, tmp_path / "fourth", *flags, "off",
            "--mask_folder", str(tmp_path / "masks"),
        )


# ------------------------------------------------------------------------ the hooks
def test_run_calls_what_drivers_wrap_with_the_arguments_it_always_had(
    tmp_path, monkeypatch
):
    """Drivers put wrappers of exactly these arguments in place of the functions.

    The stages and read_image are wrapped that way by every _run() of this file.
    """
    real_check, real_row = digitize.check_mask_rotation, digitize.append_qc_row
    seen = []

    def check_mask_rotation(mask_folder, record, rot_angle, homography):
        seen.append(("check_mask_rotation", mask_folder, record, homography))
        return real_check(mask_folder, record, rot_angle, homography)

    def append_qc_row(output_folder, record, placement, qc, max_offset_deviation):
        seen.append(("append_qc_row", record, qc["paper_normalisation"]))
        return real_row(output_folder, record, placement, qc, max_offset_deviation)

    monkeypatch.setattr(digitize, "check_mask_rotation", check_mask_rotation)
    monkeypatch.setattr(digitize, "append_qc_row", append_qc_row)

    # A mask with the record of a page that was left as it is, written by hand.
    page = _tensor(_page(SMALL, traces=False))
    data = _folder(tmp_path / "data", rec=page)
    masks = tmp_path / "masks"
    _small_masks(masks, page)

    run = _run(
        data, tmp_path / "out", "--paper_normalisation", "auto", "--time_mapping",
        "bbox", "--mask_folder", str(masks),
    )
    assert seen == [
        ("check_mask_rotation", str(masks), "rec", None),
        ("append_qc_row", "rec", "pass_chain"),
    ]
    # With a mask every stage is computed by run() itself, once.
    assert run.calls == ONCE and run.decided == []
    assert "pass_chain from the mask's record (page is the view)\n" in run.log
    assert _qc(run.out)["paper_normalisation"] == "pass_chain"


# --------------------------------------------------------------- pages of one channel
def test_a_grey_page_the_chain_reads_passes_and_is_never_searched(
    tmp_path, monkeypatch
):
    # The paper search reads colour, so a page of one channel never gets to it.
    monkeypatch.setattr(paper_normalisation, "find_paper", _never("find_paper"))
    grey = torch.from_numpy(_page(SMALL, traces=False).min(axis=2))[None]
    data = _folder(tmp_path / "data", rec=grey)
    flags = ("--time_mapping", "bbox")
    off = _run(data, tmp_path / "off", *flags, label=_flat_label())
    auto = _run(
        data, tmp_path / "auto", "--paper_normalisation", "auto", *flags,
        label=_flat_label(),
    )
    page, (view, block, meta, stages) = auto.decided[0]
    assert tuple(page.shape) == (1, SMALL[1], SMALL[0])
    assert view is page and stages is not None
    _plain_record(block, RECORD_KEYS)
    assert (block["decision"], block["reason"]) == ("pass_chain", "")
    assert block["input_sha1"] == block["view_sha1"] == digitize.page_sha1(grey)
    assert off.calls == auto.calls == ONCE
    assert auto.asked == off.asked
    assert _qc(auto.out)["paper_normalisation"] == "pass_chain"
    assert _other_columns(auto.out) == _same_columns(off.out)
    for name in ("rec.dat", "rec.hea"):
        assert _file(auto.out, name) == _file(off.out, name)


def test_a_grey_page_the_chain_cannot_read_is_kept_with_a_warning(
    tmp_path, monkeypatch
):
    monkeypatch.setattr(paper_normalisation, "find_paper", _never("find_paper"))
    grey = torch.full((1, SMALL[1], SMALL[0]), 255, dtype=torch.uint8)
    data = _folder(tmp_path / "data", rec=grey)
    run = _run(
        data, tmp_path / "out", "--paper_normalisation", "auto", "--time_mapping",
        "bbox", "--save_mask", label=_flat_label(),
    )
    page, (view, block, meta, stages) = run.decided[0]
    _plain_record(block, RECORD_KEYS)
    assert (block["decision"], block["reason"]) == ("failed", "page has 1 channels")
    assert run.log.count(
        "WARNING: paper normalisation not applied to record rec (page has 1 "
        "channels), page kept.\n"
    ) == 1
    line = "Paper normalisation for record rec: failed (page has 1 channels)\n"
    assert run.log.count(line) == 1
    assert run.log.count("aper normalisation") == 2
    assert _mask_meta(run.out)["paper_normalisation"] == block
    # The pixels of the page as they came, in the record and in front of the model.
    digest = digitize.page_sha1(grey)
    assert view is page and stages is not None
    assert block["input_size"] == block["output_size"] == list(SMALL)
    assert block["input_sha1"] == block["view_sha1"] == digest
    assert run.asked == [((1, SMALL[1], SMALL[0]), digest)]
    assert run.calls == ONCE
    assert _qc(run.out)["paper_normalisation"] == "failed"


# ------------------------------------------------------------------------------ QC
def test_append_qc_row_writes_the_decision_after_the_offset_deviation(tmp_path):
    qc = {"paper_normalisation": "normalised", "baseline_scale": 1.0}
    digitize.append_qc_row(str(tmp_path), "rec", "column", qc, 0.25)
    with open(tmp_path / "qc.csv", newline="") as f:
        reader = csv.DictReader(f)
        row = next(reader)
        # One new column, in between: the ones before and after it keep their order.
        assert reader.fieldnames[11:14] == [
            "max_offset_deviation",
            "paper_normalisation",
            "baseline_scale",
        ]
    assert row["paper_normalisation"] == "normalised"
    assert float(row["max_offset_deviation"]) == 0.25
