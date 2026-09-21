"""Unit tests for the resolution stage of src/run/digitize.py.

No model, no data files: every page is synthesised with numpy/cv2/torch.
"""
import csv
import json
from functools import lru_cache

import cv2
import numpy as np
import pytest
import torch
from torchvision.io.image import write_png

from src.run import digitize

HEIGHT, WIDTH = 1700, 2200
# 1 mm at 200 dpi, the pitch of the printed grid lines of a generator page.
PERIOD = 200 / 25.4
# Colour and width of the 5 mm and of the 1 mm lines. "printed" is a generator page,
# "flat" a grid whose 5 mm lines are not darker at all, and "bold" one whose 5 mm
# lines are so much darker that a blur leaves nothing of the 1 mm comb beside them.
GRID_STYLES = {
    "printed": (((120, 120, 255), 2), ((190, 190, 255), 1)),
    "flat": (((190, 190, 255), 1), ((190, 190, 255), 1)),
    "bold": (((60, 60, 255), 3), ((215, 215, 255), 1)),
}


@lru_cache(maxsize=None)
def _page(period=PERIOD, size=1.0, grid="printed", tilt=0.0, blur=0.0):
    """A page of size times 2200 x 1700 px with a 1 mm grid of this period.

    size is what a page of another resolution looks like: half of it is a 100 dpi
    scan of the same sheet. cv2 rotates the content, so the page comes back tilted by
    +tilt. Drawing a page takes a moment, so they are cached and no caller may write
    into one.
    """
    (heavy_colour, heavy_width), (light_colour, light_width) = GRID_STYLES[grid]
    height, width = int(HEIGHT * size), int(WIDTH * size)
    image = np.full((height, width, 3), 255, dtype=np.uint8)
    for k in range(int(width / period) + 1):
        x = int(round(k * period))
        heavy = k % 5 == 0
        if x < width:
            cv2.line(image, (x, 0), (x, height - 1),
                     heavy_colour if heavy else light_colour,
                     heavy_width if heavy else light_width)
    for k in range(int(height / period) + 1):
        y = int(round(k * period))
        heavy = k % 5 == 0
        if y < height:
            cv2.line(image, (0, y), (width - 1, y),
                     heavy_colour if heavy else light_colour,
                     heavy_width if heavy else light_width)
    # Traces and a text-like blob, so that the band median has something to reject.
    rng = np.random.default_rng(0)
    for row in range(int(300 * size), height, int(400 * size)):
        x = np.arange(0, width, 5)
        y = row + 60 * size * np.sin(2 * np.pi * x / (300 * size))
        y += rng.normal(0, 1, x.size)
        cv2.polylines(
            image, [np.stack([x, y], axis=1).astype(np.int32)], False, (0, 0, 0), 2
        )
    cv2.rectangle(image, (100, 60), (500, 110), (0, 0, 0), -1)
    if tilt:
        matrix = cv2.getRotationMatrix2D((width / 2, height / 2), tilt, 1.0)
        image = cv2.warpAffine(
            image,
            matrix,
            (width, height),
            flags=cv2.INTER_LINEAR,
            borderMode=cv2.BORDER_CONSTANT,
            borderValue=(255, 255, 255),
        )
    if blur:
        image = cv2.GaussianBlur(image, (0, 0), blur)
    return image


def _tensor(np_image):
    """The page as run() reads it: [3, H, W] uint8."""
    return torch.from_numpy(np_image.transpose(2, 0, 1).copy())


def _blank():
    return np.full((HEIGHT, WIDTH, 3), 255, dtype=np.uint8)


def _no_grid_page():
    """A page with lines the Hough transform sees, but no printed 1 mm grid."""
    image = _blank()
    for y in range(100, HEIGHT - 100, 50):
        cv2.line(image, (0, y), (WIDTH - 1, y), (0, 0, 0), 2)
    for x in (137, 402, 913, 1290, 1755, 2050):
        cv2.line(image, (x, 0), (x, HEIGHT - 1), (0, 0, 0), 3)
    return image


def _noisy(np_image, sigma=25.0):
    """The same page under sensor noise, which lifts the whole spectrum at once."""
    rng = np.random.default_rng(1)
    return np.clip(np_image + rng.normal(0, sigma, np_image.shape), 0, 255).astype(
        np.uint8
    )


def _strongest_peak(page):
    """Period of the strongest peak of the spectrum, before any hypothesis test."""
    profiles, _ = digitize._band_profiles(digitize._darkness(page))
    frequencies, amplitude = digitize._welch_spectrum(profiles)
    prominence = digitize._spectral_prominence(
        amplitude, frequencies[1] - frequencies[0]
    )
    low, high = digitize.RESOLUTION_PERIOD_RANGE
    inside = np.flatnonzero((frequencies >= 1 / high) & (frequencies <= 1 / low))
    return 1 / frequencies[inside[np.argmax(prominence[inside])]]


# -------------------------------------------------------------- _spectral_prominence
def test_spectral_prominence_keeps_the_lines_and_drops_the_hump():
    """What noise does to a spectrum is a broad hump, and a comb is not that."""
    frequencies = np.linspace(0, 0.5, 2049)
    hump = 10 * np.exp(-((frequencies - 0.1) ** 2) / (2 * 0.05**2))
    spectrum = hump.copy()
    line = int(np.argmin(np.abs(frequencies - 1 / PERIOD)))
    spectrum[line - 1 : line + 2] += [1.0, 3.0, 1.0]
    prominence = digitize._spectral_prominence(
        spectrum, frequencies[1] - frequencies[0]
    )
    assert prominence[line] == pytest.approx(3.0, abs=0.1)
    # Away from the line the hump is gone, whatever its height was there.
    assert np.all(prominence[line + 100 :] < 0.05 * hump[line + 100 :])


# --------------------------------------------------------------- measure_grid_period
@pytest.mark.parametrize("size", [0.5, 0.75, 1.0, 1.5])
def test_measure_grid_period_reads_the_printed_period(size):
    # The same sheet scanned at 100, 150, 200 and 300 dpi: both the page and the
    # period of its grid are the 200 dpi ones times the resolution.
    measured, info = digitize.measure_grid_period(
        _tensor(_page(period=PERIOD * size, size=size))
    )
    assert measured == pytest.approx(PERIOD * size, rel=0.002)
    assert info["harmonic"] == 5
    assert info["contrast"] > digitize.GRID_LINE_MIN_CONTRAST
    assert info["reason"] == ""


def test_measure_grid_period_does_not_lock_onto_a_five_mm_harmonic():
    """What the hypothesis test is for, on a page where the plain peak really fails."""
    page = _tensor(_page(grid="bold", blur=2.0))
    profiles, _ = digitize._band_profiles(digitize._darkness(page))
    # The period the rotation uses is the third harmonic of the 5 mm lines here, and
    # the strongest peak of the spectrum is the 5 mm fundamental itself.
    assert digitize._grid_line_period(profiles) == pytest.approx(
        5 * PERIOD / 3, rel=0.01
    )
    assert _strongest_peak(page) == pytest.approx(5 * PERIOD, rel=0.01)
    # The hypothesis test votes with the whole 5 mm series and lands on the 1 mm one.
    measured, info = digitize.measure_grid_period(page)
    assert measured == pytest.approx(PERIOD, rel=0.002)
    assert info["harmonic"] == 1


def test_measure_grid_period_keeps_the_peak_of_a_grid_without_darker_lines():
    page = _tensor(_page(grid="flat"))
    assert _strongest_peak(page) == pytest.approx(PERIOD, rel=0.01)
    measured, info = digitize.measure_grid_period(page)
    # No sub-harmonic votes for a 5 mm comb here, so the peak stays the 1 mm line.
    assert info["harmonic"] == 5
    assert measured == pytest.approx(PERIOD, rel=0.002)


def test_measure_grid_period_survives_heavy_noise():
    measured, info = digitize.measure_grid_period(_tensor(_noisy(_page())))
    assert measured == pytest.approx(PERIOD, rel=0.002)
    assert info["reason"] == ""


def test_measure_grid_period_straightens_the_page_first():
    # Two degrees drag a line across half a period over one band of 100 rows, which
    # is what the coarse rotation inside the measurement takes out again.
    measured, info = digitize.measure_grid_period(_tensor(_page(tilt=2.0)))
    assert measured == pytest.approx(PERIOD, rel=0.002)
    assert info["reason"] == ""


def test_measure_grid_period_takes_a_numpy_page_as_well():
    page = _page(grid="flat")
    from_numpy, _ = digitize.measure_grid_period(page.transpose(2, 0, 1))
    from_tensor, _ = digitize.measure_grid_period(_tensor(page))
    assert from_numpy == pytest.approx(from_tensor, abs=1e-9)


def test_measure_grid_period_is_nan_on_a_blank_page():
    measured, info = digitize.measure_grid_period(_tensor(_blank()))
    assert np.isnan(measured)
    assert info["reason"] == "no grid line period found"


def test_measure_grid_period_needs_grid_lines():
    measured, info = digitize.measure_grid_period(_tensor(_no_grid_page()))
    assert np.isnan(measured)
    assert "no grid lines found" in info["reason"]


def test_measure_grid_period_needs_a_whole_band_of_rows():
    thin = _tensor(_page()[: digitize.GRID_LINE_BAND_HEIGHT - 1])
    measured, info = digitize.measure_grid_period(thin)
    assert np.isnan(measured)
    assert info["reason"] == "image too small for the grid line profile"


# --------------------------------------------------------------- normalise_resolution
def test_normalise_resolution_leaves_a_200_dpi_page_untouched():
    page = _tensor(_page())
    same, info = digitize.normalise_resolution(page)
    # Inside the dead band the very object comes back, not a copy of it.
    assert same is page
    assert info["scale"] == 1.0
    assert info["period"] == pytest.approx(PERIOD, rel=0.002)
    assert info["reason"] == ""


def test_normalise_resolution_keeps_a_generator_page():
    # Two percent off, as a generator page that was cropped and turned measures.
    page = _tensor(_page(period=PERIOD * 0.98))
    same, info = digitize.normalise_resolution(page)
    assert same is page
    assert info["scale"] == 1.0
    assert info["period"] == pytest.approx(PERIOD * 0.98, rel=0.002)


def test_normalise_resolution_keeps_a_page_eight_percent_off_scale():
    """The margin condition: a sheet printed on paper of another size.

    Its grid is the 200 dpi one times 0.92, so the stage could pull it back by 1.087,
    and the scans say it should not: resampled the record reads 18.5 dB, left alone
    19.4 dB, because the interpolation blurs every pixel while the model hardly minds
    the scale. Eight percent is written out here, not derived from the band, so that
    the case that set the band cannot quietly fall out of it.
    """
    page = _tensor(_page(period=PERIOD * 0.92))
    same, info = digitize.normalise_resolution(page)
    # Not a copy of the page, not a resampling by 1.0: the very object it was given.
    assert same is page
    assert info["scale"] == 1.0
    assert info["period"] == pytest.approx(PERIOD * 0.92, rel=0.002)
    assert info["reason"] == ""


def test_normalise_resolution_resamples_a_page_twenty_percent_off_scale():
    # The other end of the measurement: a page at 0.80 of the 200 dpi size reads
    # 18.8 dB resampled against 17.7 dB kept, and keeping it drops four of its leads
    # below 10 dB, so this one is worth the blur.
    page = _tensor(_page(period=PERIOD * 0.8))
    resampled, info = digitize.normalise_resolution(page)
    assert resampled is not page
    assert info["scale"] == pytest.approx(1.25, rel=0.005)
    assert digitize.measure_grid_period(resampled)[0] == pytest.approx(PERIOD, rel=0.003)


@pytest.mark.parametrize("side", [-0.9, 0.9])
def test_normalise_resolution_keeps_a_page_just_inside_the_dead_band(side):
    # In units of the band itself, so that the boundary stays tested if it moves: a
    # page whose grid asks for a scale nine tenths of the way out still keeps its
    # pixels, and keeping them means the very object comes back.
    scale = 1 + side * digitize.RESOLUTION_DEAD_BAND
    page = _tensor(_page(period=PERIOD / scale))
    same, info = digitize.normalise_resolution(page)
    assert same is page
    assert info["scale"] == 1.0
    assert info["period"] == pytest.approx(PERIOD / scale, rel=0.002)


@pytest.mark.parametrize("side", [-1.1, 1.1])
def test_normalise_resolution_resamples_a_page_just_outside_the_dead_band(side):
    # A tenth of the band further out, and the page is brought to the 200 dpi scale.
    scale = 1 + side * digitize.RESOLUTION_DEAD_BAND
    page = _tensor(_page(period=PERIOD / scale))
    resampled, info = digitize.normalise_resolution(page)
    assert resampled is not page
    assert info["scale"] == pytest.approx(scale, rel=0.005)
    assert digitize.measure_grid_period(resampled)[0] == pytest.approx(PERIOD, rel=0.003)


@pytest.mark.parametrize("size", [0.5, 1.5])
def test_normalise_resolution_brings_a_page_to_the_200_dpi_scale(size):
    page = _tensor(_page(period=PERIOD * size, size=size))
    resampled, info = digitize.normalise_resolution(page)
    assert info["scale"] == pytest.approx(1 / size, rel=0.005)
    assert resampled.shape == (3, HEIGHT, WIDTH)
    assert resampled.dtype == torch.uint8
    # The point of the whole stage: the page now has the period of a 200 dpi one.
    assert digitize.measure_grid_period(resampled)[0] == pytest.approx(
        PERIOD, rel=0.003
    )


@pytest.mark.parametrize("size,flag", [(0.5, cv2.INTER_CUBIC), (1.5, cv2.INTER_AREA)])
def test_normalise_resolution_averages_only_when_it_shrinks(monkeypatch, size, flag):
    """Thin lines fall between the samples of a page shrunk by picking pixels."""
    flags = []
    resize = cv2.resize

    def spy(array, dsize, interpolation):
        flags.append(interpolation)
        return resize(array, dsize, interpolation=interpolation)

    monkeypatch.setattr(digitize.cv2, "resize", spy)
    digitize.normalise_resolution(_tensor(_page(period=PERIOD * size, size=size)))
    assert flags == [flag]


def test_normalise_resolution_refuses_an_absurd_scale(monkeypatch):
    absurd = PERIOD / 5
    monkeypatch.setattr(
        digitize,
        "measure_grid_period",
        lambda image: (
            absurd,
            {"period": absurd, "contrast": 99.0, "harmonic": 5, "reason": ""},
        ),
    )
    page = _tensor(_page())
    same, info = digitize.normalise_resolution(page)
    assert same is page
    assert info["scale"] == 1.0
    assert info["reason"] == "grid lines ask for a scale of 5.00"


def test_normalise_resolution_keeps_a_page_without_grid_lines():
    page = _tensor(_no_grid_page())
    same, info = digitize.normalise_resolution(page)
    assert same is page
    assert info["scale"] == 1.0
    assert np.isnan(info["period"])
    assert "no grid lines found" in info["reason"]


# ------------------------------------------------------------------ mask files and QC
def test_save_mask_files_stores_the_scale_only_when_the_page_was_resampled(tmp_path):
    mask = torch.zeros((1, 6, 8), dtype=torch.uint8)
    digitize.save_mask_files(mask, "kept", str(tmp_path), 0.0, None, 1.0)
    digitize.save_mask_files(mask, "off", str(tmp_path), 0.0, None, float("nan"))
    digitize.save_mask_files(mask, "resampled", str(tmp_path), 0.0, None, 2.0)
    with open(tmp_path / "kept_mask.json") as f:
        assert "scale" not in json.load(f)
    with open(tmp_path / "off_mask.json") as f:
        assert "scale" not in json.load(f)
    with open(tmp_path / "resampled_mask.json") as f:
        assert json.load(f)["scale"] == 2.0


def test_parser_resolution_defaults_to_lines():
    parser = digitize.get_parser()
    args = parser.parse_args(["-d", "data", "-o", "out"])
    assert args.resolution == "lines"
    args = parser.parse_args(["-d", "data", "-o", "out", "--resolution", "keep"])
    assert args.resolution == "keep"


def test_append_qc_row_writes_the_resolution_columns(tmp_path):
    qc = {"grid_period_px": 3.937, "resolution_scale": 2.0}
    digitize.append_qc_row(str(tmp_path), "rec", "column", qc, 0.0)
    with open(tmp_path / "qc.csv", newline="") as f:
        reader = csv.DictReader(f)
        row = next(reader)
        # The new columns are appended, the old ones keep their order.
        assert reader.fieldnames[:2] == ["record", "placement"]
        assert reader.fieldnames[-5:-3] == ["grid_period_px", "resolution_scale"]
    assert float(row["grid_period_px"]) == pytest.approx(3.937)
    assert float(row["resolution_scale"]) == pytest.approx(2.0)


# --------------------------------------------------------------------- end to end
def _label_mask(height=HEIGHT, width=WIDTH):
    """A flat trace per lead, in the column of the standard layout."""
    label = np.zeros((height, width), dtype=np.uint8)
    pitch = width / 5
    for lead, value in digitize.LEAD_LABEL_MAPPING.items():
        column = int(digitize.STANDARD_LEAD_OFFSETS_SEC[lead] / 2.5)
        row = int(height * (0.18 + 0.05 * (value % 12)))
        start = int(pitch / 2 + column * pitch)
        label[row, start : start + int(pitch)] = value
    return label


def _run_half_size_page(tmp_path, mask, *flags):
    """run() over one 100 dpi page with this mask, no model and no data files."""
    data_folder, mask_folder = tmp_path / "data", tmp_path / "masks"
    data_folder.mkdir()
    mask_folder.mkdir()
    page = _tensor(_page(period=PERIOD / 2, size=0.5))
    write_png(page, str(data_folder / "rec.png"))
    write_png(torch.from_numpy(mask)[None], str(mask_folder / "rec_mask.png"))
    digitize.run(
        digitize.get_parser().parse_args(
            [
                "-d", str(data_folder),
                "-o", str(tmp_path / "out"),
                "--mask_folder", str(mask_folder),
                "--time_mapping", "bbox",
                *flags,
            ]
        )
    )


def _qc_row(tmp_path):
    with open(tmp_path / "out" / "qc.csv", newline="") as f:
        return next(csv.DictReader(f))


def test_run_with_resolution_lines_brings_a_100_dpi_page_up(tmp_path, capsys):
    # The mask is predicted on the page the stage hands the model, so it is drawn at
    # the normalised size and not at the half size the file on disk has.
    _run_half_size_page(tmp_path, _label_mask(), "--resolution", "lines")

    out = capsys.readouterr().out
    assert "Resolution for record rec:" in out
    assert f"size {WIDTH} x {HEIGHT} px" in out
    row = _qc_row(tmp_path)
    assert float(row["grid_period_px"]) == pytest.approx(PERIOD / 2, rel=0.002)
    assert float(row["resolution_scale"]) == pytest.approx(2.0, rel=0.005)


def test_run_without_the_resolution_stage_leaves_its_qc_columns_empty(
    tmp_path, capsys
):
    _run_half_size_page(
        tmp_path, _label_mask(HEIGHT // 2, WIDTH // 2), "--resolution", "keep"
    )

    assert "Resolution for record rec:" not in capsys.readouterr().out
    row = _qc_row(tmp_path)
    assert np.isnan(float(row["grid_period_px"]))
    assert np.isnan(float(row["resolution_scale"]))


def test_run_refuses_a_mask_of_another_size(tmp_path):
    # The mask of the page as it came, which the resampled page has outgrown.
    with pytest.raises(ValueError, match="predicted for another --resolution"):
        _run_half_size_page(
            tmp_path, _label_mask(HEIGHT // 2, WIDTH // 2), "--resolution", "lines"
        )
