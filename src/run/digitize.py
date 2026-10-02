# Run the digitization of ECG images.
import argparse
import csv
import cv2
import json
import numpy as np
import matplotlib.pyplot as plt
# import pandas as pd
import os
import shutil
import subprocess
import sys
import tempfile
import warnings
from scipy.interpolate import CubicSpline
from scipy.signal import butter, sosfiltfilt
from tqdm import tqdm
import torch
import torch.nn.functional as F
from torchvision.io.image import read_image, write_png
from torchvision.transforms.functional import rotate
import wfdb

from config import (
    DATASET_NAME,
    IMAGE_TYPE,
    FREQUENCY,
    LONG_SIGNAL_LENGTH_SEC,
    SHORT_SIGNAL_LENGTH_SEC,
    Y_SHIFT_RATIO,
    SIGNAL_UNITS,
    LEAD_LABEL_MAPPING,
    FMT,
    ADC_GAIN,
    BASELINE,
)


# Parse arguments.
def get_parser():
    description = "Run the trained Challenge models."
    parser = argparse.ArgumentParser(description=description)
    parser.add_argument(
        "-d",
        "--data_folder",
        type=str,
        required=True,
        help="Folder containing the images to digitize.",
    )
    parser.add_argument(
        "-m",
        "--model_folder",
        type=str,
        required=False,
        default="models/M3/",
        help="Folder containing the nnUNet folder nnUNet_results.",
    )
    parser.add_argument(
        "-o",
        "--output_folder",
        type=str,
        required=True,
        help="Folder to save the digitized images.",
    )
    parser.add_argument(
        "-v",
        "--verbose",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Verbose output. Use --no-verbose to disable.",
    )
    parser.add_argument(
        "--device",
        type=str,
        choices=["auto", "cuda", "mps", "cpu"],
        default="auto",
        help="Device to run nnUNet inference on. 'auto' picks cuda, then mps, then cpu.",
    )
    parser.add_argument(
        "--enable_tta",
        action="store_true",
        default=False,
        help=(
            "Enable test time augmentation (mirroring). About 2x slower and without "
            "a measurable gain on the generated evaluation set, hence off by default."
        ),
    )
    parser.add_argument(
        "--disable_tta",
        action="store_true",
        default=False,
        help="Deprecated, test time augmentation is off unless --enable_tta is given.",
    )
    parser.add_argument(
        "--fold",
        type=str,
        default="all",
        help="nnUNet fold(s) to use. Use --fold 0 for models/M1, which only ships fold_0.",
    )
    parser.add_argument(
        "--show_image",
        action="store_true",
        default=False,
        help="Show the image with the mask.",
    )
    parser.add_argument(
        "-f",
        "--allow_failures",
        action="store_true",
        default=False,
        help="Allow failures.",
    )
    parser.add_argument(
        "--lead_placement",
        type=str,
        choices=["column", "start"],
        default="column",
        help=(
            "column = each short lead in the time window of its column, NaN elsewhere; "
            "start = legacy, all short leads at sample 0 and NaN written as 0."
        ),
    )
    parser.add_argument(
        "--time_mapping",
        type=str,
        choices=["bbox", "grid"],
        default="grid",
        help=(
            "grid = sample all leads on one shared column grid (standard 3x4 layout "
            "with rhythm strip only, falls back to bbox otherwise); "
            "bbox = legacy, stretch the bounding box of every lead to its length."
        ),
    )
    parser.add_argument(
        "--grid_pitch",
        type=str,
        choices=["page", "fit"],
        default="page",
        help=(
            "Only for --time_mapping grid. page = column pitch from the page height "
            "(same page assumption as Y_SHIFT_RATIO), checked against the fitted one; "
            "fit = pitch from the least squares fit of the column edges."
        ),
    )
    parser.add_argument(
        "--grid_origin",
        type=str,
        choices=["masks", "lines"],
        default="lines",
        help=(
            "Only for --time_mapping grid. lines = origin and pitch of the column grid "
            "refined to sub pixel accuracy with the printed 1 mm grid lines of the image "
            "(assumes that the first column starts on a grid line), falls back to masks "
            "if the image shows no such lines; masks = from the mask edges only."
        ),
    )
    # GRID_LINE_SNAP_OFFSET is defined below, get_parser() only runs after the import.
    parser.add_argument(
        "--grid_line_offset",
        type=float,
        default=GRID_LINE_SNAP_OFFSET,
        help=(
            "Only for --grid_origin lines. Pixels the printed grid lines sit right of "
            "the traces; 0.5 = the matplotlib Agg snap of the generator images, "
            "0 for scans or photographs."
        ),
    )
    parser.add_argument(
        "--column_mapping",
        type=str,
        choices=["uniform", "lines"],
        default="lines",
        help=(
            "Only for --time_mapping grid with --grid_origin lines. lines (default) = "
            "read the time axis off the printed grid lines themselves: the 1 mm lines, "
            "each put on the right line by the bold 5 mm ones, give every sample the x "
            "of its own millimetre, so a page whose grid and traces are stretched or "
            "compressed in places (printing, paper feed, scanning) is read where its "
            "traces are; keeps the uniform grid, with a warning, on a page whose map "
            "cannot be trusted, for example one without bold grid lines, and without "
            "one on a page that the map moves by less than a pixel; uniform = every "
            "column is P pixels wide from g0 on, the time axis before lines became "
            "the default."
        ),
    )
    # COLUMN_MAPPING_MEDIAN_MM is defined below as well.
    parser.add_argument(
        "--column_mapping_median",
        type=float,
        default=COLUMN_MAPPING_MEDIAN_MM,
        help=(
            "Only for --column_mapping lines. Width in grid millimetres of the running "
            "median of the map's displacement from the uniform columns, which takes out "
            "the ripple of grid lines snapped to the pixels of a page drawn at 200 dpi "
            "(13.5 = one period of it) and keeps a step or a stretch of the lines; "
            "0 = the map as measured."
        ),
    )
    parser.add_argument(
        "--trace_estimator",
        type=str,
        choices=["mask", "ink"],
        default="ink",
        help=(
            "Only for --time_mapping grid. ink = the mean row is weighted by the ink "
            "of the page under the mask, which puts the row on the core of the stroke "
            "instead of in the middle of the mask, and keeps the mask mean for a lead "
            "whose stroke is too faint to weigh with; mask = the row of a column is "
            "the mean row of the binary mask."
        ),
    )
    parser.add_argument(
        "--trace_shift",
        type=str,
        choices=["off", "page"],
        default="page",
        help=(
            "Only for --time_mapping grid. page = measure, once per page, how far the "
            "ink of the steep strokes sits right of the lead masks and read every "
            "lead on a column axis moved by that, which takes the timing error out of "
            "a resampled page and leaves a page whose masks already sit on their own "
            "ink where it is; off = read every lead where its mask is."
        ),
    )
    parser.add_argument(
        "--sharpen",
        type=str,
        choices=["none", "bandlimited"],
        default="bandlimited",
        help=(
            "Only for --time_mapping grid. bandlimited = restore the 20-40 Hz band "
            "that the column aperture and the linear interpolation attenuate: "
            "cur + LP35(R2a8 - cur), where R2a8 is the aperture-corrected (a = 1/8) "
            "cubic-spline reconstruction; only the band below 35 Hz is changed, so "
            "steep QRS spikes cannot ring; none = the column profile is linearly "
            "interpolated, the behaviour before the sharpening."
        ),
    )
    parser.add_argument(
        "--baseline",
        type=str,
        choices=["page", "leads"],
        default="leads",
        help=(
            "page = baseline from the page geometry, scaled with the column pitch; "
            "leads = additionally shifted so that the Einthoven and Goldberger sums "
            "have no DC offset (needs --time_mapping grid)."
        ),
    )
    parser.add_argument(
        "--rotation",
        type=str,
        choices=["hough", "lines"],
        default="lines",
        help=(
            "lines = a 0.1 degree Hough transform refined with the phase drift of the "
            "printed 1 mm grid lines, which resolves fractions of a degree, falls back "
            "to the whole degree angle if the image shows no such lines; hough = angle "
            "from the Hough transform of the page, in whole degrees only."
        ),
    )
    parser.add_argument(
        "--interpolation",
        type=str,
        choices=["nearest", "bicubic"],
        default="nearest",
        help=(
            "How the page is resampled when it is turned by the rotation angle. "
            "nearest = rotate() as before, which keeps every pixel as sharp as it was; "
            "bicubic = one cv2 warp that keeps thin traces and grid lines in one piece "
            "instead of breaking them into steps, at the price of a little blur. Not a "
            "clear gain: +0.5 to +0.8 dB on synthetic pages whose exact derotation lands "
            "back on their own pixel lattice, -0.3 dB on the rotated generator set, "
            "where it cannot. A page that --perspective lines rectifies is resampled "
            "bicubically either way."
        ),
    )
    parser.add_argument(
        "--perspective",
        type=str,
        choices=["off", "lines"],
        default="lines",
        help=(
            "lines = after the rotation, take out the shear and the perspective that "
            "the printed 1 mm grid lines show, in the same interpolation as the "
            "rotation, leaves the page as it is if the image shows no such lines or is "
            "straight enough already; off = keep the rotated page."
        ),
    )
    parser.add_argument(
        "--resolution",
        type=str,
        choices=["keep", "lines"],
        default="lines",
        help=(
            "lines = resample the page so that the printed 1 mm grid lines have the "
            "period of a 200 dpi page, which is the scale the model was trained on, "
            "keeps the page if the image shows no such lines; a page within about a "
            "tenth of that scale keeps its pixels as well, because resampling it "
            "costs more than the model loses to the scale. Everything after it, "
            "the rotation and the perspective included, sees the resampled "
            "page, and --grid_line_offset is a page unit, so its default stays right "
            "for generator pages of any resolution. keep = take the page as it is."
        ),
    )
    parser.add_argument(
        "--save_mask",
        action="store_true",
        default=False,
        help="Save the predicted label mask (corrected frame) as PNG plus JSON.",
    )
    parser.add_argument(
        "--mask_folder",
        type=str,
        default=None,
        help="Load masks from this folder instead of running nnUNet.",
    )
    return parser


# Offsets in seconds of the standard 3x4 layout, used as a fallback.
STANDARD_LEAD_OFFSETS_SEC = {
    "I": 0.0,
    "II": 0.0,
    "III": 0.0,
    "aVR": 2.5,
    "aVL": 2.5,
    "aVF": 2.5,
    "V1": 5.0,
    "V2": 5.0,
    "V3": 5.0,
    "V4": 7.5,
    "V5": 7.5,
    "V6": 7.5,
}


def get_rotation_angle(np_image):
    """Get the rotation angle of the image."""
    lines = get_lines(np_image, threshold_HoughLines=1200)
    filtered_lines = filter_lines(
        lines, degree_window=30, parallelism_count=3, parallelism_window=2
    )
    if filtered_lines is None:
        rot_angle = np.nan
    else:
        rot_angle = get_median_degrees(filtered_lines)
    return rot_angle


# Theta step of the fine Hough transform, in degrees. A tenth of a degree is far
# below the half degree a page has to be straightened to for the grid line phase.
ROTATION_HOUGH_THETA_STEP_DEG = 0.1
# Vote threshold of the fine Hough transform, relative to the image width. The 1200
# votes of get_rotation_angle are 0.55 of the width of a 2200 px generator page.
ROTATION_HOUGH_THRESHOLD_RATIO = 1200 / 2200
# Factor and floor of the stepwise lowering of the threshold. A page with faint or
# short lines needs fewer votes, but a floor keeps noise from passing as a line.
ROTATION_HOUGH_THRESHOLD_STEP = 0.8
ROTATION_HOUGH_THRESHOLD_FLOOR = 0.25
# Lines further than this from the horizontal are dropped, as in get_rotation_angle.
# The accumulator gets one degree more so that its border never reaches the window.
ROTATION_HOUGH_DEGREE_WINDOW = 30


def get_rotation_angle_fine(np_image):
    """Get the rotation angle of the image with a tenth of a degree resolution.

    Same lines as get_rotation_angle, but the theta step is ten times finer and the
    vote threshold scales with the image width instead of being the absolute 1200 of
    a 2200 px page. A fine theta step splits a thick line into many near duplicate
    hits, which the median over the filtered lines takes care of.
    """
    threshold = ROTATION_HOUGH_THRESHOLD_RATIO * np_image.shape[1]
    floor = ROTATION_HOUGH_THRESHOLD_FLOOR * np_image.shape[1]
    while threshold >= floor:
        lines = get_lines(
            np_image,
            threshold_HoughLines=int(round(threshold)),
            theta_resolution=np.deg2rad(ROTATION_HOUGH_THETA_STEP_DEG),
            theta_window=np.deg2rad(ROTATION_HOUGH_DEGREE_WINDOW + 1),
        )
        filtered_lines = filter_lines(
            lines,
            degree_window=ROTATION_HOUGH_DEGREE_WINDOW,
            parallelism_count=3,
            parallelism_window=2,
        )
        if filtered_lines is not None:
            return get_median_degrees(filtered_lines)
        threshold *= ROTATION_HOUGH_THRESHOLD_STEP
    return np.nan


def get_median_degrees(lines):
    """Get the median angle of the lines."""
    lines = lines[:, 0, :]
    line_angles = [-(90 - line[1] * 180 / np.pi) for line in lines]
    return round(np.median(line_angles), 4)


def is_within_x_degrees_of_horizontal(theta, degree_window):
    """Check if the line is within x degrees of horizontal (90 degrees)."""
    theta_degrees = theta * 180 / np.pi
    deviation_from_horizontal = abs(90 - theta_degrees)
    return deviation_from_horizontal < degree_window


def get_lines(
    np_image,
    threshold_HoughLines=1380,
    rho_resolution=1,
    theta_resolution=np.pi / 180,
    theta_window=None,
):
    """Get the lines in the image."""
    # Convert the image to a grayscale NumPy array
    image = cv2.cvtColor(np_image, cv2.COLOR_RGB2BGR)
    gray_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    # Apply the Canny edge detector to find edges in the image
    edges = cv2.Canny(gray_image, 50, 150, apertureSize=3)

    # Use HoughLines to find lines in the edge-detected image
    if theta_window is None:
        lines = cv2.HoughLines(
            edges, rho_resolution, theta_resolution, threshold_HoughLines, None, 0, 0
        )
    else:
        # Accumulating only around the horizontal is what makes a fine theta step
        # affordable, and the lines it leaves out are dropped by filter_lines anyway.
        lines = cv2.HoughLines(
            edges,
            rho_resolution,
            theta_resolution,
            threshold_HoughLines,
            None,
            0,
            0,
            np.pi / 2 - theta_window,
            np.pi / 2 + theta_window,
        )

    return lines


def filter_lines(lines, degree_window=20, parallelism_count=0, parallelism_window=2):
    """Filter the lines to get the rotation angle."""
    parallelism_radian = np.deg2rad(parallelism_window)
    filtered_lines = []
    line_angles = []

    # Filter lines to be within the degree window of horizontal
    if lines is not None:
        for line in lines:
            for rho, theta in line:
                if is_within_x_degrees_of_horizontal(theta, degree_window):
                    filtered_lines.append((rho, theta))
                    line_angles.append(theta)

    # Further filter lines based on parallelism
    parallel_lines = []
    if len(filtered_lines) > 0:
        # Counting the angles inside the window of every line pair by pair is
        # quadratic, and a fine theta step leaves tens of thousands of lines. Over the
        # sorted angles the count is two lookups per line: "right" at the lower edge
        # and "left" at the upper one leave out an angle sitting exactly on an edge, as
        # the strict comparisons did, and a line still counts itself. Carrying the
        # window to the angles in float64 decides every pair as the float32 difference
        # did: angles this close subtract exactly, and an edge that float64 places 1e-16
        # off still falls between two float32 angles, which lie 1e-7 apart.
        angles = np.array(line_angles, dtype=np.float64)
        ordered = np.sort(angles)
        counts = np.searchsorted(ordered, angles + parallelism_radian, side="left")
        counts -= np.searchsorted(ordered, angles - parallelism_radian, side="right")
        # The second branch of the pair count, one-sided as it was: the angles a whole
        # pi below this one, never the ones a whole pi above it.
        turned = angles - np.pi
        counts += np.searchsorted(ordered, turned + parallelism_radian, side="left")
        counts -= np.searchsorted(ordered, turned - parallelism_radian, side="right")
        for (rho, theta), count in zip(filtered_lines, counts):
            if count >= parallelism_count:
                parallel_lines.append((rho, theta))

    if len(parallel_lines) == 0:
        parallel_lines = None
    else:
        parallel_lines = np.array(parallel_lines)[:, np.newaxis, :]

    return parallel_lines


def resolve_device(device="auto"):
    """Resolve the device to run inference on."""
    if device != "auto":
        return device
    if torch.cuda.is_available():
        return "cuda"
    if torch.backends.mps.is_available():
        return "mps"
    return "cpu"


def predict_mask_nnunet(
    image, dataset_name, model_folder, device="auto", disable_tta=False, fold="all"
):
    """Predict the mask using nnUNet."""

    device_to_use = resolve_device(device)
    print(
        f"Running nnUNet on device {device_to_use} with TTA "
        f"{'off' if disable_tta else 'on'} and fold {fold}."
    )

    # Set env variabels (nnUNet needs them to be set)
    os.environ["nnUNet_results"] = os.path.join(model_folder, "nnUNet_results")

    # Define temporary folders and paths
    temp_folder_base = tempfile.mkdtemp(prefix="nnUNet_")
    try:
        temp_folder_input = os.path.join(temp_folder_base, "input")
        temp_folder_output = os.path.join(temp_folder_base, "output")
        image_path_temp = os.path.join(temp_folder_input, "00000_temp_0000.png")
        mask_path_temp = os.path.join(temp_folder_output, "00000_temp.png")

        # Create temp folders and copy image
        os.makedirs(temp_folder_input, exist_ok=True)
        os.makedirs(temp_folder_output, exist_ok=True)
        write_png(image, image_path_temp)

        # nnUNet also wants these to be set, but does not use them for inference.
        # Only set them for the subprocess and only if the user did not set them.
        env = os.environ.copy()
        env.setdefault("nnUNet_raw", temp_folder_base)
        env.setdefault("nnUNet_preprocessed", temp_folder_base)

        # Run inference
        command_run = [
            "nnUNetv2_predict",
            "-d",
            str(dataset_name),
            "-i",
            temp_folder_input,
            "-o",
            temp_folder_output,
            "-f",
            str(fold),
            "-tr",
            "nnUNetTrainer",
            "-c",
            "2d",
            "-p",
            "nnUNetPlans",
            "-device",
            device_to_use,
        ]
        if disable_tta:
            command_run.append("--disable_tta")
        try:
            subprocess.run(command_run, check=True, env=env)
        except subprocess.CalledProcessError as e:
            raise RuntimeError(
                f"nnUNet inference failed with return code {e.returncode}: "
                f"{' '.join(command_run)}"
            )

        # Get masks
        if not os.path.exists(mask_path_temp):
            raise RuntimeError(
                f"nnUNet did not produce the expected mask file {mask_path_temp}."
            )
        mask = read_image(mask_path_temp)
    finally:
        # Delete all temporary folders and files
        shutil.rmtree(temp_folder_base, ignore_errors=True)

    return mask


def cut_to_mask(img, mask, return_y1=False):
    """Cut the image to the mask."""
    coords = torch.where(mask[0] >= 1)
    y_min, y_max = coords[0].min().item(), coords[0].max().item()
    x_min, x_max = coords[1].min().item(), coords[1].max().item()
    img = img[:, y_min : y_max + 1, x_min : x_max + 1]
    if return_y1:
        return img, y_min, x_min
    else:
        return img


def cut_binary(mask_to_use, image_rotated):
    """Cut the binary mask into single binary masks."""
    signal_masks = {}
    signal_images = {}
    signal_positions = {}
    # mask_values = list(pd.Series(mask_to_use.numpy().flatten()).value_counts().index)
    possible_lead_names = LEAD_LABEL_MAPPING
    lead_names_in_mask = {
        k: v
        for k, v in possible_lead_names.items()  # if v in mask_values
    }
    for lead_name, lead_value in lead_names_in_mask.items():
        binary_mask = torch.where(mask_to_use == lead_value, 1, 0)
        if binary_mask.sum() > 0:
            signal_img, y1, x1 = cut_to_mask(image_rotated, binary_mask, True)
            signal_mask = cut_to_mask(binary_mask, binary_mask)
            signal_images[lead_name] = signal_img
            signal_masks[lead_name] = signal_mask
            signal_positions[lead_name] = {"y1": y1, "x1": x1}
        else:
            signal_images[lead_name] = None
            signal_masks[lead_name] = None
            signal_positions[lead_name] = None

    return signal_masks, signal_positions, signal_images


def vectorise(
    image_rotated, mask, signal_cropped, sec_per_pixel, mV_per_pixel, y_shift_ratio, lead
):
    """Vectorise the image."""

    # Get scaling info
    total_seconds_from_mask = round(torch.tensor(sec_per_pixel).item() * mask.shape[2], 1)
    if total_seconds_from_mask > (LONG_SIGNAL_LENGTH_SEC / 2):
        total_seconds = LONG_SIGNAL_LENGTH_SEC
        y_shift_ratio_ = y_shift_ratio["full"]
    else:
        total_seconds = SHORT_SIGNAL_LENGTH_SEC
        y_shift_ratio_ = y_shift_ratio[lead]
    values_needed = int(total_seconds * FREQUENCY)

    # Scale y
    # The code aligns and scales a signal based on a mask's non-zero regions and a vertical shift ratio. It computes the mean vertical position of non-zero elements in the mask, adjusts the signal's vertical position using y_shift_ratio_, and scales the result into physical units (e.g., millivolts) for further analysis.
    non_zero_mean = torch.tensor(
        [
            torch.mean(torch.nonzero(mask[0, :, i]).type(torch.float32))
            for i in range(mask.shape[2])
        ]
    )
    signal_cropped_shifted = (1 - y_shift_ratio_) * image_rotated.shape[
        1
    ] - signal_cropped
    predicted_signal = (signal_cropped_shifted - non_zero_mean) * mV_per_pixel

    # Scale x
    # The code reshapes the predicted signal into a 3D tensor for interpolation and resamples it to a specified size using linear interpolation. It then flattens the resampled data back into a 1D tensor for further use.
    n = predicted_signal.shape[0]
    data_reshaped = predicted_signal.view(1, 1, n)
    resampled_data = F.interpolate(
        data_reshaped, size=values_needed, mode="linear", align_corners=False
    )
    predicted_signal_sampled = resampled_data.view(-1)

    return predicted_signal_sampled


# Width of one 2.5 s column relative to the page height (62.5 mm of 215.9 mm).
PAGE_PITCH_RATIO = 25 * SHORT_SIGNAL_LENGTH_SEC / 10 / 21.59
# Maximal relative deviation between the page pitch and the fitted pitch (about
# 0.75 px). A slightly cropped or rescaled page is off by more, 0.5 % let those through.
PAGE_PITCH_TOLERANCE = 0.0015
# Gaps in a lead above this width, relative to the pitch, are reported (about 25 ms).
GRID_GAP_TOLERANCE = 0.01
# Maximal residual of a column edge to the fitted grid, relative to the pitch.
GRID_RESIDUAL_TOLERANCE = 0.02
# Maximal disagreement of the Einthoven and the Goldberger baseline estimate,
# relative to the pitch (about 2 px). Above it neither estimate is trusted.
BASELINE_DISAGREEMENT_TOLERANCE = 0.004
NUM_COLUMNS = int(LONG_SIGNAL_LENGTH_SEC / SHORT_SIGNAL_LENGTH_SEC)


def _grid_inliers(lead_edges):
    """Flag the lead edges within GRID_RESIDUAL_TOLERANCE of a Theil-Sen grid."""
    boundaries = np.array([edge[0] for edge in lead_edges], float)
    edges = np.array([edge[1] for edge in lead_edges], float)
    i, j = np.triu_indices(len(edges), 1)
    distinct = boundaries[i] != boundaries[j]
    P = np.median((edges[j] - edges[i])[distinct] / (boundaries[j] - boundaries[i])[distinct])
    residuals = edges - boundaries * P
    return np.abs(residuals - np.median(residuals)) <= GRID_RESIDUAL_TOLERANCE * P


def fit_column_grid(
    signal_masks, signal_positions, image_height, pitch="page", record=""
):
    """Fit the shared column grid of the standard 3x4 layout with rhythm strip.

    Returns (g0, P, long_leads, reason). g0 is the x position of the left edge of
    the first column, P the width of one 2.5 s column in pixels. If the layout is
    not the expected one, g0 and P are None and reason says why.
    """
    widths = {
        lead: mask.shape[2] for lead, mask in signal_masks.items() if mask is not None
    }
    if not widths:
        return None, None, [], "no leads found"
    median_width = np.median(list(widths.values()))
    long_leads = [lead for lead, width in widths.items() if width >= 2 * median_width]
    if not long_leads:
        return None, None, [], "no rhythm strip found"
    # Two merged short leads must not pass as a rhythm strip.
    if any(widths[lead] < (NUM_COLUMNS - 0.5) * median_width for lead in long_leads):
        return None, None, long_leads, "rhythm strip does not span all columns"

    # Every lead gives the start of its first and the end of its last column. The
    # rhythm strip gives the start of the first and the end of the last column.
    lead_edges = []
    for lead, width in widths.items():
        x1 = signal_positions[lead]["x1"]
        if lead in long_leads:
            first_column, last_column = 0, NUM_COLUMNS - 1
        elif lead in STANDARD_LEAD_OFFSETS_SEC:
            first_column = int(STANDARD_LEAD_OFFSETS_SEC[lead] / SHORT_SIGNAL_LENGTH_SEC)
            last_column = first_column
        else:
            return None, None, long_leads, f"unknown lead {lead}"
        lead_edges.append((first_column, x1, True))
        lead_edges.append((last_column + 1, x1 + width, False))
    short_columns = {
        int(STANDARD_LEAD_OFFSETS_SEC[lead] / SHORT_SIGNAL_LENGTH_SEC)
        for lead in widths
        if lead not in long_leads
    }
    if short_columns != set(range(NUM_COLUMNS)):
        return None, None, long_leads, "not every column has a short lead"

    # A mask that runs on past its column (into the margin or the next lead) must
    # not set the column edge, so edges far off the consensus grid are left out.
    inliers = _grid_inliers(lead_edges)
    # Column starts = min x1 per column, column ends = max x-end per column.
    starts, ends = {}, {}
    for (boundary, edge, is_start), inlier in zip(lead_edges, inliers):
        if not inlier:
            continue
        if is_start:
            starts[boundary] = min(starts.get(boundary, np.inf), edge)
        else:
            ends[boundary] = max(ends.get(boundary, -np.inf), edge)
    if set(starts) != set(range(NUM_COLUMNS)) or set(ends) != set(
        range(1, NUM_COLUMNS + 1)
    ):
        return None, None, long_leads, "a column has all its edges off the grid"

    # Pixel c covers [c, c+1), so starts and ends are both column boundaries.
    boundaries = np.array(list(starts.keys()) + list(ends.keys()), float)
    edges = np.array(list(starts.values()) + list(ends.values()), float)
    P_fit, g0 = np.polyfit(boundaries, edges, 1)
    P = P_fit
    if pitch == "page":
        P_page = image_height * PAGE_PITCH_RATIO
        if abs(P_fit - P_page) <= PAGE_PITCH_TOLERANCE * P_page:
            # With a fixed pitch no single edge pixel moves the origin by more than 1/n.
            P = P_page
            g0 = np.mean(edges - boundaries * P)
        else:
            print(
                f"Page pitch {P_page:.2f} px does not match the fitted pitch "
                f"{P_fit:.2f} px for record {record}, using the fitted one."
            )
    residual = np.max(np.abs(edges - (g0 + boundaries * P)))
    if residual > GRID_RESIDUAL_TOLERANCE * P:
        return None, None, long_leads, f"column edges are {residual:.1f} px off the grid"

    return float(g0), float(P), long_leads, ""


# Printed 1 mm grid lines in one 2.5 s column (25 mm/s).
GRID_LINES_PER_COLUMN = 25 * SHORT_SIGNAL_LENGTH_SEC
# Height of the bands whose median darkness gives the grid line profile. The median
# over a band drops the traces and the text, short bands survive a residual skew.
GRID_LINE_BAND_HEIGHT = 100
# Search range of the grid line pitch around the pitch of the masks, relative.
GRID_LINE_PITCH_RANGE = 0.005
# Minimal amplitude of the grid line comb relative to the neighbouring periods.
GRID_LINE_MIN_CONTRAST = 5.0
# Maximal move of the origin by the grid lines, relative to the pitch (about 2 px).
# The mask origin is good to about 1 px, more means the columns are not on the lines.
GRID_LINE_SHIFT_TOLERANCE = 0.004
# The generator draws with matplotlib, whose Agg backend snaps an axis parallel line
# of odd pixel width to round(x) + 0.5, on average 0.5 px right of where the unsnapped
# traces put it. A scanned page has no such offset.
GRID_LINE_SNAP_OFFSET = 0.5


def _darkness(image):
    """Darkness of a CHW image, 0 = white and 255 = black, as the grid lines show up."""
    array = image.numpy() if torch.is_tensor(image) else image
    return 255.0 - array.min(axis=0).astype(float)


def _ink(image):
    """Ink of a CHW image, 0 = white and 255 = black, as only the traces show up.

    _darkness takes the darkest channel, so that a coloured grid line counts as dark
    as a trace, which is what the grid measurements want. Here the opposite is
    wanted: the lightest channel leaves a red grid line, which is bright in its own
    channel, as white as the paper and keeps only the black trace. On a grey page
    the two are the same.
    """
    array = image.numpy() if torch.is_tensor(image) else image
    return 255.0 - array.max(axis=0).astype(float)


def _band_profiles(darkness, band_height=GRID_LINE_BAND_HEIGHT):
    """Median darkness profile of every band of rows, plus the centre row of each.

    The median over a band drops the traces and the text, so only the printed grid
    lines are left. Returns profiles of shape [bands, width] and the band centres.
    """
    starts = np.arange(0, darkness.shape[0] - band_height + 1, band_height)
    profiles = np.array(
        [np.median(darkness[start : start + band_height], axis=0) for start in starts]
    )
    return profiles, starts + band_height / 2


def _grid_line_comb(profiles, period):
    """Complex amplitude of the comb with this period in every band profile."""
    width = profiles.shape[1]
    # Pixel c covers [c, c+1). The Hann window leaves no side lobes for a pitch scan.
    x = np.arange(width) + 0.5
    centred = profiles - profiles.mean(axis=1, keepdims=True)
    return (centred * np.hanning(width)) @ np.exp(-2j * np.pi * x / period)


def refine_grid_from_lines(image_rotated, g0, P, snap_offset=GRID_LINE_SNAP_OFFSET):
    """Refine the column grid with the printed 1 mm grid lines of the image.

    The vertical grid lines are a comb over the whole page width, whose period gives
    the pitch and whose phase gives the origin far below one pixel, while the mask
    edges are whole pixels. The origin is the grid line next to the mask origin.
    snap_offset is how far in pixels the drawn lines sit right of the traces, 0.5 for
    the matplotlib generator and 0 for a scanned page.
    Returns (g0, P, info). Without usable grid lines g0 and P come back unchanged and
    info["reason"] says why.
    """
    profiles, _ = _band_profiles(_darkness(image_rotated))
    info = {"contrast": float("nan"), "shift": float("nan"), "reason": ""}
    if profiles.shape[0] == 0:
        info["reason"] = "image too small for the grid line profile"
        return g0, P, info

    def amplitude(period):
        return np.abs(_grid_line_comb(profiles, period).sum())

    # Coarse scan over the pitch range, then a fine one around the peak. The peak of
    # the Hann window is 0.4 % of the period wide on a 2200 px page.
    period = P / GRID_LINES_PER_COLUMN
    for half_range, steps in ((GRID_LINE_PITCH_RANGE, 51), (2e-4, 41)):
        candidates = period * (1 + np.linspace(-half_range, half_range, steps))
        amplitudes = np.array([amplitude(candidate) for candidate in candidates])
        best = int(np.argmax(amplitudes))
        period = candidates[best]
    if 0 < best < steps - 1:
        left, peak, right = amplitudes[best - 1 : best + 2]
        curvature = left - 2 * peak + right
        if curvature < 0:
            period += 0.5 * (left - right) / curvature * (candidates[1] - candidates[0])

    background = np.mean([amplitude(period * f) for f in (0.93, 0.96, 1.04, 1.07)])
    info["contrast"] = float(amplitude(period) / max(background, 1e-9))
    if info["contrast"] < GRID_LINE_MIN_CONTRAST:
        info["reason"] = f"no grid lines found (contrast {info['contrast']:.1f})"
        return g0, P, info

    P_lines = period * GRID_LINES_PER_COLUMN
    phase = -np.angle(_grid_line_comb(profiles, period).sum()) / (2 * np.pi) * period
    phase -= snap_offset
    # A new pitch turns the mask grid about its centre, not about its origin.
    g0_masks = g0 + NUM_COLUMNS / 2 * (P - P_lines)
    g0_lines = phase + np.round((g0_masks - phase) / period) * period
    info["shift"] = float(g0_lines - g0_masks)
    if abs(info["shift"]) > GRID_LINE_SHIFT_TOLERANCE * P:
        info["reason"] = (
            f"next grid line is {info['shift']:+.1f} px off the mask origin"
        )
        return g0, P, info
    return float(g0_lines), float(P_lines), info


# Period range of the printed 1 mm grid lines in pixels, 1 mm at about 90 to 430 dpi.
# The pitch of the masks is not known yet, so the period is searched in the image.
GRID_LINE_PERIOD_RANGE = (3.5, 17.0)
# Zero padding of the profile spectrum, which sets the resolution of the period.
GRID_LINE_PERIOD_PADDING = 8
# Bands whose comb amplitude is below this fraction of the median are left out of the
# slope fit: the borders and the black corners of a rotated page carry no grid lines.
GRID_LINE_MIN_BAND_AMPLITUDE = 0.3
# Bands further than this many robust sigmas off the fitted line are dropped once.
GRID_LINE_OUTLIER_SIGMA = 3.0
# Floor of that sigma in pixels, so that a near perfect fit keeps all its bands.
GRID_LINE_SIGMA_FLOOR = 0.05
# Fewest bands the slope fit needs, and the largest residual it accepts relative to
# the period. Above it the phase unwrapping is unreliable and the slope meaningless.
GRID_LINE_MIN_SLOPE_BANDS = 5
GRID_LINE_MAX_SLOPE_RESIDUAL = 0.1


def _grid_line_period(profiles):
    """Period in pixels of the strongest comb in the band profiles, NaN if none.

    The pitch of the masks is not known when the page is straightened, so the period
    comes from the profiles themselves. Rotation moves the comb phase from band to
    band, so the power is summed over the bands, not the complex spectra. Picking a
    harmonic of the 1 mm comb does not change the phase slope, only its unit.
    """
    width = profiles.shape[1]
    centred = profiles - profiles.mean(axis=1, keepdims=True)
    padded = GRID_LINE_PERIOD_PADDING * width
    spectrum = np.fft.rfft(centred * np.hanning(width), n=padded)
    power = (np.abs(spectrum) ** 2).sum(axis=0)
    frequencies = np.fft.rfftfreq(padded)
    low, high = GRID_LINE_PERIOD_RANGE
    inside = np.flatnonzero((frequencies >= 1 / high) & (frequencies <= 1 / low))
    if inside.size == 0 or not np.any(power[inside] > 0):
        return float("nan")
    best = int(inside[np.argmax(power[inside])])
    frequency = frequencies[best]
    if 0 < best < power.size - 1:
        left, peak, right = power[best - 1 : best + 2]
        curvature = left - 2 * peak + right
        if curvature < 0:
            frequency += (
                0.5 * (left - right) / curvature * (frequencies[1] - frequencies[0])
            )
    return float(1 / frequency)


def grid_line_slope(image, axis=0):
    """Rotation angle of the page from the phase drift of the printed grid lines.

    With the page rotated, the vertical grid lines cross the bands of rows at a
    slight angle, so the phase of the grid line comb drifts linearly with the band.
    The slope of that drift resolves far smaller angles than the Hough transform.
    axis=0 uses the vertical lines over bands of rows, axis=1 the horizontal lines
    over bands of columns; both return the angle that rotate() has to undo the tilt.
    Returns an info dict whose "angle" is NaN when "reason" says why it is unusable.
    """
    darkness = _darkness(image)
    if axis == 1:
        darkness = darkness.T
    profiles, centres = _band_profiles(darkness)
    info = {
        "angle": float("nan"),
        "period": float("nan"),
        "contrast": float("nan"),
        "residual": float("nan"),
        "bands": 0,
        "reason": "",
    }
    if profiles.shape[0] < GRID_LINE_MIN_SLOPE_BANDS:
        info["reason"] = "image too small for the grid line profile"
        return info

    period = _grid_line_period(profiles)
    if not np.isfinite(period):
        info["reason"] = "no grid line period found"
        return info
    info["period"] = period

    def amplitude(candidate):
        # Non-coherent sum: the bands of a rotated page have different phases.
        return np.abs(_grid_line_comb(profiles, candidate)).sum()

    background = np.mean([amplitude(period * f) for f in (0.93, 0.96, 1.04, 1.07)])
    comb = _grid_line_comb(profiles, period)
    info["contrast"] = float(np.abs(comb).sum() / max(background, 1e-9))
    if info["contrast"] < GRID_LINE_MIN_CONTRAST:
        info["reason"] = f"no grid lines found (contrast {info['contrast']:.1f})"
        return info

    # Position of the comb in every band, unwrapped so that it drifts continuously.
    # A band without grid lines has a random phase, so it must not take part in the
    # unwrapping, or every band after it ends up a whole period off.
    weights = np.abs(comb)
    keep = weights >= GRID_LINE_MIN_BAND_AMPLITUDE * np.median(weights)
    positions = np.zeros(len(comb))
    positions[keep] = -np.unwrap(np.angle(comb[keep])) / (2 * np.pi) * period
    for step in range(2):
        if np.count_nonzero(keep) < GRID_LINE_MIN_SLOPE_BANDS:
            info["reason"] = f"only {np.count_nonzero(keep)} bands with grid lines"
            return info
        # np.polyfit weighs the residuals, so sqrt() makes the weight an amplitude.
        slope, offset = np.polyfit(
            centres[keep], positions[keep], 1, w=np.sqrt(weights[keep])
        )
        residuals = positions - (offset + slope * centres)
        if step == 0:
            # Drop the bands off the line once, then refit on the rest.
            kept = residuals[keep]
            sigma = max(
                1.4826 * np.median(np.abs(kept - np.median(kept))),
                GRID_LINE_SIGMA_FLOOR,
            )
            keep = keep & (np.abs(residuals) <= GRID_LINE_OUTLIER_SIGMA * sigma)
    info["bands"] = int(np.count_nonzero(keep))
    info["residual"] = float(np.sqrt(np.mean(residuals[keep] ** 2)))
    if info["residual"] > GRID_LINE_MAX_SLOPE_RESIDUAL * period:
        info["reason"] = f"grid line phase is {info['residual']:.2f} px off a line"
        return info

    # Verified on rotated synthetic pages: a page that rotate(image, +a) straightens
    # has its vertical lines drifting left with the row and its horizontal ones down
    # with the column, so axis 0 and axis 1 need opposite signs.
    info["angle"] = float(np.degrees(np.arctan((-slope) if axis == 0 else slope)))
    return info


# Below this the refinement is within the noise, and an integer angle page keeps the
# Hough angle it was rotated by, so masks saved with that angle stay valid.
ROTATION_REFINE_MIN_DEG = 0.02
# A refinement above this rotates the page again and measures a second time, because
# the first measurement was made on a page that was still noticeably tilted.
ROTATION_REFINE_REPEAT_DEG = 0.2
# A refinement above this is not a residual tilt any more, so the coarse angle stands.
ROTATION_REFINE_MAX_DEG = 3.0
# Largest disagreement in degrees between a saved mask and the angle used now.
ROTATION_MASK_ANGLE_TOLERANCE = 0.005


def estimate_rotation(image, method="hough"):
    """Get the rotation angle of a page image, as read by read_image ([3, H, W]).

    method "hough" is the Hough transform in whole degrees, method "lines" refines a
    tenth of a degree Hough transform with the phase drift of the printed grid lines.
    Grid lines that cannot confirm that coarse angle get a second say on the page as
    it came, because a page turned by a spurious angle no longer shows the comb that
    would have vetoed it; with no grid lines either way the whole degree angle is
    returned, so that "lines" is never worse than "hough".
    Returns (rot_angle, info), rot_angle is NaN when no angle could be found.
    """
    np_image = image.permute(1, 2, 0).numpy().astype(np.uint8)
    info = {
        "coarse": float("nan"),
        "deltas": [],
        "period": float("nan"),
        "contrast": float("nan"),
        "residual": float("nan"),
        "reason": "",
    }
    if method != "lines":
        info["coarse"] = get_rotation_angle(np_image)
        return info["coarse"], info

    coarse = get_rotation_angle_fine(np_image)
    info["coarse"] = coarse

    def refine(start):
        """Straighten the page by start degrees and let the grid lines finish it.

        Returns the angle the page needs altogether, or None with a reason in info
        when the grid lines of the straightened page cannot be read.
        """
        total = start
        for _ in range(2):
            lines = grid_line_slope(rotate(image, total), axis=0)
            info["period"] = lines["period"]
            info["contrast"] = lines["contrast"]
            info["residual"] = lines["residual"]
            if lines["reason"]:
                info["reason"] = lines["reason"]
                return None
            delta = lines["angle"]
            if abs(delta) > ROTATION_REFINE_MAX_DEG:
                info["reason"] = f"grid lines ask for {delta:+.2f} degrees"
                return None
            info["deltas"].append(float(delta))
            total += delta
            if abs(delta) <= ROTATION_REFINE_REPEAT_DEG:
                break
        return round(total, 4)

    # Without a coarse angle the grid lines still measure a tilt of a few degrees.
    total = refine(0.0 if np.isnan(coarse) else coarse)
    if total is None and not np.isnan(coarse) and coarse != 0.0:
        # The fine Hough transform passes fainter line families than the whole degree
        # one and reads an angle off the pixel lattice of a page it should have left
        # alone, and turning the page by it smears the very comb that would have said
        # no. So the page as it came gets a second say, as it does without any coarse
        # angle at all; if its grid lines are readable, the coarse one was spurious.
        info["deltas"] = []
        info["reason"] = ""
        total = refine(0.0)
        # An angle the grid lines threw out must not come back through the dead band
        # below either; info keeps it, it is what the QC column has to show.
        coarse = float("nan")
    if total is None:
        # No grid lines either way, so hand back what --rotation hough would have
        # found: its vote threshold is the higher one, and the fine angle here is
        # exactly the one nothing could confirm.
        return get_rotation_angle(np_image), info
    if not np.isnan(coarse) and abs(total - coarse) < ROTATION_REFINE_MIN_DEG:
        return coarse, info
    return total, info


def check_mask_rotation(mask_folder, record, rot_angle, homography=None):
    """Warn when a saved mask was predicted in another frame than the one used now.

    homography is the map from the image to the frame of this run, None when the page
    is only rotated. A mask saved without one was predicted in the rotated frame of
    its angle, so both frames can be compared where they put the image corners.
    """
    meta_path = os.path.join(mask_folder, f"{record}_mask.json")
    if not os.path.exists(meta_path):
        return float("nan")
    with open(meta_path) as f:
        meta = json.load(f)
    saved = meta.get("rot_angle")
    if saved is None:
        return float("nan")
    if abs(saved - rot_angle) > ROTATION_MASK_ANGLE_TOLERANCE:
        print(
            f"WARNING: mask of record {record} was predicted at rot_angle "
            f"{saved:.4f}, the image is now rotated by {rot_angle:.4f}; "
            f"the mask does not fit."
        )
    saved_frame = meta.get("homography")
    if saved_frame is None and homography is None:
        return float(saved)
    # A frame stored without a homography is the plain rotation by its angle.
    height, width = meta["height"], meta["width"]
    then = np.array(
        saved_frame
        if saved_frame is not None
        else rotation_homography(saved, width, height)
    )
    now = (
        homography
        if homography is not None
        else rotation_homography(rot_angle, width, height)
    )
    corners = np.array([[0.0, 0.0], [width, 0.0], [width, height], [0.0, height]])
    moved = _apply_homography(then, corners) - _apply_homography(now, corners)
    distance = float(np.max(np.hypot(*moved.T)))
    if distance > PERSPECTIVE_MASK_SHIFT_TOLERANCE:
        print(
            f"WARNING: mask of record {record} was predicted in a frame whose corners "
            f"are {distance:.1f} px off the one used now; the mask does not fit."
        )
    return float(saved)


# Length of one window of the phase field, in printed grid lines. Over 32 lines the
# period of a mildly warped page is constant to a thousandth of a line, while the comb
# of a faint grid still stands out of the noise; over the whole page it does not.
PERSPECTIVE_WINDOW_LINES = 32
# Half range of the phase gradient search: a period mismatch relative to the carrier
# and a tilt of the line family in degrees. Both are far above what a page that came
# through estimate_rotation shows, and both stay inside the grating lobes of the
# window lattice (one lobe every half cycle per window step).
PERSPECTIVE_GRADIENT_RANGE = 0.03
PERSPECTIVE_GRADIENT_TILT_DEG = 1.5
# Steps of the coarse and of the fine gradient search. The coarse step stays below the
# width of the peak (one cycle over the page), the fine one leaves a fiftieth of a
# cycle over the page, far below the half cycle the unwrapping needs.
PERSPECTIVE_SEARCH_STEPS = 41
# Blocks per axis the phase gradient is measured in for the start of the fit. Over a
# third of a page the gradient is constant enough for its peak to be sharp, while one
# gradient for the whole page is a cycle off in the corners of a strongly warped one
# (measured: 1.4 cycles on the worst persp15 page), which no unwrapping can repair.
PERSPECTIVE_START_BLOCKS = 3
# Fewest windows before the gradient of a block is used, and fewest blocks before the
# start follows the drift of the gradient instead of being one plane over the page.
PERSPECTIVE_MIN_BLOCK_WINDOWS = 12
PERSPECTIVE_MIN_BLOCKS = 4
# Rounds of unwrapping, inlier selection and fit. The first round fits an affine map,
# which needs no good start, the rounds after it the full homography.
PERSPECTIVE_FIT_ROUNDS = 4
# Fewest inlier windows one line family needs, a tenth of what a 200 dpi page gives.
PERSPECTIVE_MIN_WINDOWS = 24
# Fraction of the page width and height the inlier windows have to span. The
# projective terms of a fit that only saw one half of the page are extrapolation.
PERSPECTIVE_MIN_SPAN = 0.5
# Largest displacement the rectification may ask for, relative to the page width.
# More than that is not the perspective of a page any more but a misread grid.
PERSPECTIVE_MAX_SHIFT = 0.05
# A correction above this many pixels is measured a second time on the warped page,
# where the windows of the corners are no longer washed out by the period drift.
PERSPECTIVE_REPEAT_SHIFT_PX = 1.0
# Below this displacement the page counts as straight and is left alone, so that masks
# saved for it stay valid. Undistorted pages ask for at most 0.09 px, which is the
# noise of the measurement, and the mildest perspective of the quasi real set for
# 2.4 px. In between lie generator pages whose augmentation cropped whole pixels off
# the sides and rescaled them, squashed by up to a thousandth (0.3 to 0.65 px on 15 of
# the 32 rotated pages). Warping those is a wash: a bicubic warp takes a sharp page
# off its pixel lattice, which costs about what so small a correction gains (mean
# change 0.00 dB, 40 leads worse by 1 dB and 38 better), so the band ends above them.
PERSPECTIVE_MIN_SHIFT_PX = 1.0
# Largest disagreement in pixels between the frame a mask was predicted in and the
# frame used now, measured at the image corners.
PERSPECTIVE_MASK_SHIFT_TOLERANCE = 0.1


def _grid_phase_field(darkness, period, axis):
    """Complex amplitude of the grid line comb in half overlapping local windows.

    axis=0 cuts the median profile of every band of rows into windows along x and so
    measures the vertical lines, axis=1 does the same on the transposed page for the
    horizontal ones. Under a perspective the line period drifts across the page, so
    the comb is only coherent over a short window, but every window still reports the
    phase against the absolute image coordinate. Returns the amplitudes
    [bands, windows] and the x and y of the window centres, of the same shape.
    """
    profiles, centres = _band_profiles(darkness.T if axis == 1 else darkness)
    length = int(round(PERSPECTIVE_WINDOW_LINES * period))
    if profiles.shape[0] == 0 or length < 2 or profiles.shape[1] < length:
        empty = np.zeros((0, 0))
        return empty.astype(complex), empty, empty

    starts = np.arange(0, profiles.shape[1] - length + 1, max(length // 2, 1))
    windows = profiles[:, starts[:, None] + np.arange(length)]
    # The mean of the single window, not of the page: uneven light is a slow ramp.
    centred = windows - windows.mean(axis=2, keepdims=True)
    # Pixel c covers [c, c+1). The Hann window leaves no side lobes, as in the comb.
    coordinates = starts[:, None] + np.arange(length) + 0.5
    kernel = np.hanning(length) * np.exp(-2j * np.pi * coordinates / period)
    amplitudes = np.einsum("bwl,wl->bw", centred, kernel)

    along, across = np.meshgrid(starts + length / 2, centres)
    return (amplitudes, along, across) if axis == 0 else (amplitudes, across, along)


def _phase_gradient(amplitudes, x, y, ranges):
    """Phase gradient in cycles per pixel that these windows agree on best.

    The measured phases are only known modulo a line, so the fit cannot start from
    unwrapped numbers. The coherent sum of the amplitudes against a linear model
    peaks at the gradient of the model they all follow, with no unwrapping at all.
    Returns (fx, fy, peak), the gradient relative to the carrier and the complex sum
    at it, whose size says how well the windows agree and whose angle is the offset.
    """
    fx, fy = 0.0, 0.0
    half_x, half_y = ranges
    for _ in range(2):
        grid_x = np.linspace(-half_x, half_x, PERSPECTIVE_SEARCH_STEPS)
        grid_y = np.linspace(-half_y, half_y, PERSPECTIVE_SEARCH_STEPS)
        along_x = np.exp(-2j * np.pi * np.outer(fx + grid_x, x))
        along_y = np.exp(-2j * np.pi * np.outer(fy + grid_y, y))
        sums = np.einsum("w,iw,jw->ij", amplitudes, along_x, along_y)
        peak = np.unravel_index(np.argmax(np.abs(sums)), sums.shape)
        fx, fy = fx + grid_x[peak[0]], fy + grid_y[peak[1]]
        # The second pass searches one coarse step around the peak, twenty times finer.
        half_x, half_y = grid_x[1] - grid_x[0], grid_y[1] - grid_y[0]
    return fx, fy, sums[peak]


def _phase_model_start(amplitudes, x, y, ranges):
    """Start of the phase model of one line family, without unwrapping anything.

    A perspective drifts the line period across the page, so the phase follows a
    quadratic and a single gradient cannot describe it. The gradient of a block of
    windows is unambiguous, and a plane laid through the block gradients integrates
    to that quadratic. Returns the model at every window, relative to the carrier and
    in cycles, close enough that every measurement unwraps against it.
    """

    def blocks(values):
        span = values.max() - values.min()
        edges = (values - values.min()) / max(span, 1e-9) * PERSPECTIVE_START_BLOCKS
        return np.clip(edges.astype(int), 0, PERSPECTIVE_START_BLOCKS - 1)

    index = blocks(x) * PERSPECTIVE_START_BLOCKS + blocks(y)
    # The whole page first, so that there is always one gradient to fall back on.
    parts = [np.ones(len(x), bool)] + [
        index == block
        for block in np.unique(index)
        if np.count_nonzero(index == block) >= PERSPECTIVE_MIN_BLOCK_WINDOWS
    ]
    rows, gradients, weights = [], [], []
    for inside in parts:
        fx, fy, peak = _phase_gradient(amplitudes[inside], x[inside], y[inside], ranges)
        centre_x, centre_y = x[inside].mean(), y[inside].mean()
        # The gradient of a + b x + c y + d x^2 + e x y + f y^2 at the block centre.
        rows += [[1, 0, 2 * centre_x, centre_y, 0], [0, 1, 0, centre_x, 2 * centre_y]]
        gradients += [fx, fy]
        weights += [np.abs(peak)] * 2
    terms = 5 if len(parts) > PERSPECTIVE_MIN_BLOCKS else 2
    root = np.sqrt(weights)[:, None]
    solution = np.linalg.lstsq(
        np.array(rows)[:, :terms] * root, np.array(gradients) * root[:, 0], rcond=None
    )[0]
    model = np.stack([x, y, x**2, x * y, y**2][:terms], axis=1) @ solution
    # The constant the gradients say nothing about, from the coherent sum again.
    offset = np.angle(np.sum(amplitudes * np.exp(-2j * np.pi * model))) / (2 * np.pi)
    return model + offset


def _apply_homography(homography, points):
    """Map points of shape [n, 2] through a 3x3 homography."""
    mapped = points @ homography[:, :2].T + homography[:, 2]
    return mapped[:, :2] / mapped[:, 2:3]


def _homography_shift(homography, width, height):
    """Largest displacement in pixels a homography causes on the page rectangle.

    A homography is at its most extreme on the border, so the corners, the edge
    midpoints and the centre bound the displacement over the whole page.
    """
    x, y = np.meshgrid([0.0, width / 2, width], [0.0, height / 2, height])
    points = np.stack([x.ravel(), y.ravel()], axis=1)
    return float(np.max(np.hypot(*(_apply_homography(homography, points) - points).T)))


def _unwrap_windows(values, model, period):
    """Unwrap the window measurements against a model and flag the ones that agree.

    A measurement only says where the lines of its window sit modulo one line, so it
    is put on the line of the model it is nearest to. The inliers are picked from
    scratch every time: a window the last model put a whole line off comes back as
    soon as the model is right. Returns (unwrapped, residuals, inliers) in cycles.
    """
    unwrapped = values - np.round(values - model)
    residuals = unwrapped - model
    deviation = np.abs(residuals - np.median(residuals))
    sigma = max(1.4826 * np.median(deviation), GRID_LINE_SIGMA_FLOOR / period)
    return unwrapped, residuals, deviation <= GRID_LINE_OUTLIER_SIGMA * sigma


def _grid_map_least_squares(points, values, weights, axes, projective):
    """Weighted least squares fit of the grid map to the unwrapped windows.

    points are the window centres in normalised coordinates, values their unwrapped
    grid coordinate and axes which of the two coordinates a window measured, 0 for u
    and 1 for v. The DLT form u (g x + h y + 1) = a x + b y + c is linear in the eight
    parameters and fits both line families at once, which is what ties g and h down.
    The values are centred first, as a constant in u only moves a, b and c.
    """
    centre = np.array(
        [
            np.average(values[axes == axis], weights=weights[axes == axis])
            for axis in (0, 1)
        ]
    )
    centred = values - centre[axes]
    linear = np.stack([points[:, 0], points[:, 1], np.ones(len(points))], axis=1)
    blank = np.zeros_like(linear)
    is_u = (axes == 0)[:, None]
    columns = [np.where(is_u, linear, blank), np.where(is_u, blank, linear)]
    if projective:
        columns.append(-centred[:, None] * points)
    # A least squares over squared residuals weighs the rows by the square root.
    root = np.sqrt(weights)[:, None]
    solution = np.linalg.lstsq(
        np.hstack(columns) * root, centred * root[:, 0], rcond=None
    )[0]
    projection = solution[6:] if projective else [0.0, 0.0]
    matrix = np.array([solution[:3], solution[3:6], [*projection, 1.0]])
    return np.array([[1, 0, centre[0]], [0, 1, centre[1]], [0, 0, 1]]) @ matrix


def _perspective_grid_map(darkness, width, height):
    """Projective map from image pixels to the units of the printed grid lines.

    Both line families are measured with one carrier period, the vertical one giving
    the grid coordinate u of a window modulo a line and the horizontal one its v, and
    both are fitted jointly. Every measurement is unwrapped against the current model
    rather than against its neighbours, so a single bad window cannot break a whole
    row, and the inliers are picked again in every round so that a window the first
    model put a line off comes back. Returns (M, info), M is None when info["reason"]
    says why the page gives no usable map.
    """
    info = {"period": float("nan"), "residual": float("nan"), "windows": 0, "reason": ""}
    profiles, _ = _band_profiles(darkness)
    if profiles.shape[0] == 0:
        info["reason"] = "image too small for the grid line profile"
        return None, info

    # One carrier for both families: the printed grid is square, so the unit of the
    # grid coordinates cancels in the rectification, even if this is a harmonic.
    period = _grid_line_period(profiles)
    if not np.isfinite(period):
        info["reason"] = "no grid line period found"
        return None, info
    info["period"] = period
    background = np.mean(
        [
            np.abs(_grid_line_comb(profiles, period * f)).sum()
            for f in (0.93, 0.96, 1.04, 1.07)
        ]
    )
    comb = np.abs(_grid_line_comb(profiles, period)).sum()
    contrast = float(comb / max(background, 1e-9))
    if contrast < GRID_LINE_MIN_CONTRAST:
        info["reason"] = f"no grid lines found (contrast {contrast:.1f})"
        return None, info

    mismatch = PERSPECTIVE_GRADIENT_RANGE / period
    tilt = np.tan(np.radians(PERSPECTIVE_GRADIENT_TILT_DEG)) / period
    points, values, weights, axes, model = [], [], [], [], []
    for axis in (0, 1):
        amplitudes, x, y = _grid_phase_field(darkness, period, axis)
        if amplitudes.size == 0:
            info["reason"] = "image too small for the grid line windows"
            return None, info
        amplitude = np.abs(amplitudes).ravel()
        # Borders, black wedges and blocks of text leave windows without a comb. The
        # cut is strict so that a page without this family, whose windows all have an
        # amplitude of zero, drops out instead of agreeing on a phase of zero.
        keep = amplitude > GRID_LINE_MIN_BAND_AMPLITUDE * np.median(amplitude)
        if np.count_nonzero(keep) < PERSPECTIVE_MIN_WINDOWS:
            info["reason"] = f"only {np.count_nonzero(keep)} windows with grid lines"
            return None, info
        amplitude, x, y = amplitude[keep], x.ravel()[keep], y.ravel()[keep]
        # The lines of a window sit at (psi + k) * period, so the grid coordinate of
        # its centre is x / period - psi for the vertical family, modulo one line.
        psi = -np.angle(amplitudes).ravel()[keep] / (2 * np.pi)
        carrier = (x if axis == 0 else y) / period
        start = _phase_model_start(
            amplitudes.ravel()[keep],
            x,
            y,
            (mismatch, tilt) if axis == 0 else (tilt, mismatch),
        )
        points.append(np.stack([x, y], axis=1))
        values.append(carrier - psi)
        weights.append(amplitude)
        axes.append(np.full(x.shape, axis))
        model.append(carrier + start)

    points = np.concatenate(points)
    values, weights = np.concatenate(values), np.concatenate(weights)
    axes, model = np.concatenate(axes), np.concatenate(model)
    # Centred and scaled coordinates, or the projective terms drown in the linear ones.
    scale = max(width, height) / 2
    normalised = (points - [width / 2, height / 2]) / scale
    for round_ in range(PERSPECTIVE_FIT_ROUNDS):
        unwrapped, _, keep = _unwrap_windows(values, model, period)
        for axis in (0, 1):
            if np.count_nonzero(keep & (axes == axis)) < PERSPECTIVE_MIN_WINDOWS:
                info["reason"] = (
                    f"only {np.count_nonzero(keep & (axes == axis))} windows agree "
                    f"on the {'vertical' if axis == 0 else 'horizontal'} grid lines"
                )
                return None, info
        matrix = _grid_map_least_squares(
            normalised[keep], unwrapped[keep], weights[keep], axes[keep], round_ > 0
        )
        model = _apply_homography(matrix, normalised)[np.arange(len(axes)), axes]

    _, residuals, keep = _unwrap_windows(values, model, period)
    info["windows"] = int(np.count_nonzero(keep))
    info["residual"] = float(np.sqrt(np.mean(residuals[keep] ** 2)) * period)
    if info["residual"] > GRID_LINE_MAX_SLOPE_RESIDUAL * period:
        info["reason"] = f"grid line phase is {info['residual']:.2f} px off the fit"
        return None, info
    span = points[keep].max(axis=0) - points[keep].min(axis=0)
    if span[0] < PERSPECTIVE_MIN_SPAN * width or span[1] < PERSPECTIVE_MIN_SPAN * height:
        info["reason"] = f"grid lines only span {span[0]:.0f} x {span[1]:.0f} px"
        return None, info

    # Back to pixel coordinates, so that the map reads the image as it is.
    normalise = np.array(
        [
            [1 / scale, 0, -width / (2 * scale)],
            [0, 1 / scale, -height / (2 * scale)],
            [0, 0, 1],
        ]
    )
    return matrix @ normalise, info


def _rectify_homography(matrix, width, height):
    """Pixel homography that straightens the grid, keeping the centre and the scale.

    The grid map is only known up to the unit of its coordinates and where its origin
    sits, so it is followed by the scale that keeps the area at the image centre and
    the shift that keeps the centre where it is. The same scale for x and y, the
    printed grid being square, also undoes the anisotropic foreshortening.
    """
    centre = np.array([width / 2, height / 2, 1.0])
    mapped = matrix @ centre
    grid = mapped[:2] / mapped[2]
    jacobian = (matrix[:2, :2] - np.outer(grid, matrix[2, :2])) / mapped[2]
    scale = 1 / np.sqrt(abs(np.linalg.det(jacobian)))
    similarity = np.array(
        [
            [scale, 0, centre[0] - scale * grid[0]],
            [0, scale, centre[1] - scale * grid[1]],
            [0, 0, 1],
        ]
    )
    rectified = similarity @ matrix
    return rectified / rectified[2, 2]


def rotation_homography(angle, width, height):
    """The 3x3 matrix of what rotate(image, angle) does, pixel centres at i + 0.5.

    torchvision turns the page counter-clockwise about its centre and keeps its size,
    which is what cv2.getRotationMatrix2D builds; the centre of a page of W columns
    lies at W / 2 in this convention. Verified against rotate() in the tests.
    """
    matrix = cv2.getRotationMatrix2D((width / 2, height / 2), angle, 1.0)
    return np.vstack([matrix, [0.0, 0.0, 1.0]])


def warp_page(image, homography):
    """Warp a page tensor [3, H, W] through a homography, in one bicubic pass."""
    array = np.ascontiguousarray(image.permute(1, 2, 0).numpy())
    # cv2 puts the pixel centres at integers and the pipeline at i + 0.5.
    half = np.array([[1, 0, -0.5], [0, 1, -0.5], [0, 0, 1]], float)
    warped = cv2.warpPerspective(
        array,
        half @ homography @ np.linalg.inv(half),
        (array.shape[1], array.shape[0]),
        flags=cv2.INTER_CUBIC,
        borderMode=cv2.BORDER_CONSTANT,
        # The black of a rotated page, so that both stages leave the same border.
        borderValue=(0, 0, 0),
    )
    return torch.from_numpy(warped.transpose(2, 0, 1).copy())


def estimate_perspective(image):
    """Homography that takes the shear and the perspective out of a page image.

    image is the rotation corrected page as read_image gives it ([3, H, W]) or the
    same as a numpy array. The printed 1 mm grid is a dense calibration target: the
    local phase of its two line families gives the grid coordinate of every window,
    and the homography through those coordinates rectifies the page. A correction of
    more than PERSPECTIVE_REPEAT_SHIFT_PX is measured again on the warped page, where
    the corners are no longer washed out, and the two are composed.
    Returns (H_rect, info). H_rect is None when info["reason"] says why the grid lines
    gave no homography, and also, with an empty reason, when the page is straight
    enough to be left alone.
    """
    if not torch.is_tensor(image):
        image = torch.from_numpy(np.ascontiguousarray(image))
    height, width = image.shape[1], image.shape[2]
    info = {
        "shift": float("nan"),
        "residual": float("nan"),
        "period": float("nan"),
        "windows": 0,
        "reason": "",
    }
    total = None
    for _ in range(2):
        page = image if total is None else warp_page(image, total)
        matrix, fit = _perspective_grid_map(_darkness(page), width, height)
        info.update({key: fit[key] for key in ("period", "residual", "windows")})
        if matrix is None:
            info["reason"] = fit["reason"]
            return None, info
        rectified = _rectify_homography(matrix, width, height)
        total = rectified if total is None else rectified @ total
        info["shift"] = _homography_shift(total, width, height)
        if info["shift"] > PERSPECTIVE_MAX_SHIFT * width:
            info["reason"] = f"grid lines ask for a {info['shift']:.0f} px correction"
            return None, info
        if info["shift"] <= PERSPECTIVE_REPEAT_SHIFT_PX:
            break
    if info["shift"] < PERSPECTIVE_MIN_SHIFT_PX:
        return None, info
    return total, info


# Period in pixels of the printed 1 mm grid lines of a 200 dpi page, which is the
# scale the generator draws at and the only one the model was trained on.
GRID_LINE_PERIOD_200DPI = 200 / 25.4
# Period range in pixels the 1 mm lines are searched in, 1 mm at about 75 to 1000 dpi.
# Far wider than GRID_LINE_PERIOD_RANGE of the rotation, which only needs some comb to
# follow the phase of: here the period sets the scale of the whole page, so the 5 mm
# lines of even a 1000 dpi page have to be a candidate of their own.
RESOLUTION_PERIOD_RANGE = (3.0, 40.0)
# Length in pixels of one window of the spectrum and how far the window is zero padded.
# Over 512 px the line period of a warped page is constant enough to leave its peak
# narrow, while the window still holds 65 lines of a 100 dpi page. The padding leaves
# about 30 samples over the main lobe, which is what the parabola is fitted to.
RESOLUTION_WINDOW = 512
RESOLUTION_PADDING = 8
# Half width in cycles per pixel of the running median that gives the local floor of
# the spectrum. Wider than the main lobe of a peak (0.008 at this window length), so
# that the floor steps over the comb instead of following it, and narrow enough to
# follow the broadband hump that noise and paper texture leave.
RESOLUTION_FLOOR_HALF_WIDTH = 0.01
# Harmonics of the 5 mm comb one period hypothesis is scored on: the four lines below
# the 1 mm fundamental, the fundamental itself and its second harmonic. A page whose
# 5 mm lines are darker shows all of them, a wrong hypothesis only some.
RESOLUTION_HARMONICS = (1, 2, 3, 4, 5, 10)
# Largest 5 mm harmonic the strongest peak is tried as. Above the tenth the 1 mm lines
# would be more than twice the period of the peak, which no printed grid shows.
RESOLUTION_MAX_HARMONIC = 10
# Half width of the window one harmonic is looked for in, relative to its frequency.
# It covers the period drift of a warped page, and the 5 mm series is 20 % apart.
RESOLUTION_HARMONIC_TOLERANCE = 0.004
# How much a hypothesis has to beat the one that takes the strongest peak for the 1 mm
# line itself. A grid whose 5 mm lines are not darker has no sub-harmonics to vote
# with, so every hypothesis below it scores the same and none may win on a tie.
RESOLUTION_HARMONIC_MARGIN = 1.2
# Half width in bins of the window the 1 mm fundamental is refined in, half the main
# lobe of the Hann window. It covers the bin quantisation of the strongest peak, which
# the lowest hypothesis multiplies by five, and stops well short of the next harmonic.
RESOLUTION_REFINE_BINS = 2 * RESOLUTION_PADDING
# Relative scale change below which the page keeps its pixels. Resampling by a factor
# that does not put every pixel back where it was blurs the page, and that blur costs
# about what the wrong scale does: over 8 records a condition, scans at 0.92 and 0.88 of
# the 200 dpi size read about a dB better kept (19.4 and 19.7 dB against 18.5 and 18.6),
# while 0.80, 1.15 and 1.25 read better resampled, 1.25 collapsing to -5 dB kept and
# 1.15 leaving single leads at -20 dB. The band is on the scale asked for, the inverse
# of the page scale: 0.88 asks for 1.136 and 1.15 for 0.870, as far off as each other,
# so no one band keeps the first and resamples the second. It ends between them and the
# 0.92 page, which gives the 0.88 page its dB away rather than risk the leads at 1.15,
# and it still leaves generator pages, 2 % off at most, untouched.
RESOLUTION_DEAD_BAND = 0.12
# A scale outside this is not a page of another resolution any more but a misread
# grid, so the page is left alone and the reason says so.
RESOLUTION_SCALE_RANGE = (0.2, 3.0)


def _welch_spectrum(profiles):
    """Frequencies and amplitude of the band profiles, summed over local windows.

    One window over the whole page smears the peak of a page whose period drifts, and
    it smears it in proportion to the frequency, which is what lets a harmonic of the
    5 mm lines look sharper than the 1 mm lines themselves. Half overlapping windows
    of RESOLUTION_WINDOW px keep every peak narrow. The phase of a window is its own,
    so the powers are summed over windows and bands and the amplitude is their root.
    """
    width = profiles.shape[1]
    length = min(RESOLUTION_WINDOW, width)
    padded = RESOLUTION_PADDING * length
    starts = np.arange(0, width - length + 1, max(length // 2, 1))
    power = np.zeros(padded // 2 + 1)
    for start in starts:
        window = profiles[:, start : start + length]
        # The mean of the single window, not of the page: uneven light is a slow ramp.
        centred = window - window.mean(axis=1, keepdims=True)
        spectrum = np.fft.rfft(centred * np.hanning(length), n=padded)
        power += (np.abs(spectrum) ** 2).sum(axis=0)
    return np.fft.rfftfreq(padded), np.sqrt(power)


def _spectral_prominence(amplitude, bin_width):
    """Spectrum amplitude above its local floor, clipped at zero.

    The floor is a running median, which steps over the narrow peaks of a comb instead
    of following them. Broadband noise raises the floor with the peak and so scores
    nothing, while the grid lines of the same page still stand out of it.
    """
    half = max(int(round(RESOLUTION_FLOOR_HALF_WIDTH / bin_width)), 1)
    # The edges of the spectrum have no window of their own, so they borrow one.
    windows = np.lib.stride_tricks.sliding_window_view(
        np.pad(amplitude, half, mode="edge"), 2 * half + 1
    )
    return np.clip(amplitude - np.median(windows, axis=-1), 0.0, None)


def _harmonic_score(frequencies, prominence, period):
    """How much of the 5 mm comb of a 1 mm period the page really shows.

    The 5 mm lines sit at every fifth 1 mm line, so a page drawn with this period has
    peaks at RESOLUTION_HARMONICS fifths of its fundamental. Each one is looked for in
    a window around where it belongs, and a harmonic above Nyquist simply votes zero.
    """
    score = 0.0
    for harmonic in RESOLUTION_HARMONICS:
        frequency = harmonic / (5 * period)
        if frequency >= 0.5:
            continue
        inside = np.abs(frequencies / frequency - 1) <= RESOLUTION_HARMONIC_TOLERANCE
        if np.any(inside):
            score += prominence[inside].max()
    return score


def measure_grid_period(image):
    """Period in pixels of the printed 1 mm grid lines of a page, NaN if there is none.

    image is the page as read_image gives it ([3, H, W]) or the same as a numpy array.
    The page is straightened first: a tilt of two degrees drags a line across almost a
    whole period of a 100 dpi page over one band of rows and washes the comb out.
    The strongest peak of the spectrum is not the 1 mm line by itself, as blur favours
    the low frequencies and leaves a photographed page with its 5 mm lines or their
    second harmonic on top. Every harmonic of the 5 mm comb the peak could be is
    therefore put up as a hypothesis and scored by how much of that comb the page
    shows, and only then is the 1 mm fundamental of the winner refined.
    Returns (period, info); info["harmonic"] is which 5 mm harmonic the strongest peak
    turned out to be and info["reason"] says why a period came back NaN.
    """
    if not torch.is_tensor(image):
        image = torch.from_numpy(np.ascontiguousarray(image))
    info = {
        "period": float("nan"),
        "contrast": float("nan"),
        "harmonic": 0,
        "reason": "",
    }
    coarse = get_rotation_angle_fine(image.permute(1, 2, 0).numpy().astype(np.uint8))
    # A page whose lines the Hough transform cannot find is taken as straight.
    straight = rotate(image, 0.0 if np.isnan(coarse) else float(coarse))
    profiles, _ = _band_profiles(_darkness(straight))
    if profiles.shape[0] == 0:
        info["reason"] = "image too small for the grid line profile"
        return float("nan"), info

    frequencies, amplitude = _welch_spectrum(profiles)
    bin_width = frequencies[1] - frequencies[0]
    prominence = _spectral_prominence(amplitude, bin_width)
    low, high = RESOLUTION_PERIOD_RANGE
    inside = np.flatnonzero((frequencies >= 1 / high) & (frequencies <= 1 / low))
    if inside.size == 0 or not np.any(prominence[inside] > 0):
        info["reason"] = "no grid line period found"
        return float("nan"), info
    peak = frequencies[inside[np.argmax(prominence[inside])]]

    # The peak is the m-th harmonic of a 5 mm comb of the period m / (5 peak).
    scores = {}
    for harmonic in range(1, RESOLUTION_MAX_HARMONIC + 1):
        candidate = harmonic / (5 * peak)
        if low <= candidate <= high:
            scores[harmonic] = _harmonic_score(frequencies, prominence, candidate)
    best = max(scores, key=scores.get)
    if scores[best] < RESOLUTION_HARMONIC_MARGIN * scores[5]:
        best = 5
    info["harmonic"] = best

    # Parabolic interpolation of the power peak at the 1 mm fundamental of the winner.
    centre = int(round(5 * peak / best / bin_width))
    first = max(centre - RESOLUTION_REFINE_BINS, 1)
    last = min(centre + RESOLUTION_REFINE_BINS + 1, amplitude.size - 1)
    top = first + int(np.argmax(amplitude[first:last]))
    frequency = frequencies[top]
    # Through the power, as _grid_line_period does it, not through the amplitude.
    left, middle, right = amplitude[top - 1 : top + 2] ** 2
    curvature = left - 2 * middle + right
    if curvature < 0:
        frequency += 0.5 * (left - right) / curvature * bin_width
    info["period"] = float(1 / frequency)

    def comb(candidate):
        # Non-coherent sum: the bands of a page that is still a little tilted differ.
        return np.abs(_grid_line_comb(profiles, candidate)).sum()

    background = np.mean([comb(info["period"] * f) for f in (0.93, 0.96, 1.04, 1.07)])
    info["contrast"] = float(comb(info["period"]) / max(background, 1e-9))
    if info["contrast"] < GRID_LINE_MIN_CONTRAST:
        info["reason"] = f"no grid lines found (contrast {info['contrast']:.1f})"
        return float("nan"), info
    return info["period"], info


def normalise_resolution(image):
    """Resample a page so that its printed 1 mm grid has the period of a 200 dpi page.

    The model only segments pages near the scale it was trained on, and the width of a
    page is no scale: it can be cropped, photographed or printed on another paper size.
    The printed grid is. Inside RESOLUTION_DEAD_BAND the very image object comes back,
    so that a page near enough to that scale keeps every one of its pixels: resampling
    a page that is only a little off costs more than being off the scale does.
    Returns (image, info), a tensor [3, H, W] uint8 in and out as warp_page. info is
    the measurement plus the scale the page was resampled by, 1.0 when it was left
    alone, and info["reason"] says why it was left alone, if there is a why.
    """
    if not torch.is_tensor(image):
        image = torch.from_numpy(np.ascontiguousarray(image))
    period, measured = measure_grid_period(image)
    # The whole measurement, but with the period of a rejected grid left at NaN.
    info = {**measured, "period": period, "scale": 1.0}
    if not np.isfinite(period):
        return image, info

    scale = GRID_LINE_PERIOD_200DPI / period
    low, high = RESOLUTION_SCALE_RANGE
    if not low <= scale <= high:
        info["reason"] = f"grid lines ask for a scale of {scale:.2f}"
        return image, info
    if abs(scale - 1) <= RESOLUTION_DEAD_BAND:
        return image, info

    array = np.ascontiguousarray(image.permute(1, 2, 0).numpy())
    height, width = array.shape[:2]
    resized = cv2.resize(
        array,
        (int(round(width * scale)), int(round(height * scale))),
        # Thin lines fall between the samples of a page that is shrunk by picking
        # pixels, so a shrinking page is averaged over and only a growing one is
        # interpolated, which keeps its traces smooth instead of blocky.
        interpolation=cv2.INTER_AREA if scale < 1 else cv2.INTER_CUBIC,
    )
    info["scale"] = float(scale)
    return torch.from_numpy(resized.transpose(2, 0, 1).copy()), info


def baseline_row(ratio, image_height, scale=1.0):
    """Row of the zero line of one lead, for a page rescaled about its centre."""
    centre = image_height / 2
    return centre + scale * ((1 - ratio) * image_height - centre)


# Steepness of the ink weights, which trades the timing bias of a resampled page
# against the noise of a soft or a grainy one. The square takes out a fifth of the
# bias (+0.9 dB on resampled pages), the fourth power a third (+1.6 dB) with one
# generator lead of 1152 losing more than 1 dB, the eighth more still (+2.2 dB) but
# dozens of leads on the augmented and the blurred pages lose more than 1 dB.
INK_WEIGHT_POWER = 4
# Minimal 95th percentile of the ink under a mask, relative to black, for the weights
# to say anything; below it the lead keeps the mask mean. It separates a blurred
# stroke, which spreads its ink over more and lighter pixels (p95 137 to 224, median
# 170) and only loses by being weighted, from a crisp or resampled one (200 to 255).
INK_MIN_CONTRAST = 0.75


def _ink_profile(ink_map, binary, position, mean_profile, filled, info=None):
    """Ink weighted mean row of every filled column of one lead, in image coordinates.

    A resampled page moves the mask and the ink apart: on a steep flank the binary
    mask sits 0.2 to 0.3 px left of the ink, while the ink itself is still where the
    grid says it is. Weighting the mask rows by the ink to a high power keeps the
    support of the mask, which follows the trace through a sharp peak better than the
    ink blob does, but lets the core of the stroke decide the row. A soft page is the
    other way round: blur fills a narrow peak with ink, so a lead whose stroke is too
    pale for the weights keeps the mask mean, whole and for all of its columns.
    info, if given, is filled with the measurement of this lead.
    """
    height, width = binary.shape
    patch = ink_map[
        position["y1"] : position["y1"] + height,
        position["x1"] : position["x1"] + width,
    ]
    p95 = float(np.percentile(patch[binary], 95))
    if info is not None:
        info["ink_p95"] = p95
        info["ink_used"] = p95 >= INK_MIN_CONTRAST * 255
    if p95 < INK_MIN_CONTRAST * 255:
        return mean_profile

    # Not clipped at 1: the darkest pixels of a crisp stroke are meant to outweigh.
    weight = binary * (patch / p95) ** INK_WEIGHT_POWER
    rows = np.arange(height)[:, None]
    total = weight.sum(axis=0)[filled]
    centroid = (weight * rows).sum(axis=0)[filled] / np.where(total > 0, total, 1)
    # A column that is pure white under its mask has no centroid, so it alone keeps
    # the mask mean instead of turning into a NaN.
    return np.where(total > 0, position["y1"] + centroid, mean_profile)


# Mask pixels a column needs for the stroke in it to count as steep. A flat stroke
# says nothing about the time axis: a shift along it costs slope times dx.
TRACE_SHIFT_MIN_RUN = 6
# Rows dropped at each end of such a column, where the stroke turns into its corner
# and the mask holds the pixels of the flat part as well.
TRACE_SHIFT_CORNER_ROWS = 2
# Columns added at each side of a row segment, so that the window holds the whole
# stroke and the paper it is drawn on, not the segment the mask happens to claim.
TRACE_SHIFT_PAD = 2
# The peak ink of a window has to reach this share of the lead's p95, otherwise the
# window sits on a gap the mask bridged rather than on the stroke.
TRACE_SHIFT_MIN_PEAK = 0.5
# Steepness of the ink weights inside a window. The square is flat enough to find
# the centre of a stroke a few pixels wide without the noise of a steeper one.
TRACE_SHIFT_WEIGHT_POWER = 2
# Samples a page needs before its median means anything; below it the shift is 0.
TRACE_SHIFT_MIN_ROWS = 30
# Share of the measured offset that is taken out under --trace_estimator ink. The
# ink weights already remove about a third of it, and the row wise centre of the
# weighted stroke is nearly the mask's, so the full offset overcorrects: the timing
# error of the AREA shrunk pages, +0.39 samples without a shift, goes to -0.15 with
# all of it and to 0.00 with two thirds. The optimum is broad, 0.5 to 0.8 all stay
# within 0.4 dB.
TRACE_SHIFT_INK_SHARE = 0.67
# Shifts below this are set to 0, which is what keeps a page that needs no shift
# byte identical: the measurement itself is noisy to about 0.02 px.
TRACE_SHIFT_DEAD_BAND = 0.05


def _lead_runs(sub, steep):
    """Start and end column of every horizontal run of one lead that is steep.

    Returns (rows, starts, ends), all relative to the lead's bounding box and all
    inclusive, of the maximal runs of sub that hold at least one steep pixel.
    """
    edges = np.diff(np.pad(sub, ((0, 0), (1, 1))).astype(np.int8), axis=1)
    rows, starts = np.nonzero(edges > 0)
    ends = np.nonzero(edges < 0)[1] - 1
    # Steep pixels left of every column, so a run is looked up in constant time.
    steep_before = np.zeros((steep.shape[0], steep.shape[1] + 1), dtype=np.int32)
    np.cumsum(steep, axis=1, out=steep_before[:, 1:])
    keep = steep_before[rows, ends + 1] > steep_before[rows, starts]
    return rows[keep], starts[keep], ends[keep]


def measure_trace_shift(labelled, ink_map):
    """How far right of the mask the ink of a page sits, in pixels.

    A resampled page moves the lead masks a fraction of a pixel off the ink, and on
    the time axis that costs slope times the offset. It is read row wise, which is
    the only direction that sees it: every row of a steep stroke gives the distance
    between the centre of the mask segment and the ink weighted centre of the same
    stroke, measured over a window of TRACE_SHIFT_PAD pixels more on each side. A
    window is dropped when it leaves the page, when any other mask pixel reaches
    into it, or when its ink never gets near the stroke's own darkness.
    The samples of all leads are pooled, because one scalar per page is all that
    holds up: per lead, linear across the page and per layout column were all tried
    and are noise.
    Returns (dx, info), dx the median of the pooled samples and 0.0 below
    TRACE_SHIFT_MIN_ROWS of them, info the raw median and how many there were.
    """
    height, width = labelled.shape
    # One mask pixel of any lead, summed along every row: a window holds its own
    # segment and nothing else exactly when it counts no more pixels than that.
    mask_before = np.zeros((height, width + 1), dtype=np.int32)
    np.cumsum(labelled > 0, axis=1, out=mask_before[:, 1:])

    # The labels of the page once, so that every lead is only read over its own box.
    rows, columns = np.nonzero(labelled)
    labels = labelled[rows, columns]
    pooled = []
    for value in LEAD_LABEL_MAPPING.values():
        here = labels == value
        if not here.any():
            continue
        r0, r1 = int(rows[here].min()), int(rows[here].max()) + 1
        c0, c1 = int(columns[here].min()), int(columns[here].max()) + 1
        sub = labelled[r0:r1, c0:c1] == value
        p95 = float(np.percentile(ink_map[r0:r1, c0:c1][sub], 95))
        if not np.isfinite(p95) or p95 <= 0:
            continue

        count = sub.sum(axis=0)
        first = np.argmax(sub, axis=0)
        last = sub.shape[0] - 1 - np.argmax(sub[::-1], axis=0)
        row_index = np.arange(sub.shape[0])[:, None]
        steep = (
            sub
            & (count >= TRACE_SHIFT_MIN_RUN)[None, :]
            & (row_index >= (first + TRACE_SHIFT_CORNER_ROWS)[None, :])
            & (row_index <= (last - TRACE_SHIFT_CORNER_ROWS)[None, :])
        )
        if not steep.any():
            continue

        row, start, end = _lead_runs(sub, steep)
        row, start, end = row + r0, start + c0, end + c0
        left, right = start - TRACE_SHIFT_PAD, end + TRACE_SHIFT_PAD
        # A window that reaches over an edge of the page is dropped before anything
        # is looked up in it: there is no paper there to read the stroke against.
        inside = (left >= 0) & (right < width)
        row, start, end = row[inside], start[inside], end[inside]
        left, right = left[inside], right[inside]
        keep = mask_before[row, right + 1] - mask_before[row, left] == end - start + 1
        row, start, end = row[keep], start[keep], end[keep]
        left, right = left[keep], right[keep]
        if row.size == 0:
            continue

        # The windows, laid end to end, so that every one of them is one segment.
        lengths = right - left + 1
        offsets = np.concatenate(([0], np.cumsum(lengths)))
        window = np.repeat(left - offsets[:-1], lengths) + np.arange(offsets[-1])
        ink = ink_map[np.repeat(row, lengths), window]
        weight = (ink / p95) ** TRACE_SHIFT_WEIGHT_POWER
        total = np.add.reduceat(weight, offsets[:-1])
        centre = np.add.reduceat(weight * window, offsets[:-1])
        peak = np.maximum.reduceat(ink, offsets[:-1])
        keep = (peak >= TRACE_SHIFT_MIN_PEAK * p95) & (total > 0)
        pooled.append(centre[keep] / total[keep] - 0.5 * (start + end)[keep])

    samples = np.concatenate(pooled) if pooled else np.zeros(0)
    info = {"rows": int(samples.size), "median": float("nan")}
    if samples.size == 0:
        return 0.0, info
    info["median"] = float(np.median(samples))
    if samples.size < TRACE_SHIFT_MIN_ROWS:
        return 0.0, info
    return info["median"], info


def trace_shift_px(measured, estimator):
    """The measured page shift as it is applied: shared with the estimator, dead banded.

    The ink estimator has taken part of the offset out already, so only the rest of
    it is left to move, and a shift small enough to be measurement noise is dropped.
    """
    dx = measured * (TRACE_SHIFT_INK_SHARE if estimator == "ink" else 1.0)
    return 0.0 if abs(dx) < TRACE_SHIFT_DEAD_BAND else dx


# Weight of the two neighbours in the aperture correction [-a, 1 + 2a, -a] of the
# column profile under --sharpen bandlimited. A column value is the middle of the
# rows the trace crosses in that column, the mean of its values at the two column
# edges, which damps a wave of f cycles per column width d as cos(pi f d); the
# correction lifts it by 1 + 4a sin^2(pi f d), and 1/8 is the value whose second
# order term cancels that of the cosine (1/24 would invert a box average).
SHARPEN_APERTURE_A = 0.125
# Cutoff of the low-pass that confines the correction to the band it is meant for.
# The diagnosis model reads the signal at 100 Hz, flat to 42 Hz, so little above that
# counts, while the full correction overshoots on the steep QRS spikes. 35 Hz is the
# highest cutoff screened at which no generator lead and no lead of the 12 quasi-real
# conditions of the pass line loses more than 1 dB (40 Hz: 3 such leads, 45 Hz: 5) and
# 3 of 23,994 leads of 2,000 pages do (40 Hz: 8, 45 Hz: 20), for a median gain of
# +1.41 dB on the generator pages (45 Hz: +1.69).
SHARPEN_CUTOFF_HZ = 35.0
# Order of the Butterworth low-pass. It is run forwards and backwards, so it has no
# phase and the square of the gain: 0.99 at 20 Hz, 0.78 at 30 Hz, 0.5 at the cutoff
# and 0.05 at 50 Hz.
SHARPEN_FILTER_ORDER = 4
# Shortest run of consecutive filled columns that gets the cubic spline. Through fewer
# than four columns the not-a-knot spline is a parabola or a line, so a shorter run,
# like a gap, keeps the linear values.
SHARPEN_MIN_RUN = 4
# Second order sections of the low-pass, built once per sampling rate.
_SHARPEN_SOS = {}


def _sharpen_sos(fs):
    """The --sharpen low-pass as second order sections for the sampling rate fs."""
    fs = float(fs)
    if fs not in _SHARPEN_SOS:
        _SHARPEN_SOS[fs] = butter(
            SHARPEN_FILTER_ORDER, SHARPEN_CUTOFF_HZ, fs=fs, output="sos"
        )
    return _SHARPEN_SOS[fs]


def _sharpen_padlen(sos):
    """The samples sosfiltfilt() pads a trace with at each end, its default padlen.

    Three times the taps of the sections, 15 for the fourth order filter: a trace
    needs more samples than that to be filtered at all.
    """
    taps = 2 * len(sos) + 1 - min((sos[:, 2] == 0).sum(), (sos[:, 5] == 0).sum())
    return 3 * int(taps)


def _column_runs(filled):
    """Runs of consecutive integers in the sorted filled columns of one lead.

    Returns a list of (i0, i1), half open index ranges into filled.
    """
    filled = np.asarray(filled)
    if filled.size == 0:
        return []
    cut = np.flatnonzero(np.diff(filled) != 1) + 1
    starts = np.r_[0, cut]
    ends = np.r_[cut, filled.size]
    return [(int(a), int(b)) for a, b in zip(starts, ends)]


def _aperture_fir(profile, runs, a):
    """The aperture correction [-a, 1 + 2a, -a] of a column profile, run by run.

    Computed as p - a (left - 2p + right), so that a constant stays bitwise what it
    was. The ends of a run repeat their own value instead of reaching over a gap,
    which leaves a run of one column and the inside of a straight run unchanged.
    Returns a new float64 array.
    """
    p = np.asarray(profile, dtype=float)
    y = p.copy()
    if a == 0:
        return y
    for i0, i1 in runs:
        seg = p[i0:i1]
        if seg.size < 2:
            continue
        left = np.r_[seg[0], seg[:-1]]
        right = np.r_[seg[1:], seg[-1]]
        y[i0:i1] = seg - a * (left - 2.0 * seg + right)
    return y


def _cubic_samples(columns, profile, x, runs, min_run):
    """A column profile sampled at x by a not-a-knot cubic spline through each run.

    The spline of a run of at least min_run columns replaces np.interp for the x
    between its first and its last column; the gaps, the runs that are too short
    and the ends past the outer columns keep the linear, edge held values.
    """
    out = np.interp(x, columns, profile)
    columns = np.asarray(columns, dtype=float)
    profile = np.asarray(profile, dtype=float)
    x = np.asarray(x, dtype=float)
    out = np.array(out, dtype=float, copy=True)
    for i0, i1 in runs:
        if i1 - i0 < max(min_run, 2):
            continue
        cx = columns[i0:i1]
        inside = (x >= cx[0]) & (x <= cx[-1])
        if not inside.any():
            continue
        spline = CubicSpline(cx, profile[i0:i1], bc_type="not-a-knot")
        out[inside] = spline(x[inside])
    return out


def _bandlimited_sharpen(columns, profile, filled, x, sampled, fs):
    """The linear trace of one lead with its band below the cutoff sharpened.

    The column average and the linear interpolation between the column centres both
    damp the trace, a thin one together by about 8 % at 20 Hz and 18 % at 30 Hz on a
    200 dpi page, which rounds off the QRS. The aperture correction with a spline
    through it gives that back but overshoots on the steep spikes, so only the low
    passed difference to the linear trace is added: below SHARPEN_CUTOFF_HZ the trace
    is the spline's, above it stays the linear one, with no phase shift. columns
    carries the page shift, while the runs come from the integer filled columns.
    sampled is np.interp(x, columns, profile); a trace no longer than the samples the
    filter pads its ends with (15) keeps it.
    """
    runs = _column_runs(filled)
    a_prof = _aperture_fir(profile, runs, SHARPEN_APERTURE_A)
    a_samp = _cubic_samples(columns, a_prof, x, runs, SHARPEN_MIN_RUN)
    d = a_samp - sampled
    sos = _sharpen_sos(fs)
    if len(d) > _sharpen_padlen(sos):
        return sampled + sosfiltfilt(sos, d)
    return sampled


# Window of the column mapping in printed 1 mm lines: it averages the noise of single
# lines and still follows a step of the lines within some 60 px. It does not average out
# the pixel snap of a drawn page, whose lines sit at round(x) + 0.5: at 200 dpi that
# error repeats about every eight thin lines, one cycle per window, where a Hann window
# passes half of it (its first null is at two cycles), and every 13.5 mm on the bold
# lines. COLUMN_MAPPING_MEDIAN_MM takes that ripple out of the map, and the dead band
# COLUMN_MAPPING_MIN_SHIFT_PX keeps a drawn page on the uniform columns.
COLUMN_MAPPING_FINE_LINES = 8
# Every fifth printed line is bold, and their comb over twenty lines, four of their
# periods, says which 1 mm line a window is on. The 1 mm phase alone cannot tell a step
# of the lines by half a line one way from one the other way, and on a scanned page such
# steps are common: the wrong choice leaves everything after it a whole line, 1 mm or 20
# samples, off.
COLUMN_MAPPING_BOLD_EVERY = 5
COLUMN_MAPPING_COARSE_LINES = 20
# The map is measured this share of a column beyond the first and the last column edge,
# so that its knots bracket every sample the page reads.
COLUMN_MAPPING_REACH = 0.25
# Windows whose 1 mm or 5 mm comb is weaker than this share of the median are dropped:
# they sit on text, on the QR code or on paper without grid lines.
COLUMN_MAPPING_MIN_WINDOW_AMPLITUDE = 0.3
# Largest disagreement, in 1 mm lines, between the offset of the bold lines of a window
# and that of its 1 mm lines. A window further off is dropped, and so is a window this
# far off the median of its four neighbours, where a stray 5 mm phase named another line.
COLUMN_MAPPING_BRANCH_TOLERANCE = 0.3
# Share of the windows whose two offsets have to agree. With bold lines on the page
# nearly all do (0.88 or more on every generator page with them, the 15 synthetic
# conditions and the real renders and clean scans), without them the 5 mm phase is
# noise and agrees by chance only (0.29 to 0.85 on augmented generator pages).
COLUMN_MAPPING_MIN_AGREEMENT = 0.85
# Neighbouring windows further apart than this many lines split the map into runs, and a
# run shorter than a 1 mm window is dropped: a stretch whose bold offset strayed together
# sits a whole line off both of its sides (0.96 and 1.04 lines on an augmented generator
# page). A step of the printed lines themselves is a single jump between two long runs,
# up to 0.70 lines on the real scans; a jump of more than COLUMN_MAPPING_MAX_JUMP lines
# left after that is a line off, not a step.
COLUMN_MAPPING_RUN_BREAK = 0.5
COLUMN_MAPPING_MAX_JUMP = 0.8
# Share of the windows over the columns that the map has to keep, and the longest stretch
# of a column it may bridge without a window. A clean page keeps 0.87 or more with gaps up
# to 0.17 columns; a mouldy scan loses whole stretches, over which a step goes unseen.
COLUMN_MAPPING_MIN_COVERAGE = 0.8
COLUMN_MAPPING_MAX_GAP = 0.25
# A map that moves no sample the page reads by this many pixels against the uniform
# columns is not used, which keeps such a page byte identical. After the running median
# a drawn page is off by up to 0.26 px (dev renders; 0.39 px on the generator pages and
# the synthetic conditions), the rest of the pixel snap of its lines, which the traces
# do not share, while a clean scan moves by 1.4 px at least and by 9 px in the median.
COLUMN_MAPPING_MIN_SHIFT_PX = 1.0
# Width in grid millimetres of the running median of the map's displacement from the
# uniform columns (--column_mapping_median). A page drawn at 200 dpi has its grid lines
# snapped to the pixels and its traces not: the bold lines are 39.37 px apart, so their
# snap advances by 0.37 px from one to the next and repeats every 2.7 of them, 13.5 mm.
# The real scans carry that ripple as if printed from such pages: on dev pages the map's
# displacement, less its 20 mm moving average, has its strongest period at 13.5 mm and
# follows the render of the same record (r 0.44 per record in the median). The map would
# move samples by it where the traces do not move. A running median over one period
# takes it out (r 0.06 after it) and keeps a step or a stretch of the lines as it is,
# since a run that only rises or only falls is its own median.
COLUMN_MAPPING_MEDIAN_MM = 13.5


def _running_median(knots_m, values, width):
    """Running median of values at the increasing knots_m over width grid millimetres.

    The window is centred on every knot. Beyond the outer knots the outer value is
    carried on, as _column_map_x() carries the map on, so that a run that rises or
    falls up to the end of the knots keeps its end.
    """
    half = width / 2
    spacing = float(np.median(np.diff(knots_m)))
    count = int(np.ceil(half / spacing)) + 1
    padded_m = np.r_[
        knots_m[0] - spacing * np.arange(count, 0, -1),
        knots_m,
        knots_m[-1] + spacing * np.arange(1, count + 1),
    ]
    padded = np.r_[np.full(count, values[0]), values, np.full(count, values[-1])]
    low = np.searchsorted(padded_m, knots_m - half, side="left")
    high = np.searchsorted(padded_m, knots_m + half, side="right")
    return np.array([np.median(padded[a:b]) for a, b in zip(low, high)])


def _column_mapping_windows(profile, period, low, high):
    """Complex 1 mm and 5 mm comb amplitudes of windows along one grid line profile.

    The window centres are about half a line apart and cover [low, high] as far as the
    page reaches. The 1 mm window is COLUMN_MAPPING_FINE_LINES periods long, the 5 mm
    window COLUMN_MAPPING_COARSE_LINES periods, centred on it and moved inside the page at
    its ends. Both are Hann windowed and measured against the absolute x, as in
    _grid_phase_field, so that their phases say where on the page the lines are.
    Returns (centres, fine, coarse).
    """
    width = profile.size
    fine_length = int(round(COLUMN_MAPPING_FINE_LINES * period))
    coarse_length = int(round(COLUMN_MAPPING_COARSE_LINES * period))
    step = max(int(round(period / 2)), 1)
    first = max(int(np.floor(low - fine_length / 2)), 0)
    last = min(int(np.ceil(high - fine_length / 2)), width - fine_length)
    if fine_length < 2 or coarse_length > width or last < first:
        empty = np.zeros(0)
        return empty, empty.astype(complex), empty.astype(complex)
    starts = np.arange(first, last + 1, step)
    centres = starts + fine_length / 2
    coarse_starts = np.clip(
        np.round(centres - coarse_length / 2).astype(int), 0, width - coarse_length
    )

    def comb(window_starts, length, comb_period):
        windows = profile[window_starts[:, None] + np.arange(length)]
        # The mean of the single window, not of the page: uneven light is a slow ramp.
        centred = windows - windows.mean(axis=1, keepdims=True)
        # Pixel c covers [c, c+1), so its centre is at c + 0.5.
        coordinates = window_starts[:, None] + np.arange(length) + 0.5
        kernel = np.hanning(length) * np.exp(-2j * np.pi * coordinates / comb_period)
        return (centred * kernel).sum(axis=1)

    fine = comb(starts, fine_length, period)
    coarse = comb(coarse_starts, coarse_length, COLUMN_MAPPING_BOLD_EVERY * period)
    return centres, fine, coarse


def _column_map_x(knots_m, knots_x, period, millimetres):
    """The x of grid millimetres on a column map.

    Linear between the knots (m, x), and one line per period beyond them, where nothing
    was measured, which is the uniform grid carried on from the outer knots.
    """
    m = np.asarray(millimetres, dtype=float)
    x = np.interp(m, knots_m, knots_x)
    x = np.where(m < knots_m[0], knots_x[0] + (m - knots_m[0]) * period, x)
    return np.where(m > knots_m[-1], knots_x[-1] + (m - knots_m[-1]) * period, x)


def column_mapping_edges(signal_masks, signal_positions, long_leads):
    """The mask edges of every lead with the grid millimetre each belongs to.

    A short lead starts at the start of its column and ends at the start of the next,
    the rhythm strip spans all columns; a column is 62.5 mm. The edges are x1 and
    x1 + width, as fit_column_grid() reads them: pixel c covers [c, c+1). A lead that is
    in no column of the layout gives no edge. Returns a list of (x, mm).
    """
    edges = []
    for lead, mask in signal_masks.items():
        if mask is None:
            continue
        if lead in long_leads:
            first, last = 0, NUM_COLUMNS
        elif lead in STANDARD_LEAD_OFFSETS_SEC:
            first = int(STANDARD_LEAD_OFFSETS_SEC[lead] / SHORT_SIGNAL_LENGTH_SEC)
            last = first + 1
        else:
            continue
        x1 = float(signal_positions[lead]["x1"])
        edges.append((x1, first * GRID_LINES_PER_COLUMN))
        edges.append((x1 + mask.shape[2], last * GRID_LINES_PER_COLUMN))
    return edges


def measure_column_mapping(
    image_rotated,
    g0,
    P,
    edges,
    snap_offset=GRID_LINE_SNAP_OFFSET,
    median_mm=COLUMN_MAPPING_MEDIAN_MM,
):
    """Where every millimetre of the printed grid is along x, from its vertical lines.

    The uniform grid puts every column P pixels wide from g0 on. A printed and scanned
    sheet is not that even: its grid lines, and the traces printed with them, are pushed
    about by up to a few pixels within a column, and a stretch of it can be a few per cent
    narrower. The printed 1 mm lines say where each millimetre is, so the time axis is
    read off them: the median profile of the bands of rows with grid lines, summed as the
    frame is straight, gives in windows about half a line apart the offset of the 1 mm
    lines, which is only known modulo a line, and that of the bold 5 mm lines, which,
    followed along x, names the 1 mm line each window is on. Windows the two offsets do
    not agree on are dropped, and so is a short stretch of windows a whole line off its
    sides; a page whose windows still jump by a line is left alone. The grid coordinate u(x) = (x - offset(x)) / period counts
    the lines, and its origin is the line the mask edges of all leads agree on, each read
    at its own millimetre of that same local grid, so that neither a distorted column
    nor a single late mask start picks it.
    snap_offset is how far right of the traces the lines sit, as in
    refine_grid_from_lines(). g0 and P are the column grid that function refined, P / 62.5
    is the period of the lines. edges is column_mapping_edges().
    median_mm is the width of the running median of the map's displacement from the
    uniform columns (COLUMN_MAPPING_MEDIAN_MM), 0 or less for none; info["smoothing"] is
    how far it moved a knot.
    Returns (x_map, info). x_map(m) is the x of the traces m grid millimetres after the
    start of the first column, in the coordinate of g0 (pixel c covers [c, c+1)). It is
    None when info["reason"] says why the map cannot be trusted, or when
    info["dead_band"] says that it moves no sample the page reads by
    COLUMN_MAPPING_MIN_SHIFT_PX; info["shift"] is that largest move in pixels and
    info["column_shift"] the same per column.
    """
    nan = float("nan")
    info = {
        "shift": nan,
        "windows": 0,
        "coverage": nan,
        "gap": nan,
        "agreement": nan,
        "spikes": 0,
        "islands": 0,
        "jump": nan,
        "origin_offset": nan,
        "origin_move": nan,
        "smoothing": nan,
        "column_shift": [nan] * NUM_COLUMNS,
        "dead_band": False,
        "reason": "",
        "knots_m": None,
        "knots_x": None,
    }
    period = P / GRID_LINES_PER_COLUMN
    profiles, _ = _band_profiles(_darkness(image_rotated))
    if profiles.shape[0] == 0:
        info["reason"] = "image too small for the grid line profile"
        return None, info
    # The frame is straight, so the bands add up coherently; the borders and anything
    # else without grid lines are left out, as in grid_line_slope().
    weights = np.abs(_grid_line_comb(profiles, period))
    kept_bands = weights >= GRID_LINE_MIN_BAND_AMPLITUDE * np.median(weights)
    profile = profiles[kept_bands].sum(axis=0)
    low = g0 - COLUMN_MAPPING_REACH * P
    high = g0 + (NUM_COLUMNS + COLUMN_MAPPING_REACH) * P
    centres, fine, coarse = _column_mapping_windows(profile, period, low, high)
    if centres.size < 2:
        info["reason"] = "image too small for the column map"
        return None, info

    bold = COLUMN_MAPPING_BOLD_EVERY * period
    # A blank window has no phase at all, however weak the others are.
    strong = (
        (np.abs(fine) > 0)
        & (np.abs(coarse) > 0)
        & (np.abs(fine) >= COLUMN_MAPPING_MIN_WINDOW_AMPLITUDE * np.median(np.abs(fine)))
        & (
            np.abs(coarse)
            >= COLUMN_MAPPING_MIN_WINDOW_AMPLITUDE * np.median(np.abs(coarse))
        )
    )
    index = np.flatnonzero(strong)
    if index.size < 2:
        info["reason"] = "no grid lines along the columns"
        return None, info
    # Where the lines are, modulo one line and modulo one bold line.
    fine_offset = (-np.angle(fine[index]) / (2 * np.pi) * period) % period
    coarse_offset = (-np.angle(coarse[index]) / (2 * np.pi) * bold) % bold
    # Neighbouring windows are half a line apart, and the lines move by far less than
    # half a bold period over that, so the bold offset unwraps by continuity along x.
    steps = (np.diff(coarse_offset) + bold / 2) % bold - bold / 2
    coarse_offset = coarse_offset[0] + np.r_[0.0, np.cumsum(steps)]
    # The 1 mm offset on the line the bold lines name: exact to the 1 mm phase, and on
    # the right line wherever the two agree.
    offset = fine_offset + period * np.round((coarse_offset - fine_offset) / period)
    agree = np.abs(offset - coarse_offset) <= COLUMN_MAPPING_BRANCH_TOLERANCE * period
    info["agreement"] = float(np.mean(agree))
    if info["agreement"] < COLUMN_MAPPING_MIN_AGREEMENT:
        info["reason"] = (
            f"5 mm and 1 mm lines agree on {100 * info['agreement']:.0f} % of the windows"
        )
        return None, info
    x, offset = centres[index][agree], offset[agree]
    # A window whose bold offset strayed by a bold period names another line and stands
    # out of its neighbours by that line; a step of the lines themselves does not, as
    # each side of it has two neighbours on its own level.
    if x.size >= 5:
        padded = np.pad(offset, 2, mode="edge")
        neighbours = np.stack(
            [padded[0:-4], padded[1:-3], padded[3:-1], padded[4:]], axis=1
        )
        spikes = np.abs(offset - np.median(neighbours, axis=1)) > (
            COLUMN_MAPPING_BRANCH_TOLERANCE * period
        )
        info["spikes"] = int(np.count_nonzero(spikes))
        x, offset = x[~spikes], offset[~spikes]
    runs = np.split(
        np.arange(x.size),
        np.flatnonzero(np.abs(np.diff(offset)) > COLUMN_MAPPING_RUN_BREAK * period) + 1,
    )
    # The windows are about half a line apart, so a 1 mm window holds twice its lines.
    long_runs = [run for run in runs if run.size >= 2 * COLUMN_MAPPING_FINE_LINES]
    info["islands"] = len(runs) - len(long_runs)
    if not long_runs:
        info["reason"] = "no stretch of grid lines along the columns"
        return None, info
    x, offset = x[np.concatenate(long_runs)], offset[np.concatenate(long_runs)]
    info["windows"] = int(x.size)
    info["jump"] = float(np.max(np.abs(np.diff(offset)), initial=0.0) / period)

    end = g0 + NUM_COLUMNS * P
    inside = (centres >= g0) & (centres <= end)
    kept_inside = np.count_nonzero((x >= g0) & (x <= end))
    info["coverage"] = float(kept_inside / max(np.count_nonzero(inside), 1))
    info["gap"] = float(np.max(np.diff(np.r_[g0, x[(x > g0) & (x < end)], end])) / P)
    if info["coverage"] < COLUMN_MAPPING_MIN_COVERAGE:
        info["reason"] = f"map keeps {100 * info['coverage']:.0f} % of its windows"
        return None, info
    if info["gap"] > COLUMN_MAPPING_MAX_GAP:
        info["reason"] = f"map has a gap of {info['gap']:.2f} columns"
        return None, info
    if info["jump"] > COLUMN_MAPPING_MAX_JUMP:
        info["reason"] = f"grid lines jump by {info['jump']:.2f} lines"
        return None, info
    lines = (x - offset) / period
    if np.any(np.diff(lines) <= 0):
        info["reason"] = "grid coordinate does not grow along x"
        return None, info

    # The origin: the line the mask edges agree on. Each edge is read on the local grid
    # at the millimetre it belongs to, the traces sit snap_offset left of the lines, and
    # the offset is held beyond the outer windows, as _column_map_x() carries it on.
    if not edges:
        info["reason"] = "no mask edges"
        return None, info
    edge_x = np.array([edge[0] for edge in edges], float) + snap_offset
    edge_mm = np.array([edge[1] for edge in edges], float)
    edge_lines = (edge_x - np.interp(edge_x, x, offset)) / period - edge_mm
    consensus = float(np.median(edge_lines))
    origin = np.round(consensus)
    info["origin_offset"] = float((consensus - origin) * period)
    if abs(info["origin_offset"]) > GRID_LINE_SHIFT_TOLERANCE * P:
        info["reason"] = (
            f"mask edges are {info['origin_offset']:+.1f} px off the next grid line"
        )
        return None, info
    knots_m = lines - origin
    knots_x = x - snap_offset
    if median_mm > 0:
        # The displacement from the uniform columns without the ripple of the pixel
        # snap of the lines, which the traces do not share.
        uniform_knots = g0 + knots_m * period
        smoothed = uniform_knots + _running_median(
            knots_m, knots_x - uniform_knots, median_mm
        )
        info["smoothing"] = float(np.max(np.abs(smoothed - knots_x)))
        knots_x = smoothed
        if np.any(np.diff(knots_x) <= 0):
            info["reason"] = "smoothed map does not grow along x"
            return None, info
    info["knots_m"], info["knots_x"] = knots_m, knots_x

    # Every sample the page reads: the rhythm strip covers all columns.
    samples_per_column = SHORT_SIGNAL_LENGTH_SEC * FREQUENCY
    millimetres = (
        np.arange(int(NUM_COLUMNS * samples_per_column))
        / samples_per_column
        * GRID_LINES_PER_COLUMN
    )
    mapped = _column_map_x(knots_m, knots_x, period, millimetres)
    uniform = g0 + millimetres * P / GRID_LINES_PER_COLUMN
    moved = np.abs(mapped - uniform)
    info["shift"] = float(np.max(moved))
    info["column_shift"] = [float(np.max(part)) for part in np.split(moved, NUM_COLUMNS)]
    info["origin_move"] = float(mapped[0] - g0)
    # The dead band is one for the page: a column the map moves by less than a pixel is
    # still read on it, since the uniform columns are fitted to the whole page, the
    # distorted columns included (on dev scans, holding such columns uniform cost
    # column 2 some 1.3 to 1.6 dB in the median against the map).
    if info["shift"] < COLUMN_MAPPING_MIN_SHIFT_PX:
        info["dead_band"] = True
        return None, info

    def x_map(m):
        return _column_map_x(knots_m, knots_x, period, m)

    return x_map, info


def vectorise_grid(
    image_rotated,
    mask,
    position,
    g0,
    P,
    column,
    is_long,
    y_shift_ratio,
    lead,
    scale=1.0,
    ink_map=None,
    info=None,
    x_shift=0.0,
    sharpen="none",
    x_map=None,
):
    """Vectorise one lead by sampling it on the shared column grid.

    ink_map is the ink of the whole page from _ink(), which weighs the mask rows by
    the ink under them; None is the mask on its own. x_shift is the page shift of
    measure_trace_shift(), the pixels the columns of the profile are moved right by.
    sharpen is the --sharpen mode: "none" interpolates the column profile linearly,
    "bandlimited" adds the band of _bandlimited_sharpen() to that.
    x_map is the column map of measure_column_mapping(), the x of the traces at a
    number of grid millimetres after the start of the first column, which then places
    the samples instead of the uniform columns of g0 and P; None keeps those.
    The return value stays the signal, because that is what every caller reads, so
    info, if given, is the dict the ink measurement of this lead is reported in.
    """
    if sharpen not in ("none", "bandlimited"):
        raise ValueError(f"unknown sharpen mode {sharpen!r}")
    total_seconds = LONG_SIGNAL_LENGTH_SEC if is_long else SHORT_SIGNAL_LENGTH_SEC
    y_shift_ratio_ = y_shift_ratio["full"] if is_long else y_shift_ratio[lead]
    values_needed = int(total_seconds * FREQUENCY)
    samples_per_column = SHORT_SIGNAL_LENGTH_SEC * FREQUENCY
    mV_per_pixel = 25 * SHORT_SIGNAL_LENGTH_SEC / P / 10

    # Mean row of the mask in every column, in image coordinates.
    binary = mask[0].numpy() > 0
    count = binary.sum(axis=0)
    filled = np.flatnonzero(count > 0)
    rows = np.arange(binary.shape[0])[:, None]
    profile = position["y1"] + (binary * rows).sum(axis=0)[filled] / count[filled]
    if ink_map is not None:
        profile = _ink_profile(ink_map, binary, position, profile, filled, info)

    # Pixel c covers [c, c+1), so its centre is at c + 0.5. np.interp holds the edges.
    if x_map is None:
        x = g0 + column * P + np.arange(values_needed) * P / samples_per_column - 0.5
    else:
        # A sample is 1/20 mm at 25 mm/s and 500 Hz, a column 62.5 mm.
        millimetres = (
            column + np.arange(values_needed) / samples_per_column
        ) * GRID_LINES_PER_COLUMN
        x = x_map(millimetres) - 0.5
    # The lead is read where its ink is, so the shift moves the columns of the
    # profile and not the sampling grid, which belongs to the printed grid.
    columns = position["x1"] + filled
    if x_shift:
        columns = columns + x_shift
    sampled = np.interp(x, columns, profile)
    if sharpen == "bandlimited":
        sampled = _bandlimited_sharpen(columns, profile, filled, x, sampled, FREQUENCY)
    baseline = baseline_row(y_shift_ratio_, image_rotated.shape[1], scale)

    # Pixel row r covers [r, r+1) as well, so the sampled trace sits at row + 0.5.
    return torch.from_numpy(
        ((baseline - (sampled + 0.5)) * mV_per_pixel).astype(np.float32)
    )


def estimate_baseline_shift(signals_predicted, long_leads, mV_per_pixel, P, record=""):
    """Baseline error of a record in pixels, from the limb lead sum rules.

    Goldberger (aVR + aVL + aVF = 0) and Einthoven (I + III - II = 0) hold sample
    wise, so with every lead read a px too low the medians of the two sums are
    -3 a m and -a m. Returns (shift, disagreement of the two estimates) in pixels,
    (0.0, nan) if a rule cannot be evaluated.
    """
    samples_per_column = int(SHORT_SIGNAL_LENGTH_SEC * FREQUENCY)

    def rule_median(leads, coefficients):
        # The first lead is short and names the column the whole rule lives in.
        column = int(STANDARD_LEAD_OFFSETS_SEC[leads[0]] / SHORT_SIGNAL_LENGTH_SEC)
        windows = []
        for lead in leads:
            signal = signals_predicted.get(lead)
            if signal is None:
                return np.nan
            signal = np.asarray(signal, dtype=float)
            if lead in long_leads:
                start = column * samples_per_column
                signal = signal[start : start + samples_per_column]
            if signal.shape != (samples_per_column,):
                return np.nan
            windows.append(signal)
        residual = sum(c * window for c, window in zip(coefficients, windows))
        if not np.any(np.isfinite(residual)):
            return np.nan
        return float(np.nanmedian(residual))

    einthoven = rule_median(["I", "III", "II"], [1.0, 1.0, -1.0])
    goldberger = rule_median(["aVR", "aVL", "aVF"], [1.0, 1.0, 1.0])
    if not np.isfinite(einthoven) or not np.isfinite(goldberger):
        return 0.0, float("nan")

    from_goldberger = -goldberger / (3 * mV_per_pixel)
    from_einthoven = -einthoven / mV_per_pixel
    disagreement = from_goldberger - from_einthoven
    if abs(disagreement) > BASELINE_DISAGREEMENT_TOLERANCE * P:
        print(
            f"WARNING: baseline estimates disagree for record {record}: "
            f"Goldberger {from_goldberger:.2f} px, Einthoven {from_einthoven:.2f} px, "
            f"keeping the page baseline."
        )
        return 0.0, float(disagreement)

    # Least squares over both rules, whose residuals weigh 3 a m and a m.
    shift = -(3 * goldberger + einthoven) / (10 * mV_per_pixel)
    return float(shift), float(disagreement)


def compute_grid_lead_offsets(signal_positions, signal_lengths, long_leads, g0, P):
    """Get the time offset of every lead from the column grid it was sampled on."""
    offsets = {}
    for lead in signal_lengths:
        if lead in long_leads:
            offsets[lead] = {"raw": 0.0, "snapped": 0.0}
            continue
        raw = (signal_positions[lead]["x1"] - g0) * SHORT_SIGNAL_LENGTH_SEC / P
        offsets[lead] = {
            "raw": float(raw),
            "snapped": float(STANDARD_LEAD_OFFSETS_SEC[lead]),
        }
    return offsets


def max_mask_gap(mask, x1=0, window=None):
    """Longest run of empty columns inside the cropped mask of one lead, in pixels.

    x1 is the image x of the first column of the mask. With a window (start, end)
    in image coordinates only the part of a run inside the window counts, as that
    is all the grid sampling reads: a stray label beside the lead is no gap by
    itself, but the sampling does interpolate over the empty columns leading to it.
    """
    filled = np.flatnonzero(mask[0].numpy().sum(axis=0) > 0)
    if filled.size < 2:
        return 0
    if window is None:
        return int(np.max(np.diff(filled)) - 1)
    # Pixel c covers [c, c+1), so an empty run reaches from the right edge of the
    # filled column before it to the left edge of the one after, and of that only
    # the whole pixels inside the window are sampled.
    overlaps = np.minimum(x1 + filled[1:], window[1]) - np.maximum(
        x1 + filled[:-1] + 1, window[0]
    )
    return int(max(np.max(overlaps), 0.0))


def mask_overhang(mask, x1, window):
    """Largest distance in pixels of a filled column beyond the window of a lead."""
    filled = np.flatnonzero(mask[0].numpy().sum(axis=0) > 0)
    if filled.size == 0:
        return 0.0
    centres = x1 + filled + 0.5
    return float(max(window[0] - centres[0], centres[-1] - window[1], 0.0))


def compute_lead_offsets(signal_positions, signal_lengths, sec_per_pixel, record=""):
    """Get the time offset in seconds of every predicted lead."""
    long_length = LONG_SIGNAL_LENGTH_SEC * FREQUENCY
    max_offset = LONG_SIGNAL_LENGTH_SEC - SHORT_SIGNAL_LENGTH_SEC
    long_leads = [
        lead for lead, length in signal_lengths.items() if length >= long_length
    ]

    # Without a rhythm strip there is no reference, so fall back to the layout table.
    if not long_leads:
        print(
            f"No rhythm lead found for record {record}, "
            f"falling back to the standard lead offsets."
        )
        return {
            lead: {
                "raw": float("nan"),
                "snapped": float(STANDARD_LEAD_OFFSETS_SEC.get(lead, 0.0)),
            }
            for lead in signal_lengths
        }

    x1_ref = min(signal_positions[lead]["x1"] for lead in long_leads)

    offsets = {}
    for lead in signal_lengths:
        if lead in long_leads:
            offsets[lead] = {"raw": 0.0, "snapped": 0.0}
            continue
        raw = (signal_positions[lead]["x1"] - x1_ref) * sec_per_pixel
        snapped = round(raw / SHORT_SIGNAL_LENGTH_SEC) * SHORT_SIGNAL_LENGTH_SEC
        clamped = float(min(max(snapped, 0.0), max_offset))
        if abs(raw - snapped) > 0.5 or clamped != snapped:
            print(
                f"Suspicious lead offset for record {record}, lead {lead}: "
                f"raw {raw:.3f} s, snapped {clamped:.3f} s."
            )
        offsets[lead] = {"raw": float(raw), "snapped": clamped}

    return offsets


def assemble_signals(signals, offsets, num_samples, placement):
    """Put the single lead signals into one array of shape [num_samples, n_leads]."""
    signal_list = []
    for lead, signal in signals.items():
        if placement == "start":
            if len(signal) < num_samples:
                nan_signal = np.empty(num_samples)
                nan_signal[:] = np.nan
                nan_signal[: int(len(signal))] = signal
                signal_list.append(nan_signal)
            else:
                signal_list.append(signal)
        else:
            snapped = offsets.get(lead, {}).get("snapped", 0.0)
            start = int(round(snapped * FREQUENCY))
            nan_signal = np.empty(num_samples)
            nan_signal[:] = np.nan
            end = min(start + len(signal), num_samples)
            if end > start:
                nan_signal[start:end] = signal[: end - start]
            signal_list.append(nan_signal)

    sig_names = list(signals.keys())
    return np.array(signal_list).T, sig_names


# Einthoven ratio (the RMS of I + III - II over the mean RMS of the three leads) at or
# above which a page is reported for a check. Leads I and III are read in the first
# column only, so the ratio covers at most its 2.5 s, nothing of the other columns and
# nothing of the rhythm strip II after them. Fitted on the development pages of the
# ECG-Image-Database (see the README): the highest threshold that flags at least 90 %
# of the clean scans with a page SNR below 12 dB, 0.17976, rounded down.
EINTHOVEN_RATIO_WARNING = 0.1797


def _residual_qc(signals, sig_names, leads, coefficients):
    """Residual statistics of one lead consistency rule."""
    nan_result = {"rms": np.nan, "rms_demedian": np.nan, "ratio": np.nan, "n": 0}
    if any(lead not in sig_names for lead in leads):
        return nan_result

    columns = [signals[:, sig_names.index(lead)] for lead in leads]
    valid = np.all([np.isfinite(column) for column in columns], axis=0)
    n = int(np.sum(valid))
    if n == 0:
        return nan_result

    columns = [column[valid] for column in columns]
    residual = sum(c * column for c, column in zip(coefficients, columns))
    rms = float(np.sqrt(np.mean(residual**2)))
    rms_demedian = float(np.sqrt(np.mean((residual - np.median(residual)) ** 2)))
    denominator = float(np.mean([np.sqrt(np.mean(column**2)) for column in columns]))
    ratio = float(rms / denominator) if denominator != 0 else np.nan

    return {"rms": rms, "rms_demedian": rms_demedian, "ratio": ratio, "n": n}


def compute_consistency_qc(signals, sig_names):
    """Einthoven and Goldberger consistency of the assembled signals."""
    einthoven = _residual_qc(signals, sig_names, ["I", "III", "II"], [1.0, 1.0, -1.0])
    goldberger = _residual_qc(
        signals, sig_names, ["aVR", "aVL", "aVF"], [1.0, 1.0, 1.0]
    )
    return {
        "einthoven_rms": einthoven["rms"],
        "einthoven_rms_demedian": einthoven["rms_demedian"],
        "einthoven_ratio": einthoven["ratio"],
        "einthoven_n": einthoven["n"],
        "goldberger_rms": goldberger["rms"],
        "goldberger_rms_demedian": goldberger["rms_demedian"],
        "goldberger_ratio": goldberger["ratio"],
        "goldberger_n": goldberger["n"],
    }


def einthoven_warning(qc, record, threshold=EINTHOVEN_RATIO_WARNING):
    """WARNING line of a page whose leads I, II and III disagree, None if they agree.

    qc is the dict of compute_consistency_qc(). A page without a ratio cannot be
    checked and is reported as well.
    """
    ratio = qc["einthoven_ratio"]
    if np.isnan(ratio):
        if qc["einthoven_n"] == 0:
            reason = "no sample with leads I, II and III all read"
        else:
            reason = "leads I, II and III are zero on their common samples"
        return (
            f"WARNING: Einthoven check failed for record {record} "
            f"(no ratio: {reason}), check this page."
        )
    if ratio >= threshold:
        return (
            f"WARNING: Einthoven check failed for record {record} "
            f"(ratio {ratio:.3f} >= {threshold:.4g}): leads I, II and III disagree, "
            f"check this page."
        )
    return None


def write_record(record, signals, sig_names, output_folder, placement):
    """Write the signals to a WFDB record."""
    kwargs = dict(
        fs=FREQUENCY,
        units=[SIGNAL_UNITS] * signals.shape[1],
        sig_name=sig_names,
        write_dir=output_folder,
        fmt=[FMT] * signals.shape[1],
        adc_gain=[ADC_GAIN] * signals.shape[1],
        baseline=[BASELINE] * signals.shape[1],
    )
    if placement == "start":
        wfdb.wrsamp(record, p_signal=np.nan_to_num(signals), **kwargs)
    else:
        # wfdb writes NaN as the fmt-16 sentinel, but warns while casting.
        with warnings.catch_warnings():
            warnings.filterwarnings(
                "ignore",
                message="invalid value encountered in cast",
                category=RuntimeWarning,
            )
            with np.errstate(invalid="ignore"):
                wfdb.wrsamp(record, p_signal=signals, **kwargs)


def save_mask_files(
    mask, record, output_folder, rot_angle, homography=None, scale=1.0
):
    """Save the predicted mask as PNG plus a small JSON with the frame info.

    homography is the map from the image to the frame the mask lives in, written out
    only when the page was warped: a file without one means the rotation alone.
    scale is what --resolution resampled the page by before all of that, written out
    only when it did: a file without one means the page at the size it came in.
    """
    mask_to_save = mask.to(torch.uint8)
    write_png(mask_to_save, os.path.join(output_folder, f"{record}_mask.png"))
    meta = {
        "rot_angle": float(rot_angle),
        "height": int(mask_to_save.shape[1]),
        "width": int(mask_to_save.shape[2]),
    }
    if homography is not None:
        meta["homography"] = np.asarray(homography).tolist()
    # NaN is the stage being off, 1.0 the page having been left at its own scale.
    if np.isfinite(scale) and scale != 1.0:
        meta["scale"] = float(scale)
    with open(os.path.join(output_folder, f"{record}_mask.json"), "w") as f:
        json.dump(meta, f)


def append_qc_row(output_folder, record, placement, qc, max_offset_deviation):
    """Append one QC row to qc.csv, writing the header if needed."""
    qc_path = os.path.join(output_folder, "qc.csv")
    fieldnames = [
        "record",
        "placement",
        "sharpen",
        "einthoven_rms",
        "einthoven_rms_demedian",
        "einthoven_ratio",
        "einthoven_n",
        "goldberger_rms",
        "goldberger_rms_demedian",
        "goldberger_ratio",
        "goldberger_n",
        "max_offset_deviation",
        "baseline_scale",
        "baseline_shift_px",
        "baseline_disagreement_px",
        "grid_line_contrast",
        "grid_line_shift_px",
        "rotation_angle",
        "rotation_coarse",
        "rotation_residual_px",
        "perspective_shift_px",
        "perspective_residual_px",
        "grid_period_px",
        "resolution_scale",
        "trace_estimator",
        "ink_leads",
        "ink_p95",
        "trace_shift_px",
        "trace_shift_rows",
        "column_mapping",
        "column_mapping_shift_px",
    ]
    write_header = not os.path.exists(qc_path)
    with open(qc_path, "a", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        if write_header:
            writer.writeheader()
        row = {"record": record, "placement": placement, **qc}
        row["max_offset_deviation"] = max_offset_deviation
        writer.writerow(row)


def save_plot_masks_and_signals(
    image, masks_cropped, mask_start_position, signals, sig_names, output_folder, filename="record.png"
):
    try:
        num_signals = signals.shape[1]
    except IndexError:
        print("No signals to plot.")
        print(f"Signals: {signals}")
        print(f"Image shape: {image.shape}")
        return
    fig, axs = plt.subplots(
        1 + num_signals, 1, 
        figsize=(10, 2.5 * (1 + num_signals)),
        gridspec_kw={'height_ratios': [4] + [1] * num_signals}
    )

    if hasattr(image, "numpy"):
        image = image.numpy()
    if image.ndim == 3 and image.shape[0] == 1:
        image = image.squeeze(0)
    if image.ndim == 3 and image.shape[0] in [3, 4]:
        image = image.transpose(1, 2, 0)

    mask_combined = np.zeros_like(image, dtype=np.uint8) if image.ndim == 2 else np.zeros(image.shape[:2], dtype=np.uint8)
    for lead, mask_cropped in masks_cropped.items():
        if mask_cropped is not None:
            if mask_cropped.ndim == 3 and mask_cropped.shape[0] == 1:
                mask_cropped = mask_cropped.squeeze(0)
            start_row = mask_start_position[lead]["y1"]
            start_col = mask_start_position[lead]["x1"]
            mask_height, mask_width = mask_cropped.shape
            mask_combined[start_row:start_row + mask_height, start_col:start_col + mask_width] = np.maximum(
                mask_combined[start_row:start_row + mask_height, start_col:start_col + mask_width],
                mask_cropped
            )

    axs[0].imshow(image, cmap="gray" if image.ndim == 2 else None)
    axs[0].imshow(mask_combined, cmap="jet", alpha=0.5)
    axs[0].set_title("Masks overlayed on image")
    axs[0].axis("off")

    time_axis = np.arange(signals.shape[0])
    for i, signal in enumerate(signals.T):
        axs[i + 1].plot(time_axis, signal)
        axs[i + 1].set_title(sig_names[i])
        axs[i + 1].set_xlabel("Time")
        axs[i + 1].set_ylabel("Signal amplitude")
        axs[i + 1].grid()

    plt.tight_layout()
    os.makedirs(output_folder, exist_ok=True)
    plt.savefig(os.path.join(output_folder, filename), dpi=300)
    plt.close(fig)


# Run the code.
def run(args):
    # Create a folder for the Challenge outputs if it does not already exist.
    os.makedirs(args.output_folder, exist_ok=True)

    # Run the team's models on the Challenge data.
    if args.verbose:
        print("Running digitization model...")

    # Iterate over the records.
    image_files = [
        f for f in os.listdir(args.data_folder) if f.endswith(f".{IMAGE_TYPE}")
    ]
    for _, image_file in tqdm(enumerate(image_files), total=len(image_files)):
        # Get record and header files.
        image_file_path = os.path.join(args.data_folder, image_file)
        record = image_file.replace(f".{IMAGE_TYPE}", "")
        os.makedirs(args.output_folder, exist_ok=True)
        image = read_image(image_file_path)
        image = image[:3]

        # Rescale
        resolution_info = {"period": float("nan"), "scale": float("nan")}
        if args.resolution == "lines":
            image, resolution_info = normalise_resolution(image)
            if resolution_info["reason"]:
                print(
                    f"WARNING: grid lines not used for the resolution of "
                    f"record {record} ({resolution_info['reason']}), keeping "
                    f"the page as it is."
                )
            if args.verbose:
                print(
                    f"Resolution for record {record}: period "
                    f"{resolution_info['period']:.3f} px, harmonic "
                    f"{resolution_info['harmonic']}, scale "
                    f"{resolution_info['scale']:.4f}, size "
                    f"{image.shape[2]} x {image.shape[1]} px"
                )

        # Rotate
        rot_angle, rotation_info = estimate_rotation(image, args.rotation)
        if args.rotation == "lines":
            if rotation_info["reason"]:
                print(
                    f"WARNING: grid lines not used for the rotation of record {record} "
                    f"({rotation_info['reason']}), keeping the whole degree Hough angle."
                )
            if args.verbose:
                deltas = ", ".join(f"{d:+.3f}" for d in rotation_info["deltas"])
                print(
                    f"Rotation for record {record}: {rot_angle} deg, coarse "
                    f"{rotation_info['coarse']} deg, deltas [{deltas}], "
                    f"period {rotation_info['period']:.2f} px, "
                    f"contrast {rotation_info['contrast']:.1f}, "
                    f"residual {rotation_info['residual']:.3f} px"
                )
        if rot_angle is None or np.isnan(rot_angle):
            print(
                f"No rotation angle found for record {record}, using 0.0 degrees instead."
            )
            rot_angle = 0.0
        rotation = rotation_homography(rot_angle, image.shape[2], image.shape[1])
        # Both interpolations put the page in the same frame, only the resampling
        # differs; a straight page keeps its pixels and is never warped.
        if args.interpolation == "bicubic" and rot_angle != 0.0:
            image_rotated = warp_page(image, rotation)
        else:
            image_rotated = rotate(image, rot_angle)

        # Rectify
        homography = None
        perspective_info = {"shift": float("nan"), "residual": float("nan")}
        if args.perspective == "lines":
            # The page has to be measured in the frame of the final warp, because
            # rotate() interpolates nearest, which below a twentieth of a degree moves
            # no pixel at all. The bicubic page already is that frame and the warp of a
            # straight page gives its own pixels back, so neither is warped twice.
            if rot_angle == 0.0:
                measured = image
            elif args.interpolation == "bicubic":
                measured = image_rotated
            else:
                measured = warp_page(image, rotation)
            H_rect, perspective_info = estimate_perspective(measured)
            if perspective_info["reason"]:
                print(
                    f"WARNING: grid lines not used for the perspective of record "
                    f"{record} ({perspective_info['reason']}), keeping the rotated page."
                )
            if args.verbose:
                print(
                    f"Perspective for record {record}: shift "
                    f"{perspective_info['shift']:.2f} px, residual "
                    f"{perspective_info['residual']:.3f} px, period "
                    f"{perspective_info['period']:.2f} px, "
                    f"windows {perspective_info['windows']}"
                )
            if H_rect is not None:
                # From the image, so that the page is interpolated only once.
                homography = H_rect @ rotation
                image_rotated = warp_page(image, homography)

        # Segment
        if args.mask_folder is not None:
            # A mask only fits the frame it was predicted in, so check its angle.
            check_mask_rotation(args.mask_folder, record, rot_angle, homography)
            mask_path = os.path.join(args.mask_folder, f"{record}_mask.png")
            if not os.path.exists(mask_path):
                raise FileNotFoundError(
                    f"No mask found for record {record} at {mask_path}."
                )
            mask_to_use = read_image(mask_path)
            # A mask of another size cannot be laid over this page at all, and the
            # only thing that changes the size of a page is the resolution stage.
            if mask_to_use.shape[1:] != image_rotated.shape[1:]:
                raise ValueError(
                    f"Mask of record {record} is {mask_to_use.shape[2]} x "
                    f"{mask_to_use.shape[1]} px, the page is "
                    f"{image_rotated.shape[2]} x {image_rotated.shape[1]} px; "
                    f"the mask was predicted for another --resolution."
                )
        else:
            mask_to_use = predict_mask_nnunet(
                image_rotated,
                DATASET_NAME,
                args.model_folder,
                device=args.device,
                disable_tta=not args.enable_tta,
                fold=args.fold,
            )
        if args.save_mask:
            save_mask_files(
                mask_to_use,
                record,
                args.output_folder,
                rot_angle,
                homography,
                resolution_info["scale"],
            )

        # Use mask to cut into single, binary masks
        signal_masks_cropped, signal_positions_cropped, _ = cut_binary(
            mask_to_use, image_rotated
        )

        # Vecotrise
        x_pixel_list = [
            v.shape[2] for v in signal_masks_cropped.values() if v is not None
        ]
        x_pixel_list_median = np.median(x_pixel_list)
        x_pixel_list_below_2x_median_mean = np.mean(
            [v for v in x_pixel_list if v < 2 * x_pixel_list_median]
        )
        sec_per_pixel = 2.5 / x_pixel_list_below_2x_median_mean
        g0, P, long_leads = None, None, []
        grid_lines = {"contrast": float("nan"), "shift": float("nan")}
        baseline_scale = 1.0
        if args.time_mapping == "grid":
            g0, P, long_leads, reason = fit_column_grid(
                signal_masks_cropped,
                signal_positions_cropped,
                image_rotated.shape[1],
                args.grid_pitch,
                record,
            )
            if g0 is None:
                print(
                    f"WARNING: no column grid for record {record} ({reason}), "
                    f"falling back to --time_mapping bbox."
                )
            else:
                if args.grid_origin == "lines":
                    g0, P, grid_lines = refine_grid_from_lines(
                        image_rotated, g0, P, snap_offset=args.grid_line_offset
                    )
                    if grid_lines["reason"]:
                        print(
                            f"WARNING: grid lines not used for record {record} "
                            f"({grid_lines['reason']}), keeping the grid of the masks."
                        )
                sec_per_pixel = SHORT_SIGNAL_LENGTH_SEC / P
                # A cropped page keeps its size, so its rows are rescaled with the pitch.
                baseline_scale = P / (image_rotated.shape[1] * PAGE_PITCH_RATIO)
                if args.verbose:
                    print(f"Column grid for record {record}: g0 {g0:.2f} px, P {P:.2f} px")
        mm_per_pixel = 25 * sec_per_pixel
        mV_per_pixel = mm_per_pixel / 10
        # The ink of the page, read once for all its leads. Only the grid sampling
        # knows what to do with it, so a page that falls back to bbox is left alone.
        ink_map = None
        if args.trace_estimator == "ink" and g0 is not None:
            ink_map = _ink(image_rotated)
        ink_measured = []
        # One shift for the whole page, measured on the same ink.
        trace_shift, trace_shift_rows, x_shift = float("nan"), 0, 0.0
        if args.trace_shift == "page" and g0 is not None:
            page_ink = _ink(image_rotated) if ink_map is None else ink_map
            measured, shift_info = measure_trace_shift(
                mask_to_use[0].numpy(), page_ink
            )
            trace_shift = trace_shift_px(measured, args.trace_estimator)
            trace_shift_rows = shift_info["rows"]
            x_shift = trace_shift
            if args.verbose:
                dead = ""
                if trace_shift == 0.0 and trace_shift_rows >= TRACE_SHIFT_MIN_ROWS:
                    dead = (
                        f" (median {shift_info['median']:+.2f} px, "
                        f"inside the dead band)"
                    )
                print(
                    f"Trace shift for record {record}: {trace_shift:+.2f} px from "
                    f"{trace_shift_rows} rows{dead}"
                )
        # The time axis off the printed grid lines, on a page whose columns they refined.
        # A page without a column grid or without grid lines has been warned about above.
        x_map, column_mapping, column_mapping_shift = None, "uniform", float("nan")
        if args.column_mapping == "lines":
            if g0 is None:
                column_mapping = "uniform: no column grid"
            elif args.grid_origin != "lines":
                column_mapping = "uniform: grid origin from the masks"
            elif grid_lines["reason"]:
                column_mapping = "uniform: grid lines not used"
            if column_mapping != "uniform":
                if args.verbose:
                    print(f"Column mapping for record {record}: {column_mapping}")
            else:
                x_map, map_info = measure_column_mapping(
                    image_rotated,
                    g0,
                    P,
                    column_mapping_edges(
                        signal_masks_cropped, signal_positions_cropped, long_leads
                    ),
                    snap_offset=args.grid_line_offset,
                    median_mm=args.column_mapping_median,
                )
                column_mapping_shift = map_info["shift"]
                if map_info["reason"]:
                    column_mapping = f"uniform: {map_info['reason']}"
                    print(
                        f"WARNING: column mapping not used for record {record} "
                        f"({map_info['reason']}), keeping the uniform grid."
                    )
                elif x_map is None:
                    column_mapping = "uniform: dead band"
                else:
                    column_mapping = "lines"
                if args.verbose:
                    print(
                        f"Column mapping for record {record}: {column_mapping}, shift "
                        f"{map_info['shift']:.2f} px, windows {map_info['windows']}, "
                        f"coverage {map_info['coverage']:.3f}, gap "
                        f"{map_info['gap']:.3f} columns, agreement "
                        f"{map_info['agreement']:.3f}, spikes {map_info['spikes']}, "
                        f"islands {map_info['islands']}, jump "
                        f"{map_info['jump']:.2f} lines, "
                        f"origin offset {map_info['origin_offset']:+.2f} px, origin "
                        f"move {map_info['origin_move']:+.2f} px, smoothing "
                        f"{map_info['smoothing']:.2f} px, column shifts "
                        + "/".join(f"{s:.2f}" for s in map_info["column_shift"])
                        + " px"
                    )
        signals_predicted = {}
        for lead, mask in signal_masks_cropped.items():
            if mask is None:
                signals_predicted[lead] = None
            elif g0 is not None:
                # The rhythm strip spans all columns, a short lead only its own.
                if lead in long_leads:
                    column, n_columns = 0, NUM_COLUMNS
                else:
                    column = int(
                        STANDARD_LEAD_OFFSETS_SEC[lead] / SHORT_SIGNAL_LENGTH_SEC
                    )
                    n_columns = 1
                window = (g0 + column * P, g0 + (column + n_columns) * P)
                x1 = signal_positions_cropped[lead]["x1"]
                # The grid sampling interpolates over gaps, so report them, as far as
                # they reach into the window it reads the lead over.
                gap = max_mask_gap(mask, x1, window)
                if gap > GRID_GAP_TOLERANCE * P:
                    print(
                        f"WARNING: lead {lead} of record {record} has a gap of "
                        f"{gap} px in its mask, interpolated linearly."
                    )
                if args.verbose:
                    overhang = mask_overhang(mask, x1, window)
                    if overhang > GRID_RESIDUAL_TOLERANCE * P:
                        print(
                            f"Lead {lead} of record {record} has mask pixels up to "
                            f"{overhang:.0f} px outside its column, ignored."
                        )
                ink = {}
                signals_predicted[lead] = vectorise_grid(
                    image_rotated,
                    mask,
                    signal_positions_cropped[lead],
                    g0,
                    P,
                    column,
                    lead in long_leads,
                    Y_SHIFT_RATIO,
                    lead,
                    baseline_scale,
                    ink_map=ink_map,
                    info=ink,
                    x_shift=x_shift,
                    sharpen=args.sharpen,
                    x_map=x_map,
                )
                if ink:
                    ink_measured.append(ink)
            else:
                signals_predicted[lead] = vectorise(
                    image_rotated,
                    mask,
                    signal_positions_cropped[lead]["y1"],
                    sec_per_pixel,
                    mV_per_pixel,
                    Y_SHIFT_RATIO,
                    lead,
                )

        # One number for the ink of the page: the median over the leads that were
        # measured, the gated ones included, so that a page that is too pale shows.
        ink_leads = sum(ink["ink_used"] for ink in ink_measured)
        ink_p95 = float("nan")
        if ink_measured:
            ink_p95 = float(np.median([ink["ink_p95"] for ink in ink_measured]))
        if args.trace_estimator == "ink" and args.verbose:
            print(
                f"Trace estimator for record {record}: ink on {ink_leads}/"
                f"{len(ink_measured)} leads, ink p95 {ink_p95:.0f}"
            )

        # Put the baseline of the whole record where the limb lead sums vanish.
        baseline_shift, baseline_disagreement = 0.0, float("nan")
        if g0 is not None and args.baseline == "leads":
            baseline_shift, baseline_disagreement = estimate_baseline_shift(
                signals_predicted, long_leads, mV_per_pixel, P, record
            )
            for lead, signal in signals_predicted.items():
                if signal is not None:
                    signals_predicted[lead] = signal + baseline_shift * mV_per_pixel
            if args.verbose:
                print(
                    f"Baseline for record {record}: scale {baseline_scale:.4f}, "
                    f"shift {baseline_shift:.2f} px, "
                    f"disagreement {baseline_disagreement:.2f} px"
                )

        # Save Challenge outputs.
        signals = {
            signal_name: signals_predicted[signal_name].numpy()
            for signal_name in LEAD_LABEL_MAPPING.keys()
            if signals_predicted[signal_name] is not None
        }
        num_samples = int(LONG_SIGNAL_LENGTH_SEC * FREQUENCY)
        signal_lengths = {lead: len(signal) for lead, signal in signals.items()}
        if g0 is not None:
            # Place every lead in the column it was sampled from.
            offsets = compute_grid_lead_offsets(
                signal_positions_cropped, signal_lengths, long_leads, g0, P
            )
        else:
            offsets = compute_lead_offsets(
                signal_positions_cropped, signal_lengths, sec_per_pixel, record
            )
        signals, sig_names = assemble_signals(
            signals, offsets, num_samples, args.lead_placement
        )
        if args.lead_placement == "column" and args.verbose:
            offsets_string = ", ".join(
                f"{lead} {offset['snapped']:.1f}" for lead, offset in offsets.items()
            )
            print(f"Offsets for record {record}: {offsets_string}")

        # Check if signal is empty
        if signals.shape[0] == 0:
            print(f"=========== Signal is empty for record {record}. ===========")
            if args.allow_failures:
                continue
            else:
                raise ValueError(f"Signal is empty for record {record}.")

        # Quality control of the assembled signals.
        qc = compute_consistency_qc(signals, sig_names)
        print(
            f"QC {record}: "
            f"Einthoven RMS {qc['einthoven_rms']:.4f} mV "
            f"(demedian {qc['einthoven_rms_demedian']:.4f}, "
            f"ratio {qc['einthoven_ratio']:.2f}, n={qc['einthoven_n']}) | "
            f"Goldberger RMS {qc['goldberger_rms']:.4f} mV "
            f"(demedian {qc['goldberger_rms_demedian']:.4f}, "
            f"ratio {qc['goldberger_ratio']:.2f}, n={qc['goldberger_n']})"
        )
        warning_line = einthoven_warning(qc, record)
        if warning_line is not None:
            print(warning_line)
        deviations = [
            abs(offset["raw"] - offset["snapped"])
            for offset in offsets.values()
            if np.isfinite(offset["raw"])
        ]
        max_offset_deviation = max(deviations) if deviations else np.nan
        qc["baseline_scale"] = baseline_scale
        qc["baseline_shift_px"] = baseline_shift
        qc["baseline_disagreement_px"] = baseline_disagreement
        qc["grid_line_contrast"] = grid_lines["contrast"]
        qc["grid_line_shift_px"] = grid_lines["shift"]
        qc["rotation_angle"] = rot_angle
        qc["rotation_coarse"] = rotation_info["coarse"]
        qc["rotation_residual_px"] = rotation_info["residual"]
        qc["perspective_shift_px"] = perspective_info["shift"]
        qc["perspective_residual_px"] = perspective_info["residual"]
        qc["grid_period_px"] = resolution_info["period"]
        qc["resolution_scale"] = resolution_info["scale"]
        qc["trace_estimator"] = args.trace_estimator
        qc["sharpen"] = args.sharpen
        qc["ink_leads"] = ink_leads
        qc["ink_p95"] = ink_p95
        qc["trace_shift_px"] = trace_shift
        qc["trace_shift_rows"] = trace_shift_rows
        qc["column_mapping"] = column_mapping
        qc["column_mapping_shift_px"] = column_mapping_shift
        append_qc_row(
            args.output_folder, record, args.lead_placement, qc, max_offset_deviation
        )

        # Plot and save the image with the masks and signals.
        if args.show_image:
            print(f"Storing image of shape {image_rotated.shape}")
            save_plot_masks_and_signals(
                image_rotated,
                signal_masks_cropped,
                signal_positions_cropped,
                signals,
                sig_names,
                args.output_folder,
                f"{record}.png",
            )

        # Save the signals to a WFDB record.
        if args.verbose:
            print(f"Storing signals for record {record} with shape {signals.shape}")
        if (np.nanmax(signals) > 10) or (np.nanmin(signals) < -10):
            print(f"Signal out of range for record {record}, normalizing to range between 1 and -1")
            max_val = np.nanmax(signals)
            min_val = np.nanmin(signals)
            signals = (signals - min_val) / (max_val - min_val) * 2 - 1
        write_record(
            record, signals, sig_names, args.output_folder, args.lead_placement
        )

    if args.verbose:
        print("Done.")


if __name__ == "__main__":
    run(get_parser().parse_args(sys.argv[1:]))
