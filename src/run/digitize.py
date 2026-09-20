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
        default="hough",
        help=(
            "hough = angle from the Hough transform of the page, in whole degrees; "
            "lines = a 0.1 degree Hough transform refined with the phase drift of the "
            "printed 1 mm grid lines, which resolves fractions of a degree."
        ),
    )
    parser.add_argument(
        "--save_mask",
        action="store_true",
        default=False,
        help="Save the predicted label mask (rotation-corrected frame) as PNG plus JSON.",
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
        for rho, theta in filtered_lines:
            count = 0
            for comp_rho, comp_theta in filtered_lines:
                if (
                    abs(theta - comp_theta) < parallelism_radian
                    or abs((theta - comp_theta) - np.pi) < parallelism_radian
                ):
                    count += 1
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
    # Without a coarse angle the grid lines still measure a tilt of a few degrees.
    total = 0.0 if np.isnan(coarse) else coarse
    for _ in range(2):
        lines = grid_line_slope(rotate(image, total), axis=0)
        info["period"] = lines["period"]
        info["contrast"] = lines["contrast"]
        info["residual"] = lines["residual"]
        if lines["reason"]:
            info["reason"] = lines["reason"]
            return coarse, info
        delta = lines["angle"]
        if abs(delta) > ROTATION_REFINE_MAX_DEG:
            info["reason"] = f"grid lines ask for {delta:+.2f} degrees"
            return coarse, info
        info["deltas"].append(float(delta))
        total += delta
        if abs(delta) <= ROTATION_REFINE_REPEAT_DEG:
            break

    total = round(total, 4)
    if not np.isnan(coarse) and abs(total - coarse) < ROTATION_REFINE_MIN_DEG:
        return coarse, info
    return total, info


def check_mask_rotation(mask_folder, record, rot_angle):
    """Warn when a saved mask was predicted at another rotation angle than now."""
    meta_path = os.path.join(mask_folder, f"{record}_mask.json")
    if not os.path.exists(meta_path):
        return float("nan")
    with open(meta_path) as f:
        saved = json.load(f).get("rot_angle")
    if saved is None:
        return float("nan")
    if abs(saved - rot_angle) > ROTATION_MASK_ANGLE_TOLERANCE:
        print(
            f"WARNING: mask of record {record} was predicted at rot_angle "
            f"{saved:.4f}, the image is now rotated by {rot_angle:.4f}; "
            f"the mask does not fit."
        )
    return float(saved)


def baseline_row(ratio, image_height, scale=1.0):
    """Row of the zero line of one lead, for a page rescaled about its centre."""
    centre = image_height / 2
    return centre + scale * ((1 - ratio) * image_height - centre)


def vectorise_grid(
    image_rotated, mask, position, g0, P, column, is_long, y_shift_ratio, lead, scale=1.0
):
    """Vectorise one lead by sampling it on the shared column grid."""
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

    # Pixel c covers [c, c+1), so its centre is at c + 0.5. np.interp holds the edges.
    x = g0 + column * P + np.arange(values_needed) * P / samples_per_column - 0.5
    sampled = np.interp(x, position["x1"] + filled, profile)
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


def save_mask_files(mask, record, output_folder, rot_angle):
    """Save the predicted mask as PNG plus a small JSON with the frame info."""
    mask_to_save = mask.to(torch.uint8)
    write_png(mask_to_save, os.path.join(output_folder, f"{record}_mask.png"))
    meta = {
        "rot_angle": float(rot_angle),
        "height": int(mask_to_save.shape[1]),
        "width": int(mask_to_save.shape[2]),
    }
    with open(os.path.join(output_folder, f"{record}_mask.json"), "w") as f:
        json.dump(meta, f)


def append_qc_row(output_folder, record, placement, qc, max_offset_deviation):
    """Append one QC row to qc.csv, writing the header if needed."""
    qc_path = os.path.join(output_folder, "qc.csv")
    fieldnames = [
        "record",
        "placement",
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

        # Rotate
        rot_angle, rotation_info = estimate_rotation(image, args.rotation)
        if args.rotation == "lines":
            if rotation_info["reason"]:
                print(
                    f"WARNING: grid lines not used for the rotation of record {record} "
                    f"({rotation_info['reason']}), keeping the Hough angle."
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
        image_rotated = rotate(image, rot_angle)

        # Segment
        if args.mask_folder is not None:
            # A mask only fits the frame it was predicted in, so check its angle.
            check_mask_rotation(args.mask_folder, record, rot_angle)
            mask_path = os.path.join(args.mask_folder, f"{record}_mask.png")
            if not os.path.exists(mask_path):
                raise FileNotFoundError(
                    f"No mask found for record {record} at {mask_path}."
                )
            mask_to_use = read_image(mask_path)
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
            save_mask_files(mask_to_use, record, args.output_folder, rot_angle)

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
                )
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
