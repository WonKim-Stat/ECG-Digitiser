[![License](https://img.shields.io/badge/License-BSD_2--Clause-orange.svg)](https://opensource.org/licenses/BSD-2-Clause) 
[![PWC](https://img.shields.io/endpoint.svg?url=https://paperswithcode.com/badge/combining-hough-transform-and-deep-learning/ecg-digitization-on-physionet-challenge-2024)](https://paperswithcode.com/sota/ecg-digitization-on-physionet-challenge-2024?p=combining-hough-transform-and-deep-learning)
[![Python](https://img.shields.io/badge/python-3.11-blue.svg)](https://www.python.org/downloads/release/python-3110/)

<h1 align='center'> ECG Digitiser – PhysioNet Challenge Winner (2024)<br> [<a href="https://arxiv.org/abs/2410.14185">arXiv Paper</a>] </h1>


**ECG Digitiser** is the state-of-the-art solution for converting ECG printouts into digital signals, enabling effective data extraction from legacy medical records. Our method combines the Hough Transform with deep learning, and achieved 1st place in the George B. Moody PhysioNet Challenge 2024.

This repository contains an open-source implementation, including:

- ✅ **Digitisation scripts** with straightforward usage examples.
- ✅ **Training scripts** for training [nnU-Net](https://github.com/MIC-DKFZ/nnUNet) to segment ECG printouts.
- ✅ **Pretrained ECG segmentation models** for immediate use.
- ✅ **Synthetic ECG Printout Generation** using [ecg-image-kit](https://github.com/alphanumericslab/ecg-image-kit) and the [PTB-XL dataset](https://www.nature.com/articles/s41597-020-0495-6).


## 🚀 Quick Start

### Install
```bash
https://github.com/felixkrones/ECG-Digitiser.git
cd ECG-Digitiser
git lfs install # If not already installed
git lfs pull # Only to also pull the weights
conda create -n ecgdig python=3.11 # Or any other package manager
conda activate ecgdig
pip install -r requirements.txt
```

At the moment, the official [nnU-Net](https://github.com/felixkrones/nnUNet.git) repository contains a bug and is not working with RGB png images. Please use the following for now:
```bash
cd nnUNet
pip install .
cd ..
```

### Use
```bash
python -m src.run.digitize -d data_folder -o output_folder
```

Where `data_folder` is the folder containing the ECG images to be digitized and `output_folder` is the folder where the outputs will be saved.


## 🖼️ Image assumptions for trained model weights

The pretrained segmentation weights expect ECG pages formatted like those
generated with `ecg-image-generator`:

- **Layout:** Three rows with four 2.5-second leads each. Lead names appear
  just below the left start of every trace. Rows begin at 0, 2.5 and 5 s.
- **Rhythm strip:** A fourth row contains a single 10-second lead across the
  bottom of the page.
- **Grid:** 25 mm/s horizontally and 10 mm/mV vertically.
- **Generation settings:** Images were created with random headers, random calibration pulse, random rotation up to 5°, and random black and whites.

For best results, supply images that match this layout and resolution.


## 🔍 Example Results

### ECG Segmentation
<p align="center">
    <img src="./figures/segmentation.png" alt="ECG Segmentation Example" width="800"/>
    <br><em>Example of ECG segmentation from a printed ECG image.</em>
</p>

### ECG Vectorisation
<p align="center">
    <img src="./figures/vectorisation.png" alt="ECG Vectorisation Example" width="800"/>
    <br><em>Example of vectorised ECG signal reconstructed from segmentation.</em>
</p>


## 🗂️ Repository Structure

- `ecg-image-generator`: Generate synthetic ECG images using [ecg-image-kit](https://github.com/alphanumericslab/ecg-image-kit) and the [PTB-XL dataset](https://www.nature.com/articles/s41597-020-0495-6).
- `figures`: Example images for the README.
- `models`: Trained model weights.
- `nnUNet`: Segmentation model [nnU-Net](https://github.com/MIC-DKFZ/nnUNet).
- `config.py`: Main configuration file.
- `src/run/digitize.py`: Digitisation pipeline script.
- `src/ptb_xl`: Prepare [PTB-XL dataset](https://www.nature.com/articles/s41597-020-0495-6).
- `src/utils`: Helper functions.


## 📚 Data

To run the code, you need different kind of data.
If you are using the PTB-XL dataset, see below under `Using the PTB-XL dataset` on how to prepare the data.

- **Training data for the segmentation model:**
In order to train the segmantation model, you need to have the data in the format as described in `nnUNet/documentation/dataset_format.md`. (For the following description, we assume the folder containing the data is called `Dataset500_Signals`.)

- **Data to run the digitization:**
A folder containing the ECG images to be digitized. Those images should match the ones used for training the segmentation model.
If you are using our pre-trained weights, those should match the images from the [2024 Challenge](https://physionetchallenges.org/2024/).


### Using the PTB-XL dataset

1. Download (and unzip) the [PTB-XL dataset](https://physionet.org/content/ptb-xl/) and [PTB-XL+ dataset](https://physionet.org/content/ptb-xl-plus/).
Replace the name of the folder (probably `ptb-xl-a-large-publicly-available-electrocardiography-dataset-1.0.3`) that contains a file named `ptbxl_database.csv` with `ptb-xl/`. 
Replace the name of the folder (probably `ptb-xl-a-comprehensive-electrocardiographic-feature-dataset-1.0.1`) that contains a folder called `labels` with `ptb-xl-p/`. 
Replace the paths below with the actual paths to the folders.

2. Add information from various spreadsheets from the PTB-XL dataset to the WFDB header files:

        python -m src.ptb_xl.prepare_ptbxl_data \
            -i ptb-xl/records500 \
            -pd ptb-xl/ptbxl_database.csv \
            -pm ptb-xl/scp_statements.csv \
            -sd ptb-xl-p/labels/12sl_statements.csv \
            -sm ptb-xl-p/labels/mapping/12slv23ToSNOMED.csv \
            -o ptb-xl/records500_prepared

3. [Generate synthetic ECG images](https://github.com/alphanumericslab/ecg-image-kit/tree/main/codes/ecg-image-generator) on the dataset:

    1. Move into ecg-image-generator: `cd ecg-image-generator`
    2. Deactivate the current env (`conda deactivate`) and create a new one by following the instructions in the README of the generator repo (`ecg-image-generator/README.md`) (this means running `conda env create -f environment_droplet.yml` and then `conda activate ecg_gen`).
    3. Now you can run the following code. Careful though, this can take very long, around 10 min per subfolder (approx. 1000 files) or 4h in total and will increase the necessary disk space by approx. 15x, adding another 8GB for the 500Hz data. To test, better to run it on one single subfolder (e.g., add /00000):

            python gen_ecg_images_from_data_batch.py \
                -i ptb-xl/records500_prepared \
                -o ptb-xl/records500_prepared_w_images \
                --print_header \
                --store_config 2 \
                --mask_unplotted_samples

        For example:

            python gen_ecg_images_from_data_batch.py \
                -i ptb-xl/records500_prepared/00000 \
                -o ptb-xl/records500_prepared_w_images/00000 \
                -se 10 \
                --mask_unplotted_samples \
                --print_header \
                --store_config 2 \
                --lead_name_bbox \
                --lead_bbox \
                --random_print_header 0.7 \
                --calibration_pulse 0.6 \
                --fully_random \
                -rot 5 \
                --num_images_per_ecg 4 \
                --random_bw 0.1 \
                --run_in_parallel \
                --num_workers 8
                
    4. Deactivate the environment again (`conda deactivate`), move back to your original repo (`cd ..`) and activate the environment from above again (`conda activate env-name`).

4. Add the file locations and other information for the synthetic ECG images to the WFDB header files:

        python -m src.ptb_xl.prepare_image_data \
            -i ptb-xl/records500_prepared_w_images \
            -o ptb-xl/records500_prepared_w_images

5. Create more dense pixels for the masks:

        python -m src.ptb_xl.replot_pixels \
            --resample_factor 3 \
            --dir ptb-xl/records500_prepared_w_images \
            --run_on_subdirs \
            --num_workers 12

6. Convert it into the nnUNet format:

    We use the suggested splits from `ptbxl_database.csv`:

        python -m src.ptb_xl.create_train_test \
            -i ptb-xl/records500_prepared_w_images \
            -d ptb-xl/ptbxl_database.csv \
            -o ptb-xl/Dataset500_Signals

    For example:

        python -m src.ptb_xl.create_train_test \
            -i ptb-xl/records500_prepared_w_images \
            -d ptb-xl/ptbxl_database.csv \
            -o ptb-xl/Dataset500_Signals \
            --rgba_to_rgb \
            --gray_to_rgb \
            --mask \
            --mask_multilabel \
            --rotate_image \
            --plotted_pixels_key dense_plotted_pixels \
            --num_workers 8


## ▶️ Run

You can either use our model weights for the segmentation model or train your own model following the steps under `1.` below.

### 1. Train the segmentation model

First, one needs to train the segmentation model.
We use [nnU-Net](https://github.com/felixkrones/nnUNet.git); this involves multiple steps:

1. Set the environment variables for nnU-Net (just post the following in the terminal from where you want to run the code):

        # Set environment variables
        export nnUNet_raw='path to the folder that contains the Dataset500_Signals folder'
        export nnUNet_preprocessed='any folder path to save the preprocessed data'
        export nnUNet_results='any folder path to save the results'

        # Check
        echo ${nnUNet_raw}

2. Experiment planning and preprocessing

        nnUNetv2_plan_and_preprocess -d DATASET_ID --verify_dataset_integrity

    For example:

        nnUNetv2_plan_and_preprocess -d 500 --clean -c 2d --verify_dataset_integrity

3. Model training

        nnUNetv2_train DATASET_NAME_OR_ID UNET_CONFIGURATION FOLD

    For example:

        nnUNetv2_train 500 2d 0 -device cuda --c

    Or select the device (one per fold):

        CUDA_VISIBLE_DEVICES=1 nnUNetv2_train 500 2d 1 --c

4. Determine the best configuration

        nnUNetv2_find_best_configuration DATASET_NAME_OR_ID -c CONFIGURATIONS

    For example:

        nnUNetv2_find_best_configuration 500 -c 2d --disable_ensembling


### 2. Run the digitization

1. Set the parameters in `config.py`.

2. Run the digitization pipeline by running:

        python -m src.run.digitize -d data_path -m model_path -o output_path -v

where

- `data_path` (input; required) is the folder containing the images to be digitized;
- `model_path` (input; required) is the folder containing the segmentation model folder `nnUNet_results`, e.g., `models/M3`;
- `output_path` is the folder where the outputs will be saved.


#### Pipeline stages

Under the defaults, every page goes through the stages below before its signals are
written (`run()` in `src/run/digitize.py`). Every stage that reads the printed 1 mm
grid lines falls back to the behaviour without them if the page shows none.

1. **Resolution:** the period of the printed grid lines is measured and the page is
   resampled to the period of a 200 dpi page, the scale the model was trained on.
2. **Rotation:** a 0.1° Hough transform, refined with the phase drift of the grid lines
   from the top to the bottom of the page.
3. **Perspective:** the shear and the perspective that the phase field of the grid lines
   shows are taken out, in the same warp as the rotation.
4. **Segmentation:** nnU-Net predicts the lead masks of the corrected page.
5. **Column grid:** all leads are put on one shared column grid, with the column pitch
   from the page height, the origin refined to sub-pixel accuracy on the grid lines and
   moved by the grid line offset.
6. **Column mapping:** the time axis is read off the grid lines themselves: the 1 mm
   lines, each named by the bold 5 mm lines, say where every millimetre of the page is,
   so a page whose grid and traces are stretched, compressed or stepped in places by
   printing, paper feed or scanning is read at the x of each sample's own millimetre.
   A page that the map moves by less than a pixel, or whose map cannot be trusted, keeps
   the uniform columns of the column grid.
7. **Trace reading:** every column of a lead gives one row, the mean row of its mask
   weighted by the ink of the page under it, read on a column axis moved by the trace
   shift measured on the page.
8. **Sharpening:** the 20–40 Hz band that the width of a column and the linear
   interpolation between columns attenuate is put back: the part below 35 Hz of the
   difference between an aperture-corrected cubic spline through the columns and the
   linear trace is added to the trace, so steep QRS spikes cannot ring.
9. **Baseline:** the zero line comes from the page geometry scaled with the column pitch
   and is then shifted so that the Einthoven and Goldberger sums have no DC offset.


#### Options

| Flag | Choices | Default | What it does |
| --- | --- | --- | --- |
| `--resolution` | `keep`, `lines` | `lines` | `lines` resamples the page so that the printed 1 mm grid lines have the period of a 200 dpi page, the scale the model was trained on. A page within about a tenth of that scale keeps its pixels, because resampling it costs more than the scale does. |
| `--rotation` | `hough`, `lines` | `lines` | `lines` refines a 0.1° Hough angle with the phase drift of the grid lines, which resolves fractions of a degree. `hough` is the Hough transform of the page in whole degrees only. |
| `--perspective` | `off`, `lines` | `lines` | `lines` takes the shear and the perspective that the grid lines show out of the rotated page, and leaves a page alone that is straight enough already. `off` keeps the rotated page. |
| `--time_mapping` | `bbox`, `grid` | `grid` | `grid` samples all leads on one shared column grid, for the standard 3x4 layout with rhythm strip. `bbox` stretches the bounding box of every lead to its length. |
| `--grid_pitch` | `page`, `fit` | `page` | Where the column pitch of that grid comes from: `page` from the page height, checked against the least squares fit of the column edges, `fit` from that fit. |
| `--grid_origin` | `masks`, `lines` | `lines` | `lines` refines origin and pitch of the column grid on the printed grid lines, assuming the first column starts on a grid line. `masks` uses the mask edges only. |
| `--grid_line_offset` | any number of pixels | `0.5` | How far right of the traces the printed grid lines sit. `0.5` is the matplotlib Agg snap of the generator images, `0` is for scans and photographs. It is a page unit, so it survives `--resolution`. |
| `--column_mapping` | `uniform`, `lines` | `lines` | `lines` reads the time axis off the printed grid lines themselves: in windows about half a line apart the 1 mm lines say where every millimetre is and the bold 5 mm lines which millimetre that is, so a sample is read at the x of its own millimetre (1/20 mm at 25 mm/s and 500 Hz) on a page whose grid and traces are stretched, compressed or stepped in places by printing, paper feed or scanning. The origin is the grid line the mask edges of all leads agree on, and the ripple of grid lines snapped to the pixels of a drawn page is taken out with `--column_mapping_median`. It needs `--time_mapping grid` and `--grid_origin lines` with the grid lines used, keeps the uniform grid with a `WARNING` when the map cannot be trusted (5 mm and 1 mm lines disagree, too few windows, a gap, a jump of a line, mask edges off the lines) and, byte identical, when it moves no sample by 1 px. On the development set (115 PTB-XL records, split by patient) of the ECG-Image-Database (v2, part D0: PTB-XL records printed, then scanned or photographed; references below the table), scored per lead against PTB-XL, the median lead SNR of the colour scans rises from 9.06 to 13.40 dB (paired median +3.41 dB, patient bootstrap 95 % CI 2.75 to 3.90) and that of the greyscale scans from 8.82 to 13.14 dB (+3.40, 2.84 to 3.95); the leads without a usable signal (no signal or an SNR below 0 dB) drop from 15.2 to 3.6 % and from 17.2 to 4.3 %. In the paired median of its leads, the fourth column gains 9.9 / 10.5 dB (colour / greyscale) and the rhythm strip II 6.7 / 7.5 dB, while the third column (V1–V3) changes by −0.59 / 0.00 dB. The rendered pages of the same records and all 208 synthetic pages (generator pages and synthetic defects applied to them, every page of the results below included) keep byte identical signals (dead band on 178 pages; on 26 augmented generator pages and 4 faint-grid pages the map is not trusted); on mould-damaged scans the map is used on 20–28 % of the pages, with a paired median of +0.85 to +1.06 dB on those. On the colour and the greyscale scans alike, a PTB-XL diagnostic classifier changes 3.8 % instead of 4.9 % of its record and class decisions against the digital signal. On the evaluation set of the same database (84 records of other patients, read once with this setting fixed beforehand), the colour scans rise from 8.92 to 12.83 dB (+2.55, 1.57 to 3.30) and the greyscale scans from 8.87 to 13.25 dB (+3.44, 2.70 to 4.05), the leads without a usable signal drop from 18.3 to 6.6 % and from 18.5 to 5.1 %, the rendered pages keep byte identical signals, and on the mould-damaged scans the pages that use the map gain +1.35 and +0.59 dB; the classifier's changed decisions, however, stay at 4.8 % on the colour scans and go from 2.9 to 3.6 % on the greyscale scans (pooled +0.4 points, patient bootstrap 95 % CI −0.6 to +1.3), so on these 84 records the SNR gain did not show as fewer changed decisions. It costs about 0.08 s per page. `uniform` makes every column P pixels wide from the grid origin on, the time axis before the column mapping. |
| `--column_mapping_median` | any width in grid mm, `0` = none | `13.5` | Only for `--column_mapping lines`: the width of the running median of the map's displacement from the uniform columns. The grid lines of a page drawn at 200 dpi are snapped to its pixels, its traces are not, and on the bold lines that snap repeats every 13.5 mm; scanned printouts carry the same ripple. A running median over that period takes it out and keeps a step or a stretch of the lines as it is. Without it the map follows that ripple, which the traces lack, and costs V1–V3 about 1.6 dB; the median recovers about half of that. `0` reads the samples on the map as measured. |
| `--trace_estimator` | `mask`, `ink` | `ink` | `ink` weights the mean row of a column by the ink under the mask, which puts the row on the core of the stroke; a lead whose stroke is too faint keeps the mask mean. `mask` is the mean row of the binary mask. |
| `--trace_shift` | `off`, `page` | `page` | `page` measures once per page how far right of the lead masks the ink of the steep strokes sits and reads every lead on a column axis moved by that. `off` reads every lead where its mask is. |
| `--sharpen` | `none`, `bandlimited` | `bandlimited` | `bandlimited` restores the 20–40 Hz band that the column aperture and the linear interpolation attenuate: the aperture-corrected (a = 1/8) cubic spline through the columns minus the linear trace, low-passed at 35 Hz (zero-phase Butterworth, order 4), is added to the linear trace, so only the band below 35 Hz changes and steep QRS spikes cannot ring. Paired on the same masks, the median gain per lead is 1.41 dB on the 96 generator pages below, with no lead of the 15 conditions below more than 1 dB worse, and 1.17 dB on 2,000 synthetic pages of 500 PTB-XL records (clean, augmented, rotated, simulated scan); on those pages a PTB-XL diagnostic classifier changes 138 instead of 233 of its 10,000 record and class decisions against the digital signal. It costs about 8–14 ms per page. `none` interpolates the column profile linearly, the trace before the sharpening. |
| `--baseline` | `page`, `leads` | `leads` | `page` takes the baseline from the page geometry, scaled with the column pitch. `leads` additionally shifts it so that the Einthoven and Goldberger sums have no DC offset. |

`python -m src.run.digitize --help` lists the remaining options (`--interpolation`, `--enable_tta`, `--lead_placement`, `--save_mask`, `--mask_folder`, `--fold`, `--device`).

The real-page numbers in the table and in the quality control section below are on the
ECG-Image-Database, version 2
(<https://www.kaggle.com/datasets/physionet/ecg-image-database>), whose images are
licensed CC BY-ND 4.0, so only aggregate numbers are given here. Its dataset page asks
for these citations:

1. M. A. Reyna et al., "ECG-Image-Database: large-scale paired ECG images and time-series
   with real-world artifacts; a foundation for computerized ECG digitization and
   analysis", *Physiological Measurement* 47(7), 075015, 2026. doi:10.1088/1361-6579/ae85b2
2. P. Wagner, N. Strodthoff, R.-D. Bousseljot, D. Kreiseler, F. I. Lunze, W. Samek,
   T. Schaeffter, "PTB-XL, a large publicly available electrocardiography dataset",
   *Scientific Data* 7, 154, 2020 (doi:10.1038/s41597-020-0495-6); dataset on PhysioNet,
   doi:10.13026/kfzx-aw45.
3. E. Stenhede, A. M. Bjørnstad, A. Ranjbar, "Digitizing paper ECGs at scale: an
   open-source algorithm for clinical research", *npj Digital Medicine* 9, 2026.
   doi:10.1038/s41746-025-02327-1


#### Quality control output

The digitization writes one row per record to `qc.csv` in the output folder, with these
columns:

- **Record:** `record` and `placement`, the `--lead_placement` the row was written with.
- **Lead consistency:** `einthoven_rms`, `einthoven_rms_demedian`, `einthoven_ratio`,
  `einthoven_n` for I + III − II, and `goldberger_rms`, `goldberger_rms_demedian`,
  `goldberger_ratio`, `goldberger_n` for aVR + aVL + aVF: the RMS of the sum in mV, the
  same after its median is taken out, the RMS relative to the mean RMS of the three
  leads in the sum, and the number of samples it was computed over.
- **Lead placement:** `max_offset_deviation`, the largest distance in seconds between a
  measured lead offset and the standard one it was snapped to.
- **Resolution:** `grid_period_px`, the measured grid line period, and
  `resolution_scale`, the factor the page was resampled by (1 if it was left alone).
- **Rotation:** `rotation_angle`, the angle the page was turned by, `rotation_coarse`,
  the Hough angle before the grid lines, and `rotation_residual_px`, the residual of the
  grid line phase fit.
- **Perspective:** `perspective_shift_px`, the largest displacement the rectification
  causes on the page, and `perspective_residual_px`, the residual of its fit.
- **Column grid:** `grid_line_contrast`, the amplitude of the grid line comb over the
  neighbouring periods, and `grid_line_shift_px`, how far the grid lines moved the
  origin of the mask grid.
- **Trace:** `trace_estimator`, `ink_leads`, the number of leads whose ink was dark
  enough to weight with, `ink_p95`, the median 95th percentile of that ink, and
  `trace_shift_px` and `trace_shift_rows`, the page shift that was applied and the
  number of rows it was measured on.
- **Sharpening:** `sharpen`, the `--sharpen` mode the row was written with.
- **Column mapping:** `column_mapping`, `lines` when the grid line map placed the
  samples, `uniform: <reason>` when the page kept the uniform grid under the default
  `--column_mapping lines` (`dead band`, the reason a map cannot be trusted,
  `grid lines not used`, `no column grid` or `grid origin from the masks`), or
  `uniform` with `--column_mapping uniform`, and `column_mapping_shift_px`, the largest
  move in pixels the map, after `--column_mapping_median`, makes to any sample the page
  reads against the uniform grid (NaN when it was not measured).
- **Baseline:** `baseline_scale`, the column pitch over the pitch the page height
  implies, `baseline_shift_px`, the shift that was applied, and
  `baseline_disagreement_px`, the difference between the Goldberger and the Einthoven
  estimate.

A `WARNING` line is printed whenever a stage cannot do what it is asked and falls back:
the grid lines are unusable for the resolution (the page is kept as it is), for the
rotation (the whole degree Hough angle is kept), for the perspective (the rotated page
is kept) or for the grid origin (the grid of the masks is kept); no column grid fits the
page, so `--time_mapping bbox` is used for it; the Einthoven and the Goldberger baseline
estimate disagree by more than the tolerance, so the page baseline is kept; the grid
line map of `--column_mapping lines` cannot be trusted, so the uniform grid is kept; or
a lead mask has a gap inside its column, which is interpolated linearly. A page whose fitted
column pitch does not match the pitch of the page height prints a line as well and uses
the fitted one.

One more `WARNING` line comes without a fallback, the Einthoven check. Every page with
signals prints a `QC <record>: Einthoven RMS …` line with its lead consistency values.
Right after it, a page whose `einthoven_ratio` (the RMS of I + III − II over the mean RMS
of the three leads, on the samples where all three are read) is 0.1797 or more prints
`WARNING: Einthoven check failed for record <record> (ratio <ratio> >= 0.1797): leads I, II and III disagree, check this page.`
and a page without a ratio, because the three leads share no sample or are zero on the
samples they share, prints
`WARNING: Einthoven check failed for record <record> (no ratio: <reason>), check this page.`
Leads I and III are read in the first column only, so the check covers at most its 2.5 s
(1,250 samples, `einthoven_n`) and cannot see an error confined to the other columns
(aVR to V6) or to the rhythm strip II after its first 2.5 s. It is a flag to look at the
page, not a diagnosis, and it changes neither the signals nor `qc.csv`. The threshold
was fitted on the development set of the ECG-Image-Database named above (115 records):
the highest ratio that flags at least 90 % of the clean colour and greyscale scans whose
page SNR (the median SNR of their 12 leads, a missing lead counting as the lowest) is
below 12 dB, 0.17976, rounded down to 0.1797. On that set, read with the defaults, it
flags 56 of these 62 scans and 57 of the 168 others, so 57 of the 113 flagged clean scans
are at or above 12 dB (AUROC 0.880 over the pairs of a scan below and a scan above 12 dB
of the same scan type, patient bootstrap 95 % CI 0.817 to 0.932). These numbers are
in-sample, since the threshold was chosen on these pages: fitted on one scan type alone,
it catches 25 of the 29 colour scans below 12 dB (fitted on the greyscale scans) and 32
of the 33 greyscale scans below 12 dB (fitted on the colour scans). On the evaluation set
of the same database (84 records of other patients, read once with the threshold fixed)
it flags 40 of the 48 clean scans below 12 dB and 48 of the 120 others, and 4 of the 84
rendered pages, none below 12 dB, each with a real error in lead I, II or III. On mould-damaged scans and
photographs it flags nearly every page, so there it does not tell good pages from bad
ones: all 203 mould-damaged scans below 12 dB and 23 of the other 27, and 340 of the 343
photographs below 12 dB and both others (photographs read after a paper normalisation
that is not part of this repository). It flags 1 of the 115 rendered pages, which is not
below 12 dB. On synthetic pages it flags 3 of the 192 pages of the results below, none
of them below 12 dB, and 98 of 2,000 synthetic pages of 500 PTB-XL records (read before
the column mapping), 5 of the 6 below 12 dB among them. The Goldberger ratio
(aVR + aVL + aVF) is not used: at the same sensitivity it flags 106 instead of 57 of the
clean scans at or above 12 dB.


#### Results on synthetic pages

All pages below are synthetic: pages from the image generator, and synthetic scan and
photo defects applied to generator pages. There are no real scans among them. The masks
were predicted once per condition with test time augmentation off, the signals were then
read from those saved masks with the defaults (no flags) and scored with
`python -m src.run.evaluate --placement window`. The metric is the median raw SNR over
the leads of a condition. `--sharpen none` gives the signals of the pipeline before the
sharpening byte for byte, with medians 0.9 (scale080) to 2.5 dB (gridblue) lower.
`--column_mapping uniform` gives the same signals as the defaults byte for byte: the
map moves no sample of these pages by a pixel or, on augmented pages and pages with a
faint grid whose lines it cannot read, is not trusted.

| Condition | Pages | Description | Median SNR (dB) |
| --- | --- | --- | --- |
| clean | 32 | generator pages, no defect | 25.10 |
| aug | 32 | generator augmentation on a random subset of pages: wrinkles, noise, rotation up to 5°, black and white | 23.98 |
| rot | 32 | generator augmentation with a forced rotation of ±1–5° and a crop of up to 1 % | 22.24 |
| blur | 8 | Gaussian blur, σ 1.5 | 24.10 |
| jpeg | 8 | JPEG quality 30 | 25.58 |
| gridfaint | 8 | grid ink at 30 % of its darkness | 25.98 |
| gridblue | 8 | colour channels swapped, so the red grid is printed blue | 25.96 |
| margin | 8 | page shrunk to 92 % and moved off centre | 22.15 |
| persp05 | 8 | perspective, corners moved up to 0.5 % | 24.21 |
| persp15 | 8 | perspective, corners moved up to 1.5 % | 24.13 |
| photo | 8 | rotation of 1.5°, perspective up to 1 %, uneven light, defocus, JPEG quality 60 | 23.42 |
| scale080 | 8 | page resampled ×0.80 and moved a fraction of a pixel | 21.83 |
| scale088 | 8 | page resampled ×0.88 and moved a fraction of a pixel | 22.35 |
| scale115 | 8 | page resampled ×1.15 and moved a fraction of a pixel | 25.46 |
| scale125 | 8 | page resampled ×1.25 and moved a fraction of a pixel | 25.59 |

The three generator sets are pages of 32 PTB-XL records, 384 leads each; the other twelve
conditions are defects applied to the generator pages of 8 of those records, 96 leads
each. The evaluation scripts and data are not part of this repository.


## 📄 Citation and Acknowledgements

If you use this code in your research, please cite our [paper](https://arxiv.org/abs/2410.14185):

    @article{krones2024combining,
        title={Combining Hough Transform and Deep Learning Approaches to Reconstruct ECG Signals From Printouts},
        author={Krones, Felix and Walker, Ben and Lyons, Terry and Mahdi, Adam},
        journal={arXiv:2410.14185},
        year={2024}
    }

    Krones F, Walker B, Lyons T, Mahdi A. Combining Hough Transform and Deep Learning Approaches to Reconstruct ECG Signals From Printouts. arXiv:2410.14185. 2024 Oct 18.

Additionally, please consider citing the [PhysioNet Challenge 2024](https://physionetchallenges.org/2024/), [ecg-image-kit](https://github.com/alphanumericslab/ecg-image-kit), [PTB-XL dataset](https://www.nature.com/articles/s41597-020-0495-6), and [nnU-Net](https://github.com/MIC-DKFZ/nnUNet).
