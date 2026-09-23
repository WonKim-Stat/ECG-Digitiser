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
6. **Trace reading:** every column of a lead gives one row, the mean row of its mask
   weighted by the ink of the page under it, read on a column axis moved by the trace
   shift measured on the page.
7. **Baseline:** the zero line comes from the page geometry scaled with the column pitch
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
| `--trace_estimator` | `mask`, `ink` | `ink` | `ink` weights the mean row of a column by the ink under the mask, which puts the row on the core of the stroke; a lead whose stroke is too faint keeps the mask mean. `mask` is the mean row of the binary mask. |
| `--trace_shift` | `off`, `page` | `page` | `page` measures once per page how far right of the lead masks the ink of the steep strokes sits and reads every lead on a column axis moved by that. `off` reads every lead where its mask is. |
| `--baseline` | `page`, `leads` | `leads` | `page` takes the baseline from the page geometry, scaled with the column pitch. `leads` additionally shifts it so that the Einthoven and Goldberger sums have no DC offset. |

`python -m src.run.digitize --help` lists the remaining options (`--interpolation`, `--enable_tta`, `--lead_placement`, `--save_mask`, `--mask_folder`, `--fold`, `--device`).


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
- **Baseline:** `baseline_scale`, the column pitch over the pitch the page height
  implies, `baseline_shift_px`, the shift that was applied, and
  `baseline_disagreement_px`, the difference between the Goldberger and the Einthoven
  estimate.

A `WARNING` line is printed whenever a stage cannot do what it is asked and falls back:
the grid lines are unusable for the resolution (the page is kept as it is), for the
rotation (the whole degree Hough angle is kept), for the perspective (the rotated page
is kept) or for the grid origin (the grid of the masks is kept); no column grid fits the
page, so `--time_mapping bbox` is used for it; the Einthoven and the Goldberger baseline
estimate disagree by more than the tolerance, so the page baseline is kept; or a lead
mask has a gap inside its column, which is interpolated linearly. A page whose fitted
column pitch does not match the pitch of the page height prints a line as well and uses
the fitted one.


#### Results on synthetic pages

All pages below are synthetic: pages from the image generator, and synthetic scan and
photo defects applied to generator pages. There are no real scans among them. The masks
were predicted once per condition with test time augmentation off, the signals were then
read from those saved masks with the defaults (no flags) and scored with
`python -m src.run.evaluate --placement window`. The metric is the median raw SNR over
the leads of a condition.

| Condition | Pages | Description | Median SNR (dB) |
| --- | --- | --- | --- |
| clean | 32 | generator pages, no defect | 22.99 |
| aug | 32 | generator augmentation on a random subset of pages: wrinkles, noise, rotation up to 5°, black and white | 22.36 |
| rot | 32 | generator augmentation with a forced rotation of ±1–5° and a crop of up to 1 % | 21.06 |
| blur | 8 | Gaussian blur, σ 1.5 | 21.66 |
| jpeg | 8 | JPEG quality 30 | 23.26 |
| gridfaint | 8 | grid ink at 30 % of its darkness | 23.66 |
| gridblue | 8 | colour channels swapped, so the red grid is printed blue | 23.48 |
| margin | 8 | page shrunk to 92 % and moved off centre | 20.97 |
| persp05 | 8 | perspective, corners moved up to 0.5 % | 22.54 |
| persp15 | 8 | perspective, corners moved up to 1.5 % | 22.63 |
| photo | 8 | rotation of 1.5°, perspective up to 1 %, uneven light, defocus, JPEG quality 60 | 21.56 |
| scale080 | 8 | page resampled ×0.80 and moved a fraction of a pixel | 20.94 |
| scale088 | 8 | page resampled ×0.88 and moved a fraction of a pixel | 21.32 |
| scale115 | 8 | page resampled ×1.15 and moved a fraction of a pixel | 23.50 |
| scale125 | 8 | page resampled ×1.25 and moved a fraction of a pixel | 23.47 |

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
