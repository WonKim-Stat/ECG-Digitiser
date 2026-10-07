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

Under the defaults, every page goes through the stages below, all but stage 5, which is
off unless it is asked for, before its signals are written (`run()` in
`src/run/digitize.py`). Every stage that reads the printed 1 mm grid lines falls back to
the behaviour without them if the page shows none.

1. **Paper normalisation:** a photographed page is put into the frame the model was
   trained on. A page whose grid lines stages 2 to 4 read as it is, or whose printed
   grid fills the image, is left alone; otherwise the four edges of the sheet are looked
   for and the sheet is warped onto a US Letter landscape page at 200 dpi
   (`src/run/paper_normalisation.py`). A page whose paper is not found is kept, with a
   `WARNING`. A page read with a saved mask is not decided again: the record next to
   the mask says what was done to it, and a mask without one is laid over the page as it
   is given. `--paper_normalisation off` leaves out the stage.
2. **Resolution:** the period of the printed grid lines is measured and the page is
   resampled to the period of a 200 dpi page, the scale the model was trained on.
3. **Rotation:** a 0.1° Hough transform, refined with the phase drift of the grid lines
   from the top to the bottom of the page.
4. **Perspective:** the shear and the perspective that the phase field of the grid lines
   shows are taken out, in the same warp as the rotation. A page whose grid lines are
   more than 0.15 of their period off the fit is kept as it was rotated, with a
   `WARNING`. `--perspective_tolerance 0.1` keeps every page above a tenth as it was
   rotated, the reading before 0.15 became the default.
5. **Sheet curl:** off unless `--sheet_curl lines` asks for it. A sheet that is bent is
   straightened by its grid lines: a smooth map from the pixels of the page to the
   coordinates of its grid lines, a projective part and a polynomial per line family,
   is measured in windows over the page as stage 4 left it, and the page is warped into
   the frame in which the lines are straight (`src/run/sheet_curl.py`). Every page is
   looked at, unless `--sheet_curl_scope normalised` says only a page that stage 1
   warped onto the Letter page, a photographed sheet. A page that its map moves by less
   than 15 px is left as it is, and so is, with a `WARNING`, a page whose grid lines
   carry no such map. A bend of the whole sheet is taken out, a fold or a local dent is
   not.
6. **Segmentation:** nnU-Net predicts the lead masks of the corrected page, which is
   the straightened one where stage 5 was applied.
7. **Column grid:** all leads are put on one shared column grid, with the column pitch
   from the page height, the origin refined to sub-pixel accuracy on the grid lines and
   moved by the grid line offset. A page that fails a check of this grid because one of
   its columns is distorted is tried a second time, and kept on that try only if the
   column map of stage 8 stands on it. `--grid_rescue off` does not try again.
8. **Column mapping:** the time axis is read off the grid lines themselves: the 1 mm
   lines, each named by the bold 5 mm lines, say where every millimetre of the page is,
   so a page whose grid and traces are stretched, compressed or stepped in places by
   printing, paper feed or scanning is read at the x of each sample's own millimetre.
   A page that the map moves by less than a pixel, or whose map cannot be trusted, keeps
   the uniform columns of the column grid. On a page that is read on its map, each of
   the four rows of leads is read on a map of its own, the map of the page moved by what
   the grid lines in the band of that row say. `--row_mapping off` reads every lead on
   the map of the page.
9. **Trace reading:** every column of a lead gives one row, the mean row of its mask
   weighted by the ink of the page under it, read on a column axis moved by the trace
   shift measured on the page.
10. **Sharpening:** the 20–40 Hz band that the width of a column and the linear
    interpolation between columns attenuate is put back: the part below 35 Hz of the
    difference between an aperture-corrected cubic spline through the columns and the
    linear trace is added to the trace, so steep QRS spikes cannot ring.
11. **Baseline:** the zero line comes from the page geometry scaled with the column pitch
    and is then shifted so that the Einthoven and Goldberger sums have no DC offset.


#### Options

| Flag | Choices | Default | What it does |
| --- | --- | --- | --- |
| `--paper_normalisation` | `off`, `auto` | `auto` | `auto` puts a photographed page into the frame the model was trained on, before every other stage and on the signals of the image alone. A page whose printed grid lines the resolution, the rotation and the perspective stage all read as it is is left as it is (`pass_chain`), and so is one without a background to find a paper edge in: the image reaches at most 8 mm beyond a Letter sheet, or no paper edge is found and the grid is within 12 mm of the border on all four sides, or the paper found covers 95 % of the image (`pass_fullframe`). Otherwise the four edges of the sheet are looked for (the region of the red grid, then in 24 bands per side the step from the print or its white margin to the background, then each side again at full resolution) and the sheet is warped in one bicubic pass onto a US Letter landscape page at 200 dpi (2200 × 1700 px), long side horizontal, in the orientation that puts the trace ink in the lower part, and scaled so that the 1 mm grid has the period it has on a scanned printout, 0.954 × 200 dpi (`normalised`). A page whose paper is not found, runs off the image or gives an implausible sheet or grid is kept, with a `WARNING` (`failed`); such a page, or a `pass_fullframe` one, whose grid the resolution stage could not read while the extent of its grid says it is more than 12 % above 200 dpi of paper is shrunk to that and not rectified, with a `WARNING` as well (`scaled`). The curl of a sheet is not taken out by this stage (`--sheet_curl lines`, off by default, is the stage for a sheet that is bent), and the print gets the scale of the frame, not its position. A page without a readable grid is not told from a photograph by its content, as in the view builder: a blank page is `pass_fullframe`, and a large image without a grid, blank or with content that looks like a sheet at the scale of a photograph (4000 × 3000 px, say), is taken to show a whole Letter sheet and shrunk to 200 dpi of it (`scaled`). The stage is the image code of the view builder that made the photograph results of the quality control section, function for function, with the thresholds chosen there on the 345 development photographs of the ECG-Image-Database (115 records, three photo conditions; references below the table). With that builder, outside this repository, 301 of those pages were normalised, 22 scaled, 12 passed and 10 failed; scored per lead against PTB-XL, the median lead SNR of the three conditions rose from −6.20, −3.63 and −4.53 dB to −1.87, −1.10 and −0.52 dB (paired median, pooled over the three, +3.71 dB, 95 % CI 2.87 to 4.41; +3.55 dB with the time axis left uniform, so from the normalisation alone), a PTB-XL diagnostic classifier changed 26.3, 32.7 and 30.6 % instead of 47.5, 51.3 and 40.2 % of its record and class decisions against the digital signal, and the 903 rendered pages, scans and synthetic pages it was checked on were all left as they are. So photographs are read better, not well: most of what is left is the time axis of a sheet that does not lie flat. The stage itself, run on those 345 pages with their saved masks, gives the same view, pixel for pixel, and the same signals, byte for byte, so these numbers are its numbers. With the flag itself, the 575 rendered and scanned development pages of that database and all 208 synthetic pages (every page of the results below included) were left as they are and keep byte identical signals. On a page that is left as it is, with `--resolution`, `--rotation` and `--perspective` at `lines`, the three stages are measured once and used twice, and the stage costs about 0.1 s (the digests of the page); only the perspective of a page whose grid line fit was refused in that first measurement is measured again, at `--perspective_tolerance`, when that is above 0.1, as its default is; with one of them set otherwise the stages are measured a second time, 1 to 4 s per page on generator pages, and a synthetic 12-megapixel photograph takes 5 to 8 s. What was done to a page is recorded next to its mask, see below the table. `off` takes the page as it is, the behaviour before `auto` became the default. |
| `--resolution` | `keep`, `lines` | `lines` | `lines` resamples the page so that the printed 1 mm grid lines have the period of a 200 dpi page, the scale the model was trained on. A page within about a tenth of that scale keeps its pixels, because resampling it costs more than the scale does. |
| `--rotation` | `hough`, `lines` | `lines` | `lines` refines a 0.1° Hough angle with the phase drift of the grid lines, which resolves fractions of a degree. `hough` is the Hough transform of the page in whole degrees only. |
| `--perspective` | `off`, `lines` | `lines` | `lines` takes the shear and the perspective that the grid lines show out of the rotated page, and leaves a page alone that is straight enough already. `off` keeps the rotated page. |
| `--perspective_tolerance` | a share of the grid line period, `0.1` to `0.5` | `0.15` | Only for `--perspective lines`: the largest residual of the grid line fit that the perspective stage accepts, as a share of the period of the lines it measures on. A page whose fit is above it keeps the rotated page, with a `WARNING`. `0.1` is the tolerance of the stage itself. The grid of a printed and scanned sheet is not exactly the projective image of a square one, so the fit of such a page can end a little above 0.1, and the default, `0.15`, rectifies it. Only the perspective stage of the run takes the value. The rotation stage keeps 0.1, and so does `--paper_normalisation auto` where it decides whether the stages read a page as it is, so the decisions of the paper normalisation are those at 0.1. A value below 0.1 or above 0.5 is refused when the arguments are read: the flag only widens the stage. A page that is rectified with a residual above 0.1 is in another frame than at 0.1, so a mask saved for it at 0.1 does not fit, and `--mask_folder` says so with its frame `WARNING`; what that means for masks saved before 0.15 became the default is below the table. `--verbose` names a page that is accepted above 0.1 in one line, `Perspective for record <record>: accepted at --perspective_tolerance <k>, residual <share> of the period (the stage's own tolerance is 0.1), homography applied`, or `…, inside the dead band, page kept` for a page that is accepted and straight enough to be left alone, and `qc.csv` holds the share in `perspective_residual_rel`. Development-set numbers and limits are under *Results on real pages* below: the scans gain, though 6 of the 89 mould-damaged scans that 0.15 moves into another frame lose more than 1 dB of page SNR, and so do 20 of the 100 photographs it moves, where the change of the classifier's decisions has an interval that includes zero. `0.1` is the reading before `0.15` became the default: it reproduces older outputs, and it is the frame that masks saved without the flag before the change were predicted in. |
| `--sheet_curl` | `off`, `lines` | `off` | `lines` straightens a sheet that is bent, after the perspective stage and before the segmentation. The paper normalisation puts the four corners of a photographed sheet on the page frame and the perspective stage fits one homography to its grid; both leave the bend of the sheet in the page, so its grid lines are curves, the column grid and the column map do not stand on them, and the traces are read off a time axis that is not theirs. The stage measures a smooth map from the pixels of the page, as the perspective stage left it, to the coordinates of its grid lines (`src/run/sheet_curl.py`): the phase of the 1 mm lines of both line families in windows 16 lines long, a quarter of a window apart; a projective part fitted to both families together and, per family, a polynomial of total degree 4 for what that part leaves; and the shift and the one scale for which the map moves the page least, without a rotation, since the grid gives the orientation. The page is then warped into the frame in which the lines are straight, bicubic and onto a new page of the same size, and measured again: what the straightened page still asks for is added to the map until `--sheet_curl_passes` measurements are in it, and the last measurement is the check that the lines are straight. A page comes out in one of four ways, which `qc.csv` holds in `sheet_curl`. `applied`: the page that is segmented and read is the straightened one. `dead band`: the map that would be applied, with all its passes composed, moves no measured point by `--sheet_curl_min_shift` (15 px unless given), and the page is left exactly as it is, without a `WARNING`; such a page was still measured, warped and measured again to check the map, so it costs the seconds named below. `refused: <reason>`: the page is kept as it is and prints `WARNING: sheet curl not straightened for record <record> (<reason>), keeping the page as it is.` The reasons are, in short: no grid lines, or too few windows with them; a fit that leaves more than `--sheet_curl_tolerance`; lines that span or cover less than half of the page; a map that folds, or that `stretches or squeezes the page by more than a factor of two somewhere`, which in practice is a page with holes in its grid that the map is held across; a correction of more than 5 % of the page width somewhere; a map whose inverse does not converge; a straightened page that gives no map, whose lines are not straight, or that still asks for more after the last pass; and an error inside the measurement, which is a refusal as well, so that one page does not end the run of a folder. `out of scope`: the page is not looked at, see `--sheet_curl_scope`. `applied` says that a map was accepted and the page was warped with it, not that the sheet is flat now: a smooth bend of the whole sheet is taken out, a fold or a local dent is not, since the map has no term for it, and a page with one can be `applied` without being straightened there. On synthetic generator pages bent by a known map, the estimator recovers the map to 0.03 to 0.35 px RMS for whole-sheet bends of 3 to 40 px. That was measured at the estimator's own threshold of 1 px with no dead band on top: with the default dead band of 15 px the smaller of these bends (3 px, and 10 px, which the map moves by 4 to 9 px on a clean page) are left as they are. `--verbose` prints one line per page, `Sheet curl for record <record>: <decision>, carrier …`, with the numbers of the fit (the carrier and its period, the windows of both line families, the residual of the projective part, of the field, after the warp and over all windows, the largest and the RMS shift where the lines were measured and the largest over the whole page, the scale, the smallest Jacobian determinant, the passes, the cover and the share of the windows the fit keeps), or `Sheet curl for record <record>: out of scope, not a page the paper normalisation warped onto the page frame`. In that line the decision of a page left in the dead band of `--sheet_curl_min_shift` reads `dead band of the accepted map`, with the numbers of the map that was measured, checked and not used; a plain `dead band` is a page whose first measurement already moves it by less than the 1 px of the estimator, where nothing was warped and `after the warp` is `nan`. The stage costs seconds, not fractions of one: on synthetic generator pages (2200 × 1700 px) 0.4 to 2 s for a page that is measured once and left alone and 1 to 6 s for one that is straightened with one measurement and its check; a page that takes all three passes is measured four times and warped three times. On the 345 development photographs, in the driver and in the run of *Results on real pages*, the stage took 0.9 to 9.7 s for a page it straightened (median 2.3 s, and 2.4 s for the 123 pages with three passes) and 0.2 to 6.6 s for a page it refused (median 0.6 s). A page that ends in the dead band of `--sheet_curl_min_shift` costs what a straightened one does, since its map is measured and checked before it is left alone. A mask that comes with the page is warped with it, or was saved for the straightened page and carries a record that says so: see below the table. Development-set numbers and limits are under *Results on real pages* below; the tolerance, the passes, the dead band and the scope were chosen on the development pages those numbers are on, and they were made with saved masks, each predicted on the bent page and warped with it. A run in which nnU-Net predicts on the straightened page was read once, on the development pages the stage straightens, and gives the same picture (the last paragraph before the limits there). `off` keeps the page of the perspective stage and measures nothing: signals, masks, printed lines and every column that `qc.csv` had are what they were without the stage, and only a mask that was saved for a straightened page gets a `WARNING` (below the table). |
| `--sheet_curl_tolerance` | a share of the grid line period, above `0` and up to `0.5` | `0.25` | Only for `--sheet_curl lines`: the largest RMS residual of the fit of the map that is accepted, as a share of the period of the grid lines. A page whose fit leaves more is refused, and so is one whose lines are further than that from a projective fit after the warp; both are kept as they are, with the `WARNING` of the stage. Clean synthetic pages suggest 0.1: the fit of a bent sheet ends below it there. With 0.1 the stage accepts 11 of the 345 development photographs of the ECG-Image-Database, since the fit of a photograph leaves 0.10 to 0.25 of a period on nine pages in ten (median 0.18). The default was therefore chosen on those photographs, where 0.25 accepts 135 pages with 2 passes and 188 with 3 (no dead band above the 1 px of the estimator, every page in scope); it has not been confirmed on other pages. What 0.25 lets through is not always a right map: on a drawn page with a known map, out of focus and bent by 38 px, a fit that leaves 0.17 of a period is accepted and is a wrong map, 9 px RMS and 25 px at its worst off the truth, which 0.1 refuses and the check on the straightened page does not catch. On the photographs the pages accepted between the two tolerances still gain more often than they lose (the counts are in the limits under *Results on real pages*). No residual is above half a period, so a value above 0.5, like one of 0 or less, is refused when the arguments are read. |
| `--sheet_curl_passes` | a whole number from `1` to `5` | `3` | Only for `--sheet_curl lines`: how many measurements may be composed into the map. The straightened page is measured again; while it still asks for 1 px or more, the threshold of the estimator, and this number is not reached, what it asks for is added to the map and the page is warped anew from the page as it was, so no page is resampled on top of a resampled one. The last measurement is the check that the lines are straight and adds nothing; with `1` the first map is checked and that is all. A page that does not settle, one that after the last pass still asks for 1 px or more and for a quarter or more of what the page asked for, is refused. The default was chosen on the 345 development photographs of the ECG-Image-Database: with a tolerance of 0.25, no dead band above that 1 px and every page in scope, 3 passes straighten 188 pages where 2 straighten 135. The dead band of `--sheet_curl_min_shift` does not end the passes: it is taken on the map when all its passes are composed (only a value of that flag below 1 takes the place of the 1 px here). |
| `--sheet_curl_min_shift` | any number of pixels, `0` or more | `15` | Only for `--sheet_curl lines`: the dead band of the stage. A page whose accepted map, with all its passes composed, moves no measured point by this much is left exactly as it is, without a `WARNING` (`dead band` in `qc.csv`, with the largest move of that map in `sheet_curl_shift_px`); a map that moves a point by exactly this much is applied. The dead band decides whether a map is used, not how it is measured. The estimator has a threshold of its own, 1 px, the value the stage was developed with and every development number was read with: a page whose first map moves it by less is a `dead band` page as well (all 176 flat synthetic pages that were tried stay inside it), a further pass is composed only while the straightened page asks for 1 px or more, and the check lets pass what a straightened page still asks for below it. Only a value of this flag below 1 takes the place of that threshold. So a page that is left in the dead band was still measured, warped and measured again, and a page whose grid lines carry no map that is accepted is `refused` with its `WARNING` whatever this value is, also when the map it would have had is a small one. The default was chosen on the development pages of the ECG-Image-Database, clean and mould-damaged scans included, among rules counted from the run at 1 px by leaving alone the pages whose map moves less (see *Results on real pages*). A small bend costs a page that reads well, since the page is resampled once more and a saved mask is warped nearest, which moves the edge of a trace by up to half a pixel: with no dead band above 1 px the stage straightens 193 of the 230 clean scans, and 67 of them lose more than 1 dB of page SNR where 17 gain. Counted with 15 px it straightens 2 of the clean scans, 139 of the 345 photographs, 151 of the 230 mould-damaged scans and none of the 115 rendered pages. What the dead band costs is a bend below it, which is left as it is: a drawn full-size page bent by 12 px, whose map moves it by 7.6 px, is read at 2.5 dB against the flat page with the defaults, as without the stage, and at 25.9 dB with `--sheet_curl_min_shift 2` (a drawn page, not a photograph). `0` is no dead band at all, of the stage or of the estimator: every page with a map that is accepted is warped, and every pass is used. |
| `--sheet_curl_scope` | `normalised`, `all` | `all` | Only for `--sheet_curl lines`: which pages the stage looks at. `all` looks at every page; what leaves a flat page alone is the dead band of `--sheet_curl_min_shift`, not the scope. `normalised` is only a page that the paper normalisation warped onto the Letter page, which is a photographed sheet: decided so in this run by `--paper_normalisation auto` or, with `--mask_folder`, said so by the `paper_normalisation` record of its mask, which is replayed and not decided again. Every other page is then `out of scope` and is not measured: a page the paper normalisation left as it is, only scaled or failed on, a page read with a mask that has no such record, and every page under `--paper_normalisation off` unless the record of its mask says `normalised`. So under `normalised` a normalised view that is given as the page with a mask without a record is out of scope, and only `qc.csv` and the `--verbose` line say so: such a run needs `all`. The default was chosen on the development pages (see *Results on real pages*). The mould-damaged scans, which the paper normalisation does not warp, gain most from the stage; the 20 photographs straightened at 1 px that the paper normalisation had not warped gain nothing in the median page SNR, and the two classifiers disagree on them; and the clean scans, which lose under a small dead band (67 of the 230 pages lose and 17 gain more than 1 dB at 1 px), are kept out by the dead band of 15 px, not by the scope. So `all` without that dead band is not a setting to read clean scans with. |
| `--time_mapping` | `bbox`, `grid` | `grid` | `grid` samples all leads on one shared column grid, for the standard 3x4 layout with rhythm strip. `bbox` stretches the bounding box of every lead to its length. |
| `--grid_pitch` | `page`, `fit` | `page` | Where the column pitch of that grid comes from: `page` from the page height, checked against the least squares fit of the column edges, `fit` from that fit. |
| `--grid_origin` | `masks`, `lines` | `lines` | `lines` refines origin and pitch of the column grid on the printed grid lines, assuming the first column starts on a grid line. `masks` uses the mask edges only. |
| `--grid_line_offset` | any number of pixels | `0.5` | How far right of the traces the printed grid lines sit. `0.5` is the matplotlib Agg snap of the generator images, `0` is for scans and photographs. It is a page unit, so it survives `--resolution`. |
| `--column_mapping` | `uniform`, `lines` | `lines` | `lines` reads the time axis off the printed grid lines themselves: in windows about half a line apart the 1 mm lines say where every millimetre is and the bold 5 mm lines which millimetre that is, so a sample is read at the x of its own millimetre (1/20 mm at 25 mm/s and 500 Hz) on a page whose grid and traces are stretched, compressed or stepped in places by printing, paper feed or scanning. The origin is the grid line the mask edges of all leads agree on, and the ripple of grid lines snapped to the pixels of a drawn page is taken out with `--column_mapping_median`. It needs `--time_mapping grid` and `--grid_origin lines` with the grid lines used, keeps the uniform grid with a `WARNING` when the map cannot be trusted (5 mm and 1 mm lines disagree, too few windows, a gap, a jump of a line, mask edges off the lines) and, byte identical, when it moves no sample by 1 px. On the development set (115 PTB-XL records, split by patient) of the ECG-Image-Database (v2, part D0: PTB-XL records printed, then scanned or photographed; references below the table), scored per lead against PTB-XL, the median lead SNR of the colour scans rises from 9.06 to 13.40 dB (paired median +3.41 dB, patient bootstrap 95 % CI 2.75 to 3.90) and that of the greyscale scans from 8.82 to 13.14 dB (+3.40, 2.84 to 3.95); the leads without a usable signal (no signal or an SNR below 0 dB) drop from 15.2 to 3.6 % and from 17.2 to 4.3 %. In the paired median of its leads, the fourth column gains 9.9 / 10.5 dB (colour / greyscale) and the rhythm strip II 6.7 / 7.5 dB, while the third column (V1–V3) changes by −0.59 / 0.00 dB. The rendered pages of the same records and all 208 synthetic pages (generator pages and synthetic defects applied to them, every page of the results below included) keep byte identical signals (dead band on 178 pages; on 26 augmented generator pages and 4 faint-grid pages the map is not trusted); on mould-damaged scans the map is used on 20–28 % of the pages, with a paired median of +0.85 to +1.06 dB on those. On the colour and the greyscale scans alike, a PTB-XL diagnostic classifier changes 3.8 % instead of 4.9 % of its record and class decisions against the digital signal. On the evaluation set of the same database (84 records of other patients, read once with this setting fixed beforehand), the colour scans rise from 8.92 to 12.83 dB (+2.55, 1.57 to 3.30) and the greyscale scans from 8.87 to 13.25 dB (+3.44, 2.70 to 4.05), the leads without a usable signal drop from 18.3 to 6.6 % and from 18.5 to 5.1 %, the rendered pages keep byte identical signals, and on the mould-damaged scans the pages that use the map gain +1.35 and +0.59 dB; the classifier's changed decisions, however, stay at 4.8 % on the colour scans and go from 2.9 to 3.6 % on the greyscale scans (pooled +0.4 points, patient bootstrap 95 % CI −0.6 to +1.3), so on these 84 records the SNR gain did not show as fewer changed decisions. It costs about 0.08 s per page. `uniform` makes every column P pixels wide from the grid origin on, the time axis before the column mapping. |
| `--column_mapping_median` | any width in grid mm, `0` = none | `13.5` | Only for `--column_mapping lines`: the width of the running median of the map's displacement from the uniform columns. The grid lines of a page drawn at 200 dpi are snapped to its pixels, its traces are not, and on the bold lines that snap repeats every 13.5 mm; scanned printouts carry the same ripple. A running median over that period takes it out and keeps a step or a stretch of the lines as it is. Without it the map follows that ripple, which the traces lack, and costs V1–V3 about 1.6 dB; the median recovers about half of that. `0` reads the samples on the map as measured. |
| `--grid_rescue` | `off`, `map` | `map` | Only with `--time_mapping grid`, `--grid_origin lines` and `--column_mapping lines`; with other flags it does nothing, `qc.csv` says `off` and the run says so in one line that is printed with `--verbose` only (which is on unless `--no-verbose` is given), since `map` is the default and such a run has not asked for it. A printed and scanned sheet can have a column a few per cent narrower than the others. The mask edges of such a page are off one uniform column grid, so the page falls back to `--time_mapping bbox`, or its grid lines are off the origin of the mask grid, so the page keeps the grid of the masks and gets no column map. `map` tries such a page a second time: the fit with twice the edge tolerance (4 % of the column pitch instead of 2 %), the grid lines on their own pitch, with the mask grid turned about its centre and without the origin that was refused. The second try is kept only if the column map then stands on it, and the page is read on that map (`column_mapping` is `lines`); a page whose map does not stand is exactly what it is with `off`: the same signals, the same `WARNING` lines and, but for `grid_rescue`, the same `qc.csv` row. A page that fails for another reason (no rhythm strip, a column without a short lead, no grid lines) is not tried again. A rescued page prints one line with or without `--verbose`, `Grid rescue for record <record>: <fit, phase or fit+phase>, read on the column map of the second try (…)` with the grid and the numbers of its map, in place of the `WARNING` of the check it failed; `--verbose` also says why a second try was refused. Development-set numbers and limits are under *Results on real pages* below. `off` does not try again: such a page falls back to `--time_mapping bbox` or keeps the grid of the masks, the behaviour before `map` became the default. |
| `--row_mapping` | `off`, `lines` | `lines` | Only on a page that is read on its column map. That map is one for the page, measured on all its rows together, while a printed and scanned sheet moves its rows against each other by fractions of a pixel, and a lead follows the grid lines of its own row. `lines` measures the 1 mm lines once more in the band of each of the four layout rows (three rows of short leads and the rhythm strip) against the page map, which says which line a window is on, and reads the leads of a row on the page map moved by what the lines of that row say, smoothed with `--row_mapping_median`. A row keeps the page map when its lines carry no map of its own: fewer than 80 % of its windows within 0.3 line of the page map, a quarter of a column without one, or a map that does not grow along x. A page without a column map in use (dead band, map not trusted, `--column_mapping uniform`) is read as with `off`, byte for byte. Development-set numbers and limits are under *Results on real pages* below. `off` reads every lead on the page map, the reading before `lines` became the default. |
| `--row_mapping_median` | any width in grid mm, `0` or less = none | `27` | Only for `--row_mapping lines`: the width of the running median of a row's displacement from the page map, which takes the ripple of grid lines snapped to the pixels out of it, as `--column_mapping_median` does for the page map. The default is two periods of that ripple, twice the width the page map takes: on the development scans, read with `--grid_rescue map` and `--row_mapping lines` and compared with both `off`, it left 250 leads more than 1 dB worse where one period, 13.5 mm, left 295 (only these two widths were read). `0` or less reads a row on its displacement as measured. |
| `--trace_estimator` | `mask`, `ink` | `ink` | `ink` weights the mean row of a column by the ink under the mask, which puts the row on the core of the stroke; a lead whose stroke is too faint keeps the mask mean. `mask` is the mean row of the binary mask. |
| `--trace_shift` | `off`, `page` | `page` | `page` measures once per page how far right of the lead masks the ink of the steep strokes sits and reads every lead on a column axis moved by that. `off` reads every lead where its mask is. |
| `--sharpen` | `none`, `bandlimited` | `bandlimited` | `bandlimited` restores the 20–40 Hz band that the column aperture and the linear interpolation attenuate: the aperture-corrected (a = 1/8) cubic spline through the columns minus the linear trace, low-passed at 35 Hz (zero-phase Butterworth, order 4), is added to the linear trace, so only the band below 35 Hz changes and steep QRS spikes cannot ring. Paired on the same masks, the median gain per lead is 1.41 dB on the 96 generator pages below, with no lead of the 15 conditions below more than 1 dB worse, and 1.17 dB on 2,000 synthetic pages of 500 PTB-XL records (clean, augmented, rotated, simulated scan); on those pages a PTB-XL diagnostic classifier changes 138 instead of 233 of its 10,000 record and class decisions against the digital signal. It costs about 8–14 ms per page. `none` interpolates the column profile linearly, the trace before the sharpening. |
| `--baseline` | `page`, `leads` | `leads` | `page` takes the baseline from the page geometry, scaled with the column pitch. `leads` additionally shifts it so that the Einthoven and Goldberger sums have no DC offset. |

`python -m src.run.digitize --help` lists the remaining options (`--interpolation`, `--enable_tta`, `--lead_placement`, `--save_mask`, `--mask_folder`, `--fold`, `--device`).

`--save_mask` writes the mask of a page as `<record>_mask.png` with a `<record>_mask.json`
that says which frame the mask lives in, and `--mask_folder` reads both back instead of
running the model. A page that `--paper_normalisation auto`, the default, decided gets
one more key in that JSON, `paper_normalisation`: the version of the decision rules, the
numpy and cv2 versions, the decision and its reason, the size (width, height) and the
SHA-1 of the pixels of the page as it came (`input_size`, `input_sha1`) and of the page
the model saw (`output_size`, `view_sha1`), and the numbers that build the second from
the first, the size the page is shrunk to first and the 3 × 3 warp matrix of a
`normalised` page (`pre_size`, `warp`, with `homography` and `orientation` for the
record) or the size of a `scaled` one (`resize`, with `scaled_from` and
`scale_factor`). A mask only fits the
page it was predicted on, and a normalised page has the size of every other page, so
with `--mask_folder` a page whose mask has this record is never decided again, whatever
the flag says, but checked against it before any stage runs:

- a page that already is the recorded view (size and SHA-1) is used as given;
- with `--paper_normalisation auto`, a page that is the recorded input (size and SHA-1)
  of a `normalised` or `scaled` view is warped or shrunk again with the recorded numbers,
  and the result must have the recorded size and SHA-1;
- everything else stops the run with a `ValueError` that names the record and says
  `the mask does not fit`: another page, another view, a record of an unknown version or
  decision, or the input page given with `--paper_normalisation off`. `-f` does not
  cover it, as it only skips a page without signals.

The JSON of a mask is read for this record before any stage runs, with either flag, so a
`<record>_mask.json` that cannot be parsed now stops the run there, with the error of the
JSON parser.

A mask without the record, which is every mask saved with `--paper_normalisation off`,
by a version without the stage, or again from such a mask, has no such pixel check: its page is used as given,
with `as given` in `qc.csv` under the default `--paper_normalisation auto` and `off`
under `--paper_normalisation off`, and only the size of the mask and the frame warnings
of the rotation, the perspective and, for a mask saved for a straightened page, the
sheet curl stand between it and another page. So a folder of
such masks is read to the same signals with either flag. A mask that is refused can
still be used: give the run the page it was predicted on, which is the recorded view
itself or, with `--paper_normalisation auto`, the input page the view is rebuilt from;
or, to take a page as given without the check, remove the `paper_normalisation` key
from the JSON of the mask. The message of every refusal ends
on these two ways out: `Give the page the mask was predicted on, or remove the
paper_normalisation key from the mask's JSON to take the page as given.` A mask without
the record that has another size than its page stops the run as it always did
(`the mask was predicted for another --resolution`), and under
`--paper_normalisation auto` the message adds that the mask has no record, so the page
was used as given: the page to give then is the view the mask was predicted on. With
`--mask_folder` and `--save_mask` together the record is written out again unchanged.

The frame of a mask also depends on `--perspective_tolerance`. A page whose perspective
fit lies between 0.1 and 0.15 of the grid line period was kept as it was rotated while
0.1 was the default; now it is rectified, unless it is straight enough to be left
alone. A mask that a run without the flag saved for such a page before 0.15 became the
default was therefore predicted in another frame than a run with the defaults puts the
page in. The run does not stop for that: it prints
`WARNING: mask of record <record> was predicted in a frame whose corners are <distance> px off the one used now; the mask does not fit.`
and reads the page with a mask that does not fit it. The warning compares the frame
recorded in `<record>_mask.json`, so a mask folder without that file gets no warning.
Read a mask folder that was saved at 0.1, with or without those files, with
`--perspective_tolerance 0.1`, which gives the frames and the signals it gave before;
masks saved at 0.15 are read at 0.15. A mask that is saved again by a run with
`--mask_folder` and `--save_mask` carries the frame of that run in its JSON, whatever
frame it was predicted in, so do not save masks of before the change again under the
default. On the development set of the ECG-Image-Database named below, the pages in
another frame at 0.15 are 24 of the 230 clean scans, 89 of the 230 mould-damaged scans
and 100 of the 345 photographs, and none of the 115 rendered pages. The mask of every
other page fits at either tolerance: a page whose fit is within 0.1, one that is
refused at 0.15 as well, and one that is accepted above 0.1 but straight enough to be
left alone are in the same frame at both.

The frame of a mask depends on `--sheet_curl` as well, and for this stage the JSON says
which frame it is. A run with `--sheet_curl lines` and `--save_mask` saves the mask in
the frame it goes on with, which is the straightened page where the stage was applied,
and gives the JSON one more key, `sheet_curl`: the version of the record, the decision
(`applied`, `dead band`, `refused` or `out of scope`) and the reason of a refusal, the
scope, the dead band (`dead_band`), the options and constants that define the map
(`min_shift` among the options is the threshold of the estimator, 1 px, not the dead
band), the largest shift of the map
(`shift_max`), and for an `applied` page its size and the map itself at a lattice of
9 × 7 points over the page, its borders included (`points`, rows `[x, y, X, Y]`: the
point (x, y) of the page lies at (X, Y) on the straightened one). With
`--sheet_curl off`, the default, the JSON is byte for byte what it was. With
`--mask_folder` the record says what is done with the mask:

- a mask without the record, which is every mask saved without the stage, and a mask
  whose record says anything but `applied`, was predicted on the page as the
  perspective stage left it. When the run straightens the page, the mask is warped with
  it, nearest neighbour, so that it holds no label that was not in it; when the run
  keeps the page, the mask is used as it is. So masks saved before the stage can be
  read with `--sheet_curl lines`, and the development-set numbers under *Results on
  real pages* were made that way;
- a mask whose record says `applied` was predicted on the straightened page. It is
  used as it is and never warped again, and it only fits a run that straightens the
  page by the same map, to within 0.1 px at the points of the record.

A mask of the second kind that does not fit gets a `WARNING` in the words of the frame
check, and the run goes on with it, as it does after the frame warnings of the rotation
and the perspective. When the map of the run is another one, it prints
`WARNING: mask of record <record> was predicted on a page straightened by a sheet curl map that is <distance> px off the one used now; the mask does not fit.`
and when the run does not straighten the page at all, because the stage is `off` or the
page is out of scope, inside the dead band or refused,
`WARNING: mask of record <record> was predicted on a page whose sheet curl was straightened, the page is now kept as it is (<why>); the mask does not fit.`
A record of a version this code does not read, an `applied` record that does not hold
all 63 points of its map and the size of its page, and a record that is no JSON object
are said in the same way (`has a sheet curl record of version …`,
`… without the points of its map`, `… that is no object`), and such a mask is used as it
is and not warped; `null` is no record. The map of a page depends on the page the
perspective stage hands on and on the options of the stage, so read masks that were
saved for straightened pages with the flags they were saved with. That includes the
dead band: a page that was straightened at `--sheet_curl_min_shift 1` and is left alone
at the default of 15 px gets the second `WARNING`, with `(dead band)`, so a folder of
masks saved at one dead band is read with that dead band, and with the tolerance, the
passes and the scope it was saved with; `dead_band`, `options` and `scope` in the
record say which they were. As for the perspective, the check reads what
`<record>_mask.json` holds, so a mask folder without that file is taken to hold masks
of pages that were not straightened. And a version of this code from before
`--sheet_curl` does not know the record at all: it reads a mask that was saved for a
straightened page without any `WARNING` and writes the signals of a mask that does not
fit. Keep masks of straightened pages in folders of their own, and do not read them
with an older version.
A mask that came in the straightened frame, or with a record that cannot be read,
keeps the record it came with when a run with `--mask_folder` and `--save_mask` saves
it again, also when that run keeps the page (stage `off`, out of scope, dead band,
refused) or straightens it by another map: it has printed the `WARNING` then, the mask
it saves is the one it was given, and a later run that straightens the page still uses
it as it is. Every other mask is saved with the record of the run that saves it, and
with `--sheet_curl off` a mask that came without a record is saved without one, byte
for byte as before.
In a run without `--mask_folder` nnU-Net predicts on the straightened page; what it
makes of one was read once on development pages, see *Results on real pages*.

The real-page numbers in the table, in the quality control section and under *Results on
real pages* below are on the ECG-Image-Database, version 2
(<https://www.kaggle.com/datasets/physionet/ecg-image-database>), whose images are
licensed CC BY-ND 4.0, so only aggregate numbers are given here. Those in the table and
in the quality control section were measured before `--grid_rescue map` and
`--row_mapping lines` became the defaults: unless it says otherwise (the row of
`--row_mapping_median` does), they are read with `--grid_rescue off --row_mapping off`.
What the two flags change on the same pages is under *Results on real pages*. And
every real-page number that is not about `--perspective_tolerance` itself (the numbers
in the table, those of the quality control section, the evaluation-set numbers and the
paragraphs on the two flags) was measured before 0.15 became the default of that flag,
and is read at 0.1. `--sheet_curl lines` is off unless it is asked for and is in none
of these numbers. Its own numbers, in the rows of its options in the table and in the
last paragraphs of *Results on real pages*, are the exception to both rules: they were
read, with the stage and without it, with the defaults as they are now
(`--grid_rescue map --row_mapping lines --perspective_tolerance 0.15`). The
dataset page asks for these citations:

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
- **Paper normalisation:** `paper_normalisation`, under the default
  `--paper_normalisation auto` the decision for the page, `pass_chain`,
  `pass_fullframe`, `normalised`, `scaled` or `failed`, or `as given` for a page read
  with a mask from `--mask_folder` that has no record of one; `off` with
  `--paper_normalisation off`. A mask with such a record gives its decision with either
  flag. The column is the 13th, right after `max_offset_deviation`, and rows are
  appended to a `qc.csv` that is already there under the header it has: do not append
  to a `qc.csv` written by an older version, which lacks the column, but use a fresh
  output folder.
- **Resolution:** `grid_period_px`, the measured grid line period, and
  `resolution_scale`, the factor the page was resampled by (1 if it was left alone).
- **Rotation:** `rotation_angle`, the angle the page was turned by, `rotation_coarse`,
  the Hough angle before the grid lines, and `rotation_residual_px`, the residual of the
  grid line phase fit.
- **Perspective:** `perspective_shift_px`, the largest displacement the rectification
  causes on the page, `perspective_residual_px`, the residual of its fit, and
  `perspective_residual_rel`, the residual as a share of the grid line period it was
  measured on, the largest of the rounds of the fit (NaN when the stage did not
  measure or its fit got no residual). It is the number the stage compares with
  `--perspective_tolerance`, 0.15 by default, so it reads the same on pages of any
  grid line period, and a share above 0.1 on a page without a `WARNING` for the
  perspective is a page that the stage's own tolerance of 0.1 refuses and only the
  wider one lets through, as `--verbose` says of it in one line. This column is the 35th,
  appended after `row_mapping`, so no column that was there moves; as for
  `paper_normalisation`, do not append to a `qc.csv` written by an older version, but
  use a fresh output folder.
- **Sheet curl:** `sheet_curl`, `off` without `--sheet_curl lines`. With it: `applied`
  for a page that was straightened, `dead band` for a page whose map, all passes
  composed, moves no measured point by `--sheet_curl_min_shift` (or whose first map
  moves none by the 1 px of the estimator), which is left as it is, `refused: <reason>` for a
  page whose grid lines carry no map that is accepted, which is kept as it is with a
  `WARNING`, and `out of scope` for a page that `--sheet_curl_scope` does not look at.
  And `sheet_curl_shift_px`, the largest move in pixels that the map of an `applied` or
  a `dead band` page makes where its grid lines were measured (NaN for every other
  page). `applied` says that the page was warped with a map that was accepted, not
  that the sheet is flat now: a fold or a local dent stays. These two columns are the
  36th and the 37th, appended after `perspective_residual_rel`, so no column that was
  there moves; as for `paper_normalisation`, do not append to a `qc.csv` written by an
  older version, but use a fresh output folder.
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
- **Grid rescue:** `grid_rescue`, under the default `--grid_rescue map` empty for a page
  that failed no check that is tried again, `fit`, `phase` or `fit+phase` for a page
  that is read on the column map of its second try, after the fit, the grid lines or
  both were tried again, and `refused: <reason>` for a page whose second try was not
  kept, which is then read as with `--grid_rescue off`; `off` with `--grid_rescue off`
  or in a run without the flags the rescue needs.
- **Row mapping:** `row_mapping`, under the default `--row_mapping lines`
  `lines <k>/4`, the number of layout rows read on a map of their own, the others on the
  page map, or `page map not used` for a page that has no column map in use; `off` with
  `--row_mapping off`.
  These two columns are the 33rd and the 34th, appended after `column_mapping_shift_px`,
  so no column that was there moves; as for `paper_normalisation`, do not append to a
  `qc.csv` written by an older version, but use a fresh output folder.
- **Baseline:** `baseline_scale`, the column pitch over the pitch the page height
  implies, `baseline_shift_px`, the shift that was applied, and
  `baseline_disagreement_px`, the difference between the Goldberger and the Einthoven
  estimate.

A `WARNING` line is printed whenever a stage cannot do what it is asked and falls back:
`--paper_normalisation auto` does not find the paper of a page that needs it (the page
is kept as it is, or only rescaled); the grid lines are unusable for the resolution (the
page is kept as it is), for the
rotation (the whole degree Hough angle is kept), for the perspective (the rotated page
is kept) or for the grid origin (the grid of the masks is kept); with
`--sheet_curl lines`, the grid lines of a page in its scope carry no map that is
accepted (the page of the perspective stage is kept); no column grid fits the
page, so `--time_mapping bbox` is used for it; the Einthoven and the Goldberger baseline
estimate disagree by more than the tolerance, so the page baseline is kept; the grid
line map of `--column_mapping lines` cannot be trusted, so the uniform grid is kept; or
a lead mask has a gap inside its column, which is interpolated linearly. A page whose fitted
column pitch does not match the pitch of the page height prints a line as well and uses
the fitted one. Under the default `--grid_rescue map`, a page that is rescued prints its
`Grid rescue for record <record>: …` line in place of the `WARNING` about the column grid
or the grid origin, and a page whose second try is refused keeps that `WARNING`; with
`--grid_rescue off` every such page prints it.

Two more `WARNING` lines come without a fallback. The first is the Einthoven check.
Every page with signals prints a `QC <record>: Einthoven RMS …` line with its lead
consistency values.
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
below 12 dB, 0.17976, rounded down to 0.1797. On that set, read with the defaults but
for `--grid_rescue off --row_mapping off --perspective_tolerance 0.1`, it
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
done outside this repository, by the view builder whose image code
`--paper_normalisation auto` now holds). It flags 1 of the 115 rendered pages, which is not
below 12 dB. On synthetic pages it flags 3 of the 192 pages of the results below, none
of them below 12 dB, and 98 of 2,000 synthetic pages of 500 PTB-XL records (read before
the column mapping), 5 of the 6 below 12 dB among them. The Goldberger ratio
(aVR + aVL + aVF) is not used: at the same sensitivity it flags 106 instead of 57 of the
clean scans at or above 12 dB.

The second is the rescale of an output out of range. An output with a sample beyond
±10 mV is, as it always was, rescaled to −1..1 over all its leads before it is written,
and right after the line that says so,
`Signal out of range for record <record>, normalizing to range between 1 and -1`, the
page prints
`WARNING: output rescaled for record <record> (range <lowest> to <highest> mV, outside +-10 mV): the written signals are rescaled to -1..1 and are not in millivolts, check this page.`
with the smallest and the largest sample of its signals, to three decimals (a sample
less than 0.0005 mV beyond the limit reads as the limit). The line changes neither the
signals nor `qc.csv`, which has no column for it. Such a page is always one to check,
whatever the Einthoven check, which reads the signals before the rescale, says of it. On
the development set, read as for the numbers above (the photographs after the same paper
normalisation), the output of 60 pages was rescaled: 13 of the 230 mould-damaged scans
and 47 of the 345 photographs, none of the 345 rendered pages and clean scans. The
Einthoven check flags 57 of the 60, and the other three are the only photographs it does
not flag.


#### Results on real pages

`--grid_rescue map` and `--row_mapping lines` are the defaults; both were `off` before
they became that. The numbers on the two flags, in this and the next four paragraphs,
were all read at `--perspective_tolerance 0.1`, the tolerance of the perspective stage
itself and the reading when they were made; what 0.15, the default now, changes on the
same pages comes after them, and what `--sheet_curl lines`, off by default, does on the
photographs is at the end of this section. The numbers here are **development-set
numbers**: the tolerance of the second fit, the pitch of the second try and the width
of the row median were chosen on these pages, and the evaluation set of the database
has not been read with either flag. They are on the 230 clean colour and greyscale
scans of the development set of the ECG-Image-Database (115 PTB-XL records, split by
patient; references above), read from saved masks, scored per lead against PTB-XL, and
compared, paired on record and lead, with the same pages read with
`--grid_rescue off --row_mapping off`.

With both flags as they are given above (row median 27 mm), the median lead SNR of the
scans rises from 13.24 to 15.02 dB (paired median +1.39 dB, patient bootstrap 95 % CI
1.21 to 1.58), and the leads without a usable signal (no signal or an SNR below 0 dB)
drop from 3.9 to 1.7 %. What a PTB-XL diagnostic classifier makes of the scans is read
in more than one way, since a count of changed decisions moves with the decisions that
sit on their threshold:

- of its 1,150 record and class decisions against the digital signal it changes 32
  instead of 44 (2.8 instead of 3.8 %; 95 % CI of the difference −23 to −2);
- of the 968 decisions that the digital signal puts at least 1 logit from the
  threshold it changes 7 instead of 17;
- the mean absolute change of its class probabilities falls from 0.0264 to 0.0194 (by
  0.0069, 95 % CI 0.0042 to 0.0102);
- on the pages that print no `WARNING` of a stage (resolution, rotation, perspective,
  column grid, grid origin or column mapping; 205 pages with both flags, 196 with both
  `off`) it changes 19 of 1,025 decisions (1.9 %) instead of 22 of 980 (2.2 %).

With a row median of 13.5 mm, one period of the ripple, the same reading gives
15.03 dB (+1.33 dB, 1.20 to 1.53), 34 changed decisions (−21 to 0), 8 of those at
least 1 logit from the threshold, a mean absolute change smaller by 0.0067 (0.0039 to
0.0100) and 21 of 1,025 (2.0 %) on the pages without such a `WARNING`. The 115 rendered
pages of the same records and all 208 synthetic pages keep byte identical signals with
`--grid_rescue map`, with `--row_mapping lines` and with both (read with either
median; no row is measured on them, so its width does not enter): none of those pages
fails a check that is tried again and none has a column map in use, so they show that
the flags leave such pages alone, not how well the two maps work.

Limits. With both flags 1,547 of the 2,760 leads are more than 1 dB better and 250
more than 1 dB worse (1,545 and 295 with 13.5 mm). It is the row maps that make leads
lose (with the rescue alone 2 leads are more than 1 dB worse), and on the colour scans
the leads that lose were read well before (median 17.1 to 14.8 dB). Of the 24 scans
with a page SNR below 5 dB, 12 are left, all of them pages whose perspective stage
could not use the grid lines, which a saved mask cannot mend. And 115 records are few
for a count of changed decisions: the interval of its difference is wide, and with the
13.5 mm median it includes zero.

The other pages of the development set, read the same way with both flags (row median
27 mm). On the mould-damaged scans (115 colour, 115 black and white) the flags change
102 pages, the leads without a usable signal drop from 49.9 to 45.8 %, and the
classifier changes 92 instead of 105 and 108 instead of 113 of its 575 decisions per
condition. That is the rescue: alone it gives 96 and 104, and the row maps add −4
(95 % CI −10 to +2) on the colour and +4 (−2 to +12) on the black-and-white scans. On
the 345 photographs (three conditions; original pages with the saved masks and their
`paper_normalisation` records) the flags change 79 pages and the classifier changes
493 instead of 515 of its 1,725 decisions (95 % CI of the difference −35 to −11; 502
with the rescue alone), 382 instead of 402 of the 1,452 that the digital signal puts
at least 1 logit from the threshold, and the mean absolute change of its class
probabilities falls by 0.0085 (0.0049 to 0.0127). Both kinds of page stay far from the
digital signal with the flags or with both `off` (median lead SNR below 1 dB on the
mould-damaged scans and below 0 dB on the photographs), and at
`--perspective_tolerance 0.1` so few of them print no `WARNING` of a stage (59
mould-damaged scans, 16 photographs) that the last reading above stands on few pages
there; it is given below, next to the one at 0.15.

`--perspective_tolerance 0.15` is the default; it was 0.1, the tolerance of the
perspective stage itself, before, and everything above in this section was read at 0.1.
The numbers of the tolerance are **development-set numbers** as well: the value 0.15
was chosen on the 24 clean development scans named next, and the evaluation set of the
database has not been read with it. They are on the same development pages, read with
`--grid_rescue map --row_mapping lines` at 0.15 and compared, paired, with the same
reading at 0.1 (patient bootstrap 95 % CI). A page that 0.15 rectifies and 0.1 does not
is in another frame, which its saved mask does not fit, so the masks of those pages
were predicted once more, in that frame.

On the 230 clean scans the perspective stage refuses 24 pages at 0.1, with a residual
of 0.100 to 0.139 of the grid line period, and rectifies all 24 at 0.15. On those 24
pages the median page SNR rises from 5.49 to 13.92 dB (lead SNR, paired median
+6.73 dB, 95 % CI 3.77 to 8.94), no page is more than 1 dB worse, and the classifier
changes 4 instead of 13 of its 120 record and class decisions against the digital
signal (−9, 95 % CI −15.4 to −3.3). Over the 230 pages, none instead of 12 has a page
SNR below 5 dB, the median lead SNR goes from 15.02 to 15.26 dB, the classifier changes
23 instead of 32 of its 1,150 decisions (2.00 instead of 2.78 %) and 1 instead of 7 of
the 968 that the digital signal puts at least 1 logit from the threshold, and the mean
absolute change of its class probabilities falls from 0.0194 to 0.0149. The other 206
scans keep byte identical signals.

On the 230 mould-damaged scans 89 pages change frame at 0.15, 48 colour and 41
black-and-white ones. On those 89 the median page SNR rises from −1.52 to 6.59 dB (lead
SNR, paired median +4.95 dB, 3.23 to 7.00) and the classifier changes 58 instead of 100
of its 445 decisions (−42, −65.5 to −19.4); 73 pages gain more than 1 dB of page SNR
and 6 lose more than 1 dB, 3 of them out of the 7 pages that were at or above 12 dB at
0.1. Over the 230 pages the classifier changes 158 instead of 200 of its 1,150
decisions, the leads without a usable signal drop from 45.8 to 32.5 %, the median lead
SNR goes from 0.84 to 4.21 dB, and 114 instead of 151 pages have a page SNR below 5 dB.

On the 345 photographs (three conditions; original pages, `--paper_normalisation auto`)
100 pages change frame at 0.15 (46, 44 and 10 of the three conditions). On those 100
the median page SNR rises from 0.64 to 3.33 dB (lead SNR, paired median +1.07 dB, 0.55
to 1.98), and the classifier changes 96 instead of 119 of its 500 decisions, but the
interval of that difference includes zero (−23, −45.6 to +1.9), and one page in five
gets worse: 20 pages lose more than 1 dB of page SNR, while 58 gain more than 1 dB. The
20 that lose were read better than the rest before (median page SNR 5.11 dB at 0.1,
against 0.64 dB over the 100, and −1.71 dB at 0.15): 10 of them were at or above 5 dB
at 0.1, out of 18 such pages among the 100, and 7 of those 18 are below 5 dB at 0.15.
This looks like a projective correction laid on a curled sheet breaking a page that
worked; the curl of these pages was not measured then (`--sheet_curl lines` straightens
19 of the 20 in the run at the end of this section, and 12 of those gain more than
1 dB and none loses, though a group picked for a bad outcome tends to gain under any
change). Over the 345 pages the classifier
changes 470 instead of 493 of its 1,725 decisions (−23, −48.7 to +1.0, an interval that
includes zero as well), the leads without a usable signal go from 57.2 to 53.2 %, the
median lead SNR from −0.83 to −0.34 dB, and 270 instead of 292 pages have a page SNR
below 5 dB. The paper normalisation, which keeps 0.1, decided all 100 pages as before
and gave them the same pixels.

The pages that 0.15 does not move keep byte identical signals on every kind of page, as
the 206 clean scans above do. A run of all 920 development pages from saved masks at
0.15 (the 213 pages that change frame from the masks predicted in the new frame, the
707 others from the masks saved before) gives byte identical signals to those scored
here on the 213 and to the reading at 0.1 on the 707: the 206 clean scans, the 141
other mould-damaged scans, the 245 other photographs and all 115 rendered pages of the
same records. None of the rendered pages is accepted above 0.1.

The pages that print no `WARNING` of a stage (resolution, rotation, perspective, column
grid, grid origin or column mapping) are another group at 0.15 than at 0.1. A page that
0.15 accepts no longer prints the `WARNING` of the perspective stage, so the group
grows by pages that were refused before; such a page is told by a
`perspective_residual_rel` above 0.1 in `qc.csv` and by the line of `--verbose`. On the
clean scans 229 instead of 205 of the 230 pages print none, and on them the classifier
changes 23 of 1,145 decisions (2.0 %) instead of 19 of 1,025 (1.9 %). On the
mould-damaged scans they are 104 instead of 59 of the 230 pages, with 40 of 520
decisions changed (7.7 %) instead of 15 of 295 (5.1 %), and on the photographs 45
instead of 16 of the 345 pages, with 17 of 225 (7.6 %) instead of 2 of 80 (2.5 %). So
at 0.15 the absence of a `WARNING` of a stage says less about a page than it does at
0.1, on the mould-damaged scans and the photographs above all.

Limits. The mould-damaged scans and the photographs stay far from the digital signal at
0.15 as well: 114 of the 230 scans and 270 of the 345 photographs are below 5 dB page
SNR, 166 and 331 below 12 dB, and the 331 photographs are as many as at 0.1. Eleven
further pages, 5 black-and-white mould-damaged scans and 6 photographs, that 0.1
refuses with a logged residual of 0.101 to 0.150 of the period stay refused at 0.15:
there the first round of their fit is accepted, and the second, measured on the
rectified page, ends above the tolerance. A smaller tolerance does not do better on
both kinds: counting only the pages whose residual is within it, 0.125 accounts for
−31 of the −42 changed decisions on the mould-damaged scans and for −10 of the −23 on
the photographs, and 0.13 for −43 and −9. And the change of page SNR does not go with
the size of the
residual (correlation −0.06 on the mould-damaged scans, +0.07 on the photographs), so
the pages that lose cannot be told by it. The 96 generator pages of the results below
(clean, augmented, rotated) and the other synthetic pages keep byte identical signals
at 0.15. All of this is development-set evidence: 0.15 was chosen on the 24 clean
scans, the decision to make it the default was taken with the mould-damaged scans and
the photographs of the same 115 records in view, and the evaluation set has not been
read with it. So the numbers say what the tolerance does on these pages, not what it
would do on others. `--perspective_tolerance 0.1` gives the reading before 0.15 became
the default, and a mask saved at 0.1 is read with it (see below the options table).

`--sheet_curl lines` is off by default, and everything above in this section was read
without it. Its numbers are **development-set numbers**, and the options of the stage
are fitted to the pages they are on. The options the stage was planned with were fixed
on synthetic pages before a photograph was read: a tolerance of 0.1, 2 passes and a
threshold of 1 px. With them the stage accepts 11 of the 345 development photographs,
since the fit of a photograph leaves 0.10 to 0.25 of a grid line period on nine pages
in ten (median 0.18), so the planned reading changes next to nothing. Everything below
is exploratory: the tolerance of 0.25 and the 3 passes were chosen on these
photographs, in three runs on them, and the dead band of 15 px and the scope `all` were
then picked among rules composed from those runs on the same development pages (dead
bands of 5, 10 and 15 px and both scopes among them), the clean and the mould-damaged
scans of the same records included. So the numbers say what the stage does on these
pages, not what it would do on others, and the evaluation set of the database has not
been read with it. They are on the development set (115 PTB-XL records): the 345
photographs of three conditions, and with every page in scope also the 230 clean
scans, the 230 mould-damaged scans and the 115 rendered pages; the original pages, read
from saved masks with `--paper_normalisation auto` and the defaults of the other
stages, and compared, paired, with the same pages read without the stage. All pages of
a group are in every denominator, those the stage refuses or leaves alone included; the
classifier is the one of the paragraphs above unless a second one is named, and the
intervals are patient bootstrap 95 % CIs. The stage was applied after the segmentation:
the saved masks were predicted on the pages before they were straightened and are
warped with their pages (see below the options table). So in these numbers nnU-Net has
not seen a straightened page; a run without `--mask_folder`, where the model predicts
on the straightened page, is the subject of the last paragraph before the limits. The
numbers were made with the driver the stage was developed with, outside
this repository, whose estimator `src/run/sheet_curl.py` holds. The flags themselves
give the signals of that driver, file for file, on synthetic pages and on all 920
development pages: read from the saved masks with `--sheet_curl lines
--sheet_curl_min_shift 1`, they straighten the same 188 photographs, 193 clean scans
and 207 mould-damaged scans and none of the rendered pages.

The run the numbers come from is not the default one: it has the tolerance of 0.25 and
the 3 passes of the defaults and every page in scope, but no dead band above the 1 px
of the estimator (`--sheet_curl_min_shift 1`). On the photographs 188 of the 345 pages
are straightened with it (57, 83 and 48 of the three conditions) and 157 are refused.
The leads without a usable signal drop from 53.2 to 43.7 %, the median lead SNR goes
from −0.34 to 1.97 dB, 219 instead of 270 pages have a page SNR below 5 dB and 314
instead of 331 one below 12 dB; 99 pages gain more than 1 dB of page SNR and 22 lose
more than 1 dB. The classifier changes 408 instead of 470 of its 1,725 record and class
decisions against the digital signal (−62, 95 % CI −89.8 to −35.0) and 307 instead of
361 of the 1,452 that the digital signal puts at least 1 logit from the threshold (−54,
−78.6 to −30.2), the mean absolute change of its class probabilities falls from 0.1960
to 0.1765 (by 0.0195, 0.0114 to 0.0274), and a second classifier changes 430 instead of
462 of its decisions on the same pages (−32, −57.0 to −5.1). By condition the
classifier changes 104 instead of 129, 155 instead of 174 and 149 instead of 167 of its
575 decisions; the interval of the second of these, the stained photographs, touches
zero (−19, −37.7 to 0.0), and on the mould-damaged photographs the second classifier
changes 159 instead of 153 (+6, −6.9 to +18.4).

Where that lands, among the 188 photographs straightened in that run. The 131 that were
below 5 dB page SNR go from a median page SNR of −1.65 to 4.23 dB (75 gain and 6 lose
more than 1 dB). Of the 11 that were at or above 12 dB, 5 lose and 3 gain; of the 30
that the map moves by less than 10 px, 10 lose and 5 gain; and the 20 that the paper
normalisation did not warp gain nothing in the median page SNR (−3.7 to −3.4 dB; 5 gain
and 1 loses more than 1 dB, and with them left alone the classifier changes 6 decisions
more and the second classifier 5 fewer). The first two groups are cut on the page SNR
without the stage, which is one of the two numbers compared, so part of what they show
is regression to the mean: pages picked for reading badly tend to gain, and pages
picked for reading well tend to lose, under any change. They say where the change
lands, not how large it is; its size is in the paragraph above, over all 345 pages.

Scans and rendered pages were read with the same run, every page in scope, on the 575
rendered and scanned development pages of the same records. The 115 rendered pages
stay inside the 1 px of the estimator and keep their signals. On the 230 clean scans
the stage costs: 193 pages are straightened (the map of a clean scan moves it by 6.5 px
in the median), 67 pages lose and 17 gain more than 1 dB of page SNR, the median lead
SNR goes from 15.26 to 14.75 dB, 19 instead of 10 pages are below 12 dB, and the
classifier changes 26 instead of 23 of its 1,150 decisions (+3, −4 to +10; the second
classifier 40 instead of 35). On the 230 mould-damaged scans it gains much: 207 pages
are straightened, the median lead SNR goes from 4.21 to 13.50 dB, the leads without a
usable signal drop from 32.5 to 9.3 %, 25 instead of 114 pages are below 5 dB, 159
pages gain and 9 lose more than 1 dB, and the classifier changes 68 instead of 158 of
its 1,150 decisions (−90, −121 to −61; the second classifier 73 instead of 158). So
without a dead band the stage is a gain on mould-damaged scans and a loss on clean
ones.

The defaults of the flags (every page in scope, a dead band of 15 px, the same
tolerance and passes) take the dead band on the map the stage would apply and leave
the estimator at its 1 px. So their numbers are those of the run above and of the run
without the stage, page by page: a page whose map in that run moves it by less than
15 px keeps the signals it has without the stage, and every other page has those of
that run. A reading of all 920 development pages with the flags themselves
(`--sheet_curl lines` and no other option of the stage), from the saved masks, gives
exactly these signals, file for file. With the defaults 139 of the 345 photographs
are straightened: the classifier changes 414 instead of 470 decisions (−56, about −82
to −32, from a second bootstrap whose ends differ by up to 2 from those above) and the
second classifier 430 instead of 462 (−32, about −55 to −9), the leads without a usable
signal drop from 53.2 to 44.8 %, 221 instead of 270 pages are below 5 dB, and 84 pages
gain and 11 lose more than 1 dB. Of the 230 clean scans 2 are straightened; none loses
and 1 gains more than 1 dB, and both classifiers change the decisions they change
without the stage (23 and 35 of 1,150). Of the 230 mould-damaged scans 151 are
straightened: the median lead SNR goes from 4.21 to 13.25 dB, the leads without a
usable signal drop from 32.5 to 11.1 %, 31 instead of 114 pages are below 5 dB, 135
pages gain and 2 lose more than 1 dB, and the classifier changes 72 instead of 158
decisions (−86, about −114 to −59; the second classifier 75 instead of 158, −83, about
−108 to −58). None of the 115 rendered pages is straightened. The dead band gives up a
part of the gain, 72 against 68 changed decisions on the mould-damaged scans and 414
against 408 on the photographs, to leave the clean scans as they are: with 10 px, 53
clean scans were still straightened, and 24 of them lost where 7 gained.

Straightening before the segmentation. The numbers above warp a mask that was predicted
on the bent page. The run of the stage as it stands in the digitiser, without
`--mask_folder`, lets nnU-Net predict on the straightened page; it was made once, on the
development pages that the run at a dead band of 1 px straightens (188 photographs and
207 mould-damaged scans; the pages it refuses and the clean scans and rendered pages
were not predicted again), with `--sheet_curl lines --sheet_curl_min_shift 1`, and every
page was straightened again by the same decision. Read with every page of a group in the
denominator, as above, it gives the picture of the saved masks. On the 345 photographs
the classifier changes 424 of its 1,725 decisions, where it changes 408 with the mask
warped after the segmentation and 470 without the stage (before against after: +16,
−5 to +38; the second classifier 432 against 430 and 462); the leads without a usable
signal are 45.0 % against 43.7 % and 53.2 %, 223 against 219 and 270 pages are below
5 dB, and of the 188 pages 33 gain and 23 lose more than 1 dB of page SNR against the
warped mask. On the 230 mould-damaged scans it changes 66 of its 1,150 decisions,
against 68 and 158 (−2, −17 to +12; the second classifier 68 against 73 and 158); the
median lead SNR is 14.31 dB against 13.50 dB and 4.21 dB (paired on the leads of the
207 pages: +0.62 dB, 0.53 to 0.74), 48 against 63 and 166 pages are below 12 dB, and 84
pages gain and 4 lose more than 1 dB against the warped mask. So predicting on the
straightened page is not worse than warping the mask, a little better in SNR on the
mould-damaged scans, and it does not mend what the photographs lack: the model reads a
flattened photograph no better than a bent one. These too are development-set numbers
of one run, made after the options were chosen.

Limits. The photographs stay far from the digital signal with the stage: in the run at
1 px the classifier still changes 23.7 % of its decisions and 314 of the 345 pages are
below 12 dB page SNR, and 90 pages still have no column grid (103 without the stage).
157 photographs are refused: 67 because the map is held across holes in the cover of
the grid lines and tears there (55 at the first measurement, with the reason that
starts `the map stretches or squeezes the page by more than a factor of two somewhere`,
and 12 on the straightened page, where the same words follow
`the straightened page gives no field`), 34 because three passes do not settle, 21 for
want of grid line contrast and 3 more whose straightened page shows no lines, 20
because the fit leaves more than the tolerance of 0.25 (one of them on the straightened
page), 9 because the lines are not straight after the warp, and 3 because the map
folds. The tolerance of 0.25 lets a wrong map through: on a drawn page with a known map
(the grid of the tests, out of focus and bent by 38 px), a fit that leaves 0.17 of a
period is accepted and is a wrong map, 9 px RMS and 25 px at its worst off the truth,
and the check on the straightened page does not catch it; a tolerance of 0.1 refuses
the page. Two more such pages, bent by 40 px, whose first fits leave 0.20 and 0.23 of a
period, are refused, but only because a later pass tears their map. On the development
photographs the pages accepted between the two tolerances still gain more often than
they lose. Of the 188 photographs straightened in the run
at 1 px, the 11 whose fit leaves less than 0.10 of a period have 7 pages that gain and
1 that loses more than 1 dB of page SNR; the 56 between 0.10 and 0.15 have 28 and 9;
the 80 between 0.15 and 0.20 have 47 and 5; and the 41 at 0.20 and above have 17 and
7, with a median page SNR that goes from 0.46 to 1.88 dB. On the mould-damaged scans
202 of the 207 straightened pages leave less than 0.10. So `applied` with a residual
above 0.1, which `--verbose` prints, is a map that paid on these pages on average, not
one that is known to be right. A fold or a local dent is outside the model, and a page
with one can be `applied` without being straightened. The dead band costs the small
bends: a bend whose map moves the page by less than 15 px is left as it is (the drawn
page of the options table: 2.5 dB with the defaults, 25.9 dB with a dead band of
2 px), and a page left in the dead band is still measured, warped and checked, which
takes seconds per page. Apart from the one run of the paragraph above, the stage was
measured with saved masks, each predicted on the bent page and warped with it, nearest
neighbour; the defaults (dead band 15 px) were read with saved masks only. All 176 flat
synthetic pages that were tried with the driver stay inside its 1 px, with byte
identical signals. On synthetic generator pages bent by a known map, the map is
recovered to 0.03 to 0.35 px RMS for whole-sheet bends of 3 to 40 px (at the 1 px of
the estimator with no dead band on top; the default dead band of 15 px leaves the
smaller of these bends alone), which says that the estimator finds a smooth bend on a
clean page, not how it does on a photograph. `--sheet_curl off`, the
default, is the reading of every other number in this README.


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
faint grid whose lines it cannot read, is not trusted. So does `--paper_normalisation
off`: the saved masks have no record of the paper normalisation, so their pages are
used as given. And so does `--grid_rescue off --row_mapping off`: none of these pages
fails a check that is tried again and none has a column map in use. And so does
`--perspective_tolerance 0.1`: the perspective fit of each of these pages is far inside
either tolerance, so the saved masks, predicted at 0.1, are in the frame of the
defaults.

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
