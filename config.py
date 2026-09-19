IMAGE_TYPE = "png" # Image type for the dataset
FREQUENCY = 500 # Sampling frequency of the signals
DATASET_NAME = "Dataset500_Signals" # Name of the dataset
LONG_SIGNAL_LENGTH_SEC = 10 # Length in seconds of the full signal
SHORT_SIGNAL_LENGTH_SEC = 2.5 # Length in seconds of the cropped signal
SIGNAL_UNITS = "mV" # Units of the signal for the y-axis
FMT = '16' # Format of the signal
ADC_GAIN = 1000.0 # ADC gain of the signal
BASELINE = 0 # Baseline of the signal

# Mapping of the lead labels to num for the segmentation model
LEAD_LABEL_MAPPING = {
    "I": 1,
    "II": 2,
    "III": 3,
    "aVR": 4,
    "aVL": 5,
    "aVF": 6,
    "V1": 7,
    "V2": 8,
    "V3": 9,
    "V4": 10,
    "V5": 11,
    "V6": 12,
}

# If the absolute y value matters, use the following values
# to adjusts the vertical positioning of the cropped signal within the rotated image's height.
# A value closer to 1 shifts the signal toward the top, while a value closer to 0 shifts it toward the bottom.
# The ratios are the row geometry of the image generator (ecg_plot.py): with 4 rows
# it splits the page into rows + 2 = 6 row heights and puts the short lead rows at
# 3.5, 2.5 and 1.5 row heights above the bottom edge. The rhythm strip sits at
# row_height / 2 - lead_name_offset + 0.8, i.e. half a row height plus 0.3 cm of 21.59 cm.
Y_SHIFT_RATIO = {
    "I": 3.5 / 6,
    "II": 2.5 / 6,
    "III": 1.5 / 6,
    "aVR": 3.5 / 6,
    "aVL": 2.5 / 6,
    "aVF": 1.5 / 6,
    "V1": 3.5 / 6,
    "V2": 2.5 / 6,
    "V3": 1.5 / 6,
    "V4": 3.5 / 6,
    "V5": 2.5 / 6,
    "V6": 1.5 / 6,
    "full": 1 / 12 + 0.3 / 21.59,
}
