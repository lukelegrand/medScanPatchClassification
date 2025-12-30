# config.py
import os

# ================= PATHS =================
# Raw input data directory (Folders containing 'ct.nii.gz')
INPUT_DATA_DIR = r"path/to/raw_subjects"

# Where extracted patches will be saved
OUTPUT_PATCH_DIR = r"data/patches_raw"

# Where the balanced/normalized dataset will be saved
NORMALIZED_DATA_DIR = r"data/patches_normalized"

# Temporary folder for processing slices
TEMP_SLICE_DIR = r"temp_slices"

# ================= CT IMAGE SETTINGS =================
# Hounsfield Unit (HU) Windowing
WINDOW_CENTER = 40    # Soft tissue
WINDOW_WIDTH = 400
MAX_VAL_CHECK = 1000  # Integrity check

# ================= PATCH EXTRACTION SETTINGS =================
WINDOW_SIZE = 64      # Size of the patch (64x64)
STRIDE = 32           # Overlap factor (smaller = more overlap)
PURITY_THRESHOLD = 0.95   # % of patch that must be the dominant organ
FILL_RATIO_THRESHOLD = 0.40 # % of patch that must not be background

# ================= DATASET BALANCING (Normalizer) =================
TARGET_COUNT = 500    # Target number of images per class
MIN_REQUIRED = 50     # Classes below this are dropped

# ================= INFERENCE SETTINGS =================
MODEL_INPUT_SHAPE = (1, 3, 224, 224)
DEVICE_TYPE = 'cuda'  # or 'cpu'