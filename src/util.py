"""
Shared utilities: logging setup plus dataset-specific constants, kept in one
place so every class gets consistent logging and a single source of truth for
column names / cleaning rules.
"""
import logging
import sys


def get_logger(name: str, level: str = "INFO") -> logging.Logger:
    logger = logging.getLogger(name)

    if not logger.handlers:  # avoid duplicate handlers if called more than once
        handler = logging.StreamHandler(sys.stdout)
        formatter = logging.Formatter(
            "%(asctime)s | %(levelname)-8s | %(name)s | %(message)s",
            datefmt="%Y-%m-%d %H:%M:%S",
        )
        handler.setFormatter(formatter)
        logger.addHandler(handler)
        logger.setLevel(level)

    return logger


# --------------------------------------------------------------------------
# Dataset constants (PCOS dataset). Column names are the *normalised* ones,
# i.e. after whitespace is collapsed/stripped by Preprocessor.
# --------------------------------------------------------------------------
DEFAULT_DATA_PATH = "datasets/(Main_Dataset)_PCOS_data_without_infertility.xlsx"
DEFAULT_SHEET_NAME = "Full_new"
TARGET_COLUMN = "PCOS (Y/N)"

# Identifier columns. 'Sl. No' showed up as a top feature in the notebook's
# Random Forest, which means row order leaked into the model - always drop.
DROP_COLUMNS = ["Sl. No", "Patient File No."]

# Numeric-looking codes that are really categories (one-hot encoded).
CATEGORICAL_COLUMNS = ["Blood Group", "Cycle(R/I)", "No. of abortions"]

# (lower, upper) clip bounds; None = no bound on that side.
OUTLIER_CAPS = {
    "Vit D3 (ng/mL)": (0, 100),
    "PRG(ng/mL)": (None, 30),
    "TSH (mIU/L)": (None, 20),
    "FSH/LH": (None, 20),
    "LH(mIU/mL)": (None, 30),
    "FSH(mIU/mL)": (None, 30),
    "Pulse rate(bpm)": (40, 120),
}

# Some rows record blood pressure in tens (e.g. 12 meaning 120). Any value
# below the threshold is multiplied by 10.
TENS_SCALED_COLUMNS = {
    "BP _Systolic (mmHg)": 30,
    "BP _Diastolic (mmHg)": 30,
}
