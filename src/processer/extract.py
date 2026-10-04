"""Step 1: load the raw dataset from disk (Excel or CSV)."""
from pathlib import Path

import pandas as pd

from util import get_logger

logger = get_logger(__name__)


class DataExtractor:
    """Reads a raw dataset file into a DataFrame."""

    def __init__(self, file_path: str, sheet_name: str | None = None):
        self.file_path = Path(file_path)
        self.sheet_name = sheet_name

    def extract(self) -> pd.DataFrame:
        if not self.file_path.exists():
            raise FileNotFoundError(
                f"Dataset not found at '{self.file_path}'. Pass --data-path to point to it."
            )

        suffix = self.file_path.suffix.lower()
        if suffix in {".xlsx", ".xlsm", ".xls"}:
            df = pd.read_excel(self.file_path, engine="openpyxl", sheet_name=self.sheet_name or 0)
        elif suffix == ".csv":
            df = pd.read_csv(self.file_path)
        else:
            raise ValueError(f"Unsupported file type '{suffix}' (use .xlsx or .csv).")

        logger.info(f"Extracted {df.shape[0]} rows x {df.shape[1]} columns from {self.file_path.name}")
        return df
