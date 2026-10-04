"""
Step 2: stateless cleaning (no statistics learned from the data).

Anything that learns from data (imputation, scaling, one-hot, SMOTE) lives
inside the model pipeline in model/models.py, so it is fit on training folds
only and cannot leak into the test set.
"""
import re

import numpy as np
import pandas as pd
from pandas.api.types import is_numeric_dtype

from util import (
    CATEGORICAL_COLUMNS,
    DROP_COLUMNS,
    OUTLIER_CAPS,
    TARGET_COLUMN,
    TENS_SCALED_COLUMNS,
    get_logger,
)

logger = get_logger(__name__)


class Preprocessor:
    def __init__(
        self,
        target_column: str = TARGET_COLUMN,
        drop_columns: list[str] | None = None,
        categorical_columns: list[str] | None = None,
        outlier_caps: dict | None = None,
        tens_scaled_columns: dict | None = None,
    ):
        self.target_column = target_column
        self.drop_columns = drop_columns if drop_columns is not None else DROP_COLUMNS
        self.categorical_columns = (
            categorical_columns if categorical_columns is not None else CATEGORICAL_COLUMNS
        )
        self.outlier_caps = outlier_caps if outlier_caps is not None else OUTLIER_CAPS
        self.tens_scaled_columns = (
            tens_scaled_columns if tens_scaled_columns is not None else TENS_SCALED_COLUMNS
        )

    # ------------------------------------------------------------------ #
    def clean(self, df: pd.DataFrame) -> pd.DataFrame:
        df = df.copy()
        df.columns = [self._normalise_name(c) for c in df.columns]

        df = df.drop(columns=[c for c in self.drop_columns if c in df.columns])

        before = len(df)
        df = df.drop_duplicates().reset_index(drop=True)
        if len(df) != before:
            logger.info(f"Dropped {before - len(df)} duplicate rows")

        df = self._coerce_numeric(df)

        missing_target = df[self.target_column].isna().sum()
        if missing_target:
            logger.warning(f"Dropping {missing_target} rows with missing target")
            df = df.dropna(subset=[self.target_column]).reset_index(drop=True)
        df[self.target_column] = df[self.target_column].astype(int)

        df = self._fix_tens_scaling(df)
        df = self._cap_outliers(df)
        df = self._encode_categorical_codes(df)

        logger.info(
            f"Cleaned data: {df.shape[0]} rows x {df.shape[1]} columns, "
            f"{int(df.isna().sum().sum())} NaNs left (imputed later, inside the model pipeline)"
        )
        return df

    def split_features_target(self, df: pd.DataFrame) -> tuple[pd.DataFrame, pd.Series]:
        X = df.drop(columns=[self.target_column])
        y = df[self.target_column]
        logger.info(f"Target balance: {y.value_counts(normalize=True).round(3).to_dict()}")
        return X, y

    def get_column_types(self, X: pd.DataFrame) -> tuple[list[str], list[str]]:
        categorical = [c for c in self.categorical_columns if c in X.columns]
        numeric = [c for c in X.columns if c not in categorical]
        return numeric, categorical

    # ------------------------------------------------------------------ #
    @staticmethod
    def _normalise_name(name: str) -> str:
        """'  I   beta-HCG(mIU/mL)' -> 'I beta-HCG(mIU/mL)'."""
        return re.sub(r"\s+", " ", str(name)).strip()

    def _coerce_numeric(self, df: pd.DataFrame) -> pd.DataFrame:
        # A few columns (e.g. AMH, II beta-HCG) contain stray text like '1.99.'
        for col in df.columns:
            if not is_numeric_dtype(df[col]):
                converted = pd.to_numeric(df[col], errors="coerce")
                bad = int(converted.isna().sum() - df[col].isna().sum())
                if bad:
                    logger.warning(f"'{col}': {bad} non-numeric value(s) set to NaN")
                df[col] = converted
        return df

    def _fix_tens_scaling(self, df: pd.DataFrame) -> pd.DataFrame:
        for col, threshold in self.tens_scaled_columns.items():
            if col in df.columns:
                mask = df[col] < threshold
                if mask.any():
                    logger.info(f"'{col}': multiplied {int(mask.sum())} value(s) by 10")
                    df.loc[mask, col] = df.loc[mask, col] * 10
        return df

    def _cap_outliers(self, df: pd.DataFrame) -> pd.DataFrame:
        for col, (lower, upper) in self.outlier_caps.items():
            if col in df.columns:
                df[col] = df[col].clip(lower=lower, upper=upper)
        return df

    def _encode_categorical_codes(self, df: pd.DataFrame) -> pd.DataFrame:
        """Turn code columns into clean string labels ('13.0' -> '13'), keeping NaN."""

        def to_label(v):
            if pd.isna(v):
                return np.nan
            return str(int(v)) if float(v).is_integer() else str(v)

        for col in self.categorical_columns:
            if col in df.columns:
                df[col] = df[col].map(to_label).astype(object)
        return df
