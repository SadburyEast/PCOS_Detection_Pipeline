"""
Step 3: model definitions + hyperparameter search.

Every model is a full scikit-learn/imblearn pipeline:

    ColumnTransformer (impute -> scale / one-hot)  ->  [SMOTE]  ->  classifier

wrapped in GridSearchCV. Because SMOTE and all preprocessing live inside the
pipeline, they are re-fit on the training part of each CV fold only. (The
notebook applied SMOTE before GridSearchCV, which lets synthetic copies of
validation rows leak into training and inflates the CV score.)

Note: SMOTE runs after one-hot encoding, so synthetic rows can have fractional
one-hot values. Use --no-smote to rely on class weights instead.
"""
from abc import ABC, abstractmethod

import joblib
import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import GradientBoostingClassifier, RandomForestClassifier
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import GridSearchCV, StratifiedKFold
from sklearn.pipeline import Pipeline as SkPipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

from util import get_logger

logger = get_logger(__name__)


class BaseModel(ABC):
    name: str = "base"

    def __init__(
        self,
        numeric_cols: list[str],
        categorical_cols: list[str],
        use_smote: bool = True,
        cv_folds: int = 5,
        scoring: str = "roc_auc",
        random_state: int = 42,
        n_jobs: int = -1,
    ):
        self.numeric_cols = numeric_cols
        self.categorical_cols = categorical_cols
        self.use_smote = use_smote
        self.cv_folds = cv_folds
        self.scoring = scoring
        self.random_state = random_state
        self.n_jobs = n_jobs
        self.search_: GridSearchCV | None = None

    # ---- subclasses define these ------------------------------------- #
    @abstractmethod
    def build_estimator(self):
        """Return an un-fitted classifier."""

    @abstractmethod
    def param_grid(self) -> dict:
        """Grid over classifier hyperparameters (plain names, no 'clf__' prefix)."""

    # ---- shared machinery -------------------------------------------- #
    def _build_preprocessor(self) -> ColumnTransformer:
        numeric = SkPipeline(
            [("imputer", SimpleImputer(strategy="median")), ("scaler", StandardScaler())]
        )
        categorical = SkPipeline(
            [
                ("imputer", SimpleImputer(strategy="most_frequent")),
                ("onehot", OneHotEncoder(handle_unknown="ignore", sparse_output=False)),
            ]
        )
        return ColumnTransformer(
            [("num", numeric, self.numeric_cols), ("cat", categorical, self.categorical_cols)],
            verbose_feature_names_out=False,
        )

    def _build_pipeline(self):
        steps = [("prep", self._build_preprocessor())]
        if self.use_smote:
            try:
                from imblearn.over_sampling import SMOTE
                from imblearn.pipeline import Pipeline as ImbPipeline
            except ImportError as e:
                raise ImportError(
                    "SMOTE needs imbalanced-learn (pip install imbalanced-learn) "
                    "or run with --no-smote."
                ) from e
            steps.append(("smote", SMOTE(random_state=self.random_state)))
            steps.append(("clf", self.build_estimator()))
            return ImbPipeline(steps)
        steps.append(("clf", self.build_estimator()))
        return SkPipeline(steps)

    def fit(self, X: pd.DataFrame, y: pd.Series) -> "BaseModel":
        grid = {f"clf__{k}": v for k, v in self.param_grid().items()}
        cv = StratifiedKFold(self.cv_folds, shuffle=True, random_state=self.random_state)
        self.search_ = GridSearchCV(
            self._build_pipeline(), grid, cv=cv, scoring=self.scoring,
            n_jobs=self.n_jobs, refit=True,
        )
        self.search_.fit(X, y)
        logger.info(
            f"{self.name}: best CV {self.scoring}={self.search_.best_score_:.4f} "
            f"params={self.best_params_}"
        )
        return self

    # ---- results ----------------------------------------------------- #
    @property
    def estimator_(self):
        return self.search_.best_estimator_

    @property
    def best_params_(self) -> dict:
        return {k.removeprefix("clf__"): v for k, v in self.search_.best_params_.items()}

    @property
    def cv_score_(self) -> float:
        return float(self.search_.best_score_)

    def predict(self, X):
        return self.estimator_.predict(X)

    def predict_proba(self, X):
        return self.estimator_.predict_proba(X)[:, 1]

    def feature_importances(self) -> pd.Series:
        """Importances per ORIGINAL feature (one-hot columns summed back), summing to 1."""
        clf = self.estimator_.named_steps["clf"]
        raw = clf.feature_importances_ if hasattr(clf, "feature_importances_") else np.abs(clf.coef_).ravel()
        names = self.estimator_.named_steps["prep"].get_feature_names_out()
        series = pd.Series(raw, index=names)

        def origin(feature: str) -> str:
            for col in self.categorical_cols:
                if feature.startswith(f"{col}_"):
                    return col
            return feature

        series = series.groupby(origin).sum().sort_values(ascending=False)
        return series / series.sum()

    def save(self, path: str) -> None:
        joblib.dump(self.estimator_, path)
        logger.info(f"Saved {self.name} to {path}")

    def __repr__(self) -> str:
        return self.name


class LogisticRegressionModel(BaseModel):
    name = "Logistic Regression"

    def build_estimator(self):
        return LogisticRegression(
            max_iter=5000, random_state=self.random_state,
            class_weight=None if self.use_smote else "balanced",
        )

    def param_grid(self) -> dict:
        return {"C": [0.01, 0.1, 1, 10]}


class RandomForestModel(BaseModel):
    name = "Random Forest"

    def build_estimator(self):
        return RandomForestClassifier(
            random_state=self.random_state, n_jobs=1,
            class_weight=None if self.use_smote else "balanced",
        )

    def param_grid(self) -> dict:
        return {
            "n_estimators": [100, 300],
            "max_depth": [None, 5, 10],
            "min_samples_leaf": [1, 2],
        }


class GradientBoostingModel(BaseModel):
    name = "Gradient Boosting"

    def build_estimator(self):
        return GradientBoostingClassifier(random_state=self.random_state)

    def param_grid(self) -> dict:
        return {
            "n_estimators": [100, 200],
            "learning_rate": [0.05, 0.1],
            "max_depth": [2, 3],
        }


class XGBoostModel(BaseModel):
    name = "XGBoost"

    def build_estimator(self):
        from xgboost import XGBClassifier  # imported lazily so xgboost stays optional

        return XGBClassifier(
            random_state=self.random_state, eval_metric="logloss", n_jobs=1, verbosity=0
        )

    def param_grid(self) -> dict:
        return {
            "max_depth": [3, 5, 7],
            "learning_rate": [0.01, 0.1],
            "n_estimators": [50, 100, 150],
            "subsample": [0.8, 1.0],
        }


MODEL_REGISTRY: dict[str, type[BaseModel]] = {
    "logistic": LogisticRegressionModel,
    "random_forest": RandomForestModel,
    "gradient_boosting": GradientBoostingModel,
    "xgboost": XGBoostModel,
}
