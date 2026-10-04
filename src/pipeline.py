"""
Wires the whole pipeline together:
extract -> preprocess -> split -> tune/train all models -> evaluate -> save -> plot.
"""
import json
from pathlib import Path

import joblib
from sklearn.model_selection import train_test_split

from processer.extract import DataExtractor
from processer.preprocess import Preprocessor
from model.models import MODEL_REGISTRY
from model.evaluate import Evaluator
from model.visualize import Visualizer
from util import DEFAULT_DATA_PATH, DEFAULT_SHEET_NAME, TARGET_COLUMN, get_logger

logger = get_logger(__name__)


class Pipeline:
    """End-to-end pipeline: raw PCOS file in, tuned + compared models out."""

    def __init__(
        self,
        data_path: str = DEFAULT_DATA_PATH,
        sheet_name: str | None = DEFAULT_SHEET_NAME,
        target_column: str = TARGET_COLUMN,
        model_names: list[str] | None = None,
        test_size: float = 0.2,
        random_state: int = 42,
        cv_folds: int = 5,
        use_smote: bool = True,
        select_by: str = "cv",
        top_n_features: int = 15,
        output_dir: str = "outputs",
        show_plots: bool = True,
    ):
        self.test_size = test_size
        self.random_state = random_state
        self.cv_folds = cv_folds
        self.use_smote = use_smote
        self.output_dir = Path(output_dir)
        self.model_names = model_names or list(MODEL_REGISTRY)

        self.extractor = DataExtractor(data_path, sheet_name)
        self.preprocessor = Preprocessor(target_column)
        self.evaluator = Evaluator(select_by=select_by, top_n_features=top_n_features)
        self.visualizer = Visualizer(output_dir, show=show_plots, top_n_features=top_n_features)

        self.models = {}
        self.results = None

    def run(self) -> dict:
        logger.info("=== Pipeline started ===")

        raw_df = self.extractor.extract()
        cleaned_df = self.preprocessor.clean(raw_df)
        X, y = self.preprocessor.split_features_target(cleaned_df)
        numeric_cols, categorical_cols = self.preprocessor.get_column_types(X)

        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=self.test_size, random_state=self.random_state, stratify=y
        )
        logger.info(f"Split: {len(X_train)} train / {len(X_test)} test rows")

        for key in self._available_models():
            model = MODEL_REGISTRY[key](
                numeric_cols, categorical_cols,
                use_smote=self.use_smote, cv_folds=self.cv_folds,
                random_state=self.random_state,
            )
            logger.info(f"Tuning {model.name} ...")
            model.fit(X_train, y_train)
            self.models[model.name] = model

        self.results = self.evaluator.evaluate(self.models, X_test, y_test)
        logger.info(self.evaluator.format_report(self.results))

        self._save()
        self.visualizer.plot_all(self.results)  # pop-ups open here

        logger.info("=== Pipeline finished ===")
        return self.results

    # ------------------------------------------------------------------ #
    def _available_models(self) -> list[str]:
        unknown = [m for m in self.model_names if m not in MODEL_REGISTRY]
        if unknown:
            raise ValueError(f"Unknown model(s) {unknown}. Choose from {list(MODEL_REGISTRY)}")

        keys = []
        for key in self.model_names:
            if key == "xgboost":
                try:
                    import xgboost  # noqa: F401
                except ImportError:
                    logger.warning("xgboost not installed - skipping it (pip install xgboost)")
                    continue
            keys.append(key)
        return keys

    def _save(self) -> None:
        self.output_dir.mkdir(parents=True, exist_ok=True)
        best = self.models[self.results["best_model_name"]]
        best.save(str(self.output_dir / "best_model.joblib"))

        summary = {
            "best_model": self.results["best_model_name"],
            "best_params": self.results["best_model_params"],
            "best_metrics": self.results["best_model_metrics"],
            "comparison": self.results["comparison"].reset_index().to_dict(orient="records"),
            "feature_importance": self.results["feature_importance"].to_dict(),
        }
        with open(self.output_dir / "results.json", "w") as f:
            json.dump(summary, f, indent=2, default=str)
        self.results["feature_importance"].to_csv(self.output_dir / "feature_importance.csv")
        logger.info(f"Saved model, results.json and feature_importance.csv to {self.output_dir}/")


if __name__ == "__main__":
    Pipeline().run()
