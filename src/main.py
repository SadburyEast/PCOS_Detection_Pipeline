"""
CLI entry point. Run with:

    python main.py
    python main.py --data-path data/other.xlsx --sheet-name Sheet1
    python main.py --models random_forest xgboost --no-smote --no-show
"""
import argparse

from pipeline import Pipeline
from model.models import MODEL_REGISTRY
from util import DEFAULT_DATA_PATH, DEFAULT_SHEET_NAME, TARGET_COLUMN, get_logger

logger = get_logger(__name__)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run the PCOS classification pipeline.")
    parser.add_argument("--data-path", type=str, default=DEFAULT_DATA_PATH, help="Path to the .xlsx/.csv file.")
    parser.add_argument("--sheet-name", type=str, default=DEFAULT_SHEET_NAME, help="Excel sheet (ignored for csv).")
    parser.add_argument("--target", type=str, default=TARGET_COLUMN, help="Target column name.")
    parser.add_argument("--models", nargs="+", default=list(MODEL_REGISTRY), choices=list(MODEL_REGISTRY),
                        help="Which models to tune and compare.")
    parser.add_argument("--select-by", choices=["cv", "test"], default="cv",
                        help="Pick best model by cross-validated or test ROC-AUC.")
    parser.add_argument("--cv-folds", type=int, default=5)
    parser.add_argument("--test-size", type=float, default=0.2)
    parser.add_argument("--top-n", type=int, default=15, help="Features shown in importance comparison.")
    parser.add_argument("--no-smote", action="store_true", help="Disable SMOTE (use class weights instead).")
    parser.add_argument("--no-show", action="store_true", help="Save plots but don't open pop-up windows.")
    parser.add_argument("--output-dir", type=str, default="outputs")
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    pipeline = Pipeline(
        data_path=args.data_path,
        sheet_name=args.sheet_name,
        target_column=args.target,
        model_names=args.models,
        test_size=args.test_size,
        cv_folds=args.cv_folds,
        use_smote=not args.no_smote,
        select_by=args.select_by,
        top_n_features=args.top_n,
        output_dir=args.output_dir,
        show_plots=not args.no_show,
    )
    results = pipeline.run()

    logger.info(f"Best model: {results['best_model_name']} | params: {results['best_model_params']}")


if __name__ == "__main__":
    main()
