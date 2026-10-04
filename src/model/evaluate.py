"""Step 4: score every tuned model on the held-out test set and pick the best."""
import pandas as pd
from sklearn.metrics import (
    accuracy_score, confusion_matrix, f1_score, precision_score, recall_score,
    roc_auc_score, roc_curve,
)

from model.models import BaseModel
from util import get_logger

logger = get_logger(__name__)


class Evaluator:
    """
    select_by:
      "cv"   - best model = highest cross-validated ROC-AUC on the training data
               (default; the test set is small, so this avoids picking a winner by luck)
      "test" - best model = highest ROC-AUC on the test set
    """

    def __init__(self, select_by: str = "cv", top_n_features: int = 15):
        if select_by not in {"cv", "test"}:
            raise ValueError("select_by must be 'cv' or 'test'")
        self.select_by = select_by
        self.top_n_features = top_n_features

    def evaluate(self, models: dict[str, BaseModel], X_test: pd.DataFrame, y_test: pd.Series) -> dict:
        rows, roc_curves, confusion, importances = [], {}, {}, {}

        for name, model in models.items():
            proba = model.predict_proba(X_test)
            pred = model.predict(X_test)
            fpr, tpr, _ = roc_curve(y_test, proba)
            auc = roc_auc_score(y_test, proba)

            rows.append({
                "Model": name,
                "CV ROC-AUC": model.cv_score_,
                "Test ROC-AUC": auc,
                "Test Accuracy": accuracy_score(y_test, pred),
                "Precision": precision_score(y_test, pred, zero_division=0),
                "Recall": recall_score(y_test, pred, zero_division=0),
                "F1": f1_score(y_test, pred, zero_division=0),
            })
            roc_curves[name] = (fpr, tpr, auc)
            confusion[name] = confusion_matrix(y_test, pred)
            importances[name] = model.feature_importances()

        comparison = pd.DataFrame(rows).set_index("Model")
        sort_col = "CV ROC-AUC" if self.select_by == "cv" else "Test ROC-AUC"
        comparison = comparison.sort_values(sort_col, ascending=False)
        best_name = comparison.index[0]

        importance_df = pd.DataFrame(importances).fillna(0.0)
        importance_df = importance_df.loc[importance_df.mean(axis=1).sort_values(ascending=False).index]

        return {
            "comparison": comparison,
            "best_model_name": best_name,
            "best_model_params": models[best_name].best_params_,
            "best_model_metrics": comparison.loc[best_name].to_dict(),
            "feature_importance": importance_df,
            "roc_curves": roc_curves,
            "confusion_matrices": confusion,
            "select_by": self.select_by,
        }

    def format_report(self, results: dict) -> str:
        line = "=" * 70
        basis = "cross-validated ROC-AUC" if results["select_by"] == "cv" else "test ROC-AUC"
        m = results["best_model_metrics"]
        top = results["feature_importance"].head(self.top_n_features).round(4)

        return "\n".join([
            "", line, "RESULTS", line,
            f"Best model (selected by {basis}): {results['best_model_name']}",
            f"Best parameters: {results['best_model_params']}",
            f"  CV ROC-AUC {m['CV ROC-AUC']:.3f} | Test ROC-AUC {m['Test ROC-AUC']:.3f} "
            f"| Test accuracy {m['Test Accuracy']:.3f}",
            "", "Model comparison:", results["comparison"].round(3).to_string(),
            "", f"Feature importance comparison (top {self.top_n_features}, normalised to sum to 1):",
            top.to_string(), line,
        ])
