"""Step 5: result plots. All figures are built first, then shown as pop-up windows together."""
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

from util import get_logger

logger = get_logger(__name__)


class Visualizer:
    def __init__(self, output_dir: str = "outputs", show: bool = True, top_n_features: int = 15):
        self.output_dir = Path(output_dir) / "plots"
        self.show = show
        self.top_n_features = top_n_features

    def plot_all(self, results: dict) -> None:
        self.output_dir.mkdir(parents=True, exist_ok=True)
        figures = {
            "metric_comparison": self._metric_comparison(results),
            "roc_curves": self._roc_curves(results),
            "feature_importance_comparison": self._feature_importance(results),
            "best_model_confusion_matrix": self._confusion_matrix(results),
        }
        for name, fig in figures.items():
            fig.savefig(self.output_dir / f"{name}.png", dpi=150, bbox_inches="tight")
        logger.info(f"Saved plots to {self.output_dir}")

        if self.show:
            plt.show()  # opens every figure as its own pop-up window
        else:
            plt.close("all")

    # ------------------------------------------------------------------ #
    def _metric_comparison(self, results):
        comp = results["comparison"]
        metrics = ["CV ROC-AUC", "Test ROC-AUC", "Test Accuracy"]
        x = np.arange(len(comp))
        width = 0.26

        fig, ax = plt.subplots(figsize=(10, 5.5))
        fig.canvas.manager.set_window_title("Model comparison")
        for i, metric in enumerate(metrics):
            bars = ax.bar(x + (i - 1) * width, comp[metric], width, label=metric)
            ax.bar_label(bars, fmt="%.3f", fontsize=8, padding=2)
        ax.set_xticks(x)
        ax.set_xticklabels(comp.index)
        ax.set_ylim(0, 1.08)
        ax.set_ylabel("Score")
        ax.set_title(f"ROC-AUC and accuracy comparison (best: {results['best_model_name']})")
        ax.legend(loc="lower right")
        ax.grid(axis="y", alpha=0.3)
        fig.tight_layout()
        return fig

    def _roc_curves(self, results):
        fig, ax = plt.subplots(figsize=(7, 6))
        fig.canvas.manager.set_window_title("ROC curves")
        for name, (fpr, tpr, auc) in results["roc_curves"].items():
            best = name == results["best_model_name"]
            ax.plot(fpr, tpr, lw=2.8 if best else 1.6, label=f"{name} (AUC {auc:.3f})")
        ax.plot([0, 1], [0, 1], "k--", lw=1, label="Chance")
        ax.set_xlabel("False positive rate")
        ax.set_ylabel("True positive rate")
        ax.set_title("ROC curves on the test set")
        ax.legend(loc="lower right")
        ax.grid(alpha=0.3)
        fig.tight_layout()
        return fig

    def _feature_importance(self, results):
        imp = results["feature_importance"].head(self.top_n_features)
        models = list(imp.columns)
        y = np.arange(len(imp))
        height = 0.8 / len(models)

        fig, ax = plt.subplots(figsize=(11, 8))
        fig.canvas.manager.set_window_title("Feature importance comparison")
        for i, model in enumerate(models):
            ax.barh(y + (i - (len(models) - 1) / 2) * height, imp[model], height, label=model)
        ax.set_yticks(y)
        ax.set_yticklabels(imp.index)
        ax.invert_yaxis()
        ax.set_xlabel("Normalised importance (each model sums to 1)")
        ax.set_title(f"Top {len(imp)} features by mean importance across models")
        ax.legend()
        ax.grid(axis="x", alpha=0.3)
        fig.tight_layout()
        return fig

    def _confusion_matrix(self, results):
        name = results["best_model_name"]
        cm = results["confusion_matrices"][name]

        fig, ax = plt.subplots(figsize=(5.5, 5))
        fig.canvas.manager.set_window_title("Best model confusion matrix")
        im = ax.imshow(cm, cmap="Blues")
        for (i, j), v in np.ndenumerate(cm):
            ax.text(j, i, int(v), ha="center", va="center",
                    color="white" if v > cm.max() / 2 else "black", fontsize=14)
        ax.set_xticks([0, 1])
        ax.set_yticks([0, 1])
        ax.set_xticklabels(["No PCOS", "PCOS"])
        ax.set_yticklabels(["No PCOS", "PCOS"])
        ax.set_xlabel("Predicted")
        ax.set_ylabel("Actual")
        ax.set_title(f"{name} - confusion matrix (test set)")
        fig.colorbar(im, ax=ax, fraction=0.046)
        fig.tight_layout()
        return fig
