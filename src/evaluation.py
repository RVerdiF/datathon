"""
Evaluation module for Passos Mágicos student risk prediction.
Provides metrics, reports, and feature importance analysis.
"""

import pandas as pd
import numpy as np
import json
import logging
from typing import Dict, Any, Optional
from pathlib import Path

from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score,
    classification_report,
    confusion_matrix,
)

logger = logging.getLogger("passos_magicos")


class ModelEvaluator:
    """Evaluates ML models and generates reports."""

    TARGET_NAMES = ["Sem Risco", "Risco"]

    def evaluate(
        self,
        model: Any,
        X: pd.DataFrame,
        y: pd.Series,
        dataset_name: str = "test",
    ) -> Dict[str, Any]:
        """
        Evaluate a model and return comprehensive metrics.

        Args:
            model: Trained sklearn model
            X: Feature dataframe
            y: True labels
            dataset_name: Name of the dataset for logging

        Returns:
            Dictionary with all metrics
        """
        y_pred = model.predict(X)
        y_proba = (
            model.predict_proba(X)[:, 1]
            if hasattr(model, "predict_proba")
            else None
        )

        metrics = {
            "dataset": dataset_name,
            "accuracy": float(accuracy_score(y, y_pred)),
            "precision": float(precision_score(y, y_pred, zero_division=0)),
            "recall": float(recall_score(y, y_pred, zero_division=0)),
            "f1_score": float(f1_score(y, y_pred, zero_division=0)),
            "confusion_matrix": confusion_matrix(y, y_pred).tolist(),
            "classification_report": classification_report(
                y, y_pred, target_names=self.TARGET_NAMES, output_dict=True
            ),
        }

        if y_proba is not None:
            metrics["roc_auc"] = float(roc_auc_score(y, y_proba))

        # Log summary
        self._log_summary(metrics)

        return metrics

    def _log_summary(self, metrics: Dict[str, Any]):
        """Log a formatted summary of metrics."""
        logger.info(f"  Accuracy:  {metrics['accuracy']:.4f}")
        logger.info(f"  Precision: {metrics['precision']:.4f}")
        logger.info(f"  Recall:    {metrics['recall']:.4f}")
        logger.info(f"  F1-Score:  {metrics['f1_score']:.4f}")
        if "roc_auc" in metrics:
            logger.info(f"  ROC-AUC:   {metrics['roc_auc']:.4f}")

        cm = np.array(metrics["confusion_matrix"])
        logger.info("  Confusion Matrix:")
        logger.info(f"  {'':>15} Pred:Sem  Pred:Risco")
        logger.info(f"  {'Real:Sem':>15}  {cm[0][0]:>6}  {cm[0][1]:>6}")
        logger.info(f"  {'Real:Risco':>15}  {cm[1][0]:>6}  {cm[1][1]:>6}")

    def get_feature_importance(
        self, model: Any, feature_names: list
    ) -> pd.DataFrame:
        """
        Extract feature importance from the model.

        Args:
            model: Trained model (must have feature_importances_ or coef_)
            feature_names: List of feature names

        Returns:
            DataFrame with features sorted by importance
        """
        if hasattr(model, "feature_importances_"):
            importances = model.feature_importances_
        elif hasattr(model, "coef_"):
            importances = np.abs(model.coef_[0])
        else:
            logger.warning("Model does not support feature importance extraction.")
            return pd.DataFrame()

        importance_df = pd.DataFrame(
            {"feature": feature_names, "importance": importances}
        )
        importance_df = importance_df.sort_values("importance", ascending=False)
        importance_df = importance_df.reset_index(drop=True)

        return importance_df

    def print_feature_importance(
        self, model: Any, feature_names: list, top_n: int = 15
    ):
        """Print top N most important features."""
        importance_df = self.get_feature_importance(model, feature_names)

        if importance_df.empty:
            return

        logger.info(f"Top {top_n} Feature Importances:")
        logger.info("-" * 40)
        for _, row in importance_df.head(top_n).iterrows():
            bar = "█" * int(row["importance"] * 50)
            logger.info(f"  {row['feature']:>25}: {row['importance']:.4f} {bar}")

    def save_report(
        self,
        metrics: Dict[str, Any],
        filepath: str = "models/evaluation_report.json",
    ):
        """Save evaluation report to JSON."""
        Path(filepath).parent.mkdir(parents=True, exist_ok=True)

        # Convert numpy types to native Python
        report = json.loads(json.dumps(metrics, default=str))

        with open(filepath, "w") as f:
            json.dump(report, f, indent=2)

        logger.info(f"Report saved to: {filepath}")


def evaluate_model(
    model_path: str = "models/model.joblib",
    test_path: str = "data/processed/test.csv",
    report_path: str = "models/evaluation_report.json",
) -> Dict[str, Any]:
    """
    Evaluate a saved model on test data.

    Args:
        model_path: Path to saved model
        test_path: Path to test data CSV
        report_path: Path to save evaluation report

    Returns:
        Dictionary with evaluation metrics
    """
    import joblib

    # Load model and data
    model = joblib.load(model_path)
    test_df = pd.read_csv(test_path)

    X_test = test_df.drop(columns=["target"])
    y_test = test_df["target"]

    # Evaluate
    evaluator = ModelEvaluator()
    metrics = evaluator.evaluate(model, X_test, y_test, dataset_name="test")

    # Feature importance
    evaluator.print_feature_importance(model, X_test.columns.tolist())

    # Save report
    evaluator.save_report(metrics, report_path)

    return metrics


if __name__ == "__main__":
    metrics = evaluate_model()
