"""
Training module for Passos Mágicos student risk prediction.
Trains multiple models, compares performance, and selects the best one.
"""

import pandas as pd
import numpy as np
import joblib
import logging
from pathlib import Path
from typing import Dict, Any, Tuple, Optional

from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import cross_val_score, StratifiedKFold
from sklearn.pipeline import Pipeline

from src.evaluation import ModelEvaluator

logger = logging.getLogger("passos_magicos")


class ModelTrainer:
    """Trains and selects the best model for student risk prediction."""

    # Available model configurations
    MODEL_REGISTRY = {
        "logistic_regression": {
            "class": LogisticRegression,
            "params": {
                "max_iter": 1000,
                "class_weight": "balanced",
                "random_state": 42,
                "solver": "lbfgs",
            },
        },
        "random_forest": {
            "class": RandomForestClassifier,
            "params": {
                "n_estimators": 200,
                "max_depth": 10,
                "min_samples_split": 5,
                "min_samples_leaf": 2,
                "class_weight": "balanced",
                "random_state": 42,
                "n_jobs": -1,
            },
        },
        "gradient_boosting": {
            "class": GradientBoostingClassifier,
            "params": {
                "n_estimators": 200,
                "max_depth": 5,
                "learning_rate": 0.1,
                "min_samples_split": 5,
                "min_samples_leaf": 2,
                "subsample": 0.8,
                "random_state": 42,
            },
        },
    }

    def __init__(self, random_state: int = 42):
        self.random_state = random_state
        self.trained_models: Dict[str, Any] = {}
        self.cv_results: Dict[str, Dict[str, float]] = {}
        self.best_model_name: Optional[str] = None
        self.best_model: Optional[Any] = None
        self.evaluator = ModelEvaluator()

    def train_model(
        self, model_name: str, X_train: pd.DataFrame, y_train: pd.Series
    ) -> Any:
        """Train a single model."""
        if model_name not in self.MODEL_REGISTRY:
            raise ValueError(
                f"Unknown model: {model_name}. "
                f"Available: {list(self.MODEL_REGISTRY.keys())}"
            )

        config = self.MODEL_REGISTRY[model_name]
        model = config["class"](**config["params"])
        model.fit(X_train, y_train)
        self.trained_models[model_name] = model

        logger.info(f"Trained model: {model_name}")
        return model

    def cross_validate(
        self,
        model_name: str,
        X: pd.DataFrame,
        y: pd.Series,
        cv: int = 5,
        scoring: str = "f1",
    ) -> Dict[str, float]:
        """Perform cross-validation for a model."""
        config = self.MODEL_REGISTRY[model_name]
        model = config["class"](**config["params"])

        skf = StratifiedKFold(n_splits=cv, shuffle=True, random_state=self.random_state)

        scores_f1 = cross_val_score(model, X, y, cv=skf, scoring="f1")
        scores_acc = cross_val_score(model, X, y, cv=skf, scoring="accuracy")
        scores_roc = cross_val_score(model, X, y, cv=skf, scoring="roc_auc")

        results = {
            "f1_mean": scores_f1.mean(),
            "f1_std": scores_f1.std(),
            "accuracy_mean": scores_acc.mean(),
            "accuracy_std": scores_acc.std(),
            "roc_auc_mean": scores_roc.mean(),
            "roc_auc_std": scores_roc.std(),
        }

        self.cv_results[model_name] = results
        logger.info(
            f"CV {model_name}: F1={results['f1_mean']:.4f} ± {results['f1_std']:.4f}"
        )
        return results

    def train_all_models(
        self, X_train: pd.DataFrame, y_train: pd.Series, cv: int = 5
    ) -> Dict[str, Dict[str, float]]:
        """Train and cross-validate all available models."""
        print("=" * 60)
        print("Training and Cross-Validating Models")
        print("=" * 60)

        for name in self.MODEL_REGISTRY:
            print(f"\n--- {name} ---")

            # Cross-validation
            cv_results = self.cross_validate(name, X_train, y_train, cv=cv)
            print(f"  CV F1-Score:  {cv_results['f1_mean']:.4f} ± {cv_results['f1_std']:.4f}")
            print(f"  CV Accuracy:  {cv_results['accuracy_mean']:.4f} ± {cv_results['accuracy_std']:.4f}")
            print(f"  CV ROC-AUC:   {cv_results['roc_auc_mean']:.4f} ± {cv_results['roc_auc_std']:.4f}")

            # Train on full training set
            self.train_model(name, X_train, y_train)

        return self.cv_results

    def select_best_model(self, metric: str = "f1_mean") -> str:
        """Select the best model based on cross-validation metric."""
        if not self.cv_results:
            raise ValueError("No CV results. Run train_all_models first.")

        best_name = max(self.cv_results, key=lambda k: self.cv_results[k][metric])
        self.best_model_name = best_name
        self.best_model = self.trained_models[best_name]

        print(f"\n{'=' * 60}")
        print(f"Best model: {best_name}")
        print(f"  {metric}: {self.cv_results[best_name][metric]:.4f}")
        print(f"{'=' * 60}")

        return best_name

    def evaluate_best_model(
        self,
        X_val: pd.DataFrame,
        y_val: pd.Series,
        X_test: pd.DataFrame,
        y_test: pd.Series,
    ) -> Dict[str, Any]:
        """Evaluate the best model on validation and test sets."""
        if self.best_model is None:
            raise ValueError("No best model selected. Run select_best_model first.")

        print(f"\nEvaluating best model: {self.best_model_name}")
        print("-" * 40)

        # Validation set
        print("\n[Validation Set]")
        val_metrics = self.evaluator.evaluate(
            self.best_model, X_val, y_val, dataset_name="validation"
        )

        # Test set
        print("\n[Test Set]")
        test_metrics = self.evaluator.evaluate(
            self.best_model, X_test, y_test, dataset_name="test"
        )

        return {
            "model_name": self.best_model_name,
            "cv_results": self.cv_results[self.best_model_name],
            "validation_metrics": val_metrics,
            "test_metrics": test_metrics,
        }

    def save_model(self, filepath: str = "models/model.joblib"):
        """Save the best model to disk."""
        if self.best_model is None:
            raise ValueError("No best model to save.")

        Path(filepath).parent.mkdir(parents=True, exist_ok=True)
        joblib.dump(self.best_model, filepath)
        print(f"Model saved to: {filepath}")

    @staticmethod
    def load_model(filepath: str = "models/model.joblib"):
        """Load a model from disk."""
        return joblib.load(filepath)


def train_pipeline(
    train_path: str = "data/processed/train.csv",
    val_path: str = "data/processed/val.csv",
    test_path: str = "data/processed/test.csv",
    model_path: str = "models/model.joblib",
    metrics_path: str = "models/metrics.json",
) -> Dict[str, Any]:
    """
    Full training pipeline.
    Loads data, trains models, selects best, evaluates, and saves.
    """
    from src.utils import save_metrics

    # Load data
    train_df = pd.read_csv(train_path)
    val_df = pd.read_csv(val_path)
    test_df = pd.read_csv(test_path)

    X_train = train_df.drop(columns=["target"])
    y_train = train_df["target"]
    X_val = val_df.drop(columns=["target"])
    y_val = val_df["target"]
    X_test = test_df.drop(columns=["target"])
    y_test = test_df["target"]

    # Train
    trainer = ModelTrainer()
    trainer.train_all_models(X_train, y_train)

    # Select best
    trainer.select_best_model(metric="f1_mean")

    # Evaluate
    results = trainer.evaluate_best_model(X_val, y_val, X_test, y_test)

    # Save model and metrics
    trainer.save_model(model_path)
    save_metrics(results, metrics_path)
    print(f"Metrics saved to: {metrics_path}")

    return results


if __name__ == "__main__":
    results = train_pipeline()
