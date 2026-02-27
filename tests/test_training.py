"""
Tests for the training module.
"""

import pytest
import pandas as pd
import numpy as np
import tempfile
from pathlib import Path

from src.training import ModelTrainer
from src.evaluation import ModelEvaluator


class TestModelTrainer:
    """Tests for the ModelTrainer class."""

    def test_init(self):
        """Test trainer initialization."""
        trainer = ModelTrainer()
        assert trainer.random_state == 42
        assert trainer.trained_models == {}
        assert trainer.cv_results == {}
        assert trainer.best_model is None

    def test_model_registry(self):
        """Test that model registry has expected models."""
        trainer = ModelTrainer()
        assert "logistic_regression" in trainer.MODEL_REGISTRY
        assert "random_forest" in trainer.MODEL_REGISTRY
        assert "gradient_boosting" in trainer.MODEL_REGISTRY

    def test_train_single_model(self, processed_train_data):
        """Test training a single model."""
        X, y, _ = processed_train_data
        trainer = ModelTrainer()
        model = trainer.train_model("logistic_regression", X, y)
        assert model is not None
        assert "logistic_regression" in trainer.trained_models

    def test_train_invalid_model(self, processed_train_data):
        """Test training with an invalid model name."""
        X, y, _ = processed_train_data
        trainer = ModelTrainer()
        with pytest.raises(ValueError, match="Unknown model"):
            trainer.train_model("invalid_model", X, y)

    def test_cross_validate(self, processed_train_data):
        """Test cross-validation."""
        X, y, _ = processed_train_data
        trainer = ModelTrainer()
        results = trainer.cross_validate("logistic_regression", X, y, cv=3)
        assert "f1_mean" in results
        assert "accuracy_mean" in results
        assert "roc_auc_mean" in results
        assert 0 <= results["f1_mean"] <= 1

    def test_train_all_models(self, processed_train_data):
        """Test training all available models."""
        X, y, _ = processed_train_data
        trainer = ModelTrainer()
        cv_results = trainer.train_all_models(X, y, cv=3)
        assert len(cv_results) == 3
        assert len(trainer.trained_models) == 3

    def test_select_best_model(self, processed_train_data):
        """Test best model selection."""
        X, y, _ = processed_train_data
        trainer = ModelTrainer()
        trainer.train_all_models(X, y, cv=3)
        best_name = trainer.select_best_model()
        assert best_name in trainer.MODEL_REGISTRY
        assert trainer.best_model is not None

    def test_select_best_no_cv(self):
        """Test selecting best model without prior CV results."""
        trainer = ModelTrainer()
        with pytest.raises(ValueError, match="No CV results"):
            trainer.select_best_model()

    def test_save_model(self, processed_train_data, tmp_path):
        """Test saving the best model."""
        X, y, _ = processed_train_data
        trainer = ModelTrainer()
        trainer.train_all_models(X, y, cv=3)
        trainer.select_best_model()

        filepath = str(tmp_path / "model.joblib")
        trainer.save_model(filepath)
        assert Path(filepath).exists()

    def test_save_model_no_best(self):
        """Test saving without a best model."""
        trainer = ModelTrainer()
        with pytest.raises(ValueError, match="No best model"):
            trainer.save_model()

    def test_load_model(self, processed_train_data, tmp_path):
        """Test loading a saved model."""
        X, y, _ = processed_train_data
        trainer = ModelTrainer()
        trainer.train_all_models(X, y, cv=3)
        trainer.select_best_model()

        filepath = str(tmp_path / "model.joblib")
        trainer.save_model(filepath)
        loaded = ModelTrainer.load_model(filepath)
        assert loaded is not None

    def test_model_predictions(self, processed_train_data):
        """Test that trained model can make predictions."""
        X, y, _ = processed_train_data
        trainer = ModelTrainer()
        model = trainer.train_model("random_forest", X, y)
        predictions = model.predict(X)
        assert len(predictions) == len(X)
        assert set(predictions).issubset({0, 1})


class TestModelEvaluator:
    """Tests for the ModelEvaluator class."""

    def test_evaluate(self, processed_train_data):
        """Test model evaluation."""
        X, y, _ = processed_train_data
        trainer = ModelTrainer()
        model = trainer.train_model("logistic_regression", X, y)

        evaluator = ModelEvaluator()
        metrics = evaluator.evaluate(model, X, y, dataset_name="test")
        assert "accuracy" in metrics
        assert "precision" in metrics
        assert "recall" in metrics
        assert "f1_score" in metrics
        assert "confusion_matrix" in metrics

    def test_evaluate_roc_auc(self, processed_train_data):
        """Test ROC-AUC is included for models with predict_proba."""
        X, y, _ = processed_train_data
        trainer = ModelTrainer()
        model = trainer.train_model("random_forest", X, y)

        evaluator = ModelEvaluator()
        metrics = evaluator.evaluate(model, X, y)
        assert "roc_auc" in metrics

    def test_feature_importance_tree(self, processed_train_data):
        """Test feature importance extraction for tree models."""
        X, y, _ = processed_train_data
        trainer = ModelTrainer()
        model = trainer.train_model("random_forest", X, y)

        evaluator = ModelEvaluator()
        importance = evaluator.get_feature_importance(model, X.columns.tolist())
        assert len(importance) > 0
        assert "feature" in importance.columns
        assert "importance" in importance.columns

    def test_feature_importance_linear(self, processed_train_data):
        """Test feature importance extraction for linear models."""
        X, y, _ = processed_train_data
        trainer = ModelTrainer()
        model = trainer.train_model("logistic_regression", X, y)

        evaluator = ModelEvaluator()
        importance = evaluator.get_feature_importance(model, X.columns.tolist())
        assert len(importance) > 0

    def test_save_report(self, processed_train_data, tmp_path):
        """Test saving evaluation report."""
        X, y, _ = processed_train_data
        trainer = ModelTrainer()
        model = trainer.train_model("logistic_regression", X, y)

        evaluator = ModelEvaluator()
        metrics = evaluator.evaluate(model, X, y)

        filepath = str(tmp_path / "report.json")
        evaluator.save_report(metrics, filepath)
        assert Path(filepath).exists()
