"""
Tests for utility functions and pipeline integration.
"""

import pytest
import pandas as pd
import numpy as np
import json
import tempfile
from pathlib import Path

from src.utils import setup_logging, get_project_root, ensure_dir, save_metrics, load_metrics
from src.preprocessing import DataPreprocessor, preprocess_data
from src.evaluation import ModelEvaluator, evaluate_model
from src.training import ModelTrainer


class TestUtils:
    """Tests for utility functions."""

    def test_setup_logging(self):
        """Test logger creation."""
        logger = setup_logging(log_level="DEBUG")
        assert logger is not None
        assert logger.name == "passos_magicos"

    def test_setup_logging_with_file(self, tmp_path):
        """Test logger with file handler."""
        log_file = str(tmp_path / "test.log")
        logger = setup_logging(log_level="INFO", log_file=log_file)
        assert len(logger.handlers) >= 2

    def test_get_project_root(self):
        """Test project root path."""
        root = get_project_root()
        assert isinstance(root, Path)

    def test_ensure_dir(self, tmp_path):
        """Test directory creation."""
        new_dir = tmp_path / "new" / "nested" / "dir"
        result = ensure_dir(str(new_dir))
        assert result.exists()

    def test_save_metrics(self, tmp_path):
        """Test saving metrics to JSON."""
        filepath = str(tmp_path / "metrics.json")
        metrics = {"accuracy": 0.85, "f1": 0.80}
        save_metrics(metrics, filepath)
        assert Path(filepath).exists()

        with open(filepath) as f:
            loaded = json.load(f)
        assert loaded["accuracy"] == 0.85
        assert "timestamp" in loaded

    def test_load_metrics(self, tmp_path):
        """Test loading metrics from JSON."""
        filepath = str(tmp_path / "metrics.json")
        metrics = {"accuracy": 0.85}
        save_metrics(metrics, filepath)

        loaded = load_metrics(filepath)
        assert loaded["accuracy"] == 0.85

    def test_ensure_dir_existing(self, tmp_path):
        """Test ensure_dir on existing directory."""
        result = ensure_dir(str(tmp_path))
        assert result.exists()


class TestPreprocessDataFunction:
    """Tests for the preprocess_data top-level function."""

    def test_preprocess_data(self, sample_raw_data, tmp_path):
        """Test the full preprocess_data pipeline."""
        # Save raw data to a temp Excel file
        input_path = str(tmp_path / "test_data.xlsx")
        sample_raw_data.to_excel(input_path, index=False)

        output_dir = str(tmp_path / "processed")
        preprocessor_path = str(tmp_path / "preprocessor.joblib")

        train_df, val_df, test_df = preprocess_data(
            input_path, output_dir, preprocessor_path
        )

        assert len(train_df) > 0
        assert len(val_df) > 0
        assert len(test_df) > 0
        assert "target" in train_df.columns
        assert Path(f"{output_dir}/train.csv").exists()
        assert Path(f"{output_dir}/val.csv").exists()
        assert Path(f"{output_dir}/test.csv").exists()
        assert Path(preprocessor_path).exists()


class TestEvaluateModelFunction:
    """Tests for the evaluate_model top-level function."""

    def test_evaluate_model(self, processed_train_data, tmp_path):
        """Test the evaluate_model standalone function."""
        X, y, preprocessor = processed_train_data

        # Train and save a model
        trainer = ModelTrainer()
        model = trainer.train_model("logistic_regression", X, y)

        import joblib
        model_path = str(tmp_path / "model.joblib")
        joblib.dump(model, model_path)

        # Save test data
        test_df = X.copy()
        test_df["target"] = y
        test_path = str(tmp_path / "test.csv")
        test_df.to_csv(test_path, index=False)

        # Evaluate
        report_path = str(tmp_path / "report.json")
        metrics = evaluate_model(model_path, test_path, report_path)

        assert "accuracy" in metrics
        assert "f1_score" in metrics
        assert Path(report_path).exists()


class TestTrainPipelineFunction:
    """Tests for the train_pipeline top-level function."""

    def test_train_pipeline(self, sample_raw_data, tmp_path):
        """Test the full train_pipeline function."""
        # Prepare data using preprocessor
        preprocessor = DataPreprocessor()
        df_processed = preprocessor.fit_transform(sample_raw_data)
        X_train, X_val, X_test, y_train, y_val, y_test = preprocessor.split_data(df_processed)

        # Save CSVs
        for name, X, y in [
            ("train", X_train, y_train),
            ("val", X_val, y_val),
            ("test", X_test, y_test),
        ]:
            df = X.copy()
            df["target"] = y
            df.to_csv(str(tmp_path / f"{name}.csv"), index=False)

        from src.training import train_pipeline

        results = train_pipeline(
            train_path=str(tmp_path / "train.csv"),
            val_path=str(tmp_path / "val.csv"),
            test_path=str(tmp_path / "test.csv"),
            model_path=str(tmp_path / "model.joblib"),
            metrics_path=str(tmp_path / "metrics.json"),
        )

        assert "model_name" in results
        assert "validation_metrics" in results
        assert "test_metrics" in results
        assert Path(tmp_path / "model.joblib").exists()
        assert Path(tmp_path / "metrics.json").exists()


class TestModelEvaluatorEdgeCases:
    """Additional edge case tests for ModelEvaluator."""

    @pytest.mark.filterwarnings("ignore::sklearn.exceptions.UndefinedMetricWarning")
    def test_evaluate_all_same_predictions(self):
        """Test evaluation when all predictions are the same class."""
        evaluator = ModelEvaluator()

        class ConstantModel:
            def predict(self, X):
                return np.ones(len(X))
            def predict_proba(self, X):
                return np.column_stack([np.zeros(len(X)), np.ones(len(X))])

        X = pd.DataFrame({"f1": [1, 2, 3, 4, 5]})
        y = pd.Series([0, 1, 1, 1, 1])  # mixed classes for report compatibility
        metrics = evaluator.evaluate(ConstantModel(), X, y)
        assert metrics["recall"] == 1.0  # all class-1 predictions correct

    def test_feature_importance_no_support(self):
        """Test feature importance with unsupported model."""
        evaluator = ModelEvaluator()

        class DummyModel:
            pass

        result = evaluator.get_feature_importance(DummyModel(), ["f1", "f2"])
        assert result.empty
