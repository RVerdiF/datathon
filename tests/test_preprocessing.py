"""
Tests for the preprocessing module.
"""

import pytest
import pandas as pd
import numpy as np
import tempfile
import os

from src.preprocessing import DataPreprocessor, preprocess_data


class TestDataPreprocessor:
    """Tests for the DataPreprocessor class."""

    def test_init(self, preprocessor):
        """Test preprocessor initialization."""
        assert preprocessor.target_column == "target"
        assert preprocessor.feature_columns == []
        assert preprocessor.label_encoders == {}

    def test_create_target_positive_defas(self, preprocessor):
        """Test target creation with non-negative defasagem (no risk)."""
        df = pd.DataFrame({"Defas": [0, 1, 2]})
        result = preprocessor.create_target(df)
        assert all(result["target"] == 0)

    def test_create_target_negative_defas(self, preprocessor):
        """Test target creation with negative defasagem (risk)."""
        df = pd.DataFrame({"Defas": [-1, -2, -3]})
        result = preprocessor.create_target(df)
        assert all(result["target"] == 1)

    def test_create_target_mixed(self, preprocessor):
        """Test target creation with mixed defasagem values."""
        df = pd.DataFrame({"Defas": [-2, -1, 0, 1, 2]})
        result = preprocessor.create_target(df)
        expected = [1, 1, 0, 0, 0]
        assert result["target"].tolist() == expected

    def test_create_target_does_not_modify_original(self, preprocessor):
        """Test that create_target does not modify the original DataFrame."""
        df = pd.DataFrame({"Defas": [-1, 0, 1]})
        preprocessor.create_target(df)
        assert "target" not in df.columns

    def test_handle_missing_numeric(self, preprocessor):
        """Test handling of missing numeric values."""
        df = pd.DataFrame({"col1": [1.0, np.nan, 3.0], "col2": [4.0, 5.0, np.nan]})
        result = preprocessor.handle_missing_values(df)
        assert not result.isnull().any().any()

    def test_handle_missing_numeric_fills_with_median(self, preprocessor):
        """Test that numeric missing values are filled with median."""
        df = pd.DataFrame({"col1": [1.0, np.nan, 5.0]})
        result = preprocessor.handle_missing_values(df)
        assert result["col1"].iloc[1] == 3.0  # median of [1.0, 5.0]

    def test_handle_missing_categorical(self, preprocessor):
        """Test handling of missing categorical values."""
        df = pd.DataFrame({"col1": ["A", None, "A", "B"]})
        result = preprocessor.handle_missing_values(df)
        assert not result.isnull().any().any()
        assert result["col1"].iloc[1] == "A"  # mode

    def test_handle_missing_does_not_modify_original(self, preprocessor):
        """Test that handle_missing_values does not modify the original."""
        df = pd.DataFrame({"col1": [1.0, np.nan, 3.0]})
        preprocessor.handle_missing_values(df)
        assert df["col1"].isnull().sum() == 1

    def test_encode_categorical(self, preprocessor):
        """Test categorical encoding."""
        df = pd.DataFrame({
            "Gênero": ["Masculino", "Feminino", "Masculino"],
            "Indicado": ["Sim", "Não", "Sim"],
        })
        result = preprocessor.encode_categorical(df, fit=True)
        assert result["Gênero"].dtype in [np.int32, np.int64]
        assert result["Indicado"].dtype in [np.int32, np.int64]

    def test_encode_categorical_stores_encoders(self, preprocessor):
        """Test that encoders are stored after fitting."""
        df = pd.DataFrame({"Gênero": ["Masculino", "Feminino"]})
        preprocessor.encode_categorical(df, fit=True)
        assert "Gênero" in preprocessor.label_encoders

    def test_encode_categorical_transform_mode(self, preprocessor):
        """Test categorical encoding in transform mode."""
        df_fit = pd.DataFrame({"Gênero": ["Masculino", "Feminino"]})
        preprocessor.encode_categorical(df_fit, fit=True)

        df_transform = pd.DataFrame({"Gênero": ["Feminino", "Masculino"]})
        result = preprocessor.encode_categorical(df_transform, fit=False)
        assert result["Gênero"].dtype in [np.int32, np.int64]

    def test_select_features_drops_id_columns(self, preprocessor):
        """Test that identifier columns are dropped."""
        df = pd.DataFrame({
            "RA": ["001"], "Nome": ["Test"], "Turma": ["T1"],
            "Defas": [-1], "Fase": [3], "target": [1],
        })
        result = preprocessor.select_features(df)
        assert "RA" not in result.columns
        assert "Nome" not in result.columns
        assert "Defas" not in result.columns
        assert "Fase" in result.columns

    def test_select_features_drops_leaky_columns(self, preprocessor):
        """Test that data-leaky columns are dropped."""
        df = pd.DataFrame({
            "IAN": [5.0], "Fase ideal": ["Fase 3"],
            "Idade 22": [14], "Ano nasc": [2008],
            "INDE 22": [7.0], "Fase": [3], "target": [1],
        })
        result = preprocessor.select_features(df)
        assert "IAN" not in result.columns
        assert "Fase ideal" not in result.columns
        assert "Idade 22" not in result.columns
        assert "Ano nasc" not in result.columns
        assert "INDE 22" not in result.columns

    def test_select_features_stores_feature_columns(self, preprocessor):
        """Test that feature columns are stored."""
        df = pd.DataFrame({"Fase": [3], "IEG": [8.0], "target": [1]})
        preprocessor.select_features(df)
        assert "Fase" in preprocessor.feature_columns
        assert "IEG" in preprocessor.feature_columns
        assert "target" not in preprocessor.feature_columns

    def test_scale_features(self, preprocessor):
        """Test feature scaling."""
        df = pd.DataFrame({
            "col1": [1.0, 2.0, 3.0, 4.0, 5.0],
            "target": [0, 1, 0, 1, 0],
        })
        preprocessor.target_column = "target"
        result = preprocessor.scale_features(df, fit=True)
        # Scaled values should have mean ~0 and std ~1
        assert abs(result["col1"].mean()) < 0.01
        assert abs(result["col1"].std() - 1.0) < 0.2

    def test_fit_transform_full_pipeline(self, preprocessor, sample_raw_data):
        """Test the full fit_transform pipeline."""
        result = preprocessor.fit_transform(sample_raw_data)
        assert "target" in result.columns
        assert "RA" not in result.columns
        assert "Nome" not in result.columns
        assert not result.isnull().any().any()
        assert len(preprocessor.feature_columns) > 0

    def test_fit_transform_returns_correct_shape(self, preprocessor, sample_raw_data):
        """Test that fit_transform returns the expected number of rows."""
        result = preprocessor.fit_transform(sample_raw_data)
        assert len(result) == len(sample_raw_data)

    def test_split_data_sizes(self, fitted_preprocessor, sample_raw_data):
        """Test data splitting proportions."""
        df = fitted_preprocessor.fit_transform(sample_raw_data)
        X_train, X_val, X_test, y_train, y_val, y_test = fitted_preprocessor.split_data(df)
        total = len(X_train) + len(X_val) + len(X_test)
        assert total == len(df)

    def test_split_data_stratified(self, fitted_preprocessor, sample_raw_data):
        """Test that split preserves class proportions roughly."""
        df = fitted_preprocessor.fit_transform(sample_raw_data)
        _, _, _, y_train, _, y_test = fitted_preprocessor.split_data(df)
        # Both sets should have both classes
        assert len(y_train.unique()) > 1 or len(sample_raw_data) < 10
        assert len(y_test.unique()) > 1 or len(sample_raw_data) < 10

    def test_save_and_load(self, fitted_preprocessor, tmp_path):
        """Test saving and loading the preprocessor."""
        filepath = str(tmp_path / "preprocessor.joblib")
        fitted_preprocessor.save(filepath)

        loaded = DataPreprocessor.load(filepath)
        assert loaded.feature_columns == fitted_preprocessor.feature_columns
        assert loaded.target_column == fitted_preprocessor.target_column
        assert set(loaded.label_encoders.keys()) == set(fitted_preprocessor.label_encoders.keys())
