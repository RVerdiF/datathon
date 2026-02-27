"""
Tests for the feature engineering module.
"""

import pytest
import pandas as pd
import numpy as np

from src.feature_engineering import FeatureEngineer, engineer_features


class TestFeatureEngineer:
    """Tests for the FeatureEngineer class."""

    def test_init(self, feature_engineer):
        """Test initialization."""
        assert feature_engineer.created_features == []

    def test_create_performance_features_avg(self, feature_engineer):
        """Test average indicator creation."""
        df = pd.DataFrame({"IAN": [5.0, 8.0], "IDA": [6.0, 7.0], "IEG": [7.0, 9.0]})
        result = feature_engineer.create_performance_features(df)
        assert "avg_indicators" in result.columns
        assert abs(result["avg_indicators"].iloc[0] - 6.0) < 0.01

    def test_create_performance_features_std(self, feature_engineer):
        """Test standard deviation indicator creation."""
        df = pd.DataFrame({"IAN": [5.0], "IDA": [5.0], "IEG": [5.0]})
        result = feature_engineer.create_performance_features(df)
        assert "std_indicators" in result.columns
        assert result["std_indicators"].iloc[0] == 0.0

    def test_create_performance_features_grades(self, feature_engineer):
        """Test average grades feature."""
        df = pd.DataFrame({"Matem": [8.0, 6.0], "Portug": [7.0, 5.0]})
        result = feature_engineer.create_performance_features(df)
        assert "avg_grades" in result.columns
        assert result["avg_grades"].iloc[0] == 7.5

    def test_create_performance_no_indicators(self, feature_engineer):
        """Test with no indicator columns present."""
        df = pd.DataFrame({"other_col": [1, 2, 3]})
        result = feature_engineer.create_performance_features(df)
        assert "avg_indicators" not in result.columns

    def test_create_engagement_features(self, feature_engineer):
        """Test engagement feature creation."""
        df = pd.DataFrame({"Nº Av": [1, 2, 3, 4, 5]})
        result = feature_engineer.create_engagement_features(df)
        assert "high_engagement" in result.columns
        assert result["high_engagement"].isin([0, 1]).all()

    def test_create_trajectory_pedra(self, feature_engineer):
        """Test Pedra numeric conversion."""
        df = pd.DataFrame({
            "Pedra 20": ["Quartzo", "Ágata"],
            "Pedra 21": ["Ágata", "Ametista"],
            "Pedra 22": ["Ametista", "Topázio"],
        })
        result = feature_engineer.create_trajectory_features(df)
        assert "pedra_num_22" in result.columns
        assert result["pedra_num_22"].iloc[0] == 3  # Ametista = 3

    def test_create_trajectory_improvement(self, feature_engineer):
        """Test Pedra improvement feature."""
        df = pd.DataFrame({
            "Pedra 21": ["Quartzo", "Topázio"],
            "Pedra 22": ["Ametista", "Quartzo"],
        })
        result = feature_engineer.create_trajectory_features(df)
        assert "pedra_improvement" in result.columns
        assert result["pedra_improvement"].iloc[0] == 2  # 3 - 1

    def test_create_trajectory_years_in_program(self, feature_engineer):
        """Test years in program calculation."""
        df = pd.DataFrame({"Ano ingresso": [2020, 2018]})
        result = feature_engineer.create_trajectory_features(df)
        assert "years_in_program" in result.columns
        assert result["years_in_program"].iloc[0] == 2
        assert result["years_in_program"].iloc[1] == 4

    def test_create_phase_features(self, feature_engineer):
        """Test phase-related features."""
        df = pd.DataFrame({
            "Fase": [3, 7],
            "Fase ideal": ["Fase 4 (4º Fund)", "Fase 7 (3º EM)"],
        })
        result = feature_engineer.create_phase_features(df)
        assert "advanced_phase" in result.columns
        assert result["advanced_phase"].iloc[0] == 0
        assert result["advanced_phase"].iloc[1] == 1

    def test_create_risk_indicators(self, feature_engineer):
        """Test risk indicator creation."""
        df = pd.DataFrame({
            "IDA": [3.0, 8.0, 5.0, 9.0],
            "IPS": [2.0, 7.0, 6.0, 8.0],
        })
        result = feature_engineer.create_risk_indicators(df)
        assert "risk_factor_count" in result.columns
        # Low values should have higher risk count
        assert result["risk_factor_count"].iloc[0] >= result["risk_factor_count"].iloc[3]

    def test_create_all_features(self, feature_engineer, sample_raw_data):
        """Test that create_all_features adds new columns."""
        original_cols = len(sample_raw_data.columns)
        result = feature_engineer.create_all_features(sample_raw_data)
        assert len(result.columns) > original_cols

    def test_get_feature_names(self, feature_engineer, sample_raw_data):
        """Test that feature names are tracked."""
        feature_engineer.create_all_features(sample_raw_data)
        names = feature_engineer.get_feature_names()
        assert len(names) > 0
        assert isinstance(names, list)

    def test_create_all_features_resets(self, feature_engineer):
        """Test that create_all_features resets feature list."""
        df = pd.DataFrame({"Nº Av": [1, 2, 3]})
        feature_engineer.create_all_features(df)
        first_run = feature_engineer.get_feature_names()

        feature_engineer.create_all_features(df)
        second_run = feature_engineer.get_feature_names()

        assert first_run == second_run

    def test_does_not_modify_original(self, feature_engineer):
        """Test that features are created on copies."""
        df = pd.DataFrame({"IDA": [5.0, 8.0], "IPS": [6.0, 7.0]})
        original_cols = set(df.columns)
        feature_engineer.create_risk_indicators(df)
        assert set(df.columns) == original_cols


class TestEngineerFeaturesFunction:
    """Tests for the engineer_features convenience function."""

    def test_engineer_features(self, sample_raw_data):
        """Test the top-level engineer_features function."""
        result = engineer_features(sample_raw_data)
        assert len(result.columns) > len(sample_raw_data.columns)

    def test_engineer_features_returns_dataframe(self, sample_raw_data):
        """Test that the function returns a DataFrame."""
        result = engineer_features(sample_raw_data)
        assert isinstance(result, pd.DataFrame)
