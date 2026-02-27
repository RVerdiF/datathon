"""
Tests for the FastAPI application and endpoints.
"""

import pytest
import pandas as pd
import numpy as np
from unittest.mock import MagicMock, patch
from fastapi.testclient import TestClient

from api.main import app
from api.schemas import StudentData, PredictionResponse
from api.routers.predict import student_to_dataframe, predict_single


@pytest.fixture
def client():
    """Create a test client with mocked model."""
    # Mock a simple model
    mock_model = MagicMock()
    mock_model.predict.return_value = np.array([1])
    mock_model.predict_proba.return_value = np.array([[0.3, 0.7]])

    # Mock preprocessor
    mock_preprocessor = MagicMock()
    mock_preprocessor.feature_columns = [
        "Fase", "Gênero", "Ano ingresso", "Instituição de ensino",
        "Pedra 20", "Pedra 21", "Pedra 22", "Nº Av",
        "IAA", "IEG", "IPS", "IDA", "Matem", "Portug", "Inglês",
        "Indicado", "Atingiu PV", "IPV",
        "Destaque IEG", "Destaque IDA", "Destaque IPV",
    ]
    mock_preprocessor.encode_categorical.return_value = pd.DataFrame({
        col: [0] for col in mock_preprocessor.feature_columns
    })
    mock_preprocessor.handle_missing_values.return_value = pd.DataFrame({
        col: [0.0] for col in mock_preprocessor.feature_columns
    })
    mock_preprocessor.scaler = MagicMock()
    mock_preprocessor.scaler.transform.return_value = np.zeros((1, len(mock_preprocessor.feature_columns)))

    app.state.model = mock_model
    app.state.preprocessor = mock_preprocessor
    app.state.model_loaded = True

    with TestClient(app, raise_server_exceptions=False) as c:
        yield c


class TestHealthCheck:
    """Tests for the health check endpoint."""

    def test_health_check(self, client):
        """Test GET / returns healthy status."""
        response = client.get("/")
        assert response.status_code == 200
        data = response.json()
        assert data["status"] == "healthy"
        assert data["model_loaded"] is True
        assert "version" in data


class TestStudentToDataframe:
    """Tests for the student_to_dataframe utility."""

    def test_converts_to_dataframe(self, sample_student_input):
        """Test conversion from StudentData to DataFrame."""
        student = StudentData(**sample_student_input)
        df = student_to_dataframe(student)
        assert isinstance(df, pd.DataFrame)
        assert len(df) == 1

    def test_column_names_match(self, sample_student_input):
        """Test that DataFrame columns use original Portuguese names."""
        student = StudentData(**sample_student_input)
        df = student_to_dataframe(student)
        assert "Fase" in df.columns
        assert "Gênero" in df.columns
        assert "Instituição de ensino" in df.columns
        assert "Nº Av" in df.columns

    def test_values_preserved(self, sample_student_input):
        """Test that values are preserved in conversion."""
        student = StudentData(**sample_student_input)
        df = student_to_dataframe(student)
        assert df["Fase"].iloc[0] == 3
        assert df["IAA"].iloc[0] == 7.5


class TestPredictEndpoint:
    """Tests for the /predict endpoint."""

    def test_predict_returns_200(self, client, sample_student_input):
        """Test that /predict returns 200 for valid input."""
        response = client.post("/predict", json=sample_student_input)
        assert response.status_code == 200

    def test_predict_response_format(self, client, sample_student_input):
        """Test response contains expected fields."""
        response = client.post("/predict", json=sample_student_input)
        data = response.json()
        assert "risco" in data
        assert "risco_label" in data
        assert "probabilidade_risco" in data
        assert "probabilidade_sem_risco" in data

    def test_predict_risco_values(self, client, sample_student_input):
        """Test that risco is 0 or 1."""
        response = client.post("/predict", json=sample_student_input)
        data = response.json()
        assert data["risco"] in [0, 1]

    def test_predict_label_matches_risco(self, client, sample_student_input):
        """Test that label matches numeric prediction."""
        response = client.post("/predict", json=sample_student_input)
        data = response.json()
        if data["risco"] == 1:
            assert data["risco_label"] == "Risco"
        else:
            assert data["risco_label"] == "Sem Risco"

    def test_predict_probabilities_sum_to_one(self, client, sample_student_input):
        """Test that probabilities sum to approximately 1."""
        response = client.post("/predict", json=sample_student_input)
        data = response.json()
        total = data["probabilidade_risco"] + data["probabilidade_sem_risco"]
        assert abs(total - 1.0) < 0.01

    def test_predict_missing_required_field(self, client):
        """Test that missing required fields cause 422."""
        response = client.post("/predict", json={"Fase": 3})
        assert response.status_code == 422

    def test_predict_invalid_fase(self, client, sample_student_input):
        """Test validation for invalid Fase value."""
        invalid = sample_student_input.copy()
        invalid["Fase"] = 99
        response = client.post("/predict", json=invalid)
        assert response.status_code == 422

    def test_predict_with_null_optional(self, client, sample_student_input):
        """Test prediction with null optional fields."""
        input_data = sample_student_input.copy()
        input_data["Pedra 20"] = None
        input_data["Pedra 21"] = None
        response = client.post("/predict", json=input_data)
        assert response.status_code == 200


class TestBatchPredictEndpoint:
    """Tests for the /predict/batch endpoint."""

    def test_batch_predict(self, client, sample_student_input):
        """Test batch prediction with multiple students."""
        response = client.post(
            "/predict/batch",
            json={"students": [sample_student_input, sample_student_input]},
        )
        assert response.status_code == 200
        data = response.json()
        assert data["total"] == 2
        assert len(data["predictions"]) == 2

    def test_batch_predict_empty_list(self, client):
        """Test batch prediction with empty list."""
        response = client.post("/predict/batch", json={"students": []})
        assert response.status_code == 200
        data = response.json()
        assert data["total"] == 0


class TestModelNotLoaded:
    """Tests for when the model is not loaded."""

    def test_predict_model_not_loaded(self, client, sample_student_input):
        """Test /predict when model is not loaded."""
        original = app.state.model_loaded
        app.state.model_loaded = False
        try:
            response = client.post("/predict", json=sample_student_input)
            assert response.status_code == 503
        finally:
            app.state.model_loaded = original
