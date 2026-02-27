"""
Shared pytest fixtures for the Passos Mágicos test suite.
"""

import pytest
import pandas as pd
import numpy as np
import joblib
from pathlib import Path
from unittest.mock import MagicMock

from src.preprocessing import DataPreprocessor
from src.feature_engineering import FeatureEngineer


@pytest.fixture
def sample_raw_data():
    """Create a sample raw DataFrame mimicking the real dataset."""
    np.random.seed(42)
    n = 50
    return pd.DataFrame({
        "RA": [f"RA{i:04d}" for i in range(n)],
        "Fase": np.random.randint(0, 8, n),
        "Turma": [f"T{i % 5}" for i in range(n)],
        "Nome": [f"Aluno {i}" for i in range(n)],
        "Ano nasc": np.random.randint(2004, 2014, n),
        "Idade 22": np.random.randint(9, 19, n),
        "Gênero": np.random.choice(["Masculino", "Feminino"], n),
        "Ano ingresso": np.random.randint(2016, 2022, n),
        "Instituição de ensino": np.random.choice(
            ["Escola Municipal", "Escola Estadual", "Escola Particular"], n
        ),
        "Pedra 20": np.random.choice(["Quartzo", "Ágata", "Ametista", "Topázio", None], n),
        "Pedra 21": np.random.choice(["Quartzo", "Ágata", "Ametista", "Topázio", None], n),
        "Pedra 22": np.random.choice(["Quartzo", "Ágata", "Ametista", "Topázio"], n),
        "INDE 22": np.random.uniform(3.0, 9.5, n),
        "Cg": np.random.randint(1, 100, n),
        "Cf": np.random.randint(1, 50, n),
        "Ct": np.random.randint(1, 30, n),
        "Nº Av": np.random.randint(1, 6, n),
        "Avaliador1": [f"Av1_{i}" for i in range(n)],
        "Rec Av1": [f"Rec1_{i}" for i in range(n)],
        "Avaliador2": [f"Av2_{i}" for i in range(n)],
        "Rec Av2": [f"Rec2_{i}" for i in range(n)],
        "Avaliador3": [f"Av3_{i}" if i % 3 != 0 else None for i in range(n)],
        "Rec Av3": [f"Rec3_{i}" for i in range(n)],
        "Avaliador4": [f"Av4_{i}" if i % 2 != 0 else None for i in range(n)],
        "Rec Av4": [f"Rec4_{i}" if i % 2 != 0 else None for i in range(n)],
        "IAA": np.random.uniform(2.0, 10.0, n),
        "IEG": np.random.uniform(2.0, 10.0, n),
        "IPS": np.random.uniform(2.0, 10.0, n),
        "Rec Psicologia": [f"RecPsi_{i}" for i in range(n)],
        "IDA": np.random.uniform(2.0, 10.0, n),
        "Matem": np.random.uniform(3.0, 10.0, n),
        "Portug": np.random.uniform(3.0, 10.0, n),
        "Inglês": np.where(np.random.random(n) > 0.3, np.random.uniform(3.0, 10.0, n), np.nan),
        "Indicado": np.random.choice(["Sim", "Não"], n),
        "Atingiu PV": np.random.choice(["Sim", "Não"], n),
        "IPV": np.random.uniform(2.0, 10.0, n),
        "IAN": np.random.uniform(2.0, 10.0, n),
        "Fase ideal": np.random.choice(
            ["Fase 1 (1º Fund)", "Fase 2 (2º Fund)", "Fase 3 (3º Fund)",
             "Fase 4 (4º Fund)", "Fase 5 (1º EM)", "Fase 7 (3º EM)"], n
        ),
        "Defas": np.random.choice([-3, -2, -1, 0, 1, 2], n, p=[0.03, 0.15, 0.45, 0.30, 0.05, 0.02]),
        "Destaque IEG": np.random.choice(["Sim", "Não"], n, p=[0.15, 0.85]),
        "Destaque IDA": np.random.choice(["Sim", "Não"], n, p=[0.15, 0.85]),
        "Destaque IPV": np.random.choice(["Sim", "Não"], n, p=[0.15, 0.85]),
    })


@pytest.fixture
def sample_raw_data_with_missing(sample_raw_data):
    """Sample data with explicit missing values."""
    df = sample_raw_data.copy()
    df.loc[0, "Matem"] = np.nan
    df.loc[1, "Portug"] = np.nan
    df.loc[2, "Inglês"] = np.nan
    df.loc[0:5, "Pedra 20"] = None
    return df


@pytest.fixture
def preprocessor():
    """Create a fresh DataPreprocessor instance."""
    return DataPreprocessor()


@pytest.fixture
def fitted_preprocessor(preprocessor, sample_raw_data):
    """Create a fitted DataPreprocessor."""
    preprocessor.fit_transform(sample_raw_data)
    return preprocessor


@pytest.fixture
def feature_engineer():
    """Create a fresh FeatureEngineer instance."""
    return FeatureEngineer()


@pytest.fixture
def processed_train_data(sample_raw_data):
    """Create processed train data for training tests."""
    preprocessor = DataPreprocessor()
    df_processed = preprocessor.fit_transform(sample_raw_data)
    X = df_processed.drop(columns=["target"])
    y = df_processed["target"]
    return X, y, preprocessor


@pytest.fixture
def sample_student_input():
    """Create a sample student input dict for API testing."""
    return {
        "Fase": 3,
        "Gênero": "Feminino",
        "Ano ingresso": 2020,
        "Instituição de ensino": "Escola Municipal",
        "Pedra 20": "Quartzo",
        "Pedra 21": "Ágata",
        "Pedra 22": "Ametista",
        "Nº Av": 4,
        "IAA": 7.5,
        "IEG": 8.0,
        "IPS": 6.5,
        "IDA": 7.0,
        "Matem": 7.5,
        "Portug": 8.0,
        "Inglês": 6.0,
        "Indicado": "Não",
        "Atingiu PV": "Não",
        "IPV": 5.0,
        "Destaque IEG": "Não",
        "Destaque IDA": "Não",
        "Destaque IPV": "Não",
    }
