"""
Pydantic schemas for the Passos Mágicos API.
Defines request and response models for the /predict endpoint.
"""

from pydantic import BaseModel, Field
from typing import Optional, List
from enum import Enum


class GenderEnum(str, Enum):
    MASCULINO = "Masculino"
    FEMININO = "Feminino"


class PedraEnum(str, Enum):
    QUARTZO = "Quartzo"
    AGATA = "Ágata"
    AMETISTA = "Ametista"
    TOPAZIO = "Topázio"


class SimNaoEnum(str, Enum):
    SIM = "Sim"
    NAO = "Não"


class StudentData(BaseModel):
    """Input data for a single student prediction."""
    Fase: int = Field(..., ge=0, le=8, description="Fase do aluno no programa (0-8)")
    Genero: str = Field(..., alias="Gênero", description="Gênero do aluno")
    Ano_ingresso: int = Field(..., alias="Ano ingresso", ge=2010, le=2025, description="Ano de ingresso no programa")
    Instituicao_ensino: str = Field(..., alias="Instituição de ensino", description="Instituição de ensino do aluno")
    Pedra_20: Optional[str] = Field(None, alias="Pedra 20", description="Classificação Pedra em 2020")
    Pedra_21: Optional[str] = Field(None, alias="Pedra 21", description="Classificação Pedra em 2021")
    Pedra_22: str = Field(..., alias="Pedra 22", description="Classificação Pedra em 2022")
    Num_Av: int = Field(..., alias="Nº Av", ge=0, description="Número de avaliações")
    IAA: float = Field(..., ge=0, description="Indicador de Autoavaliação")
    IEG: float = Field(..., ge=0, description="Indicador de Engajamento")
    IPS: float = Field(..., ge=0, description="Indicador Psicossocial")
    IDA: float = Field(..., ge=0, description="Indicador de Desempenho Acadêmico")
    Matem: Optional[float] = Field(None, ge=0, le=10, description="Nota de Matemática")
    Portug: Optional[float] = Field(None, ge=0, le=10, description="Nota de Português")
    Ingles: Optional[float] = Field(None, alias="Inglês", description="Nota de Inglês")
    Indicado: str = Field("Não", description="Indicado para bolsa (Sim/Não)")
    Atingiu_PV: str = Field("Não", alias="Atingiu PV", description="Atingiu Ponto de Virada (Sim/Não)")
    IPV: float = Field(..., ge=0, description="Indicador de Ponto de Virada")
    Destaque_IEG: str = Field("Não", alias="Destaque IEG", description="Destaque em Engajamento (Sim/Não)")
    Destaque_IDA: str = Field("Não", alias="Destaque IDA", description="Destaque em Desempenho (Sim/Não)")
    Destaque_IPV: str = Field("Não", alias="Destaque IPV", description="Destaque em Ponto de Virada (Sim/Não)")

    model_config = {
        "populate_by_name": True,
        "json_schema_extra": {
            "examples": [
                {
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
                    "Destaque IPV": "Não"
                }
            ]
        }
    }


class PredictionResponse(BaseModel):
    """Response model for a single prediction."""
    risco: int = Field(..., description="Previsão: 1 = Risco de defasagem, 0 = Sem risco")
    risco_label: str = Field(..., description="Label da previsão")
    probabilidade_risco: float = Field(..., ge=0, le=1, description="Probabilidade de risco (0-1)")
    probabilidade_sem_risco: float = Field(..., ge=0, le=1, description="Probabilidade sem risco (0-1)")


class BatchPredictionRequest(BaseModel):
    """Request model for batch predictions."""
    students: List[StudentData] = Field(..., description="Lista de alunos para predição")


class BatchPredictionResponse(BaseModel):
    """Response model for batch predictions."""
    predictions: List[PredictionResponse]
    total: int


class HealthResponse(BaseModel):
    """Response model for health check."""
    status: str
    model_loaded: bool
    version: str


class ErrorResponse(BaseModel):
    """Response model for errors."""
    detail: str
