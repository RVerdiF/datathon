"""
FastAPI application for Passos Mágicos student risk prediction.
Provides endpoints for predicting school lag risk.
"""

import logging
from pathlib import Path
from contextlib import asynccontextmanager

import joblib
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from api.schemas import HealthResponse
from api.routers import predict

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
)
logger = logging.getLogger("passos_magicos")

# Paths
MODEL_PATH = Path("models/model.joblib")
PREPROCESSOR_PATH = Path("models/preprocessor.joblib")


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Load model and preprocessor on startup."""
    logger.info("Loading model and preprocessor...")

    try:
        from src.preprocessing import DataPreprocessor

        app.state.model = joblib.load(MODEL_PATH)
        app.state.preprocessor = DataPreprocessor.load(PREPROCESSOR_PATH)
        app.state.model_loaded = True
        logger.info("Model and preprocessor loaded successfully.")
    except Exception as e:
        logger.error(f"Failed to load model: {e}")
        app.state.model_loaded = False

    yield

    logger.info("Shutting down application.")


# Create FastAPI app
app = FastAPI(
    title="Passos Mágicos - Student Risk Prediction API",
    description=(
        "API para predição de risco de defasagem escolar de estudantes "
        "da Associação Passos Mágicos. Utiliza um modelo de Machine Learning "
        "treinado com dados educacionais para identificar estudantes em risco."
    ),
    version="1.0.0",
    lifespan=lifespan,
)

# CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Include routers
app.include_router(predict.router)


@app.get("/", response_model=HealthResponse)
async def root():
    """Health check endpoint."""
    return HealthResponse(
        status="healthy",
        model_loaded=app.state.model_loaded,
        version="1.0.0",
    )
