"""
Predict router for the Passos Mágicos API.
Handles single and batch prediction endpoints.
"""

import pandas as pd
import numpy as np
import logging
from fastapi import APIRouter, HTTPException

from api.schemas import (
    StudentData,
    PredictionResponse,
    BatchPredictionRequest,
    BatchPredictionResponse,
)

logger = logging.getLogger("passos_magicos")

router = APIRouter(tags=["predictions"])


def student_to_dataframe(student: StudentData) -> pd.DataFrame:
    """Convert a StudentData Pydantic model to a DataFrame row."""
    data = {
        "Fase": student.Fase,
        "Gênero": student.Genero,
        "Ano ingresso": student.Ano_ingresso,
        "Instituição de ensino": student.Instituicao_ensino,
        "Pedra 20": student.Pedra_20,
        "Pedra 21": student.Pedra_21,
        "Pedra 22": student.Pedra_22,
        "Nº Av": student.Num_Av,
        "IAA": student.IAA,
        "IEG": student.IEG,
        "IPS": student.IPS,
        "IDA": student.IDA,
        "Matem": student.Matem,
        "Portug": student.Portug,
        "Inglês": student.Ingles,
        "Indicado": student.Indicado,
        "Atingiu PV": student.Atingiu_PV,
        "IPV": student.IPV,
        "Destaque IEG": student.Destaque_IEG,
        "Destaque IDA": student.Destaque_IDA,
        "Destaque IPV": student.Destaque_IPV,
    }
    return pd.DataFrame([data])


def predict_single(student_df: pd.DataFrame, app_state: dict) -> PredictionResponse:
    """Run prediction for a single student DataFrame row."""
    preprocessor = app_state["preprocessor"]
    model = app_state["model"]

    # Apply preprocessing (encode + scale)
    processed = preprocessor.encode_categorical(student_df, fit=False)
    
    # Handle missing values
    processed = preprocessor.handle_missing_values(processed)
    
    # Select only the feature columns the model expects
    feature_cols = preprocessor.feature_columns
    
    # Ensure all feature columns exist
    for col in feature_cols:
        if col not in processed.columns:
            processed[col] = 0
    
    processed = processed[feature_cols]
    
    # Scale features
    numeric_cols = processed.select_dtypes(include=[np.number]).columns.tolist()
    if numeric_cols:
        processed[numeric_cols] = preprocessor.scaler.transform(processed[numeric_cols])

    # Predict
    prediction = model.predict(processed)[0]
    probabilities = model.predict_proba(processed)[0]

    return PredictionResponse(
        risco=int(prediction),
        risco_label="Risco" if prediction == 1 else "Sem Risco",
        probabilidade_risco=float(probabilities[1]),
        probabilidade_sem_risco=float(probabilities[0]),
    )


@router.post("/predict", response_model=PredictionResponse)
async def predict(student: StudentData):
    """
    Predict the risk of school lag for a single student.
    
    Returns the prediction (0 = no risk, 1 = risk) and probabilities.
    """
    from api.main import app

    if not app.state.model_loaded:
        raise HTTPException(status_code=503, detail="Model not loaded")

    try:
        student_df = student_to_dataframe(student)
        result = predict_single(
            student_df,
            {"preprocessor": app.state.preprocessor, "model": app.state.model},
        )
        
        logger.info(
            f"Prediction: {result.risco_label} "
            f"(prob={result.probabilidade_risco:.3f})"
        )
        return result

    except Exception as e:
        logger.error(f"Prediction error: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Prediction error: {str(e)}")


@router.post("/predict/batch", response_model=BatchPredictionResponse)
async def predict_batch(request: BatchPredictionRequest):
    """
    Predict the risk of school lag for multiple students.
    
    Accepts a list of student data and returns predictions for each.
    """
    from api.main import app

    if not app.state.model_loaded:
        raise HTTPException(status_code=503, detail="Model not loaded")

    try:
        predictions = []
        for student in request.students:
            student_df = student_to_dataframe(student)
            result = predict_single(
                student_df,
                {"preprocessor": app.state.preprocessor, "model": app.state.model},
            )
            predictions.append(result)

        return BatchPredictionResponse(
            predictions=predictions, total=len(predictions)
        )

    except Exception as e:
        logger.error(f"Batch prediction error: {str(e)}")
        raise HTTPException(
            status_code=500, detail=f"Batch prediction error: {str(e)}"
        )
