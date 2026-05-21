"""
app.py — FastAPI inference server for the Titanic Logistic Regression model.

Endpoints:
  GET  /              → redirect to docs
  GET  /health        → liveness check
  GET  /ready         → readiness check (model loaded?)
  POST /predict       → single passenger prediction
  POST /predict/batch → batch of passengers
  GET  /model/info    → model metadata + feature list

Auto-generated docs available at:
  http://localhost:8000/docs      ← Swagger UI
  http://localhost:8000/redoc     ← ReDoc

Run locally:
  uvicorn scripts.app:app --reload --port 8000

Docker:
  docker-compose up
"""

import json
import logging
import os
import sys
import time
from contextlib import asynccontextmanager
from pathlib import Path
from typing import Optional

import joblib
from fastapi import FastAPI, HTTPException, Request, status
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse, RedirectResponse
from pydantic import BaseModel, Field, field_validator

# ── Path setup ────────────────────────────────────────────────────
sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))
from inference import ModelInference  # noqa: E402

# ── Logging ───────────────────────────────────────────────────────
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(levelname)s  %(name)s  %(message)s",
)
logger = logging.getLogger(__name__)

# ── Global model state ────────────────────────────────────────────
_inference_engine: Optional[ModelInference] = None
_model_load_time: Optional[float] = None
_model_path: str = os.getenv("MODEL_PATH", "/model/model.pkl")
_preprocessor_path: str = os.getenv("PREPROCESSOR_PATH", "/model/preprocessor.pkl")


# ══════════════════════════════════════════════════════════════════
# Pydantic schemas
# ══════════════════════════════════════════════════════════════════

class PassengerFeatures(BaseModel):
    """Input features for a single passenger prediction."""

    Pclass: int = Field(..., ge=1, le=3, description="Ticket class: 1, 2, or 3")
    Name: str   = Field(..., min_length=1, description="Passenger full name (used to extract title)")
    Sex: str    = Field(..., description="Sex: 'male' or 'female'")
    Age: Optional[float] = Field(None, ge=0, le=120, description="Age in years (nullable — will be imputed)")
    SibSp: int  = Field(0, ge=0, description="Number of siblings/spouses aboard")
    Parch: int  = Field(0, ge=0, description="Number of parents/children aboard")
    Ticket: str = Field("", description="Ticket number")
    Fare: float = Field(..., ge=0, description="Passenger fare")
    Cabin: Optional[str] = Field(None, description="Cabin number (nullable)")
    Embarked: Optional[str] = Field("S", description="Port: S, C, or Q")

    @field_validator("Sex")
    @classmethod
    def validate_sex(cls, v: str) -> str:
        if v.lower() not in {"male", "female"}:
            raise ValueError("Sex must be 'male' or 'female'")
        return v.lower()

    @field_validator("Embarked")
    @classmethod
    def validate_embarked(cls, v: Optional[str]) -> Optional[str]:
        if v and v.upper() not in {"S", "C", "Q"}:
            raise ValueError("Embarked must be S, C, or Q")
        return v.upper() if v else "S"

    model_config = {"json_schema_extra": {
        "example": {
            "Pclass": 3,
            "Name": "Braund, Mr. Owen Harris",
            "Sex": "male",
            "Age": 22.0,
            "SibSp": 1,
            "Parch": 0,
            "Ticket": "A/5 21171",
            "Fare": 7.25,
            "Cabin": None,
            "Embarked": "S",
        }
    }}


class PredictionResponse(BaseModel):
    """Response for a single passenger prediction."""
    prediction: int                  = Field(..., description="0 = did not survive, 1 = survived")
    prediction_label: str            = Field(..., description="Human-readable label")
    survival_probability: float      = Field(..., description="Probability of survival (0–1)")
    non_survival_probability: float  = Field(..., description="Probability of not surviving (0–1)")


class BatchPredictionRequest(BaseModel):
    """Request body for batch predictions."""
    passengers: list[PassengerFeatures] = Field(..., min_length=1, max_length=500)

    model_config = {"json_schema_extra": {
        "example": {
            "passengers": [
                {"Pclass": 1, "Name": "Cumings, Mrs. John Bradley", "Sex": "female",
                 "Age": 38, "SibSp": 1, "Parch": 0, "Ticket": "PC 17599",
                 "Fare": 71.28, "Cabin": "C85", "Embarked": "C"},
                {"Pclass": 3, "Name": "Heikkinen, Miss. Laina", "Sex": "female",
                 "Age": 26, "SibSp": 0, "Parch": 0, "Ticket": "STON/O2 3101282",
                 "Fare": 7.92, "Cabin": None, "Embarked": "S"},
            ]
        }
    }}


class BatchPredictionResponse(BaseModel):
    predictions:    list[PredictionResponse]
    total:          int
    survived_count: int
    survival_rate:  float


class HealthResponse(BaseModel):
    status:       str
    model_loaded: bool


class ReadyResponse(BaseModel):
    ready:          bool
    model_path:     str
    load_time_sec:  Optional[float]
    feature_count:  Optional[int]
    features:       Optional[list[str]]


class ModelInfoResponse(BaseModel):
    model_type:    str
    dataset:       str
    features:      list[str]
    feature_count: int
    model_path:    str


# ══════════════════════════════════════════════════════════════════
# App lifecycle
# ══════════════════════════════════════════════════════════════════

@asynccontextmanager
async def lifespan(app: FastAPI):
    """Load model on startup, clean up on shutdown."""
    global _inference_engine, _model_load_time
    t0 = time.time()
    try:
        logger.info("Loading model from %s", _model_path)
        logger.info("Loading preprocessor from %s", _preprocessor_path)
        _inference_engine = ModelInference(_model_path, _preprocessor_path)
        _model_load_time = round(time.time() - t0, 3)
        logger.info("✅ Model loaded in %.3fs", _model_load_time)
    except Exception as exc:
        logger.error("❌ Model failed to load: %s", exc)
        # Don't crash — /health will report model_loaded=False
    yield
    logger.info("Shutting down inference server")


# ══════════════════════════════════════════════════════════════════
# FastAPI app
# ══════════════════════════════════════════════════════════════════

app = FastAPI(
    title="Titanic Survival Prediction API",
    description=(
        "Logistic Regression model trained on the Titanic dataset. "
        "Predicts passenger survival probability with per-feature explainability. "
        "Built with FastAPI + Azure ML."
    ),
    version="1.0.0",
    contact={"name": "Suriya Divi", "email": "suriya.divi@outlook.com"},
    lifespan=lifespan,
    docs_url="/docs",
    redoc_url="/redoc",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["GET", "POST"],
    allow_headers=["*"],
)


# ── Request timing middleware ─────────────────────────────────────
@app.middleware("http")
async def add_process_time_header(request: Request, call_next):
    t0 = time.time()
    response = await call_next(request)
    response.headers["X-Process-Time-Ms"] = str(round((time.time() - t0) * 1000, 2))
    return response


# ══════════════════════════════════════════════════════════════════
# Endpoints
# ══════════════════════════════════════════════════════════════════

@app.get("/", include_in_schema=False)
async def root():
    """Redirect to Swagger docs."""
    return RedirectResponse(url="/docs")


@app.get("/health", response_model=HealthResponse, tags=["Monitoring"])
async def health():
    """Liveness probe — always returns 200 if the process is running."""
    return HealthResponse(
        status="healthy",
        model_loaded=_inference_engine is not None,
    )


@app.get("/ready", response_model=ReadyResponse, tags=["Monitoring"])
async def ready():
    """
    Readiness probe — returns 503 if model isn't loaded yet.
    Use this in Kubernetes/ACA readinessProbe.
    """
    if _inference_engine is None:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="Model not loaded yet",
        )
    features = _inference_engine.processor.feature_names
    return ReadyResponse(
        ready=True,
        model_path=_model_path,
        load_time_sec=_model_load_time,
        feature_count=len(features) if features else 0,
        features=features,
    )


@app.post("/predict", response_model=PredictionResponse, tags=["Inference"])
async def predict(passenger: PassengerFeatures):
    """
    Predict survival for a single passenger.

    Returns survival probability (0–1) and a human-readable label.
    """
    if _inference_engine is None:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="Model not loaded",
        )
    try:
        result = _inference_engine.predict_single(passenger.model_dump())
        return PredictionResponse(**result)
    except Exception as exc:
        logger.error("Prediction failed: %s", exc, exc_info=True)
        raise HTTPException(status_code=status.HTTP_422_UNPROCESSABLE_ENTITY, detail=str(exc))


@app.post("/predict/batch", response_model=BatchPredictionResponse, tags=["Inference"])
async def predict_batch(body: BatchPredictionRequest):
    """
    Predict survival for a batch of passengers (max 500).

    Returns individual predictions plus aggregated survival statistics.
    """
    if _inference_engine is None:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="Model not loaded",
        )
    predictions = []
    for p in body.passengers:
        try:
            result = _inference_engine.predict_single(p.model_dump())
            predictions.append(PredictionResponse(**result))
        except Exception as exc:
            logger.warning("Skipping passenger due to error: %s", exc)
            predictions.append(PredictionResponse(
                prediction=0,
                prediction_label="Error",
                survival_probability=0.0,
                non_survival_probability=1.0,
            ))

    survived = sum(p.prediction for p in predictions)
    return BatchPredictionResponse(
        predictions=predictions,
        total=len(predictions),
        survived_count=survived,
        survival_rate=round(survived / len(predictions), 4) if predictions else 0.0,
    )


@app.get("/model/info", response_model=ModelInfoResponse, tags=["Model"])
async def model_info():
    """Get model metadata — type, dataset, feature names."""
    if _inference_engine is None:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="Model not loaded",
        )
    features = _inference_engine.processor.feature_names or []
    return ModelInfoResponse(
        model_type="logistic_regression",
        dataset="titanic",
        features=features,
        feature_count=len(features),
        model_path=_model_path,
    )


# ── Entrypoint for local dev ──────────────────────────────────────
if __name__ == "__main__":
    import uvicorn
    uvicorn.run(
        "app:app",
        host="0.0.0.0",
        port=int(os.getenv("PORT", "8000")),
        reload=False,
        log_level="info",
    )
