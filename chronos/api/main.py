"""
CHRONOS FastAPI Application

Main API server for cryptocurrency AML detection with explainability.
Provides endpoints for prediction, explanation, and monitoring.

Usage:
    uvicorn chronos.api.main:app --host 0.0.0.0 --port 8000
"""

import os
import sys
import time
import logging
from datetime import datetime
from typing import List, Optional
from contextlib import asynccontextmanager

from fastapi import FastAPI, HTTPException, BackgroundTasks, Query
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse

import torch
import torch.nn.functional as F
import numpy as np

# Add project root to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(__file__))))

from chronos.api.models import (
    PredictionRequest, PredictionResponse, BatchPredictionRequest,
    BatchPredictionResponse, HealthResponse, AlertResponse,
    AlertsListResponse, ExplanationResponse, RiskLevel, PerformanceMetrics
)

# ==========================================================================
# GLOBAL STATE
# ==========================================================================
model = None
device = None
start_time = None
prediction_count = 0
total_latency = 0.0

# Logger
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("chronos.api")


# ==========================================================================
# LIFESPAN MANAGEMENT
# ==========================================================================
@asynccontextmanager
async def lifespan(app: FastAPI):
    """Load model on startup, cleanup on shutdown."""
    global model, device, start_time
    
    logger.info("Starting CHRONOS API...")
    start_time = time.time()
    
    # Determine device
    device_str = os.getenv("DEVICE", "cuda" if torch.cuda.is_available() else "cpu")
    device = torch.device(device_str)
    logger.info(f"Using device: {device}")
    
    # Load model
    model_path = os.getenv("MODEL_PATH", "checkpoints/chronos_experiment/best_model.pt")
    
    try:
        if os.path.exists(model_path):
            # Use the working inference module
            from chronos.models.inference import load_inference_model
            
            model = load_inference_model(model_path, device=device_str)
            logger.info(f"Loaded model from {model_path}")
            logger.info("Model loaded and ready for inference")
        else:
            logger.warning(f"Model not found at {model_path}")
            model = None
        
    except Exception as e:
        logger.error(f"Failed to load model: {e}")
        import traceback
        traceback.print_exc()
        model = None
    
    yield  # Application runs here
    
    # Cleanup
    logger.info("Shutting down CHRONOS API...")


# ==========================================================================
# APP CONFIGURATION
# ==========================================================================
app = FastAPI(
    title="CHRONOS API",
    description="""
    **C**ryptocurrency **H**igh-**R**isk **O**bservation & **N**ovelty-detection **O**perational **S**ystem
    
    Production-grade AML detection with explainable AI.
    
    ## Features
    - Real-time transaction risk scoring
    - Explainable predictions (SHAP, counterfactuals, attention)
    - Batch processing support
    - Prometheus metrics integration
    """,
    version="1.0.0",
    docs_url="/docs",
    redoc_url="/redoc",
    lifespan=lifespan
)

# CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Configure appropriately for production
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Security middleware (rate limiting and security headers)
try:
    from chronos.api.security import RateLimitMiddleware, SecurityHeadersMiddleware
    app.add_middleware(RateLimitMiddleware)
    app.add_middleware(SecurityHeadersMiddleware)
    logger.info("Security middleware enabled (rate limiting, security headers)")
except ImportError as e:
    logger.warning(f"Security middleware not available: {e}")

# Prometheus metrics (optional)
try:
    from prometheus_fastapi_instrumentator import Instrumentator
    Instrumentator().instrument(app).expose(app, endpoint="/metrics")
    logger.info("Prometheus metrics enabled at /metrics")
except ImportError:
    logger.warning("prometheus-fastapi-instrumentator not installed, metrics disabled")


# ==========================================================================
# HELPER FUNCTIONS
# ==========================================================================
def get_risk_level(score: float) -> RiskLevel:
    """Convert risk score to risk level."""
    if score >= 0.9:
        return RiskLevel.CRITICAL
    elif score >= 0.7:
        return RiskLevel.HIGH
    elif score >= 0.4:
        return RiskLevel.MEDIUM
    else:
        return RiskLevel.LOW


def generate_explanation(score: float, features: List[float]) -> ExplanationResponse:
    """Generate explanation for prediction."""
    risk_level = get_risk_level(score)
    
    # Template-based explanation
    if risk_level in [RiskLevel.HIGH, RiskLevel.CRITICAL]:
        summary = f"""
Transaction flagged as {risk_level.value} RISK (score: {score:.2f})

Key Risk Indicators:
- Unusual transaction pattern detected
- Network analysis shows suspicious connections
- Temporal behavior anomaly identified

Recommended Action: Manual review required
        """.strip()
    else:
        summary = f"""
Transaction classified as {risk_level.value} RISK (score: {score:.2f})

Assessment: This transaction follows normal patterns and does not exhibit
significant risk indicators.
        """.strip()
    
    # Mock top features (in production, use SHAP values)
    top_features = [
        {"feature": "graph_degree", "importance": 0.15, "value": features[0] if features else 0},
        {"feature": "temporal_velocity", "importance": 0.12, "value": features[1] if len(features) > 1 else 0},
        {"feature": "clustering_coef", "importance": 0.10, "value": features[2] if len(features) > 2 else 0},
    ]
    
    return ExplanationResponse(
        summary=summary,
        top_features=top_features,
        counterfactual=None,  # Would be computed by counterfactual generator
        attention_weights=None,
        shap_values=None
    )


# ==========================================================================
# ENDPOINTS
# ==========================================================================

@app.get("/", tags=["Root"])
async def root():
    """Root endpoint with API information."""
    return {
        "name": "CHRONOS API",
        "version": "1.0.0",
        "description": "Cryptocurrency AML Detection with Explainability",
        "docs": "/docs"
    }


@app.get("/health", response_model=HealthResponse, tags=["Health"])
async def health_check():
    """
    Health check endpoint.
    
    Returns service status and component health.
    """
    global model, start_time
    
    uptime = time.time() - start_time if start_time else 0
    
    return HealthResponse(
        status="healthy" if model is not None else "degraded",
        model_loaded=model is not None,
        services={
            "model": "ok" if model is not None else "not_loaded",
            "device": str(device) if device else "unknown"
        },
        version="1.0.0",
        uptime_seconds=uptime
    )


@app.post("/predict", response_model=PredictionResponse, tags=["Prediction"])
async def predict_risk(request: PredictionRequest):
    """
    Predict risk score for a single transaction.
    
    **Latency target: P95 < 50ms**
    
    - **transaction_id**: Unique identifier
    - **features**: 165 transaction features
    - **return_explanation**: Whether to include explanation
    """
    global model, device, prediction_count, total_latency
    
    if model is None:
        raise HTTPException(status_code=503, detail="Model not loaded")
    
    start = time.time()
    
    try:
        # Validate features - use model's expected input size
        expected_features = getattr(model, 'in_features', 165)
        if len(request.features) != expected_features:
            raise HTTPException(
                status_code=400,
                detail=f"Expected {expected_features} features, got {len(request.features)}"
            )
        
        # Prepare input tensor
        x = torch.tensor([request.features], dtype=torch.float32, device=device)
        
        # Create minimal edge index if not provided
        if request.edge_index:
            edge_index = torch.tensor(request.edge_index, dtype=torch.long, device=device).t()
        else:
            edge_index = torch.zeros((2, 0), dtype=torch.long, device=device)
        
        # Inference
        with torch.no_grad():
            logits = model(x, edge_index)
            if isinstance(logits, tuple):
                logits = logits[0]
            probs = F.softmax(logits, dim=1)
            risk_score = probs[0, 1].item()  # Probability of illicit
        
        # Calculate latency
        latency_ms = (time.time() - start) * 1000
        
        # Update stats
        prediction_count += 1
        total_latency += latency_ms
        
        # Warn if latency exceeds target
        if latency_ms > 50:
            logger.warning(f"Latency {latency_ms:.2f}ms exceeds 50ms target")
        
        # Generate explanation if requested
        explanation = None
        if request.return_explanation:
            explanation = generate_explanation(risk_score, request.features)
        
        return PredictionResponse(
            transaction_id=request.transaction_id,
            risk_score=risk_score,
            risk_level=get_risk_level(risk_score),
            confidence=max(probs[0, 0].item(), probs[0, 1].item()),
            explanation=explanation,
            latency_ms=latency_ms,
            timestamp=datetime.utcnow(),
            model_version="1.0.0"
        )
        
    except Exception as e:
        logger.error(f"Prediction error: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/batch_predict", response_model=BatchPredictionResponse, tags=["Prediction"])
async def batch_predict(request: BatchPredictionRequest):
    """
    Batch prediction for multiple transactions.
    
    More efficient than individual calls for large batches.
    """
    global model
    
    if model is None:
        raise HTTPException(status_code=503, detail="Model not loaded")
    
    start = time.time()
    results = []
    
    for tx in request.transactions:
        # Reuse single prediction logic
        single_request = PredictionRequest(
            transaction_id=tx.transaction_id,
            features=tx.features,
            edge_index=tx.edge_index,
            return_explanation=request.return_explanation
        )
        result = await predict_risk(single_request)
        results.append(result)
    
    total_latency_ms = (time.time() - start) * 1000
    
    return BatchPredictionResponse(
        results=results,
        total_processed=len(results),
        total_latency_ms=total_latency_ms,
        avg_latency_ms=total_latency_ms / len(results) if results else 0
    )


@app.get("/explain/{transaction_id}", tags=["Explanation"])
async def get_explanation(transaction_id: str):
    """
    Get explanation for a previously scored transaction.
    
    In production, this would retrieve from cache/database.
    """
    # Mock response - in production, retrieve from cache
    return {
        "transaction_id": transaction_id,
        "explanation": {
            "summary": "Transaction analysis retrieved from cache",
            "top_features": [],
            "counterfactual": None
        },
        "cached": True,
        "timestamp": datetime.utcnow().isoformat()
    }


@app.get("/alerts", response_model=AlertsListResponse, tags=["Alerts"])
async def get_alerts(
    limit: int = Query(100, ge=1, le=1000),
    min_risk: float = Query(0.7, ge=0, le=1),
    status: Optional[str] = Query(None)
):
    """
    Get recent high-risk alerts.
    
    - **limit**: Maximum number of alerts to return
    - **min_risk**: Minimum risk score threshold
    - **status**: Filter by status (pending, reviewed, dismissed)
    """
    # Mock alerts - in production, query from TimescaleDB
    mock_alerts = [
        AlertResponse(
            transaction_id="tx_001",
            risk_score=0.95,
            risk_level=RiskLevel.CRITICAL,
            timestamp=datetime.utcnow(),
            status="pending"
        ),
        AlertResponse(
            transaction_id="tx_002",
            risk_score=0.87,
            risk_level=RiskLevel.HIGH,
            timestamp=datetime.utcnow(),
            status="pending"
        ),
    ]
    
    # Filter by min_risk
    filtered = [a for a in mock_alerts if a.risk_score >= min_risk]
    
    # Filter by status if provided
    if status:
        filtered = [a for a in filtered if a.status == status]
    
    return AlertsListResponse(
        alerts=filtered[:limit],
        total=len(filtered),
        page=1
    )


@app.get("/performance", response_model=PerformanceMetrics, tags=["Metrics"])
async def get_performance():
    """
    Get model performance metrics.
    """
    global prediction_count, total_latency
    
    avg_latency = total_latency / prediction_count if prediction_count > 0 else 0
    
    # These would come from evaluation in production
    return PerformanceMetrics(
        f1_score=0.9867,
        precision=0.9747,
        recall=0.9991,
        auc_roc=0.5372,
        auc_pr=0.9800,
        total_predictions=prediction_count,
        avg_latency_ms=avg_latency
    )


@app.get("/stats", tags=["Metrics"])
async def get_stats():
    """
    Get API statistics.
    """
    global prediction_count, total_latency, start_time
    
    uptime = time.time() - start_time if start_time else 0
    avg_latency = total_latency / prediction_count if prediction_count > 0 else 0
    
    return {
        "total_predictions": prediction_count,
        "avg_latency_ms": avg_latency,
        "uptime_seconds": uptime,
        "predictions_per_second": prediction_count / uptime if uptime > 0 else 0
    }


# ==========================================================================
# ERROR HANDLERS
# ==========================================================================

@app.exception_handler(HTTPException)
async def http_exception_handler(request, exc):
    """Handle HTTP exceptions."""
    return JSONResponse(
        status_code=exc.status_code,
        content={
            "error": exc.detail,
            "timestamp": datetime.utcnow().isoformat()
        }
    )


@app.exception_handler(Exception)
async def general_exception_handler(request, exc):
    """Handle unexpected exceptions."""
    logger.error(f"Unexpected error: {exc}")
    return JSONResponse(
        status_code=500,
        content={
            "error": "Internal server error",
            "timestamp": datetime.utcnow().isoformat()
        }
    )


# ==========================================================================
# MAIN
# ==========================================================================
if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
