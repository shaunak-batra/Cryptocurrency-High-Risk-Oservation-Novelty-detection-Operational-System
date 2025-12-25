"""
Pydantic models for CHRONOS API requests and responses.
"""

from pydantic import BaseModel, Field
from typing import List, Optional, Dict, Any
from datetime import datetime
from enum import Enum


class RiskLevel(str, Enum):
    """Risk level classification."""
    LOW = "LOW"
    MEDIUM = "MEDIUM"
    HIGH = "HIGH"
    CRITICAL = "CRITICAL"


# ==========================================================================
# REQUEST MODELS
# ==========================================================================

class PredictionRequest(BaseModel):
    """Request model for single transaction prediction."""
    transaction_id: str = Field(..., description="Unique transaction identifier")
    features: List[float] = Field(..., description="Transaction features (165 values)")
    edge_index: Optional[List[List[int]]] = Field(None, description="Graph edge indices")
    return_explanation: bool = Field(False, description="Whether to return explanation")
    
    class Config:
        json_schema_extra = {
            "example": {
                "transaction_id": "tx_12345",
                "features": [0.1] * 165,
                "return_explanation": True
            }
        }


class BatchPredictionRequest(BaseModel):
    """Request model for batch predictions."""
    transactions: List[PredictionRequest] = Field(..., description="List of transactions")
    return_explanation: bool = Field(False, description="Return explanations for all")


class TransactionQuery(BaseModel):
    """Query parameters for transaction lookup."""
    transaction_id: str = Field(..., description="Transaction ID to look up")


# ==========================================================================
# RESPONSE MODELS
# ==========================================================================

class ExplanationResponse(BaseModel):
    """Explanation details for a prediction."""
    summary: str = Field(..., description="Natural language summary")
    top_features: List[Dict[str, Any]] = Field(default_factory=list, description="Top contributing features")
    counterfactual: Optional[Dict[str, Any]] = Field(None, description="Counterfactual explanation")
    attention_weights: Optional[List[float]] = Field(None, description="Attention weights for edges")
    shap_values: Optional[List[float]] = Field(None, description="SHAP values for features")


class PredictionResponse(BaseModel):
    """Response model for prediction endpoint."""
    transaction_id: str = Field(..., description="Transaction identifier")
    risk_score: float = Field(..., ge=0, le=1, description="Risk score (0-1)")
    risk_level: RiskLevel = Field(..., description="Risk classification")
    confidence: float = Field(..., ge=0, le=1, description="Prediction confidence")
    explanation: Optional[ExplanationResponse] = Field(None, description="Explanation if requested")
    latency_ms: float = Field(..., description="Inference latency in milliseconds")
    timestamp: datetime = Field(default_factory=datetime.utcnow, description="Prediction timestamp")
    model_version: str = Field("1.0.0", description="Model version used")
    
    class Config:
        json_schema_extra = {
            "example": {
                "transaction_id": "tx_12345",
                "risk_score": 0.89,
                "risk_level": "HIGH",
                "confidence": 0.94,
                "latency_ms": 42.3,
                "timestamp": "2025-01-15T10:30:00Z",
                "model_version": "1.0.0"
            }
        }


class BatchPredictionResponse(BaseModel):
    """Response model for batch predictions."""
    results: List[PredictionResponse] = Field(..., description="Prediction results")
    total_processed: int = Field(..., description="Number of transactions processed")
    total_latency_ms: float = Field(..., description="Total processing time")
    avg_latency_ms: float = Field(..., description="Average latency per transaction")


class AlertResponse(BaseModel):
    """Alert for high-risk transaction."""
    transaction_id: str
    risk_score: float
    risk_level: RiskLevel
    timestamp: datetime
    status: str = "pending"


class AlertsListResponse(BaseModel):
    """List of alerts response."""
    alerts: List[AlertResponse]
    total: int
    page: int = 1


class HealthResponse(BaseModel):
    """Health check response."""
    status: str = Field(..., description="Service status")
    model_loaded: bool = Field(..., description="Whether model is loaded")
    services: Dict[str, str] = Field(default_factory=dict, description="Service statuses")
    version: str = Field("1.0.0", description="API version")
    uptime_seconds: float = Field(0, description="Server uptime")


class PerformanceMetrics(BaseModel):
    """Model performance metrics."""
    f1_score: float
    precision: float
    recall: float
    auc_roc: float
    auc_pr: float
    total_predictions: int
    avg_latency_ms: float


class ErrorResponse(BaseModel):
    """Error response model."""
    error: str = Field(..., description="Error message")
    detail: Optional[str] = Field(None, description="Detailed error information")
    timestamp: datetime = Field(default_factory=datetime.utcnow)
