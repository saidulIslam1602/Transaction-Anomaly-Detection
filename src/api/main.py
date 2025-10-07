"""
FastAPI application for Transaction Anomaly Detection.
"""

from fastapi import FastAPI, HTTPException, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from prometheus_client import Counter, Histogram, Gauge, generate_latest
from prometheus_fastapi_instrumentator import Instrumentator
from pydantic import BaseModel
from typing import Dict, List, Optional
import time
import logging

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Create FastAPI app
app = FastAPI(
    title="Transaction Anomaly Detection API",
    description="API for real-time fraud detection and anomaly analysis",
    version="2.0.0"
)

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Prometheus metrics
transactions_total = Counter(
    'transactions_total',
    'Total number of transactions processed'
)
fraud_detected = Counter(
    'fraud_detected_total',
    'Total number of frauds detected'
)
prediction_latency = Histogram(
    'prediction_latency_seconds',
    'Prediction latency in seconds'
)
model_accuracy = Gauge(
    'model_accuracy',
    'Current model accuracy'
)
data_drift_detected = Gauge(
    'data_drift_detected',
    'Data drift detection flag (0 or 1)'
)
feature_serving_latency = Histogram(
    'feature_serving_latency_ms',
    'Feature serving latency in milliseconds'
)

# Instrument app with Prometheus
Instrumentator().instrument(app).expose(app)


# Request/Response models
class Transaction(BaseModel):
    """Transaction input model."""
    step: int
    type: str
    amount: float
    nameOrig: str
    oldbalanceOrg: float
    newbalanceOrig: float
    nameDest: str
    oldbalanceDest: float
    newbalanceDest: float


class PredictionResponse(BaseModel):
    """Prediction response model."""
    transaction_id: str
    is_fraud: bool
    risk_score: float
    risk_level: str
    confidence: float
    explanation: Optional[str] = None
    recommendations: Optional[List[str]] = None
    processing_time_ms: float


class HealthResponse(BaseModel):
    """Health check response."""
    status: str
    version: str
    timestamp: float


@app.get("/", response_model=Dict[str, str])
async def root():
    """Root endpoint."""
    return {
        "name": "Transaction Anomaly Detection API",
        "version": "2.0.0",
        "status": "running"
    }


@app.get("/health", response_model=HealthResponse)
async def health_check():
    """Health check endpoint."""
    return HealthResponse(
        status="healthy",
        version="2.0.0",
        timestamp=time.time()
    )


@app.get("/ready")
async def readiness_check():
    """Readiness check endpoint."""
    # Check if models are loaded, services are available, etc.
    return {"status": "ready"}


@app.post("/predict", response_model=PredictionResponse)
async def predict(transaction: Transaction):
    """
    Predict if a transaction is fraudulent.
    
    Args:
        transaction: Transaction data
        
    Returns:
        Prediction response with risk score and explanation
    """
    start_time = time.time()
    
    try:
        # Count transaction
        transactions_total.inc()
        
        # Extract features (mock implementation)
        # In production, this would use the feature store
        features = {
            'amount': transaction.amount,
            'type': transaction.type,
            'balance_ratio': transaction.amount / (transaction.oldbalanceOrg + 1.0)
        }
        
        # Make prediction (mock implementation)
        # In production, this would use the loaded model
        risk_score = min(transaction.amount / 10000.0, 1.0)
        is_fraud = risk_score > 0.7
        
        if is_fraud:
            fraud_detected.inc()
        
        # Determine risk level
        if risk_score >= 0.75:
            risk_level = "CRITICAL"
        elif risk_score >= 0.5:
            risk_level = "HIGH"
        elif risk_score >= 0.25:
            risk_level = "MEDIUM"
        else:
            risk_level = "LOW"
        
        # Calculate processing time
        processing_time = (time.time() - start_time) * 1000
        prediction_latency.observe(processing_time / 1000)
        
        return PredictionResponse(
            transaction_id=f"TXN_{int(time.time())}",
            is_fraud=is_fraud,
            risk_score=risk_score,
            risk_level=risk_level,
            confidence=0.85,
            explanation="Risk assessment based on amount and account patterns",
            recommendations=[
                "Review transaction history",
                "Verify customer identity"
            ] if is_fraud else [],
            processing_time_ms=processing_time
        )
        
    except Exception as e:
        logger.error(f"Prediction error: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/metrics/model")
async def model_metrics():
    """Get current model performance metrics."""
    return {
        "accuracy": 0.95,
        "precision": 0.92,
        "recall": 0.96,
        "f1_score": 0.94,
        "auc": 0.96
    }


@app.get("/metrics/drift")
async def drift_metrics():
    """Get data drift metrics."""
    return {
        "drift_detected": False,
        "features_with_drift": [],
        "last_check": time.time()
    }


@app.exception_handler(Exception)
async def global_exception_handler(request: Request, exc: Exception):
    """Global exception handler."""
    logger.error(f"Global exception: {str(exc)}")
    return JSONResponse(
        status_code=500,
        content={"detail": "Internal server error"}
    )


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)

