"""
Pydantic models for API request/response validation
"""
from pydantic import BaseModel, Field
from typing import List, Dict, Optional
from datetime import datetime


class EmotionPrediction(BaseModel):
    """Single emotion prediction result"""
    category: str = Field(..., description="Emotion category (Boredom, Engagement, etc.)")
    level: int = Field(..., ge=0, le=3, description="Intensity level (0-3)")
    confidence: float = Field(..., ge=0.0, le=1.0, description="Prediction confidence")
    probabilities: List[float] = Field(..., description="Probability distribution over levels")


class InferenceRequest(BaseModel):
    """Request for emotion detection"""
    image: str = Field(..., description="Base64 encoded image (face crop)")
    timestamp: Optional[float] = Field(None, description="Client timestamp")
    session_id: Optional[str] = Field(None, description="Session identifier for caching")


class InferenceResponse(BaseModel):
    """Response with emotion predictions"""
    predictions: Dict[str, EmotionPrediction] = Field(..., description="Predictions for all categories")
    processing_time_ms: float = Field(..., description="Server processing time in milliseconds")
    cached: bool = Field(False, description="Whether result was retrieved from cache")
    timestamp: float = Field(default_factory=lambda: datetime.now().timestamp())


class BatchInferenceRequest(BaseModel):
    """Batch request for multiple faces"""
    images: List[str] = Field(..., max_items=8, description="Base64 encoded images (max 8)")
    session_id: Optional[str] = Field(None, description="Session identifier")


class BatchInferenceResponse(BaseModel):
    """Batch response"""
    results: List[InferenceResponse] = Field(..., description="Results for each input image")
    total_processing_time_ms: float = Field(..., description="Total batch processing time")


class HealthResponse(BaseModel):
    """Health check response"""
    status: str = Field(..., description="Service status")
    model_loaded: bool = Field(..., description="Whether model is loaded")
    gpu_available: bool = Field(..., description="Whether GPU is available")
    cache_size: int = Field(..., description="Current cache size")
    uptime_seconds: float = Field(..., description="Server uptime")
