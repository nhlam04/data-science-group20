"""
FastAPI application for emotion detection API
Optimized for real-time webcam inference
"""
from fastapi import FastAPI, HTTPException, BackgroundTasks
from fastapi.middleware.cors import CORSMiddleware
from contextlib import asynccontextmanager
import logging
import time
from typing import Dict

from config import *
from models import (
    InferenceRequest, InferenceResponse, 
    BatchInferenceRequest, BatchInferenceResponse,
    HealthResponse, EmotionPrediction
)
from inference_engine import get_inference_engine

# Configure logging
logging.basicConfig(
    level=LOG_LEVEL,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Lifecycle manager for model loading"""
    logger.info("Starting up - loading models...")
    engine = get_inference_engine()
    engine.load_models()
    logger.info("Models loaded successfully")
    yield
    logger.info("Shutting down...")


# Create FastAPI app
app = FastAPI(
    title="Emotion Detection API",
    description="Real-time emotion detection from webcam faces using Qwen2.5-VL",
    version="1.0.0",
    lifespan=lifespan
)

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=CORS_ORIGINS,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.get("/", tags=["Root"])
async def root():
    """Root endpoint"""
    return {
        "message": "Emotion Detection API",
        "version": "1.0.0",
        "endpoints": {
            "health": "/health",
            "predict": "/predict",
            "batch": "/predict/batch",
            "docs": "/docs"
        }
    }


@app.get("/health", response_model=HealthResponse, tags=["Health"])
async def health_check():
    """Health check endpoint"""
    engine = get_inference_engine()
    
    return HealthResponse(
        status="healthy" if engine.model_loaded else "loading",
        model_loaded=engine.model_loaded,
        gpu_available=torch.cuda.is_available(),
        cache_size=engine.get_cache_stats().get("size", 0),
        uptime_seconds=engine.get_uptime()
    )


@app.get("/cache/stats", tags=["Monitoring"])
async def get_cache_stats():
    """Get cache statistics"""
    engine = get_inference_engine()
    return engine.get_cache_stats()


@app.post("/predict", response_model=InferenceResponse, tags=["Inference"])
async def predict_emotion(request: InferenceRequest):
    """
    Predict emotions from a single face image
    
    - **image**: Base64 encoded face crop (RGB, 112x112 or similar)
    - **session_id**: Optional session ID for better caching
    """
    engine = get_inference_engine()
    
    if not engine.model_loaded:
        raise HTTPException(status_code=503, detail="Models not loaded yet")
    
    try:
        # Run inference
        predictions_dict, cached, processing_time = engine.predict(
            request.image,
            session_id=request.session_id
        )
        
        # Format response
        predictions = {}
        for category, pred_data in predictions_dict.items():
            predictions[category] = EmotionPrediction(
                category=category,
                level=pred_data["level"],
                confidence=pred_data["confidence"],
                probabilities=pred_data["probabilities"]
            )
        
        return InferenceResponse(
            predictions=predictions,
            processing_time_ms=processing_time,
            cached=cached,
            timestamp=time.time()
        )
    
    except Exception as e:
        logger.error(f"Error during inference: {str(e)}", exc_info=True)
        raise HTTPException(status_code=500, detail=f"Inference error: {str(e)}")


@app.post("/predict/batch", response_model=BatchInferenceResponse, tags=["Inference"])
async def predict_emotion_batch(request: BatchInferenceRequest):
    """
    Predict emotions from batch of face images (max 8)
    
    - **images**: List of base64 encoded face crops
    - **session_id**: Optional session ID for better caching
    """
    engine = get_inference_engine()
    
    if not engine.model_loaded:
        raise HTTPException(status_code=503, detail="Models not loaded yet")
    
    if len(request.images) > BATCH_SIZE:
        raise HTTPException(
            status_code=400, 
            detail=f"Batch size {len(request.images)} exceeds maximum {BATCH_SIZE}"
        )
    
    try:
        # Run batch inference
        predictions_list, cached_flags, total_time = engine.predict_batch(
            request.images,
            session_id=request.session_id
        )
        
        # Format responses
        results = []
        for predictions_dict, cached in zip(predictions_list, cached_flags):
            predictions = {}
            for category, pred_data in predictions_dict.items():
                predictions[category] = EmotionPrediction(
                    category=category,
                    level=pred_data["level"],
                    confidence=pred_data["confidence"],
                    probabilities=pred_data["probabilities"]
                )
            
            results.append(InferenceResponse(
                predictions=predictions,
                processing_time_ms=total_time / len(request.images),  # Approximate
                cached=cached,
                timestamp=time.time()
            ))
        
        return BatchInferenceResponse(
            results=results,
            total_processing_time_ms=total_time
        )
    
    except Exception as e:
        logger.error(f"Error during batch inference: {str(e)}", exc_info=True)
        raise HTTPException(status_code=500, detail=f"Batch inference error: {str(e)}")


@app.get("/metrics", tags=["Monitoring"])
async def get_metrics():
    """Get performance metrics"""
    engine = get_inference_engine()
    
    return {
        "uptime_seconds": engine.get_uptime(),
        "cache_stats": engine.get_cache_stats(),
        "gpu_info": {
            "available": torch.cuda.is_available(),
            "device_name": torch.cuda.get_device_name(0) if torch.cuda.is_available() else None,
            "memory_allocated_gb": torch.cuda.memory_allocated(0) / 1e9 if torch.cuda.is_available() else 0,
            "memory_reserved_gb": torch.cuda.memory_reserved(0) / 1e9 if torch.cuda.is_available() else 0,
        },
        "model_info": {
            "model_name": MODEL_NAME,
            "classifiers_loaded": len(engine.classifiers),
            "categories": list(engine.classifiers.keys())
        }
    }


if __name__ == "__main__":
    import uvicorn
    
    uvicorn.run(
        app,
        host=API_HOST,
        port=API_PORT,
        workers=API_WORKERS,
        log_level=LOG_LEVEL.lower()
    )
