"""
Configuration for emotion detection API deployment
Optimized for RTX 4000 (20GB VRAM, 8 vCPU, 32GB RAM)
"""
import os
from pathlib import Path

# Server specs
GPU_MEMORY_GB = 20
VCPU_COUNT = 8
RAM_GB = 32

# Model configuration
MODEL_NAME = "Qwen/Qwen2.5-VL-7B-Instruct"
MODELS_DIR = Path(__file__).parent.parent / "models"
CLASSIFIER_CHECKPOINTS_DIR = MODELS_DIR / "classifiers"

# Emotion categories
CATEGORIES = ["Boredom", "Engagement", "Confusion", "Frustration"]
NUM_CLASSES = 4  # Levels 0-3

# Inference configuration
BATCH_SIZE = 8  # Process up to 8 faces simultaneously
MAX_QUEUE_SIZE = 32  # Maximum number of requests in queue
FPS = 1  # Frames per second for video processing

# Model optimization
USE_FP16 = True  # Use half precision (saves ~50% VRAM)
USE_FLASH_ATTENTION = True  # Faster attention computation
COMPILE_MODEL = False  # torch.compile (experimental, may not work on all systems)

# Caching configuration
ENABLE_EMBEDDING_CACHE = True
CACHE_SIMILARITY_THRESHOLD = 0.95  # Cosine similarity threshold
MAX_CACHE_SIZE = 1000  # Maximum cached embeddings
CACHE_TTL_SECONDS = 300  # Cache time-to-live (5 minutes)

# Redis configuration (optional, for distributed caching)
USE_REDIS = False
REDIS_HOST = os.getenv("REDIS_HOST", "localhost")
REDIS_PORT = int(os.getenv("REDIS_PORT", 6379))
REDIS_DB = int(os.getenv("REDIS_DB", 0))

# API configuration
API_HOST = "0.0.0.0"
API_PORT = 8000
API_WORKERS = 1  # Single worker to avoid loading model multiple times
CORS_ORIGINS = ["*"]  # Update with your frontend URL in production

# Image preprocessing
MAX_IMAGE_SIZE = 224  # Maximum dimension for input images
FACE_CROP_PADDING = 0.2  # 20% padding around detected face
JPEG_QUALITY = 85  # Quality for image compression

# Performance monitoring
ENABLE_METRICS = True
LOG_LEVEL = "INFO"
