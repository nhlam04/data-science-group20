# Emotion Detection API - Deployment Guide

Real-time emotion detection system using Qwen2.5-VL and trained classifier heads, optimized for RTX 4000 GPU.

## Architecture

### Two-Part System

1. **Training (Kaggle Notebook)**: Train classifier heads on pre-extracted Qwen embeddings
2. **Deployment (Server)**: FastAPI backend + Web frontend for real-time inference

### Technology Stack

- **Backend**: FastAPI, PyTorch, Transformers (Qwen2.5-VL)
- **Frontend**: Vanilla JavaScript, MediaPipe Face Detection
- **Deployment**: RTX 4000 GPU (20GB VRAM, 8 vCPU, 32GB RAM)

## Directory Structure

```
deployment/
├── backend/
│   ├── main.py                 # FastAPI application
│   ├── inference_engine.py     # Qwen + classifier inference
│   ├── config.py               # Configuration
│   ├── models.py               # Pydantic models
│   └── requirements.txt        # Python dependencies
├── frontend/
│   ├── index.html              # Web UI
│   └── app.js                  # Face detection + API client
└── models/
    └── classifiers/            # Trained classifier checkpoints
        ├── mlp_Boredom_best.pth
        ├── mlp_Engagement_best.pth
        ├── mlp_Confusion_best.pth
        └── mlp_Frustration_best.pth
```

## Setup Instructions

### 1. Training Phase (Kaggle)

1. Open `qwen2-5-embedding-classifier.ipynb` in Kaggle
2. Ensure dataset is mounted at `/kaggle/input/daisee/`
3. Run all cells to:
   - Extract embeddings from videos
   - Train MLP classifier heads
   - Save models to `/kaggle/working/models/`
4. Download trained models from Kaggle:
   ```python
   # In Kaggle notebook, run:
   !zip -r models.zip /kaggle/working/models/
   ```
5. Download `models.zip` and extract to `deployment/models/`

### 2. Server Setup

#### Install Dependencies

```bash
cd deployment/backend

# Create virtual environment
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install packages
pip install -r requirements.txt
```

#### Configure Settings

Edit `backend/config.py`:

```python
# Update paths if needed
MODELS_DIR = Path(__file__).parent.parent / "models"

# API configuration
API_HOST = "0.0.0.0"  # Listen on all interfaces
API_PORT = 8000

# CORS (update with your frontend URL)
CORS_ORIGINS = ["http://localhost:3000", "https://your-domain.com"]
```

#### Download Qwen Model (First Time)

The Qwen model will auto-download on first run (~15GB). To pre-download:

```python
from transformers import AutoProcessor, AutoModelForVision2Seq

processor = AutoProcessor.from_pretrained("Qwen/Qwen2.5-VL-7B-Instruct")
model = AutoModelForVision2Seq.from_pretrained("Qwen/Qwen2.5-VL-7B-Instruct")
```

### 3. Run Backend

```bash
cd deployment/backend
python main.py
```

Expected output:
```
INFO: Starting up - loading models...
INFO: Loading Qwen2.5-VL model...
INFO: Vision model loaded with dtype: torch.float16
INFO: Loaded classifier for Boredom (Val F1: 0.xxxx)
INFO: Loaded classifier for Engagement (Val F1: 0.xxxx)
...
INFO: Models loaded successfully
INFO: Uvicorn running on http://0.0.0.0:8000
```

### 4. Serve Frontend

#### Option A: Simple HTTP Server

```bash
cd deployment/frontend
python -m http.server 3000
```

Access at: `http://localhost:3000`

#### Option B: Production (Nginx)

```nginx
server {
    listen 80;
    server_name your-domain.com;
    
    root /path/to/deployment/frontend;
    index index.html;
    
    location / {
        try_files $uri $uri/ =404;
    }
}
```

### 5. Update Frontend Config

Edit `frontend/app.js`:

```javascript
const CONFIG = {
    API_URL: 'http://your-server-ip:8000',  // Update this
    ...
};
```

## Performance Optimization

### Memory Usage (20GB VRAM)

- Qwen2.5-VL-7B FP16: ~14GB
- Classifier heads: <50MB
- Inference buffers: ~2GB
- **Total**: ~16GB (80% utilization)

### Throughput Estimates

- Single face inference: ~200-300ms
- Batch (8 faces): ~800-1200ms
- **Effective throughput**: ~5-10 FPS per user

### Caching Benefits

With `CACHE_SIMILARITY_THRESHOLD = 0.95`:
- Cache hit rate: 60-80% (for similar faces)
- Cached inference: <10ms
- **Cache speedup**: 20-30x

### Scaling Recommendations

For multiple concurrent users:

1. **Horizontal scaling**: Deploy multiple backend instances with load balancer
2. **Redis caching**: Enable distributed cache
   ```python
   USE_REDIS = True
   REDIS_HOST = "your-redis-server"
   ```
3. **Batch processing**: Process multiple users' faces in single batch

## API Endpoints

### Health Check
```bash
curl http://localhost:8000/health
```

### Single Face Prediction
```bash
curl -X POST http://localhost:8000/predict \
  -H "Content-Type: application/json" \
  -d '{
    "image": "<base64_encoded_face>",
    "session_id": "user123"
  }'
```

Response:
```json
{
  "predictions": {
    "Boredom": {
      "category": "Boredom",
      "level": 2,
      "confidence": 0.87,
      "probabilities": [0.05, 0.08, 0.87, 0.00]
    },
    ...
  },
  "processing_time_ms": 245.3,
  "cached": false
}
```

### Batch Prediction
```bash
curl -X POST http://localhost:8000/predict/batch \
  -H "Content-Type: application/json" \
  -d '{
    "images": ["<base64_1>", "<base64_2>"],
    "session_id": "user123"
  }'
```

### Metrics
```bash
curl http://localhost:8000/metrics
```

## Troubleshooting

### Issue: CUDA Out of Memory

**Solution**: Reduce batch size in `config.py`
```python
BATCH_SIZE = 4  # Reduce from 8
```

### Issue: Slow inference (>1s)

**Solutions**:
1. Enable FP16: `USE_FP16 = True`
2. Check GPU utilization: `nvidia-smi`
3. Ensure no other processes using GPU

### Issue: Models not found

**Solution**: Verify model paths
```bash
ls deployment/models/classifiers/
# Should show: mlp_Boredom_best.pth, etc.
```

### Issue: CORS errors in browser

**Solution**: Update CORS settings in `config.py`
```python
CORS_ORIGINS = ["http://localhost:3000"]
```

## Production Deployment

### Security Recommendations

1. **API Authentication**: Add JWT tokens
2. **Rate limiting**: Prevent abuse
3. **HTTPS**: Use SSL certificates
4. **Input validation**: Sanitize all inputs

### Monitoring

1. **Prometheus metrics**: Add `/metrics` endpoint
2. **Logging**: Use structured logging (JSON)
3. **Alerting**: Monitor GPU memory, latency

### Docker Deployment

```dockerfile
FROM pytorch/pytorch:2.1.2-cuda11.8-cudnn8-runtime

WORKDIR /app
COPY deployment/backend /app
RUN pip install -r requirements.txt

CMD ["python", "main.py"]
```

Build and run:
```bash
docker build -t emotion-detection .
docker run --gpus all -p 8000:8000 emotion-detection
```

## Client-Side Integration

### Browser Requirements

- Modern browser with WebRTC support
- Camera permissions
- Stable internet connection (for API calls)

### Network Bandwidth

- Face crop (112x112 JPEG): ~3-5 KB
- Request every 2 seconds: ~2.5 KB/s
- **Very low bandwidth usage**

### Mobile Support

Frontend works on mobile browsers with camera access. Consider:
- Reduce detection interval to 3-4 seconds on mobile
- Use lower JPEG quality (0.7 instead of 0.85)

## License

See main project license.

## Support

For issues, check logs:
```bash
# Backend logs
journalctl -u emotion-api -f

# Check GPU
nvidia-smi -l 1
```
