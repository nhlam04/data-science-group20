"""
Optimized inference engine for emotion detection
Uses Qwen2.5-VL for embedding extraction + trained classifier heads
"""
import torch
import torch.nn as nn
import numpy as np
from PIL import Image
import io
import base64
import time
from pathlib import Path
from typing import Dict, List, Tuple, Optional
import logging
from collections import OrderedDict
from transformers import AutoProcessor, AutoModelForVision2Seq

from config import *

logger = logging.getLogger(__name__)


class EmbeddingCache:
    """LRU cache for embeddings with similarity checking"""
    
    def __init__(self, max_size: int = MAX_CACHE_SIZE, similarity_threshold: float = CACHE_SIMILARITY_THRESHOLD):
        self.cache = OrderedDict()
        self.max_size = max_size
        self.similarity_threshold = similarity_threshold
        self.hits = 0
        self.misses = 0
    
    def _cosine_similarity(self, a: np.ndarray, b: np.ndarray) -> float:
        """Compute cosine similarity between two embeddings"""
        return np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b))
    
    def get(self, image_hash: str, embedding: Optional[np.ndarray] = None) -> Optional[np.ndarray]:
        """Get cached embedding if exists and similar"""
        # Direct hash lookup
        if image_hash in self.cache:
            self.hits += 1
            self.cache.move_to_end(image_hash)
            return self.cache[image_hash]
        
        # Similarity search (only if embedding provided)
        if embedding is not None:
            for key, cached_emb in list(self.cache.items())[:10]:  # Check last 10
                if self._cosine_similarity(embedding, cached_emb) > self.similarity_threshold:
                    self.hits += 1
                    return cached_emb
        
        self.misses += 1
        return None
    
    def put(self, image_hash: str, embedding: np.ndarray):
        """Add embedding to cache"""
        if image_hash in self.cache:
            self.cache.move_to_end(image_hash)
        else:
            self.cache[image_hash] = embedding
            if len(self.cache) > self.max_size:
                self.cache.popitem(last=False)
    
    def get_stats(self) -> Dict:
        """Get cache statistics"""
        total = self.hits + self.misses
        hit_rate = self.hits / total if total > 0 else 0
        return {
            "size": len(self.cache),
            "hits": self.hits,
            "misses": self.misses,
            "hit_rate": hit_rate
        }


class MLPClassifier(nn.Module):
    """MLP classifier head (same as training notebook)"""
    
    def __init__(self, input_dim: int, hidden_dim: int = 512, num_classes: int = 4, dropout: float = 0.3):
        super(MLPClassifier, self).__init__()
        
        self.pooling = nn.AdaptiveAvgPool1d(1)
        
        self.classifier = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim // 2, num_classes)
        )
    
    def forward(self, x):
        # x shape: (batch, num_frames, embedding_dim) or (batch, embedding_dim)
        if x.dim() == 3:
            x = x.transpose(1, 2)
            x = self.pooling(x).squeeze(-1)
        logits = self.classifier(x)
        return logits


class EmotionInferenceEngine:
    """Main inference engine for emotion detection"""
    
    def __init__(self):
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.processor = None
        self.vision_model = None
        self.classifiers = {}
        self.embedding_cache = EmbeddingCache() if ENABLE_EMBEDDING_CACHE else None
        self.model_loaded = False
        self.start_time = time.time()
        
        logger.info(f"Inference engine initialized on device: {self.device}")
    
    def load_models(self):
        """Load Qwen vision model and classifier heads"""
        logger.info("Loading Qwen2.5-VL model...")
        
        # Load processor
        self.processor = AutoProcessor.from_pretrained(MODEL_NAME)
        
        # Load vision model with optimization
        dtype = torch.float16 if USE_FP16 and torch.cuda.is_available() else torch.float32
        
        self.vision_model = AutoModelForVision2Seq.from_pretrained(
            MODEL_NAME,
            torch_dtype=dtype,
            device_map="auto" if torch.cuda.is_available() else None,
            # attn_implementation="flash_attention_2" if USE_FLASH_ATTENTION else "eager",
        )
        self.vision_model.eval()
        
        logger.info(f"Vision model loaded with dtype: {dtype}")
        
        # Load classifier heads
        self._load_classifiers()
        
        self.model_loaded = True
        logger.info("All models loaded successfully")
    
    def _load_classifiers(self):
        """Load trained classifier heads for each emotion category"""
        logger.info("Loading classifier heads...")
        
        # Determine embedding dimension from a dummy forward pass
        dummy_embedding_dim = 3584  # Qwen2.5-VL-7B hidden size
        
        for category in CATEGORIES:
            category_clean = category.strip()
            checkpoint_path = CLASSIFIER_CHECKPOINTS_DIR / f"mlp_{category_clean}_best.pth"
            
            if not checkpoint_path.exists():
                logger.warning(f"Checkpoint not found for {category_clean}: {checkpoint_path}")
                continue
            
            # Create classifier
            classifier = MLPClassifier(input_dim=dummy_embedding_dim, num_classes=NUM_CLASSES)
            
            # Load weights
            checkpoint = torch.load(checkpoint_path, map_location=self.device)
            classifier.load_state_dict(checkpoint['model_state_dict'])
            classifier.to(self.device)
            classifier.eval()
            
            self.classifiers[category_clean] = classifier
            logger.info(f"Loaded classifier for {category_clean} (Val F1: {checkpoint.get('val_f1', 'N/A')})")
        
        logger.info(f"Loaded {len(self.classifiers)} classifiers")
    
    def _preprocess_image(self, image_b64: str) -> Image.Image:
        """Decode and preprocess base64 image"""
        # Decode base64
        image_bytes = base64.b64decode(image_b64)
        image = Image.open(io.BytesIO(image_bytes)).convert('RGB')
        
        # Resize if needed (face crops should already be small)
        if max(image.size) > MAX_IMAGE_SIZE:
            image.thumbnail((MAX_IMAGE_SIZE, MAX_IMAGE_SIZE), Image.Resampling.LANCZOS)
        
        return image
    
    def _extract_embeddings(self, images: List[Image.Image]) -> np.ndarray:
        """Extract embeddings from batch of images using Qwen model"""
        embeddings_list = []
        
        with torch.no_grad():
            for image in images:
                # Create message structure for single image
                messages = [
                    {
                        "role": "user",
                        "content": [
                            {"type": "image", "image": image},
                            {"type": "text", "text": "Analyze this face."}
                        ]
                    }
                ]
                
                # Process image
                inputs = self.processor.apply_chat_template(
                    messages,
                    add_generation_prompt=True,
                    tokenize=True,
                    return_dict=True,
                    return_tensors="pt",
                )
                
                # Move to device
                inputs = {k: v.to(self.device) for k, v in inputs.items()}
                
                # Forward pass
                outputs = self.vision_model(**inputs, output_hidden_states=True)
                
                # Extract hidden states
                hidden_states = outputs.hidden_states[-1]  # (1, seq_len, hidden_dim)
                
                # Pool over sequence dimension
                embedding = hidden_states.mean(dim=1).squeeze(0)  # (hidden_dim,)
                
                embeddings_list.append(embedding.cpu().numpy())
        
        return np.array(embeddings_list)
    
    def _predict_emotions(self, embeddings: np.ndarray) -> Dict[str, Dict]:
        """Run classifier heads on embeddings"""
        results = {}
        
        # Convert to tensor
        embeddings_tensor = torch.FloatTensor(embeddings).to(self.device)
        
        with torch.no_grad():
            for category, classifier in self.classifiers.items():
                # Forward pass
                logits = classifier(embeddings_tensor)
                probabilities = torch.softmax(logits, dim=-1)
                
                # Get predictions
                predicted_levels = torch.argmax(probabilities, dim=-1)
                confidences = torch.max(probabilities, dim=-1)[0]
                
                # Store results for each sample in batch
                for i in range(len(embeddings)):
                    if i not in results:
                        results[i] = {}
                    
                    results[i][category] = {
                        "level": int(predicted_levels[i].cpu().item()),
                        "confidence": float(confidences[i].cpu().item()),
                        "probabilities": probabilities[i].cpu().numpy().tolist()
                    }
        
        return results
    
    def predict(self, image_b64: str, session_id: Optional[str] = None) -> Tuple[Dict, bool, float]:
        """
        Predict emotions from single face image
        
        Returns:
            (predictions_dict, was_cached, processing_time_ms)
        """
        start_time = time.time()
        cached = False
        
        # Preprocess image
        image = self._preprocess_image(image_b64)
        
        # Generate cache key
        image_hash = f"{session_id}_{hash(image_b64[:100])}" if session_id else hash(image_b64[:100])
        
        # Check cache
        cached_embedding = None
        if self.embedding_cache:
            cached_embedding = self.embedding_cache.get(image_hash)
        
        if cached_embedding is not None:
            embeddings = np.array([cached_embedding])
            cached = True
        else:
            # Extract embeddings
            embeddings = self._extract_embeddings([image])
            
            # Cache embedding
            if self.embedding_cache:
                self.embedding_cache.put(image_hash, embeddings[0])
        
        # Predict emotions
        predictions = self._predict_emotions(embeddings)
        
        processing_time = (time.time() - start_time) * 1000  # Convert to ms
        
        return predictions[0], cached, processing_time
    
    def predict_batch(self, images_b64: List[str], session_id: Optional[str] = None) -> Tuple[List[Dict], List[bool], float]:
        """
        Predict emotions from batch of face images
        
        Returns:
            (list_of_predictions, list_of_cached_flags, total_processing_time_ms)
        """
        start_time = time.time()
        
        # Preprocess images
        images = [self._preprocess_image(img_b64) for img_b64 in images_b64]
        
        # Extract embeddings (with caching)
        embeddings_list = []
        cached_flags = []
        
        for i, (image, img_b64) in enumerate(zip(images, images_b64)):
            image_hash = f"{session_id}_{i}_{hash(img_b64[:100])}" if session_id else f"{i}_{hash(img_b64[:100])}"
            
            cached_embedding = None
            if self.embedding_cache:
                cached_embedding = self.embedding_cache.get(image_hash)
            
            if cached_embedding is not None:
                embeddings_list.append(cached_embedding)
                cached_flags.append(True)
            else:
                # Extract for single image
                emb = self._extract_embeddings([image])[0]
                embeddings_list.append(emb)
                cached_flags.append(False)
                
                if self.embedding_cache:
                    self.embedding_cache.put(image_hash, emb)
        
        embeddings = np.array(embeddings_list)
        
        # Predict emotions
        predictions = self._predict_emotions(embeddings)
        
        processing_time = (time.time() - start_time) * 1000
        
        # Convert to list format
        predictions_list = [predictions[i] for i in range(len(images))]
        
        return predictions_list, cached_flags, processing_time
    
    def get_cache_stats(self) -> Dict:
        """Get cache statistics"""
        if self.embedding_cache:
            return self.embedding_cache.get_stats()
        return {"enabled": False}
    
    def get_uptime(self) -> float:
        """Get server uptime in seconds"""
        return time.time() - self.start_time


# Global inference engine instance
inference_engine = None


def get_inference_engine() -> EmotionInferenceEngine:
    """Get or create global inference engine"""
    global inference_engine
    if inference_engine is None:
        inference_engine = EmotionInferenceEngine()
    return inference_engine
