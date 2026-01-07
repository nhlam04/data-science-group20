// Configuration
const CONFIG = {
    API_URL: 'http://localhost:8000',  // Update with your server URL
    DETECTION_INTERVAL: 2000,  // Send frame every 2 seconds
    FACE_CROP_SIZE: 112,  // Face crop size (pixels)
    FACE_PADDING: 0.2,  // 20% padding around face
    MAX_RETRIES: 3,
    CATEGORIES: ['Boredom', 'Engagement', 'Confusion', 'Frustration'],
    EMOTION_ICONS: {
        'Boredom': '😴',
        'Engagement': '🎯',
        'Confusion': '😕',
        'Frustration': '😤'
    }
};

// Global state
let webcamStream = null;
let faceDetection = null;
let detectionInterval = null;
let sessionId = null;
let frameCount = 0;
let lastFrameTime = Date.now();

// DOM elements
const webcamElement = document.getElementById('webcam');
const canvasElement = document.getElementById('canvas');
const canvasCtx = canvasElement.getContext('2d');
const startBtn = document.getElementById('startBtn');
const stopBtn = document.getElementById('stopBtn');
const statusElement = document.getElementById('status');
const resultsElement = document.getElementById('results');

// Initialize MediaPipe Face Detection
async function initializeFaceDetection() {
    faceDetection = new FaceDetection({
        locateFile: (file) => {
            return `https://cdn.jsdelivr.net/npm/@mediapipe/face_detection/${file}`;
        }
    });

    faceDetection.setOptions({
        model: 'short',  // short-range model (better for webcam)
        minDetectionConfidence: 0.5
    });

    await faceDetection.initialize();
    console.log('Face detection initialized');
}

// Start webcam
async function startWebcam() {
    try {
        webcamStream = await navigator.mediaDevices.getUserMedia({
            video: {
                width: { ideal: 640 },
                height: { ideal: 480 },
                facingMode: 'user'
            }
        });
        
        webcamElement.srcObject = webcamStream;
        
        // Set canvas size to match video
        webcamElement.addEventListener('loadedmetadata', () => {
            canvasElement.width = webcamElement.videoWidth;
            canvasElement.height = webcamElement.videoHeight;
        });
        
        return true;
    } catch (error) {
        console.error('Error accessing webcam:', error);
        updateStatus('Error accessing webcam', 'error');
        return false;
    }
}

// Stop webcam
function stopWebcam() {
    if (webcamStream) {
        webcamStream.getTracks().forEach(track => track.stop());
        webcamStream = null;
    }
    
    if (detectionInterval) {
        clearInterval(detectionInterval);
        detectionInterval = null;
    }
}

// Crop face from frame
function cropFace(imageData, detection) {
    const canvas = document.createElement('canvas');
    const ctx = canvas.getContext('2d');
    
    // Get bounding box
    const bbox = detection.boundingBox;
    const x = bbox.xCenter - bbox.width / 2;
    const y = bbox.yCenter - bbox.height / 2;
    const w = bbox.width;
    const h = bbox.height;
    
    // Add padding
    const padding = CONFIG.FACE_PADDING;
    const paddedX = Math.max(0, x - w * padding);
    const paddedY = Math.max(0, y - h * padding);
    const paddedW = Math.min(imageData.width - paddedX, w * (1 + 2 * padding));
    const paddedH = Math.min(imageData.height - paddedY, h * (1 + 2 * padding));
    
    // Set canvas size to face crop size
    canvas.width = CONFIG.FACE_CROP_SIZE;
    canvas.height = CONFIG.FACE_CROP_SIZE;
    
    // Draw and resize face
    ctx.drawImage(
        imageData,
        paddedX, paddedY, paddedW, paddedH,
        0, 0, CONFIG.FACE_CROP_SIZE, CONFIG.FACE_CROP_SIZE
    );
    
    return canvas;
}

// Convert canvas to base64
function canvasToBase64(canvas) {
    return canvas.toDataURL('image/jpeg', 0.85).split(',')[1];
}

// Detect faces and send to API
async function detectAndAnalyze() {
    if (!webcamElement.videoWidth) return;
    
    // Create temporary canvas for face detection
    const tempCanvas = document.createElement('canvas');
    tempCanvas.width = webcamElement.videoWidth;
    tempCanvas.height = webcamElement.videoHeight;
    const tempCtx = tempCanvas.getContext('2d');
    tempCtx.drawImage(webcamElement, 0, 0);
    
    // Detect faces with MediaPipe
    const results = await faceDetection.send({ image: webcamElement });
    
    // Clear canvas
    canvasCtx.clearRect(0, 0, canvasElement.width, canvasElement.height);
    
    if (results.detections && results.detections.length > 0) {
        const detection = results.detections[0];  // Use first detected face
        
        // Draw bounding box
        drawFaceBoundingBox(detection);
        
        // Crop face
        const faceCrop = cropFace(tempCanvas, detection);
        const faceBase64 = canvasToBase64(faceCrop);
        
        // Send to API
        await analyzeFace(faceBase64);
        
        // Update stats
        document.getElementById('facesDetected').textContent = results.detections.length;
    } else {
        document.getElementById('facesDetected').textContent = '0';
        updateStatus('No face detected', 'idle');
    }
    
    // Update FPS
    frameCount++;
    const now = Date.now();
    const elapsed = (now - lastFrameTime) / 1000;
    if (elapsed >= 1) {
        const fps = frameCount / elapsed;
        document.getElementById('fps').textContent = fps.toFixed(1);
        frameCount = 0;
        lastFrameTime = now;
    }
}

// Draw face bounding box on canvas
function drawFaceBoundingBox(detection) {
    const bbox = detection.boundingBox;
    const x = bbox.xCenter - bbox.width / 2;
    const y = bbox.yCenter - bbox.height / 2;
    
    // Convert normalized coords to pixel coords
    const pixelX = x * canvasElement.width;
    const pixelY = y * canvasElement.height;
    const pixelW = bbox.width * canvasElement.width;
    const pixelH = bbox.height * canvasElement.height;
    
    canvasCtx.strokeStyle = '#667eea';
    canvasCtx.lineWidth = 3;
    canvasCtx.strokeRect(pixelX, pixelY, pixelW, pixelH);
}

// Send face to API for emotion analysis
async function analyzeFace(faceBase64) {
    try {
        updateStatus('Analyzing...', 'running');
        
        const response = await fetch(`${CONFIG.API_URL}/predict`, {
            method: 'POST',
            headers: {
                'Content-Type': 'application/json'
            },
            body: JSON.stringify({
                image: faceBase64,
                session_id: sessionId,
                timestamp: Date.now()
            })
        });
        
        if (!response.ok) {
            throw new Error(`API error: ${response.status}`);
        }
        
        const data = await response.json();
        
        // Update UI with results
        displayResults(data);
        updateStats(data);
        updateStatus('Detection active', 'running');
        
    } catch (error) {
        console.error('Error analyzing face:', error);
        updateStatus(`Error: ${error.message}`, 'error');
    }
}

// Display emotion results
function displayResults(data) {
    const predictions = data.predictions;
    
    resultsElement.innerHTML = '';
    
    for (const category of CONFIG.CATEGORIES) {
        const pred = predictions[category];
        if (!pred) continue;
        
        const card = document.createElement('div');
        card.className = 'emotion-card';
        
        const icon = CONFIG.EMOTION_ICONS[category] || '😐';
        
        card.innerHTML = `
            <h3>
                <span class="emotion-icon">${icon}</span>
                ${category}
            </h3>
            <div class="level-display">Level ${pred.level}</div>
            <div class="confidence">Confidence: ${(pred.confidence * 100).toFixed(1)}%</div>
            <div class="probability-bars">
                ${pred.probabilities.map((prob, idx) => `
                    <div class="prob-bar">
                        <label>Level ${idx}</label>
                        <div class="prob-bar-fill">
                            <div class="prob-bar-value" style="width: ${prob * 100}%">
                                <span class="prob-bar-text">${(prob * 100).toFixed(0)}%</span>
                            </div>
                        </div>
                    </div>
                `).join('')}
            </div>
        `;
        
        resultsElement.appendChild(card);
    }
}

// Update statistics
function updateStats(data) {
    document.getElementById('processingTime').textContent = 
        `${data.processing_time_ms.toFixed(0)}ms`;
    document.getElementById('cached').textContent = 
        data.cached ? 'Yes' : 'No';
}

// Update status message
function updateStatus(message, type) {
    statusElement.textContent = message;
    statusElement.className = `status ${type}`;
}

// Start detection
async function startDetection() {
    // Generate session ID
    sessionId = `session_${Date.now()}_${Math.random().toString(36).substr(2, 9)}`;
    
    // Initialize face detection
    if (!faceDetection) {
        updateStatus('Initializing face detection...', 'running');
        await initializeFaceDetection();
    }
    
    // Start webcam
    updateStatus('Starting webcam...', 'running');
    const success = await startWebcam();
    
    if (!success) return;
    
    // Wait for video to be ready
    await new Promise(resolve => {
        webcamElement.addEventListener('loadeddata', resolve, { once: true });
    });
    
    // Start detection loop
    updateStatus('Detection active', 'running');
    detectionInterval = setInterval(detectAndAnalyze, CONFIG.DETECTION_INTERVAL);
    
    // UI updates
    startBtn.disabled = true;
    stopBtn.disabled = false;
}

// Stop detection
function stopDetection() {
    stopWebcam();
    
    updateStatus('Stopped', 'idle');
    
    // Clear canvas
    canvasCtx.clearRect(0, 0, canvasElement.width, canvasElement.height);
    
    // UI updates
    startBtn.disabled = false;
    stopBtn.disabled = true;
}

// Event listeners
startBtn.addEventListener('click', startDetection);
stopBtn.addEventListener('click', stopDetection);

// Check API health on load
async function checkAPIHealth() {
    try {
        const response = await fetch(`${CONFIG.API_URL}/health`);
        const data = await response.json();
        console.log('API Health:', data);
        
        if (!data.model_loaded) {
            updateStatus('API models loading...', 'idle');
        }
    } catch (error) {
        console.error('API health check failed:', error);
        updateStatus('Cannot connect to API', 'error');
    }
}

// Initialize on page load
window.addEventListener('load', () => {
    checkAPIHealth();
});
