#!/bin/bash

# Quick start script for emotion detection deployment

echo "==================================="
echo "Emotion Detection - Quick Start"
echo "==================================="

# Check if virtual environment exists
if [ ! -d "backend/venv" ]; then
    echo "Creating virtual environment..."
    cd backend
    python3 -m venv venv
    source venv/bin/activate
    pip install -r requirements.txt
    cd ..
else
    echo "Activating virtual environment..."
    source backend/venv/bin/activate
fi

# Check if models exist
if [ ! -d "models/classifiers" ]; then
    echo ""
    echo "WARNING: Classifier models not found!"
    echo "Please:"
    echo "1. Train models in Kaggle notebook"
    echo "2. Run export_models.py in Kaggle"
    echo "3. Download and extract to deployment/models/"
    echo ""
    read -p "Continue anyway? (y/n) " -n 1 -r
    echo
    if [[ ! $REPLY =~ ^[Yy]$ ]]; then
        exit 1
    fi
fi

# Check GPU
echo ""
echo "Checking GPU..."
if command -v nvidia-smi &> /dev/null; then
    nvidia-smi --query-gpu=name,memory.total,memory.free --format=csv,noheader
else
    echo "WARNING: nvidia-smi not found. GPU may not be available."
fi

# Start backend
echo ""
echo "Starting backend server..."
echo "API will be available at: http://localhost:8000"
echo "API docs: http://localhost:8000/docs"
echo ""

cd backend
python main.py &
BACKEND_PID=$!

# Wait for backend to start
echo "Waiting for backend to start..."
sleep 5

# Check if backend is running
if curl -s http://localhost:8000/health > /dev/null; then
    echo "✓ Backend is running"
else
    echo "✗ Backend failed to start"
    kill $BACKEND_PID
    exit 1
fi

# Start frontend
echo ""
echo "Starting frontend server..."
echo "Frontend will be available at: http://localhost:3000"
echo ""

cd ../frontend
python3 -m http.server 3000 &
FRONTEND_PID=$!

# Wait a bit
sleep 2

echo ""
echo "==================================="
echo "Servers started successfully!"
echo "==================================="
echo "Backend:  http://localhost:8000"
echo "Frontend: http://localhost:3000"
echo "API Docs: http://localhost:8000/docs"
echo ""
echo "Press Ctrl+C to stop all servers"
echo ""

# Cleanup on exit
trap "echo 'Stopping servers...'; kill $BACKEND_PID $FRONTEND_PID 2>/dev/null; exit" INT TERM

# Keep script running
wait
