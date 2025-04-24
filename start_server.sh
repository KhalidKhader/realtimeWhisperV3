#!/bin/bash

# Activate the virtual environment
source whisper_env/bin/activate

# Set environment variables for better debugging
export PYTHONUNBUFFERED=1

# Start the FastAPI server with increased verbosity
echo "Starting transcription server..."
# Change to server directory
cd server
# Run the server
python -m uvicorn app:app --host 0.0.0.0 --port 8000 --log-level debug 