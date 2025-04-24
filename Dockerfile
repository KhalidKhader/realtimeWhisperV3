FROM nvidia/cuda:11.8.0-cudnn8-runtime-ubuntu22.04

# Prevent interactive prompts
ENV DEBIAN_FRONTEND=noninteractive

# Install system dependencies
RUN apt-get update && \
    apt-get install -y --no-install-recommends \
        build-essential \
        python3.10 \
        python3.10-dev \
        python3-pip \
        python3-setuptools \
        libsndfile1 \
        portaudio19-dev \
        ffmpeg \
        sox \
        git \
        cmake \
        curl \
        ca-certificates \
        gnupg && \
    apt-get clean && \
    rm -rf /var/lib/apt/lists/*

# Install Node.js and npm
RUN mkdir -p /etc/apt/keyrings && \
    curl -fsSL https://deb.nodesource.com/gpgkey/nodesource-repo.gpg.key | gpg --dearmor -o /etc/apt/keyrings/nodesource.gpg && \
    echo "deb [signed-by=/etc/apt/keyrings/nodesource.gpg] https://deb.nodesource.com/node_18.x nodistro main" | tee /etc/apt/sources.list.d/nodesource.list && \
    apt-get update && \
    apt-get install -y nodejs && \
    apt-get clean && \
    rm -rf /var/lib/apt/lists/*

# Set up Python
RUN ln -sf /usr/bin/python3.10 /usr/bin/python && \
    ln -sf /usr/bin/pip3 /usr/bin/pip

# Set working directory
WORKDIR /app

# Install basic Python packages
RUN pip install --no-cache-dir --upgrade pip && \
    pip install --no-cache-dir setuptools wheel

# Install core dependencies first
RUN pip install --no-cache-dir numpy torch torchaudio typing_extensions psutil

# Copy requirements and install dependencies
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt || \
    (grep -v -E "sox==|OpenCC==|PyAudio==" requirements.txt > requirements_filtered.txt && \
     pip install --no-cache-dir -r requirements_filtered.txt && \
     pip install --no-cache-dir sox PyAudio || echo "Some optional packages failed to install")

# Build the React client
COPY client/package.json client/package-lock.json ./client/
WORKDIR /app/client
RUN npm install
COPY client/ ./
RUN npm run build

# Copy server files (including app.py and real_time_diarization.py)
WORKDIR /app
COPY server/ ./server/

# Create data directory
RUN mkdir -p /app/data

# Serve React app with a simple HTTP server
RUN pip install --no-cache-dir fastapi uvicorn aiofiles

# Create a startup script to serve both the API and static files
RUN echo '#!/bin/bash\n\
cd /app/server\n\
python -m uvicorn app:app --host 0.0.0.0 --port 80\n\
' > /app/start.sh && \
chmod +x /app/start.sh

# Expose port for the API
EXPOSE 80

# Set up environment variables
ENV PYTHONUNBUFFERED=1
ENV NVIDIA_VISIBLE_DEVICES=all
ENV NVIDIA_DRIVER_CAPABILITIES=compute,utility

# Start the FastAPI application
CMD ["/app/start.sh"] 