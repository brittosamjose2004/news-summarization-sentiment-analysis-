FROM python:3.9-slim

WORKDIR /app

# Install dependencies
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Install additional dependencies needed for NLP tasks and TTS
RUN apt-get update && apt-get install -y \
    build-essential \
    curl \
    software-properties-common \
    git \
    ffmpeg \
    espeak \
    libespeak-dev \
    alsa-utils \
    python3-pyaudio \
    libasound2-dev \
    && rm -rf /var/lib/apt/lists/*

# Copy app files
COPY . .

# Create directory for audio files
RUN mkdir -p audio_files

# Set environment variables
ENV PYTHONDONTWRITEBYTECODE=1
ENV PYTHONUNBUFFERED=1

# Download NLTK data
RUN python -c "import nltk; nltk.download('punkt'); nltk.download('stopwords')"

# Expose ports
EXPOSE 8000
EXPOSE 8501

# Create a shell script to run both services
RUN echo '#!/bin/bash\n\
uvicorn api:app --host 0.0.0.0 --port 8000 &\n\
streamlit run app.py --server.port 8501 --server.address 0.0.0.0\n'\
> /app/start.sh

RUN chmod +x /app/start.sh

# Start the application
CMD ["/app/start.sh"] 