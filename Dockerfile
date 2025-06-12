# Use official lightweight Python image
FROM python:3.10-slim

# Install system dependencies for librosa & ffmpeg
RUN apt-get update && \
    apt-get install -y ffmpeg libsndfile1 && \
    rm -rf /var/lib/apt/lists/*

# Set working directory inside container
WORKDIR /app

# Copy requirements first to leverage caching
COPY requirements.txt .

# Install Python dependencies
RUN pip install --no-cache-dir -r requirements.txt

# Copy the rest of your app code and model file
COPY . .

# Expose port 8000 for uvicorn
EXPOSE 8000

# Run your FastAPI app with uvicorn
CMD ["python", "main.py"]