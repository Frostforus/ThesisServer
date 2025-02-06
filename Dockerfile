# Use NVIDIA CUDA base image with Python
FROM nvidia/cuda:12.6.1-base-ubuntu24.04

# Install system dependencies as root
RUN apt-get update && apt-get install -y \
    python3.10 \
    python3-pip \
    python3.10-venv \
    ffmpeg \
    && rm -rf /var/lib/apt/lists/*

# Create a non-root user and set up environment
RUN useradd -m -u 1001 appuser
USER appuser
WORKDIR /home/appuser/app

# Create virtual environment
RUN python3 -m venv /home/appuser/app/venv
ENV PATH="/home/appuser/app/venv/bin:$PATH"

# Install PyTorch with CUDA 12.6 support first
RUN pip install --no-cache-dir torch --index-url https://download.pytorch.org/whl/cu126

# Install other requirements
COPY --chown=appuser:appuser requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy application files with proper permissions
COPY --chown=appuser:appuser . .

# Expose FastAPI port
EXPOSE 8000

# Command to run the application
CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "8000"]