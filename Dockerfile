FROM python:3.10-slim

WORKDIR /app

# Install ALL system libs OpenCV needs
RUN apt-get update && apt-get install -y \
    libglib2.0-0 \
    libgomp1 \
    libxcb1 \
    libxext6 \
    libsm6 \
    libxrender1 \
    gcc \
    g++ \
    && rm -rf /var/lib/apt/lists/*

COPY requirements.txt .

# Upgrade pip first
RUN pip install --no-cache-dir --upgrade pip setuptools wheel

# Install numpy FIRST at the version TF 2.13 needs
RUN pip install --no-cache-dir "numpy==1.24.3"

# Install TensorFlow (must come before ultralytics to lock numpy)
RUN pip install --no-cache-dir "tensorflow==2.13.0"

# Install OpenCV headless (no GUI libs needed)
RUN pip install --no-cache-dir "opencv-python-headless==4.9.0.80"

# Install the rest
RUN pip install --no-cache-dir \
    "fastapi==0.111.0" \
    "uvicorn[standard]==0.29.0" \
    "python-multipart==0.0.9" \
    "pydantic==2.7.1" \
    "ultralytics==8.2.0"

COPY . .

EXPOSE 8000

CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "8000"]
