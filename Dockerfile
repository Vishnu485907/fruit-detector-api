FROM python:3.10-slim

WORKDIR /app

# ALL system libs OpenCV needs (including libGL)
RUN apt-get update && apt-get install -y \
    libglib2.0-0 \
    libgomp1 \
    libxcb1 \
    libxext6 \
    libsm6 \
    libxrender1 \
    libgl1 \
    gcc \
    g++ \
    && rm -rf /var/lib/apt/lists/*

RUN pip install --no-cache-dir --upgrade pip setuptools wheel

# Step 1: pin numpy + tensorflow FIRST
RUN pip install --no-cache-dir \
    "numpy==1.24.3" \
    "typing-extensions==4.5.0"

RUN pip install --no-cache-dir "tensorflow==2.13.0"

# Step 2: install ultralytics (this will upgrade numpy — we fix that next)
RUN pip install --no-cache-dir \
    "ultralytics==8.2.0" \
    "opencv-python-headless==4.9.0.80"

# Step 3: FORCE numpy back to TF-compatible version AFTER ultralytics
RUN pip install --no-cache-dir --force-reinstall \
    "numpy==1.24.3" \
    "typing-extensions==4.5.0"

# Step 4: FastAPI stack
RUN pip install --no-cache-dir \
    "fastapi==0.111.0" \
    "uvicorn[standard]==0.29.0" \
    "python-multipart==0.0.9" \
    "pydantic==2.7.1"

COPY . .

EXPOSE 8000

CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "8000"]
