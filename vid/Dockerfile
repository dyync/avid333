FROM python:3.9-slim

# Install system dependencies for OpenCV and build tools
RUN apt-get update && apt-get install -y \
    libglib2.0-0 \
    libsm6 \
    libxext6 \
    libxrender-dev \
    wget \
    && rm -rf /var/lib/apt/lists/*

# Set pip timeout to handle large downloads
RUN pip config set global.timeout 300
RUN pip config set global.retries 5

# Copy requirements first to leverage Docker cache
COPY requirements.txt .

# Install Python dependencies with retry logic
RUN pip install --no-cache-dir -r requirements.txt || \
    pip install --no-cache-dir -r requirements.txt || \
    pip install --no-cache-dir -r requirements.txt

# Copy your application code
COPY . .

EXPOSE 1375

CMD ["python", "app.py"]