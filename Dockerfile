FROM python:3.11-slim

WORKDIR /app

# CPU-only PyTorch keeps the image small; the other dependencies come from requirements.txt.
COPY requirements.txt .
RUN pip install --no-cache-dir torch --index-url https://download.pytorch.org/whl/cpu \
    && pip install --no-cache-dir -r requirements.txt

COPY . .
ENV PYTHONPATH=/app/src

# Mount a host folder at /app/data and /app/results to keep downloads and outputs:
#   docker run --rm -v "$PWD/data:/app/data" -v "$PWD/results:/app/results" gexpr
CMD ["python", "scripts/run_pipeline.py"]
