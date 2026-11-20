# Phase 9: the price service in one container (CPU only).
# Build after running Phases 1-8 and `python -m src.serve build`, because the image copies models/.
#   docker build -t airbnb-price .
#   docker run -p 8000:8000 -v ${PWD}/data/monitoring:/app/data/monitoring airbnb-price
FROM python:3.11-slim

# LightGBM needs the OpenMP runtime
RUN apt-get update && apt-get install -y --no-install-recommends libgomp1 \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app
ENV PYTHONUNBUFFERED=1 HF_HOME=/app/.hf

# CPU-only PyTorch first (about 5x smaller than the default GPU build), then everything else
RUN pip install --no-cache-dir torch --index-url https://download.pytorch.org/whl/cpu
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Download CLIP once at build time, so the first photo upload is fast and works offline
RUN python -c "from transformers import CLIPModel, CLIPProcessor; \
m='openai/clip-vit-base-patch32'; CLIPModel.from_pretrained(m); CLIPProcessor.from_pretrained(m)"

COPY config.yaml .
COPY src/ src/
COPY web/ web/
COPY models/ models/

EXPOSE 8000
HEALTHCHECK CMD python -c "import urllib.request; urllib.request.urlopen('http://127.0.0.1:8000/health')"
CMD ["uvicorn", "src.serve:create_app", "--factory", "--host", "0.0.0.0", "--port", "8000"]