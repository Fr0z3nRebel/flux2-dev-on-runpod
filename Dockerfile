# RunPod Serverless: FLUX.2 [dev] image generation worker
# Build: docker build --platform linux/amd64 -t your-dockerhub-user/flux2-dev-runpod:latest .

FROM pytorch/pytorch:2.5.1-cuda12.4-cudnn9-runtime

ENV PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1 \
    MODEL_MODE=bnb4_remote

WORKDIR /app

COPY requirements.txt .
RUN pip install -r requirements.txt

COPY rp_handler.py .

# RunPod GitHub import expects this filename (same pattern as runpod-workers/worker-basic)
CMD ["python", "-u", "rp_handler.py"]
