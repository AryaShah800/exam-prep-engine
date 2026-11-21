# Hugging Face Space (Docker SDK) for the offline demo. The Space holds only this file and README.md;
# the app is cloned from GitHub at build time, so the Space always runs a known commit of main.
FROM python:3.12-slim
RUN apt-get update && apt-get install -y --no-install-recommends git && rm -rf /var/lib/apt/lists/*
RUN useradd --create-home --uid 1000 user
USER user
WORKDIR /home/user/app
ARG REPO=https://github.com/AryaShah800/exam-prep-engine.git
ARG REF=main
RUN git clone --depth 1 --branch "$REF" "$REPO" . && git log -1 --format='%H %cs' > COMMIT
RUN pip install --no-cache-dir --user -r requirements-web.txt
ENV PATH=/home/user/.local/bin:$PATH PYTHONUNBUFFERED=1 \
    EMBED_PROVIDER=local HF_HOME=/home/user/.cache/hf \
    WEB_ALLOWED_HOSTS=arya2389-exam-prep-engine.hf.space,localhost,127.0.0.1
# Download the local embedding model and embed the sample bank at build time (deploy/hf-space/warmup.py).
RUN WEB_ALLOWED_HOSTS=testserver PYTHONPATH=. python deploy/hf-space/warmup.py
EXPOSE 7860
CMD ["python", "-m", "exam_prep", "demo", "--host", "0.0.0.0", "--port", "7860"]
