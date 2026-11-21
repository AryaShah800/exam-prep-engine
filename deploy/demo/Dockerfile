# Hosted offline demo (Render free web service, see render.yaml). Synthetic sample bank only: no papers, no keys.
# Render's free instance has 512 MB RAM; bge-base peaks at about 650 MB in the warm-up, bge-small at about 300 MB,
# so the hosted demo uses bge-small. The retrieval evaluation in the README was run with bge-base.
FROM python:3.12-slim
ENV PYTHONDONTWRITEBYTECODE=1 PYTHONUNBUFFERED=1 \
    EMBED_PROVIDER=local LOCAL_EMBED_MODEL=BAAI/bge-small-en-v1.5
WORKDIR /app
COPY requirements.txt requirements-web.txt ./
RUN pip install --no-cache-dir -r requirements-web.txt
COPY exam_prep ./exam_prep
COPY demo ./demo
COPY deploy/demo/warmup.py ./deploy/demo/warmup.py
RUN useradd --create-home appuser && chown -R appuser:appuser /app
USER appuser
# Download the embedding model and embed the sample bank at build time, so the first search is not a cold download.
RUN WEB_ALLOWED_HOSTS=testserver PYTHONPATH=. python deploy/demo/warmup.py
# PORT is set by the platform; WEB_ALLOWED_HOSTS must list the public host name (render.yaml).
CMD ["python", "-m", "exam_prep", "demo", "--host", "0.0.0.0"]
