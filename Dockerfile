FROM python:3.12-slim
WORKDIR /app
ENV PYTHONDONTWRITEBYTECODE=1 PYTHONUNBUFFERED=1
COPY requirements.txt requirements-web.txt ./
RUN pip install --no-cache-dir -r requirements-web.txt
COPY requirements-agent.txt ./
ARG INCLUDE_AGENT=1
RUN if [ "$INCLUDE_AGENT" = "1" ]; then pip install --no-cache-dir -r requirements-agent.txt; fi
COPY exam_prep ./exam_prep
RUN useradd --create-home appuser && mkdir -p data && chown -R appuser:appuser /app
USER appuser
EXPOSE 7860
CMD ["uvicorn", "exam_prep.web:app", "--host", "0.0.0.0", "--port", "7860"]
