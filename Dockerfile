FROM python:3.11-slim
ENV PYTHONDONTWRITEBYTECODE=1 PYTHONUNBUFFERED=1
WORKDIR /app
COPY requirements.txt ./
RUN pip install --no-cache-dir -r requirements.txt
COPY turn_detector/ ./turn_detector/
COPY artifacts/turn_model.joblib ./artifacts/turn_model.joblib
RUN useradd --create-home --uid 10001 appuser
USER appuser
EXPOSE 8000
HEALTHCHECK --interval=30s --timeout=5s --start-period=10s --retries=3 CMD python -c "import urllib.request; urllib.request.urlopen('http://localhost:8000/healthz', timeout=3)" || exit 1
CMD ["uvicorn", "turn_detector.api:app", "--host", "0.0.0.0", "--port", "8000", "--workers", "1"]
