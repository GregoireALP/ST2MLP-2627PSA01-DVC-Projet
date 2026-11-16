# syntax=docker/dockerfile:1
FROM python:3.12-slim

WORKDIR /app
ENV PYTHONDONTWRITEBYTECODE=1 PYTHONUNBUFFERED=1 MODEL_DIR=/app/model

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY app/ ./app/

# Fetch the latest trained model from DagsHub through MLflow.
# Credentials are passed as BuildKit secrets so they are NOT stored in the image.
RUN --mount=type=secret,id=dagshub_user \
    --mount=type=secret,id=dagshub_token \
    MLFLOW_TRACKING_USERNAME="$(cat /run/secrets/dagshub_user)" \
    MLFLOW_TRACKING_PASSWORD="$(cat /run/secrets/dagshub_token)" \
    python app/download_model.py

EXPOSE 8000
CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8000"]
