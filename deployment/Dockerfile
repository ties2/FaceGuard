FROM python:3.12-slim
LABEL authors="Nirvana Fanaelahi"

ENV PYTHONUNBUFFERED=1 APP_HOME=/app
WORKDIR $APP_HOME

RUN apt-get update && apt-get install -y --no-install-recommends curl && rm -rf /var/lib/apt/lists/*
COPY requirements.txt .
# serving image: no PyTorch — the CNN is served through ONNX Runtime
RUN pip install --no-cache-dir -r requirements.txt

COPY src ./src
COPY configs ./configs
COPY assets ./assets
COPY web ./web
COPY tests ./tests
COPY pyproject.toml .

EXPOSE 8000
HEALTHCHECK --interval=30s --timeout=5s CMD curl -f http://localhost:8000/health || exit 1
CMD ["uvicorn", "src.serving.app:app", "--host", "0.0.0.0", "--port", "8000"]
