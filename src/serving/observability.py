import json
import logging
from contextvars import ContextVar

from prometheus_client import Counter, Histogram

# Simple logger configuration (in real projects, libraries like structlog can be used)
logger = logging.getLogger("msp_observability")
logger.setLevel(logging.INFO)
if not logger.handlers:
    handler = logging.StreamHandler()
    logger.addHandler(handler)

# Save Request ID in the current execution context
# request_id_var = contextvars.ContextVar("request_id", default="unknown")
request_id_ctx_var: ContextVar[str] = ContextVar("request_id", default="")

# Metrics Definition
PREDICTION_REQUESTS = Counter("prediction_requests_total", "Total prediction requests", ["model", "version", "status"])

PREDICTION_ERRORS = Counter("prediction_errors_total", "Total prediction errors", ["model", "version", "error_type"])

PREDICTION_LATENCY = Histogram("prediction_latency_seconds", "Prediction latency in seconds", ["model", "version"])

MODEL_LOAD_TOTAL = Counter(
    "model_load_total", "Total model loads from artifact storage", ["model", "version", "status"]
)

MODEL_LOAD_LATENCY = Histogram("model_load_latency_seconds", "Model loading latency in seconds", ["model", "version"])

CACHE_HITS = Counter("model_cache_hits_total", "Total model cache hits", ["model", "version"])

CACHE_MISSES = Counter("model_cache_misses_total", "Total model cache misses", ["model", "version"])


MODEL_PROMOTIONS = Counter("model_promotions_total", "Total model promotions", ["model", "version", "status"])


# Abstraction Layer
class Observability:
    @staticmethod
    def set_request_id(request_id: str):
        request_id_ctx_var.set(request_id)

    @staticmethod
    def get_request_id() -> str:
        return request_id_ctx_var.get()

    @staticmethod
    def log_prediction_event(model: str, version: str, status: str, latency_ms: float = None, error_type: str = None):
        """Generate structured JSON log without PII"""
        log_data = {
            "event": "prediction",
            "request_id": Observability.get_request_id(),
            "model_name": model,
            "version": version,
            "status": status,
        }
        if latency_ms is not None:
            log_data["latency_ms"] = round(latency_ms, 2)
        if error_type is not None:
            log_data["error_type"] = error_type

        logger.info(json.dumps(log_data))

    # --- Metrics Wrapper Methods ---
    @staticmethod
    def record_prediction_success(model: str, version: str, latency_sec: float):
        PREDICTION_REQUESTS.labels(model=model, version=version, status="success").inc()
        PREDICTION_LATENCY.labels(model=model, version=version).observe(latency_sec)
        Observability.log_prediction_event(model, version, "success", latency_ms=latency_sec * 1000)

    @staticmethod
    def record_prediction_error(model: str, version: str, error_type: str):
        PREDICTION_REQUESTS.labels(model=model, version=version, status="error").inc()
        PREDICTION_ERRORS.labels(model=model, version=version, error_type=error_type).inc()
        Observability.log_prediction_event(model, version, "error", error_type=error_type)

    @staticmethod
    def record_cache_hit(model: str, version: str):
        CACHE_HITS.labels(model=model, version=version).inc()

    @staticmethod
    def record_cache_miss(model: str, version: str):
        CACHE_MISSES.labels(model=model, version=version).inc()

    @staticmethod
    def record_model_load(model: str, version: str, status: str, latency_sec: float = 0.0):
        MODEL_LOAD_TOTAL.labels(model=model, version=version, status=status).inc()
        if status == "success":
            MODEL_LOAD_LATENCY.labels(model=model, version=version).observe(latency_sec)

    @staticmethod
    def record_promotion(model_name: str, version: str, status: str, error_type: str = None):
        MODEL_PROMOTIONS.labels(model=model_name, version=version, status=status).inc()
        log_data = {
            "event": "model_promotion",
            "request_id": request_id_ctx_var.get(),
            "model_name": model_name,
            "version": version,
            "status": status,
        }
        if error_type:
            log_data["error_type"] = error_type
        logger.info(json.dumps(log_data))
