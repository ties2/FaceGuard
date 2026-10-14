import uuid
from contextlib import asynccontextmanager
from pathlib import Path

from fastapi import FastAPI, Request
from fastapi.responses import RedirectResponse, Response
from fastapi.staticfiles import StaticFiles
from prometheus_client import CONTENT_TYPE_LATEST, generate_latest

import src.liveness.db_models  # noqa: F401  (register liveness tables with the shared metadata)
import src.serving.db_models  # noqa: F401
from src.liveness.api import router as liveness_router
from src.serving.api import router as models_router
from src.serving.database import Base, engine
from src.serving.observability import Observability


@asynccontextmanager
async def lifespan(app: FastAPI):
    # Create database tables when the application starts (not during import)
    Base.metadata.create_all(bind=engine)
    yield


app = FastAPI(
    title="FaceGuard — liveness defence layer (on ML Model Serving Platform)",
    description="Deepfake, injection and presentation-attack defence around a FaceTec-style liveness flow.",
    lifespan=lifespan,
)


# Middleware to generate and set Request ID
@app.middleware("http")
async def request_id_middleware(request: Request, call_next):
    req_id = request.headers.get("X-Request-ID", str(uuid.uuid4()))
    Observability.set_request_id(req_id)
    response = await call_next(request)
    response.headers["X-Request-ID"] = req_id
    return response


@app.get("/metrics", tags=["Observability"])
def get_metrics():
    """Standard and Prometheus-compatible output"""
    return Response(content=generate_latest(), media_type=CONTENT_TYPE_LATEST)


@app.get("/health", tags=["Health"])
def health_check():
    return {"status": "ok"}


app.include_router(models_router)
app.include_router(liveness_router)

_demo = Path(__file__).resolve().parents[2] / "web"
if _demo.exists():
    app.mount("/demo", StaticFiles(directory=_demo, html=True), name="demo")

    @app.get("/", include_in_schema=False)
    def root():
        return RedirectResponse("/demo/")
