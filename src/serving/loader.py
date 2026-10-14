"""Model loaders (MSP-004) + multi-runtime registry (FG-001).

MSP shipped a single scikit-learn/joblib loader and hard-coded the format check
in the service layer. FaceGuard serves PyTorch-trained CNNs exported to ONNX
next to scikit-learn models, so loaders are now looked up in a registry keyed by
``(framework, model_format)``. Adding a runtime is one class + one registry line.
"""

from __future__ import annotations

import io
from typing import Any, Protocol

import joblib
import numpy as np


# Custom Exceptions for Loading
class ModelLoadError(Exception):
    pass


class UnsupportedModelFormat(Exception):
    pass


class InvalidModelArtifact(Exception):
    pass


# Abstraction (Interface)
class ModelLoader(Protocol):
    """Interface for different model loaders."""

    def load(self, artifact_bytes: bytes) -> Any: ...


# Implementation for Scikit-Learn + Joblib
class SklearnJoblibLoader:
    def load(self, artifact_bytes: bytes) -> Any:
        try:
            virtual_file = io.BytesIO(artifact_bytes)
            model = joblib.load(virtual_file)

            # Validation: Is this actually a Scikit-Learn model?
            if not hasattr(model, "predict") and not hasattr(model, "transform"):
                raise InvalidModelArtifact(
                    "The loaded object is not a valid scikit-learn model (missing 'predict' or 'transform' methods)."
                )
            return model

        except InvalidModelArtifact:
            raise
        except Exception as e:
            raise ModelLoadError(f"Failed to load scikit-learn model: {e!s}") from e


class OnnxModel:
    """Thin wrapper that gives an ONNX Runtime session the MSP model contract.

    * ``predict(x)`` runs the first graph input and returns the first output,
      so the generic ``/predict`` endpoint keeps working for tabular ONNX models.
    * ``run(feeds)`` exposes the full multi-input/multi-output graph for the
      liveness pipeline (e.g. image tensors).
    """

    def __init__(self, session: Any):
        self.session = session
        self.input_names = [i.name for i in session.get_inputs()]
        self.output_names = [o.name for o in session.get_outputs()]
        shape = session.get_inputs()[0].shape
        # Only meaningful for 2-D tabular graphs: [batch, n_features]
        if len(shape) == 2 and isinstance(shape[1], int):
            self.n_features_in_ = shape[1]

    def run(self, feeds: dict[str, np.ndarray]) -> list[np.ndarray]:
        return self.session.run(self.output_names, feeds)

    def predict(self, x: Any) -> np.ndarray:
        arr = np.asarray(x, dtype=np.float32)
        return self.session.run(self.output_names[:1], {self.input_names[0]: arr})[0]


class OnnxRuntimeLoader:
    def load(self, artifact_bytes: bytes) -> Any:
        try:
            import onnxruntime as ort
        except ImportError as e:  # pragma: no cover - dependency is in requirements.txt
            raise ModelLoadError("onnxruntime is not installed") from e
        try:
            opts = ort.SessionOptions()
            opts.intra_op_num_threads = 2
            session = ort.InferenceSession(artifact_bytes, sess_options=opts, providers=["CPUExecutionProvider"])
        except Exception as e:
            raise ModelLoadError(f"Failed to load ONNX model: {e!s}") from e
        return OnnxModel(session)


# ---------------------------------------------------------------------------
# Registry
# ---------------------------------------------------------------------------
_LOADERS: dict[tuple[str, str], ModelLoader] = {
    ("scikit-learn", "joblib"): SklearnJoblibLoader(),
    ("onnx", "onnx"): OnnxRuntimeLoader(),
    ("pytorch", "onnx"): OnnxRuntimeLoader(),  # trained in PyTorch, exported to ONNX
}


def supported_formats() -> list[tuple[str, str]]:
    return sorted(_LOADERS)


def get_loader(framework: str, model_format: str) -> ModelLoader:
    try:
        return _LOADERS[(framework, model_format)]
    except KeyError:
        raise UnsupportedModelFormat(
            f"Unsupported format ({framework}, {model_format}). Supported: {supported_formats()}"
        ) from None
