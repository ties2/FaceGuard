"""Shared test helpers: a public-domain test face and tiny stand-in models.

The face is NASA astronaut Eileen Collins (``skimage.data.astronaut``, public
domain), so the suite needs no dataset download.
"""

from __future__ import annotations

import base64
import io
from functools import lru_cache

import cv2
import joblib
import numpy as np
import onnx
from onnx import TensorProto, helper
from sklearn.linear_model import LogisticRegression

from src.liveness.face import FaceDetector, context_crop
from src.liveness.signals.capture_integrity import CI_FEATURES
from src.liveness.signals.texture import PAD_FEATURE_DIM
from src.serving.config import settings


@lru_cache
def astronaut_face() -> tuple[np.ndarray, np.ndarray]:
    from skimage import data

    img = cv2.cvtColor(data.astronaut(), cv2.COLOR_RGB2BGR)
    det = FaceDetector(settings.faceguard_yunet_path)
    box = det.detect(img)
    assert box is not None
    return context_crop(img, box)


def b64jpeg(img: np.ndarray, q: int = 92) -> str:
    ok, buf = cv2.imencode(".jpg", img, [cv2.IMWRITE_JPEG_QUALITY, q])
    return base64.b64encode(buf.tobytes()).decode()


def sklearn_bytes(n_features: int, attack_if_low_first_feature: bool = False) -> bytes:
    rng = np.random.default_rng(0)
    x = rng.normal(size=(200, n_features))
    y = (x[:, 0] > 0).astype(int)
    m = LogisticRegression(max_iter=200).fit(x, y)
    buf = io.BytesIO()
    joblib.dump(m, buf)
    return buf.getvalue()


def stand_in_ci_bytes() -> bytes:
    """Capture-integrity stand-in: attack probability falls with the early
    challenge correlation (feature 0) — enough to exercise the API path."""
    rng = np.random.default_rng(1)
    x = rng.normal(size=(400, len(CI_FEATURES)))
    x[:, 0] = rng.uniform(-0.2, 1.0, 400)
    y = (x[:, 0] < 0.4).astype(int)
    m = LogisticRegression(max_iter=500).fit(x, y)
    buf = io.BytesIO()
    joblib.dump(m, buf)
    return buf.getvalue()


def stand_in_forensics_bytes() -> bytes:
    from src.liveness.signals.artifacts import ART_FEATURES

    rng = np.random.default_rng(2)
    x = rng.normal(size=(300, len(ART_FEATURES)))
    y = (x[:, 0] > 2.5).astype(int)  # rarely fires
    y[:5] = 1
    m = LogisticRegression(max_iter=500).fit(x, y)
    buf = io.BytesIO()
    joblib.dump(m, buf)
    return buf.getvalue()


def stand_in_pad_bytes() -> bytes:
    return sklearn_bytes(PAD_FEATURE_DIM)


def tiny_image_onnx(bias: float = -3.0) -> bytes:
    """image[N,3,128,128] -> GlobalAveragePool -> Flatten -> Gemm(3->1): logit."""
    w = helper.make_tensor("W", TensorProto.FLOAT, [3, 1], [0.1, 0.1, 0.1])
    b = helper.make_tensor("B", TensorProto.FLOAT, [1], [bias])
    nodes = [
        helper.make_node("GlobalAveragePool", ["image"], ["gap"]),
        helper.make_node("Flatten", ["gap"], ["flat"]),
        helper.make_node("Gemm", ["flat", "W", "B"], ["logit"]),
    ]
    g = helper.make_graph(
        nodes,
        "tiny",
        [helper.make_tensor_value_info("image", TensorProto.FLOAT, ["N", 3, 128, 128])],
        [helper.make_tensor_value_info("logit", TensorProto.FLOAT, ["N", 1])],
        [w, b],
    )
    model = helper.make_model(g, opset_imports=[helper.make_opsetid("", 17)])
    model.ir_version = 8
    onnx.checker.check_model(model)
    return model.SerializeToString()


def tiny_tabular_onnx(n_features: int = 4) -> bytes:
    """features[N,n] -> MatMul -> logit[N,1] (for the generic /predict path)."""
    w = helper.make_tensor("W", TensorProto.FLOAT, [n_features, 1], [0.5] * n_features)
    g = helper.make_graph(
        [helper.make_node("MatMul", ["x", "W"], ["y"])],
        "tab",
        [helper.make_tensor_value_info("x", TensorProto.FLOAT, ["N", n_features])],
        [helper.make_tensor_value_info("y", TensorProto.FLOAT, ["N", 1])],
        [w],
    )
    model = helper.make_model(g, opset_imports=[helper.make_opsetid("", 17)])
    model.ir_version = 8
    return model.SerializeToString()
