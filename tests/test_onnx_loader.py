"""FG-001: multi-runtime loader registry on top of MSP."""

import pytest
from fastapi.testclient import TestClient

from src.serving.app import app
from src.serving.cache import model_cache
from src.serving.dependency import get_artifact_storage
from src.serving.loader import (
    ModelLoadError,
    OnnxModel,
    OnnxRuntimeLoader,
    SklearnJoblibLoader,
    UnsupportedModelFormat,
    get_loader,
    supported_formats,
)
from tests.helpers import tiny_image_onnx, tiny_tabular_onnx

client = TestClient(app)


def test_registry_resolves_known_formats():
    assert isinstance(get_loader("scikit-learn", "joblib"), SklearnJoblibLoader)
    assert isinstance(get_loader("onnx", "onnx"), OnnxRuntimeLoader)
    assert isinstance(get_loader("pytorch", "onnx"), OnnxRuntimeLoader)
    assert ("pytorch", "onnx") in supported_formats()


def test_registry_rejects_unknown_format():
    with pytest.raises(UnsupportedModelFormat):
        get_loader("tensorflow", "savedmodel")


def test_onnx_loader_wraps_session():
    m = OnnxRuntimeLoader().load(tiny_tabular_onnx(4))
    assert isinstance(m, OnnxModel)
    assert m.n_features_in_ == 4
    assert m.predict([[1, 1, 1, 1]]).shape == (1, 1)


def test_onnx_image_model_has_no_tabular_contract():
    m = OnnxRuntimeLoader().load(tiny_image_onnx())
    assert not hasattr(m, "n_features_in_")
    assert m.input_names == ["image"]


def test_onnx_loader_corrupt_bytes():
    with pytest.raises(ModelLoadError):
        OnnxRuntimeLoader().load(b"not an onnx graph")


def test_onnx_model_served_through_msp_predict():
    model_cache.clear()
    name, uri = "onnx-tabular", "local://onnx-tabular/1.0.0/model.onnx"
    get_artifact_storage().save(uri, tiny_tabular_onnx(4))
    client.post("/models", json={"name": name})
    r = client.post(
        f"/models/{name}/versions",
        json={"version": "1.0.0", "framework": "onnx", "model_format": "onnx", "artifact_uri": uri},
    )
    assert r.status_code in (200, 400)
    assert client.post(f"/models/{name}/versions/1.0.0/promote").status_code == 200
    r = client.post(f"/models/{name}/predict", json={"features": [1.0, 2.0, 3.0, 4.0]})
    assert r.status_code == 200
    assert r.json()["prediction"] == pytest.approx(5.0)
    # wrong feature count still rejected by the MSP contract
    assert client.post(f"/models/{name}/versions/1.0.0/predict", json={"features": [1.0]}).status_code == 422


def test_unsupported_format_load_returns_400():
    name = "weird-format"
    client.post("/models", json={"name": name})
    client.post(
        f"/models/{name}/versions",
        json={"version": "1.0.0", "framework": "tensorflow", "model_format": "savedmodel", "artifact_uri": "local://x"},
    )
    assert client.post(f"/models/{name}/versions/1.0.0/load").status_code == 400
