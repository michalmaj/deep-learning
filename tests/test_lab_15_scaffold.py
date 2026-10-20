"""The lab 15 scaffold (provided code, not the student TODOs) must run.

Unlike other labs, module 15's "notebook" is really a small application
(`app/`, `benchmark.py`, `monitoring.py`) exercised from `lab.ipynb`. The
full lab.ipynb scaffold-to-first-TODO execution needs a real EuroSAT +
ResNet18 artifact, so it is skipped under CI=true, matching modules
07/12/14. The always-on tests below import the REAL application modules
(never an independent copy) and exercise them two ways:

1. The GIVEN parts (load_artifact, health/ready/model-info endpoints,
   send_requests, histogram_counts) must work correctly -- verified here
   with a tiny, randomly-initialized ResNet18 artifact built with
   `weights=None` (no network access, no download).
2. The TODO parts (preprocess_image_bytes, predict, the /predict endpoint,
   compute_stats, psi) must fail with NotImplementedError -- confirming
   the scaffold is in the expected "not yet implemented" state, without
   publishing the private solution's actual implementations.
"""

import asyncio
import io
import json
import os
import sys
from pathlib import Path

import nbformat
import pytest
import torch
import torch.nn as nn
from fastapi import UploadFile
from fastapi.testclient import TestClient
from nbclient import NotebookClient
from torch.utils.data import TensorDataset
from torchvision.models import resnet18

REPO_ROOT = Path(__file__).resolve().parent.parent
LAB_DIR = REPO_ROOT / "labs" / "15_production"
NOTEBOOK_PATH = LAB_DIR / "lab.ipynb"

sys.path.insert(0, str(LAB_DIR))


def _artifact_metadata():
    return {
        "model_version": "test-0.0.1",
        "architecture": "resnet18",
        "num_classes": 3,
        "class_names": ["A", "B", "C"],
        "input_size": [224, 224],
        "input_channels": 3,
        "preprocessing": "ResNet18_Weights.IMAGENET1K_V1.transforms()",
    }


@pytest.mark.skipif(
    os.environ.get("CI") == "true",
    reason="needs a real EuroSAT + ResNet18 artifact; run locally before opening a PR",
)
def test_lab_15_scaffold_runs_until_first_todo():
    nb = nbformat.read(NOTEBOOK_PATH, as_version=4)
    client = NotebookClient(
        nb,
        timeout=180,
        kernel_name="python3",
        allow_errors=True,
        resources={"metadata": {"path": str(LAB_DIR)}},
    )
    client.execute()

    code_cells = [c for c in nb.cells if c.cell_type == "code"]

    def _cell_error(cell):
        for output in cell.get("outputs", []):
            if output.get("output_type") == "error":
                return output
        return None

    errors = [(i, _cell_error(c)) for i, c in enumerate(code_cells)]
    first_error_index, first_error = next(e for e in errors if e[1] is not None)

    for i, error in errors:
        if i < first_error_index:
            assert error is None, f"scaffold cell {i} failed unexpectedly: {error}"

    assert first_error["ename"] == "NotImplementedError", (
        f"expected the first failure to be the artifact-consistency TODO, "
        f"got {first_error['ename']}: {first_error['evalue']}"
    )


@pytest.fixture
def fake_artifact_dir(tmp_path):
    """A tiny, randomly-initialized ResNet18 artifact -- no network access,
    no download, built purely from a local architecture (weights=None)."""
    model = resnet18(weights=None)
    model.fc = nn.Linear(model.fc.in_features, 3)
    torch.save(model.state_dict(), tmp_path / "weights.pt")
    metadata = _artifact_metadata()
    with open(tmp_path / "metadata.json", "w") as f:
        json.dump(metadata, f)
    return str(tmp_path)


def test_load_artifact_works_with_no_network_access(fake_artifact_dir):
    from app import model as model_module

    loaded = model_module.load_artifact(fake_artifact_dir)
    assert loaded.class_names == ["A", "B", "C"]
    assert loaded.model_version == "test-0.0.1"
    assert loaded.input_size == [224, 224]


@pytest.mark.parametrize(
    ("field", "value", "expected_fragment"),
    [
        ("model_version", "", "model_version"),
        ("architecture", "mobilenet_v3_small", "architecture"),
        ("num_classes", 0, "num_classes"),
        ("class_names", ["A", "A", "C"], "class_names must be unique"),
        ("class_names", ["A", "B"], "length must match num_classes"),
        ("input_size", [32, 32], "input_size"),
        ("input_channels", 1, "input_channels"),
        ("preprocessing", "manual resize", "preprocessing"),
    ],
)
def test_artifact_metadata_contract_rejects_inconsistent_fields(
    field, value, expected_fragment
):
    from app.artifact_contract import validate_artifact_metadata

    metadata = _artifact_metadata()
    metadata[field] = value

    with pytest.raises(ValueError, match=expected_fragment):
        validate_artifact_metadata(metadata)


def test_load_artifact_validates_metadata_before_loading_weights(fake_artifact_dir):
    from app import model as model_module

    metadata_path = Path(fake_artifact_dir) / "metadata.json"
    metadata = json.loads(metadata_path.read_text())
    metadata["input_size"] = [32, 32]
    metadata_path.write_text(json.dumps(metadata))

    with pytest.raises(ValueError, match="input_size"):
        model_module.load_artifact(fake_artifact_dir)


def test_extract_features_streams_selected_samples_in_batches():
    from prepare_artifact import extract_features

    class TrackingModel(nn.Module):
        def __init__(self):
            super().__init__()
            self.batch_sizes = []

        def forward(self, inputs):
            self.batch_sizes.append(len(inputs))
            return inputs.flatten(1)

    images = torch.arange(20, dtype=torch.float32).reshape(5, 1, 2, 2)
    labels = torch.tensor([10, 11, 12, 13, 14])
    indices = [4, 1, 3]
    model = TrackingModel()

    features, selected_labels = extract_features(
        model, TensorDataset(images, labels), indices, batch_size=2
    )

    assert model.batch_sizes == [2, 1]
    assert torch.equal(features, images[indices].flatten(1))
    assert torch.equal(selected_labels, labels[indices])
    assert not features.requires_grad


def test_preprocess_image_bytes_is_not_yet_implemented():
    from app import model as model_module

    with pytest.raises(NotImplementedError):
        model_module.preprocess_image_bytes(b"irrelevant bytes")


def test_predict_is_not_yet_implemented(fake_artifact_dir):
    from app import model as model_module

    loaded = model_module.load_artifact(fake_artifact_dir)
    with pytest.raises(NotImplementedError):
        model_module.predict(loaded, b"irrelevant bytes")


def test_server_given_endpoints_work_with_fake_artifact(fake_artifact_dir, monkeypatch):
    monkeypatch.setenv("MODULE15_ARTIFACT_DIR", fake_artifact_dir)
    # server.py reads the env var at import time via a module-level default;
    # reload to pick up the patched artifact dir for this test.
    import importlib

    from app import server as server_module

    importlib.reload(server_module)

    with TestClient(server_module.app) as client:
        r_health = client.get("/health")
        r_ready = client.get("/ready")
        r_info = client.get("/model-info")

    assert r_health.status_code == 200
    assert r_health.json() == {"status": "ok"}
    assert r_ready.status_code == 200
    assert r_ready.json() == {"ready": True}
    assert r_info.status_code == 200
    assert r_info.json()["class_names"] == ["A", "B", "C"]


def test_server_lifespan_loads_once_and_clears_state(fake_artifact_dir, monkeypatch):
    monkeypatch.setenv("MODULE15_ARTIFACT_DIR", fake_artifact_dir)
    import importlib

    from app import server as server_module

    importlib.reload(server_module)
    original_load_artifact = server_module.model_module.load_artifact
    load_calls = []

    def tracked_load_artifact(artifact_dir):
        load_calls.append(artifact_dir)
        return original_load_artifact(artifact_dir)

    monkeypatch.setattr(server_module.model_module, "load_artifact", tracked_load_artifact)

    with TestClient(server_module.app) as client:
        assert client.get("/ready").json() == {"ready": True}
        assert client.get("/health").json() == {"status": "ok"}
        assert client.get("/model-info").status_code == 200
        assert load_calls == [fake_artifact_dir]

    assert server_module.app.state.loaded is None


def test_model_info_returns_503_when_model_is_not_ready():
    from app import server as server_module

    server_module.app.state.loaded = None
    client = TestClient(server_module.app, raise_server_exceptions=False)
    response = client.get("/model-info")

    assert response.status_code == 503
    assert response.json() == {"detail": "model not ready"}


def test_read_upload_limited_accepts_a_bounded_payload():
    from app import server as server_module

    upload = UploadFile(io.BytesIO(b"four"), filename="x.png")
    data = asyncio.run(server_module.read_upload_limited(upload, max_bytes=4))

    assert data == b"four"


@pytest.mark.parametrize(
    ("payload", "expected_status"),
    [(b"", 400), (b"five!", 413)],
)
def test_read_upload_limited_rejects_empty_and_oversized_payloads(payload, expected_status):
    from app import server as server_module

    upload = UploadFile(io.BytesIO(payload), filename="x.png")
    with pytest.raises(server_module.HTTPException) as exc_info:
        asyncio.run(server_module.read_upload_limited(upload, max_bytes=4))

    assert exc_info.value.status_code == expected_status


def test_predict_endpoint_is_not_yet_implemented(fake_artifact_dir, monkeypatch):
    monkeypatch.setenv("MODULE15_ARTIFACT_DIR", fake_artifact_dir)
    import importlib

    from app import server as server_module

    importlib.reload(server_module)

    with TestClient(server_module.app) as client:
        with pytest.raises(NotImplementedError):
            client.post("/predict", files={"file": ("x.png", b"irrelevant", "image/png")})


def test_compute_stats_is_not_yet_implemented():
    from benchmark import compute_stats

    with pytest.raises(NotImplementedError):
        compute_stats([0.001, 0.002, 0.0015])


def test_send_requests_given_scaffold_propagates_until_predict_is_implemented(
    fake_artifact_dir, monkeypatch
):
    """send_requests itself is given, but it calls client.post("/predict", ...)
    with no try/except, so TestClient re-raises the endpoint's own
    NotImplementedError straight through it until /predict is implemented."""
    monkeypatch.setenv("MODULE15_ARTIFACT_DIR", fake_artifact_dir)
    import importlib

    from app import server as server_module

    importlib.reload(server_module)

    from benchmark import send_requests

    with TestClient(server_module.app) as client:
        with pytest.raises(NotImplementedError):
            send_requests(client, b"irrelevant", 3)


def test_send_requests_rejects_failed_http_responses():
    from benchmark import send_requests

    class FailingResponse:
        def raise_for_status(self):
            raise RuntimeError("HTTP 500")

    class FailingClient:
        def __init__(self):
            self.calls = 0

        def post(self, *args, **kwargs):
            self.calls += 1
            return FailingResponse()

    client = FailingClient()
    with pytest.raises(RuntimeError, match="HTTP 500"):
        send_requests(client, b"irrelevant", 3)

    assert client.calls == 1


def test_psi_is_not_yet_implemented():
    from monitoring import psi

    with pytest.raises(NotImplementedError):
        psi(torch.tensor([50.0, 50.0]), torch.tensor([70.0, 30.0]))


def test_histogram_counts_given_scaffold_works():
    from monitoring import histogram_counts

    values = torch.tensor([0.1, 0.2, 0.3, 0.9])
    bin_edges = torch.linspace(0.0, 1.0, 3)
    counts = histogram_counts(values, bin_edges)
    assert counts.sum().item() == 4
    assert counts[0].item() == 3  # 0.1, 0.2, 0.3 in [0, 0.5)
    assert counts[1].item() == 1  # 0.9 in [0.5, 1.0]


def test_brightness_feature_given_scaffold_works():
    from monitoring import brightness_feature

    images = torch.full((2, 3, 4, 4), 0.5)
    result = brightness_feature(images)
    assert torch.allclose(result, torch.tensor([0.5, 0.5]))
