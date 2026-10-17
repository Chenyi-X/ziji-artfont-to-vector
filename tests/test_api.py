"""接口层测试：素材预检、异步任务、预设一致性。"""
from __future__ import annotations

import time
from pathlib import Path

from fastapi.testclient import TestClient

from app.main import app
from app.pipeline.service import inspect_bytes, validate_input
from app.pipeline.vectorize import VectorizeOptions

ROOT = Path(__file__).resolve().parent.parent
SAMPLE = ROOT / "samples" / "images" / "13_cartoon.png"

client = TestClient(app)


def _sample_bytes() -> bytes:
    return SAMPLE.read_bytes()


def test_validate_input_rejects_bad_ext_and_oversize():
    assert validate_input("a.gif", b"x")
    assert validate_input("a.png", b"0" * (5 * 1024 * 1024 + 1))
    assert not validate_input("a.png", b"0" * 1024)


def test_inspect_bytes_reports_quality():
    out = inspect_bytes("13_cartoon.png", _sample_bytes())
    assert out["problems"] == []
    assert out["size"]["width"] > 0
    assert out["quality"]["severity"] in {"ok", "warning", "reject"}


def test_inspect_bytes_rejects_junk():
    out = inspect_bytes("bad.png", b"not an image")
    assert out["problems"]


def test_api_inspect_endpoint():
    res = client.post("/api/inspect", files={"file": ("13_cartoon.png", _sample_bytes(), "image/png")})
    assert res.status_code == 200
    data = res.json()
    assert data["filename"] == "13_cartoon.png"
    assert "quality" in data


def test_api_job_flow_to_done():
    res = client.post("/api/jobs", files={"file": ("13_cartoon.png", _sample_bytes(), "image/png")},
                      data={"preset": "balanced"})
    assert res.status_code == 200
    job_id = res.json()["job_id"]
    deadline = time.time() + 60
    payload = {}
    while time.time() < deadline:
        payload = client.get(f"/api/jobs/{job_id}").json()
        if payload["status"] in {"done", "error", "cancelled"}:
            break
        time.sleep(0.3)
    assert payload["status"] == "done", payload
    assert "<svg" in payload["result"]["svg"]
    assert payload["result"]["metrics"]["contour_fidelity"] > 0
    assert payload["stage"] in {"preprocess", "vectorize", "verify"}


def test_api_job_unknown_id():
    assert client.get("/api/jobs/nope").status_code == 404


def test_preset_desc_matches_options():
    """档位文案与真实参数同源，防止再次出现文案与生效值不一致。"""
    data = client.get("/api/presets").json()
    descs = {p["key"]: p["desc"] for p in data["presets"]}
    q = VectorizeOptions.PRESETS["quality"]
    c = VectorizeOptions.PRESETS["compact"]
    assert str(q["tolerance"]) in descs["quality"]
    assert str(q["max_colors"]) in descs["quality"]
    assert str(c["tolerance"]) in descs["compact"]
    assert str(c["max_colors"]) in descs["compact"]
