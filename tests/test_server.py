import pytest
from fastapi.testclient import TestClient

import r2d2.server as server
from tests.test_streaming import Backend


@pytest.fixture
def client(monkeypatch):
    engine = server.Engine()
    backend = Backend()
    backend.close = lambda: None
    engine.backend = backend
    engine.state = "ready"
    monkeypatch.setattr(server, "engine", engine)
    with TestClient(server.app) as client:
        yield client


def connect(client):
    return client.websocket_connect("/api/stream")


def start(ws):
    ws.send_json({"backend": "gguf", "language": "Chinese"})
    assert ws.receive_json()["type"] == "loading"
    assert ws.receive_json()["type"] == "ready"


def test_real_protocol_flush_and_exclusive_model_switch(client):
    with connect(client) as ws:
        start(ws)
        assert client.post("/api/backend", json={"backend": "mlx"}).status_code == 409
        ws.send_bytes(bytes(5120))
        ws.send_bytes(bytes(5120))
        update = ws.receive_json()
        assert update["text"] == "甲" and not update["final"]
        ws.send_text("stop")
        final = ws.receive_json()
        assert final["final"] and final["text"].endswith("乙")
        assert ws.receive_json()["type"] == "done"


def test_second_session_is_rejected(client):
    with connect(client) as first:
        start(first)
        with connect(client) as second:
            second.send_json({"backend": "gguf"})
            assert second.receive_json()["type"] == "error"
        first.send_text("stop")
        assert first.receive_json()["final"]
        assert first.receive_json()["type"] == "done"


@pytest.mark.parametrize("packet", [b"x", bytes(5122), b""])
def test_invalid_audio_returns_error(client, packet):
    with connect(client) as ws:
        start(ws)
        ws.send_bytes(packet)
        assert ws.receive_json()["type"] == "error"


def test_unknown_model_rejected(client):
    assert client.post("/api/backend", json={"backend": "other"}).status_code == 422
    with connect(client) as ws:
        ws.send_json({"backend": "other"})
        assert ws.receive_json()["type"] == "error"


def test_first_text_metric_includes_text_only_emitted_on_stop(client):
    with connect(client) as ws:
        start(ws)
        ws.send_bytes(bytes(320))
        ws.send_text("stop")
        final = ws.receive_json()
        assert final["text"] and final["first_text_ms"] is not None
        assert final["final"]
        assert ws.receive_json()["type"] == "done"


@pytest.mark.parametrize("variant", ["gguf_q8", "gguf_q4"])
def test_quantized_switch_loads_distinct_backend_and_stream_keeps_identity(client, monkeypatch, variant):
    loaded = []
    closed = []
    server.engine.backend.close = lambda: closed.append("gguf")

    def create(name):
        loaded.append(name)
        backend = Backend()
        backend.load = lambda: None
        backend.close = lambda: closed.append(name)
        return backend

    monkeypatch.setattr(server, "GGUFBackend", create)
    response = client.post("/api/backend", json={"backend": variant})
    assert response.status_code == 200
    assert response.json()["backend"] == variant
    assert loaded == [variant] and closed == ["gguf"]
    with connect(client) as ws:
        ws.send_json({"backend": variant, "language": "Chinese"})
        assert ws.receive_json()["type"] == "loading"
        ready = ws.receive_json()
        assert ready["type"] == "ready" and ready["backend"] == variant
        assert client.post("/api/backend", json={"backend": "gguf"}).status_code == 409
        ws.send_bytes(bytes(5120))
        ws.send_bytes(bytes(5120))
        assert ws.receive_json()["text"] == "甲"
        ws.send_text("stop")
        assert ws.receive_json()["final"]
        assert ws.receive_json() == {"type": "done", "backend": variant, "audio_ms": 320}
    assert loaded == [variant]
    assert client.post("/api/backend", json={"backend": "gguf"}).status_code == 200
    assert loaded == [variant, "gguf"]
    assert closed == ["gguf", variant]


@pytest.mark.parametrize("language", ["Chinese", "English", "Japanese", "Korean", "Spanish", "auto"])
def test_every_offered_language_is_accepted(client, language):
    with connect(client) as ws:
        ws.send_json({"backend": "gguf", "language": language})
        assert ws.receive_json()["type"] == "loading"
        assert ws.receive_json()["type"] == "ready"


@pytest.mark.parametrize("language", ["Klingon", "chinese", "Cantonese", ""])
def test_languages_outside_the_offered_set_are_rejected(client, language):
    with connect(client) as ws:
        ws.send_json({"backend": "gguf", "language": language})
        assert ws.receive_json()["type"] == "error"


def test_hint_accepts_upstreams_full_length(client):
    with connect(client) as ws:
        ws.send_json({"backend": "gguf", "language": "Chinese", "context": "词" * 4000})
        assert ws.receive_json()["type"] == "loading"
    with connect(client) as ws:
        ws.send_json({"backend": "gguf", "language": "Chinese", "context": "词" * 4001})
        assert ws.receive_json()["type"] == "error"
