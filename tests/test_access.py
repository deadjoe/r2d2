from fastapi.testclient import TestClient
import pytest
from starlette.websockets import WebSocketDisconnect

import r2d2.server as server
from r2d2.access import AccessGate
from tests.test_server import start
from tests.test_streaming import Backend

KEY = "k3y_for-tests-0123456789"


@pytest.fixture
def gated(monkeypatch):
    engine = server.Engine()
    engine.backend = Backend()
    engine.backend.close = lambda: None
    engine.name, engine.state = "gguf", "ready"
    monkeypatch.setattr(server, "engine", engine)
    with TestClient(AccessGate(server.app, KEY)) as gated:
        yield gated


def test_everything_is_closed_without_the_key(gated):
    page = gated.get("/")
    assert page.status_code == 401 and "access key" in page.text
    assert gated.get("/api/status").status_code == 401
    assert gated.get("/static/app.js").status_code == 401
    with pytest.raises(WebSocketDisconnect) as closed:
        with gated.websocket_connect("/api/stream") as ws:
            ws.receive_json()
    assert closed.value.code == 1008


def test_a_wrong_key_sets_no_cookie(gated):
    response = gated.get("/?k=wrong", follow_redirects=False)
    assert response.status_code == 303 and "set-cookie" not in response.headers
    assert gated.get("/").status_code == 401


def test_the_key_in_the_url_becomes_a_cookie_and_leaves_the_address_bar(gated):
    response = gated.get(f"/?k={KEY}&lang=ko", follow_redirects=False)
    assert response.status_code == 303 and response.headers["location"] == "/?lang=ko"
    assert "HttpOnly" in response.headers["set-cookie"] and "Secure" not in response.headers["set-cookie"]
    gated.cookies.set("r2d2_key", KEY)
    assert gated.get("/").status_code == 200
    assert gated.get("/api/status").json()["version"]
    with gated.websocket_connect("/api/stream") as ws:
        start(ws)
        ws.send_text("stop")
        assert ws.receive_json()["final"]


def test_behind_an_https_proxy_the_cookie_is_secure(gated):
    response = gated.get(f"/?k={KEY}", headers={"x-forwarded-proto": "https"}, follow_redirects=False)
    assert "Secure" in response.headers["set-cookie"]


def test_the_audio_socket_also_takes_the_key_as_a_query_parameter(gated):
    with gated.websocket_connect(f"/api/stream?k={KEY}") as ws:
        start(ws)
        ws.send_text("stop")
        assert ws.receive_json()["final"]
