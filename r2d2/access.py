"""Optional access key for a server reachable from outside the machine.

A cloud GPU's proxy URL (RunPod's https://<pod>-8765.proxy.runpod.net) is public
and the app has no accounts. With R2D2_ACCESS_KEY set, every page, API call and
the audio socket needs the key: open /?k=<key> once and a cookie carries it from
then on. Without the variable the app is open, as on a laptop.
"""
from __future__ import annotations

import hmac
from http.cookies import SimpleCookie
from urllib.parse import parse_qsl, urlencode

COOKIE = "r2d2_key"

LOCK_PAGE = """<!doctype html><html lang="en"><head><meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1"><meta name="color-scheme" content="dark light">
<title>R2D2 // Listening room</title>
<style>body{margin:0;min-height:100vh;display:flex;align-items:center;justify-content:center;
font:14px/1.45 ui-monospace,Menlo,monospace;background:#0b0a09;color:#f1ece2;padding:16px;box-sizing:border-box}
@media(prefers-color-scheme:light){body{background:#f1ece2;color:#16140f}}
form{width:100%;max-width:340px}label{display:block;font-size:11px;letter-spacing:.12em;text-transform:uppercase;opacity:.7;margin:0 0 4px}
input,button{width:100%;box-sizing:border-box;font:inherit;font-size:16px;padding:10px;border-radius:8px;margin-bottom:10px}
input{border:1px solid #7a746a;background:transparent;color:inherit}button{border:0;font-weight:600;letter-spacing:.08em}</style></head>
<body><form method="get" action="/"><h1 style="font-size:15px;letter-spacing:.06em">R2D2 // LISTENING ROOM</h1>
<label for="k">access key</label><input id="k" name="k" type="password" autocomplete="current-password" autofocus required>
<button>Unlock</button></form></body></html>"""


def _headers(scope):
    return {k.decode("latin-1").lower(): v.decode("latin-1") for k, v in scope.get("headers", [])}


class AccessGate:
    """ASGI middleware: pass requests that carry the key, turn the rest away."""

    def __init__(self, app, key: str):
        self.app, self.key = app, key

    def _valid(self, value: str | None) -> bool:
        return value is not None and hmac.compare_digest(value.encode(), self.key.encode())

    async def __call__(self, scope, receive, send):
        if scope["type"] not in ("http", "websocket"):
            return await self.app(scope, receive, send)
        headers = _headers(scope)
        cookie = SimpleCookie()
        try:
            cookie.load(headers.get("cookie", ""))
        except Exception:
            pass
        query = parse_qsl(scope.get("query_string", b"").decode("latin-1"), keep_blank_values=True)
        given = next((v for k, v in query if k == "k"), None)
        if self._valid(cookie[COOKIE].value if COOKIE in cookie else None):
            return await self.app(scope, receive, send)

        if scope["type"] == "websocket":
            if self._valid(given):
                return await self.app(scope, receive, send)
            # Refuse the handshake; the browser sees a failed connection.
            await send({"type": "websocket.close", "code": 1008})
            return

        if given is not None:
            # Trade the key in the URL for a cookie, and drop it from the address bar.
            rest = urlencode([(k, v) for k, v in query if k != "k"])
            location = scope["path"] + (f"?{rest}" if rest else "")
            response = [(b"location", location.encode()), (b"cache-control", b"no-store")]
            if self._valid(given):
                secure = "https" in (headers.get("x-forwarded-proto", ""), scope.get("scheme"))
                value = f"{COOKIE}={given}; Path=/; Max-Age=2592000; HttpOnly; SameSite=Lax" + ("; Secure" if secure else "")
                response.append((b"set-cookie", value.encode()))
            await send({"type": "http.response.start", "status": 303, "headers": response})
            await send({"type": "http.response.body", "body": b""})
            return

        api = scope["path"].startswith("/api/")
        body = b'{"error":"access key required"}' if api else LOCK_PAGE.encode()
        kind = b"application/json" if api else b"text/html; charset=utf-8"
        await send({"type": "http.response.start", "status": 401,
                    "headers": [(b"content-type", kind), (b"cache-control", b"no-store")]})
        await send({"type": "http.response.body", "body": body})
