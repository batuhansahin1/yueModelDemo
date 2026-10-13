#!/usr/bin/env python3
"""YuE2 web UI server: the static site in public/ plus an /api proxy to the model server.

The browser talks only to this process. Requests under /api/ go to the model server
(server/yue2_server.py), so the model server can stay on 127.0.0.1 and no CORS setup is
needed. Only the endpoints the site uses are forwarded. Reference analyses run in the
background here (see AnalysisTasks). Standard library only: any Python 3.9+ runs it.
"""
from __future__ import annotations

import argparse
import base64
import binascii
import hmac
import http.client
import http.server
import json
import os
import re
import secrets
import socket
import sys
import threading
import time
import urllib.parse
from pathlib import Path

ROOT = Path(__file__).resolve().parent
PUBLIC = ROOT / "public"
PROJECT = ROOT.parent
# Shared with the Gradio UI: the style tag list and the example song.
DATA_FILES = {
    "/data/tags.json": PROJECT / "top_200_tags.json",
    "/data/example.json": PROJECT / "examples" / "example.json",
}
JOB_ID = r"[A-Za-z0-9_-]{1,64}"
TASK_ID = r"[0-9a-f]{32}"
# (method, model server path, timeout in seconds) of the endpoints forwarded as they are.
ROUTES = [
    ("GET", "/health", 15),
    ("GET", "/stats", 15),
    ("GET", "/jobs", 30),
    ("POST", "/jobs", 30),
    ("GET", f"/jobs/{JOB_ID}", 30),
    ("POST", f"/jobs/{JOB_ID}/cancel", 30),
    ("GET", f"/jobs/{JOB_ID}/audio", 300),
    ("POST", "/uploads", 600),
]
# POST /analyze: Demucs + Whisper (Turkish model ~4 min on a Mac) + SheetSage2 on a whole song.
ANALYZE_TIMEOUT = 3 * 3600
# A finished analysis stays available this long, so a lost answer can be asked for again.
FINISHED_TASK_SECONDS = 15 * 60
FORWARDED_REQUEST_HEADERS = ("Content-Type", "Range", "If-Range", "Accept")
FORWARDED_RESPONSE_HEADERS = ("Content-Type", "Content-Range", "Accept-Ranges", "Last-Modified", "ETag",
                              "Content-Disposition")
# Everything comes from this server: no CDN, fonts or analytics. blob: plays the reference file before upload.
SECURITY_HEADERS = {
    "Content-Security-Policy": ("default-src 'self'; script-src 'self'; style-src 'self'; img-src 'self' data:; "
                                "media-src 'self' blob:; connect-src 'self'; object-src 'none'; base-uri 'self'; "
                                "form-action 'self'"),
    "X-Content-Type-Options": "nosniff",
    "Referrer-Policy": "no-referrer",
}
LOCAL_HOSTS = {"127.0.0.1", "localhost", "::1"}
# How the model server is started on this system, for the messages that ask for it.
START_MODEL_SERVER, START_MOCK = (("start_model_server.bat", r"frontend\start.bat --mock") if os.name == "nt"
                                  else ("./start_model_server.sh", "./frontend/start.sh --mock"))


def open_connection(server, timeout):
    api = server.api
    connection_class = http.client.HTTPSConnection if api.scheme == "https" else http.client.HTTPConnection
    return connection_class(api.hostname, api.port, timeout=timeout)


def upstream_error(server, exc):
    """(status, JSON) the site shows when the model server cannot be reached or does not answer in time."""
    if isinstance(exc, (socket.timeout, TimeoutError)):
        return 504, {"detail": "Model sunucusu zamanında yanıt vermedi; işlem sunucuda sürüyor olabilir, biraz sonra "
                               "tekrar deneyin."}
    return 502, {"unreachable": True,
                 "detail": f"Model sunucusuna bağlanılamadı ({server.api_url}). Sunucuyu başlatın: "
                           f"{START_MODEL_SERVER} (modelsiz test: {START_MOCK})"}


def model_server_call(server, method, path, body=None, headers=None, timeout=30):
    """(status, content type, body) of one model server request, or of the error the site shows instead."""
    connection = open_connection(server, timeout)
    try:
        connection.request(method, server.api.path.rstrip("/") + path, body=body, headers=headers or {})
        response = connection.getresponse()
        return response.status, response.getheader("Content-Type") or "application/json", response.read()
    except (OSError, http.client.HTTPException) as exc:
        status, data = upstream_error(server, exc)
        return status, "application/json", json.dumps(data, ensure_ascii=False).encode()
    finally:
        connection.close()


class AnalysisTask:
    def __init__(self):
        self.started = time.time()
        self.finished = None
        self.result = None   # (status, content type, body) of the model server's answer
        self.done = threading.Event()


class AnalysisTasks:
    """Reference analyses that run longer than --analyze-wait.

    The model server answers POST /analyze only when Demucs, Whisper and SheetSage2 are done, which
    takes minutes for a whole song. Browsers (Safari) and reverse proxies (nginx: 60 s, Cloudflare:
    100 s) cut requests that stay silent that long, so this server waits for the answer itself and
    the page asks for it with GET /api/analyze/<task>.
    """

    def __init__(self):
        self.items = {}
        self.lock = threading.Lock()

    def add(self, task):
        with self.lock:
            self.purge()
            task_id = secrets.token_hex(16)
            self.items[task_id] = task
            return task_id

    def get(self, task_id):
        with self.lock:
            self.purge()
            return self.items.get(task_id)

    def purge(self):
        now = time.time()
        for task_id, task in list(self.items.items()):
            if task.finished and now - task.finished > FINISHED_TASK_SECONDS:
                del self.items[task_id]


class Handler(http.server.SimpleHTTPRequestHandler):
    server_version = "YuE2Web"
    extensions_map = {
        **http.server.SimpleHTTPRequestHandler.extensions_map,
        # Module scripts need a JavaScript type; some systems map .js to text/plain.
        ".js": "text/javascript; charset=utf-8",
        ".css": "text/css; charset=utf-8",
        ".html": "text/html; charset=utf-8",
        ".json": "application/json",
        ".svg": "image/svg+xml",
    }

    def __init__(self, *args, **kwargs):
        super().__init__(*args, directory=str(PUBLIC), **kwargs)

    # --- Dispatch ------------------------------------------------------------
    def do_GET(self):
        self.dispatch("GET")

    def do_HEAD(self):
        self.dispatch("HEAD")

    def do_POST(self):
        self.dispatch("POST")

    def dispatch(self, method):
        if not self.authorized():
            return self.ask_password()
        path = urllib.parse.urlsplit(self.path).path
        if path.startswith("/api/"):
            return self.handle_api(method, path[len("/api"):])
        if method == "POST":
            return self.send_json(405, {"detail": "Bu adres POST kabul etmez"})
        if path in DATA_FILES:
            return self.send_data_file(DATA_FILES[path])
        return super().do_GET() if method == "GET" else super().do_HEAD()

    def handle_api(self, method, path):
        if method == "POST" and path == "/analyze":
            return self.start_analysis()
        task = re.fullmatch(f"/analyze/({TASK_ID})", path)
        if method == "GET" and task:
            return self.analysis_result(task.group(1))
        route = next((r for r in ROUTES if r[0] == method and re.fullmatch(r[1], path)), None)
        if route is None:
            return self.send_json(404, {"detail": f"Bilinmeyen API adresi: {method} {path}"})
        self.proxy(method, path, timeout=route[2])

    def end_headers(self):
        for name, value in SECURITY_HEADERS.items():
            self.send_header(name, value)
        # No path yet when the request line itself was rejected (400, 414).
        if not urllib.parse.urlsplit(getattr(self, "path", "")).path.startswith("/api/"):
            # Revalidate the site's files on every load, so an update never leaves a stale page behind.
            self.send_header("Cache-Control", "no-cache")
        super().end_headers()

    def list_directory(self, path):
        self.send_error(404, "Not found")
        return None

    # --- Password (optional, HTTP Basic) ------------------------------------------
    def authorized(self):
        password = self.server.password
        if not password:
            return True
        scheme, _, encoded = self.headers.get("Authorization", "").partition(" ")
        if scheme.lower() != "basic":
            return False
        try:
            _, _, given = base64.b64decode(encoded, validate=True).decode("utf-8").partition(":")
        except (binascii.Error, UnicodeDecodeError):
            return False
        return hmac.compare_digest(given.encode(), password.encode())

    def ask_password(self):
        body = "Bu demo parola ile korunuyor.".encode()
        self.send_response(401)
        self.send_header("WWW-Authenticate", 'Basic realm="YuE2 Demo", charset="UTF-8"')
        self.send_header("Content-Type", "text/plain; charset=utf-8")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        if self.command != "HEAD":
            self.wfile.write(body)

    # --- Responses ---------------------------------------------------------------
    def send_result(self, result, cache="no-store"):
        status, content_type, body = result
        self.send_response(status)
        self.send_header("Content-Type", content_type)
        self.send_header("Content-Length", str(len(body)))
        if cache:
            self.send_header("Cache-Control", cache)
        self.end_headers()
        if self.command != "HEAD":
            self.wfile.write(body)

    def send_json(self, status, data):
        self.send_result((status, "application/json", json.dumps(data, ensure_ascii=False).encode()))

    def send_data_file(self, path):
        try:
            body = path.read_bytes()
        except OSError:
            return self.send_json(404, {"detail": f"{path.name} bulunamadı"})
        self.send_result((200, "application/json", body), cache=None)

    def read_body(self):
        """The request body; None after an error was sent (bad length, too large)."""
        try:
            length = int(self.headers.get("Content-Length") or 0)
        except ValueError:
            self.send_json(400, {"detail": "Geçersiz Content-Length"})
            return None
        if length > self.server.max_upload:
            self.discard_body(length)
            self.send_json(413, {"detail": f"Dosya çok büyük; en fazla {self.server.max_upload // 2**20} MB "
                                           "yüklenebilir."})
            return None
        return self.rfile.read(length) if length else b""

    def discard_body(self, length):
        """Read a rejected upload first: closing on unread data resets the connection before the browser
        sees the 413."""
        remaining = min(length, 4 * 2**30)
        while remaining > 0:
            chunk = self.rfile.read(min(remaining, 2**20))
            if not chunk:
                break
            remaining -= len(chunk)

    # --- /api proxy -----------------------------------------------------------------
    def proxy(self, method, path, timeout):
        body = self.read_body()
        if body is None:
            return
        headers = {name: self.headers[name] for name in FORWARDED_REQUEST_HEADERS if self.headers.get(name)}
        if method == "POST":
            headers["Content-Length"] = str(len(body))
        query = urllib.parse.urlsplit(self.path).query
        target = self.server.api.path.rstrip("/") + path + (f"?{query}" if query else "")
        connection = open_connection(self.server, timeout)
        try:
            connection.request(method, target, body=body if method == "POST" else None, headers=headers)
            upstream = connection.getresponse()
        except (OSError, http.client.HTTPException) as exc:
            connection.close()
            return self.send_json(*upstream_error(self.server, exc))
        try:
            self.relay(upstream)
        except (BrokenPipeError, ConnectionResetError):
            pass   # the browser went away, e.g. while seeking in the audio
        except (OSError, http.client.HTTPException):
            self.close_connection = True   # the model server stopped mid-response
        finally:
            connection.close()

    def relay(self, upstream):
        """Stream the model server's response (audio can be large)."""
        data = None
        length = upstream.getheader("Content-Length")
        if length is None:
            data = upstream.read()
            length = str(len(data))
        self.send_response(upstream.status)
        for name in FORWARDED_RESPONSE_HEADERS:
            value = upstream.getheader(name)
            if value:
                self.send_header(name, value)
        self.send_header("Content-Length", length)
        self.send_header("Cache-Control", "no-store")
        self.end_headers()
        if data is not None:
            self.wfile.write(data)
            return
        while True:
            chunk = upstream.read(64 * 1024)
            if not chunk:
                break
            self.wfile.write(chunk)

    # --- Reference analysis in the background -------------------------------------------
    def start_analysis(self):
        body = self.read_body()
        if body is None:
            return
        headers = {"Content-Type": self.headers.get("Content-Type") or "application/json",
                   "Content-Length": str(len(body))}
        server, task = self.server, AnalysisTask()

        def run():
            task.result = model_server_call(server, "POST", "/analyze", body, headers, ANALYZE_TIMEOUT)
            task.finished = time.time()
            task.done.set()

        threading.Thread(target=run, daemon=True).start()
        # Cached analyses and refusals (409 busy GPU, 422, 503) come back at once and are answered directly.
        if task.done.wait(server.analyze_wait):
            return self.send_result(task.result)
        self.send_json(202, {"pending": True, "task": server.tasks.add(task)})

    def analysis_result(self, task_id):
        task = self.server.tasks.get(task_id)
        if task is None:
            return self.send_json(404, {"detail": "Analiz bulunamadı; web sunucusu yeniden başlatılmış olabilir. "
                                                  "Tekrar deneyin: biten analizler önbellekten hemen gelir."})
        if not task.done.is_set():
            return self.send_json(202, {"pending": True, "task": task_id,
                                        "elapsed": round(time.time() - task.started)})
        self.send_result(task.result)

    # --- Log ---------------------------------------------------------------------------
    def log_request(self, code="-", size="-"):
        # Status polls (every second while a song is generated) and static files would flood the log.
        try:
            status = int(code)
        except (TypeError, ValueError):
            status = 0
        if self.command in ("GET", "HEAD") and 0 < status < 400:
            return
        super().log_request(code, size)


class WebServer(http.server.ThreadingHTTPServer):
    daemon_threads = True

    def __init__(self, address, api_url, password=None, max_upload_mb=200, analyze_wait=1.5):
        if ":" in address[0]:
            self.address_family = socket.AF_INET6
        super().__init__(address, Handler)
        self.api_url = api_url.rstrip("/")
        self.api = urllib.parse.urlsplit(self.api_url)
        self.password = password or None
        self.max_upload = max_upload_mb * 2**20
        self.analyze_wait = analyze_wait
        self.tasks = AnalysisTasks()


def model_server_status(server):
    """One line about the model server for the start message."""
    status, _, body = model_server_call(server, "GET", "/health", timeout=3)
    if status != 200:
        return (f"Model sunucusu şu an kapalı ({server.api_url}); başlatmak için: {START_MODEL_SERVER}\n"
                "Arayüz 5 saniyede bir kontrol eder ve sunucu açılınca kendiliğinden bağlanır.")
    try:
        health = json.loads(body)
    except ValueError:
        return f"{server.api_url} bir YuE2 model sunucusu gibi yanıt vermiyor; --api adresini kontrol edin."
    if health.get("mode") == "mock":
        return "Model sunucusu: test modu (mock); gerçek model yok, deneme tonu üretilir."
    device = health.get("device_name") or health.get("device") or "?"
    state = "bellekte" if health.get("model_loaded") else "ilk üretimde yüklenecek"
    return f"Model sunucusu bağlı: YuE2 ({state}), cihaz: {device}."


def main():
    parser = argparse.ArgumentParser(description="YuE2 web arayüzü (statik site + model sunucusu vekili)")
    parser.add_argument("--host", default=os.environ.get("YUE2_WEB_HOST", "127.0.0.1"),
                        help="Dinlenecek adres. Ağa/dışarıya açmak için 0.0.0.0 (varsayılan: 127.0.0.1)")
    parser.add_argument("--port", type=int, default=int(os.environ.get("YUE2_WEB_PORT", "7870")))
    parser.add_argument("--api", default=os.environ.get("YUE2_SERVER", "http://127.0.0.1:7861"),
                        help="Model sunucusunun adresi (server/yue2_server.py; varsayılan: http://127.0.0.1:7861)")
    parser.add_argument("--password", default=os.environ.get("YUE2_WEB_PASSWORD"),
                        help="Siteyi parolayla koru (HTTP Basic; kullanıcı adı önemsiz). Ortam değişkeni: "
                             "YUE2_WEB_PASSWORD")
    parser.add_argument("--max-upload-mb", type=int, default=200, help="Referans şarkı yükleme sınırı (MB)")
    parser.add_argument("--analyze-wait", type=float, default=1.5,
                        help="Referans analizi bu kadar saniyede biterse sonuç doğrudan döner; daha uzun sürerse "
                             "arka planda sürer ve sayfa sonucu sorar (varsayılan: 1.5)")
    args = parser.parse_args()

    api = urllib.parse.urlsplit(args.api)
    if api.scheme not in ("http", "https") or not api.hostname:
        parser.error(f"--api bir http(s) adresi olmalı, örn. http://127.0.0.1:7861 (verilen: {args.api})")
    try:
        server = WebServer((args.host, args.port), args.api, args.password, args.max_upload_mb, args.analyze_wait)
    except OSError as exc:
        sys.exit(f"{args.host}:{args.port} dinlenemedi: {exc}. Başka bir port deneyin: --port 7871")
    host, port = server.server_address[:2]
    shown = "127.0.0.1" if host in ("0.0.0.0", "") else host
    print(f"YuE2 web arayüzü: http://{shown}:{port}  (model sunucusu: {server.api_url})", flush=True)
    print(model_server_status(server), flush=True)
    if args.host not in LOCAL_HOSTS:
        print(f"Uyarı: {args.host} dinleniyor; site bu bilgisayarın dışından da açılabilir."
              + ("" if args.password else " Parola yok: herkes şarkı üretebilir (--password)."), flush=True)
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        pass
    finally:
        server.server_close()


if __name__ == "__main__":
    main()
