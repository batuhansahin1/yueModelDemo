"""Tests of frontend/server.py: static site, data files, the /api proxy and the password, plus the site's
requests end to end against the mock model server (server/yue2_server.py --mock).

    ./frontend/test.sh server
    ../venv/bin/python -m unittest discover -s frontend/tests -v

The mock model server runs with yue-ui/venv's Python (fastapi + uvicorn); without it those tests are skipped.
"""
from __future__ import annotations

import base64
import http.server
import io
import json
import math
import os
import re
import socket
import struct
import subprocess
import sys
import tempfile
import threading
import time
import unittest
import urllib.error
import urllib.request
import wave
from pathlib import Path

FRONTEND = Path(__file__).resolve().parents[1]
PROJECT = FRONTEND.parent
MOCK_PYTHON = os.environ.get("YUE2_MOCK_PYTHON", str(PROJECT / "venv" / "bin" / "python"))


def free_port():
    with socket.socket() as sock:
        sock.bind(("127.0.0.1", 0))
        return sock.getsockname()[1]


def request(url, data=None, method=None, headers=None, timeout=30):
    """(status, headers, body); HTTP errors are returned like any other response."""
    req = urllib.request.Request(url, data=data, method=method, headers=headers or {})
    try:
        with urllib.request.urlopen(req, timeout=timeout) as response:
            return response.status, response.headers, response.read()
    except urllib.error.HTTPError as error:
        return error.code, error.headers, error.read()


def call(url, payload=None, method=None, headers=None):
    """(status, decoded JSON) of an API call."""
    data = None if payload is None else json.dumps(payload).encode()
    headers = {**({"Content-Type": "application/json"} if data else {}), **(headers or {})}
    status, _, body = request(url, data, method or ("POST" if data else "GET"), headers)
    return status, json.loads(body) if body else None


def analyze(api, params, timeout=30):
    """POST /api/analyze as public/js/api.js does: a long analysis answers {pending, task} until it is done."""
    status, data = call(api + "/analyze", params)
    deadline = time.time() + timeout
    while data and data.get("pending") and time.time() < deadline:
        time.sleep(0.2)
        status, data = call(f"{api}/analyze/{data['task']}")
    return status, data


def wait_until(check, timeout=30):
    deadline = time.time() + timeout
    while time.time() < deadline:
        try:
            if check():
                return True
        except OSError:
            pass
        time.sleep(0.1)
    return False


def start_web(api, *extra):
    port = free_port()
    process = subprocess.Popen([sys.executable, str(FRONTEND / "server.py"), "--port", str(port), "--api", api, *extra],
                               stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
    base = f"http://127.0.0.1:{port}"
    if not wait_until(lambda: request(base + "/favicon.svg", timeout=2)[0] in (200, 401)):
        process.kill()
        raise RuntimeError("web server did not start")
    return process, base


def stop(process):
    process.terminate()
    try:
        process.wait(10)
    except subprocess.TimeoutExpired:
        process.kill()


def tone_wav(seconds=1.0, rate=16000):
    buffer = io.BytesIO()
    with wave.open(buffer, "wb") as out:
        out.setnchannels(1)
        out.setsampwidth(2)
        out.setframerate(rate)
        out.writeframes(b"".join(struct.pack("<h", int(8000 * math.sin(2 * math.pi * 440 * i / rate)))
                                 for i in range(int(seconds * rate))))
    return buffer.getvalue()


def mock_available():
    try:
        return subprocess.run([MOCK_PYTHON, "-c", "import fastapi, uvicorn, numpy"], capture_output=True).returncode == 0
    except OSError:
        return False


class SiteTest(unittest.TestCase):
    """The web server alone; nothing listens at its model server address."""

    @classmethod
    def setUpClass(cls):
        cls.web, cls.base = start_web(f"http://127.0.0.1:{free_port()}")

    @classmethod
    def tearDownClass(cls):
        stop(cls.web)

    def test_index_and_security_headers(self):
        status, headers, body = request(self.base + "/")
        self.assertEqual(status, 200)
        self.assertIn("text/html", headers["Content-Type"])
        self.assertIn(b"<title>YuE2 Demo</title>", body)
        self.assertIn("default-src 'self'", headers["Content-Security-Policy"])
        self.assertEqual(headers["X-Content-Type-Options"], "nosniff")
        self.assertEqual(headers["Cache-Control"], "no-cache")

    def test_every_page_asset_is_served_with_its_type(self):
        _, _, body = request(self.base + "/")
        assets = re.findall(rb'(?:src|href)="([^"#:]+)"', body)
        self.assertIn(b"js/main.js", assets)
        for asset in assets:
            status, headers, _ = request(self.base + "/" + asset.decode())
            self.assertEqual(status, 200, asset)
        for module in (FRONTEND / "public" / "js").glob("*.js"):
            status, headers, _ = request(f"{self.base}/js/{module.name}")
            self.assertEqual((status, headers["Content-Type"].split(";")[0]), (200, "text/javascript"), module.name)

    def test_data_files_come_from_the_project(self):
        status, _, body = request(self.base + "/data/tags.json")
        self.assertEqual(status, 200)
        self.assertEqual(len(json.loads(body)["genre"]), 200)
        status, _, body = request(self.base + "/data/example.json")
        self.assertIn("[chorus]", json.loads(body)["lyrics_text"])

    def test_no_listing_and_no_files_outside_public(self):
        self.assertEqual(request(self.base + "/js/")[0], 404)
        for path in ("/../server.py", "/%2e%2e/server.py", "/..%2fserver.py", "/server.py"):
            status, _, body = request(self.base + path)
            self.assertEqual(status, 404, path)
            self.assertNotIn(b"import http.server", body)

    def test_only_known_api_endpoints_are_forwarded(self):
        for method, path in (("GET", "/api/docs"), ("POST", "/api/health"), ("GET", "/api/jobs/x/../../health"),
                             ("GET", "/api/uploads"), ("POST", "/api/jobs/abc/audio")):
            status, data = call(self.base + path, method=method)
            self.assertEqual(status, 404, f"{method} {path}")
            self.assertIn("Bilinmeyen API adresi", data["detail"])
        self.assertEqual(call(self.base + "/", {"x": 1})[0], 405)

    def test_unreachable_model_server(self):
        status, data = call(self.base + "/api/health")
        self.assertEqual(status, 502)
        self.assertTrue(data["unreachable"])
        self.assertIn("Model sunucusuna bağlanılamadı", data["detail"])
        # A refused connection fails at once, so the analysis answers directly too.
        status, data = call(self.base + "/api/analyze", {"upload_id": "0" * 32})
        self.assertEqual((status, data["unreachable"]), (502, True))


class PasswordAndLimitTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.web, cls.base = start_web(f"http://127.0.0.1:{free_port()}", "--password", "gizli parola",
                                      "--max-upload-mb", "1")

    @classmethod
    def tearDownClass(cls):
        stop(cls.web)

    @staticmethod
    def auth(password):
        return {"Authorization": "Basic " + base64.b64encode(f"herhangi:{password}".encode()).decode()}

    def test_password_is_required(self):
        status, headers, _ = request(self.base + "/")
        self.assertEqual(status, 401)
        self.assertIn("Basic", headers["WWW-Authenticate"])
        self.assertEqual(request(self.base + "/", headers=self.auth("yanlış"))[0], 401)
        self.assertEqual(request(self.base + "/api/health", headers={"Authorization": "Basic !!!"})[0], 401)
        self.assertEqual(request(self.base + "/", headers=self.auth("gizli parola"))[0], 200)

    def test_large_uploads_get_a_readable_413(self):
        status, _, body = request(self.base + "/api/uploads?filename=big.wav", data=b"\0" * (2 * 2**20), method="POST",
                                  headers={**self.auth("gizli parola"), "Content-Type": "application/octet-stream"})
        self.assertEqual(status, 413)
        self.assertIn("en fazla 1 MB", json.loads(body)["detail"])


class SlowAnalysisTest(unittest.TestCase):
    """An analysis that takes longer than --analyze-wait runs on in the web server; the page asks for it."""

    ANALYSIS_SECONDS = 1.5

    @classmethod
    def setUpClass(cls):
        class Stub(http.server.BaseHTTPRequestHandler):
            """Model server whose /analyze is slow, or refuses at once like a busy GPU (409)."""

            def do_POST(self):
                params = json.loads(self.rfile.read(int(self.headers["Content-Length"])))
                if params["upload_id"] == "busy":
                    return self.reply(409, {"detail": "Şu an şarkı üretiliyor; bitince tekrar deneyin."})
                time.sleep(SlowAnalysisTest.ANALYSIS_SECONDS)
                self.reply(200, {"abc": "X:1\n", "lyrics": None, "summary": {"seconds": 1.0},
                                 "upload_id": params["upload_id"]})

            def reply(self, status, data):
                body = json.dumps(data).encode()
                self.send_response(status)
                self.send_header("Content-Type", "application/json")
                self.send_header("Content-Length", str(len(body)))
                self.end_headers()
                self.wfile.write(body)

            def log_message(self, *args):
                pass

        cls.stub = http.server.ThreadingHTTPServer(("127.0.0.1", 0), Stub)
        threading.Thread(target=cls.stub.serve_forever, daemon=True).start()
        cls.web, base = start_web(f"http://127.0.0.1:{cls.stub.server_address[1]}", "--analyze-wait", "0.3")
        cls.api = base + "/api"

    @classmethod
    def tearDownClass(cls):
        stop(cls.web)
        cls.stub.shutdown()
        cls.stub.server_close()

    def test_long_analysis_is_asked_for_until_done(self):
        status, data = call(self.api + "/analyze", {"upload_id": "a" * 32})
        self.assertEqual((status, data["pending"]), (202, True))
        task = data["task"]
        status, data = call(f"{self.api}/analyze/{task}")
        self.assertEqual((status, data["pending"]), (202, True))
        self.assertTrue(wait_until(lambda: call(f"{self.api}/analyze/{task}")[0] == 200, timeout=10))
        # The answer stays available for a while, so a lost response can be asked for again.
        status, data = call(f"{self.api}/analyze/{task}")
        self.assertEqual((status, data["upload_id"]), (200, "a" * 32))

    def test_quick_answers_come_directly(self):
        status, data = call(self.api + "/analyze", {"upload_id": "busy"})
        self.assertEqual(status, 409)
        self.assertIn("şarkı üretiliyor", data["detail"])

    def test_unknown_task(self):
        status, data = call(f"{self.api}/analyze/{'0' * 32}")
        self.assertEqual(status, 404)
        self.assertIn("Analiz bulunamadı", data["detail"])
        self.assertEqual(call(self.api + "/analyze/../health")[0], 404)


@unittest.skipUnless(mock_available(), f"mock model sunucusu için fastapi/uvicorn gerekli: {MOCK_PYTHON}")
class MockModelServerTest(unittest.TestCase):
    """The site's requests through the proxy, as public/js/*.js sends them."""

    @classmethod
    def setUpClass(cls):
        cls.outputs = tempfile.TemporaryDirectory()
        port = free_port()
        cls.mock = subprocess.Popen([MOCK_PYTHON, str(PROJECT / "server" / "yue2_server.py"), "--mock", "--port", str(port),
                                     "--outputs", cls.outputs.name], stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
        if not wait_until(lambda: request(f"http://127.0.0.1:{port}/health", timeout=2)[0] == 200):
            stop(cls.mock)
            raise RuntimeError("mock model server did not start")
        cls.web, base = start_web(f"http://127.0.0.1:{port}")
        cls.api = base + "/api"

    @classmethod
    def tearDownClass(cls):
        stop(cls.web)
        stop(cls.mock)
        cls.outputs.cleanup()

    def follow(self, job_id, timeout=60):
        deadline = time.time() + timeout
        while time.time() < deadline:
            status, job = call(f"{self.api}/jobs/{job_id}")
            self.assertEqual(status, 200)
            if job["status"] in ("complete", "failed", "cancelled"):
                return job
            time.sleep(0.2)
        self.fail(f"job {job_id} did not finish")

    def payload(self, **overrides):
        """A request as logic.buildT2SPayload makes it with default settings."""
        return {"style": "pop, female vocal, Gece", "lyrics": "[verse]\nNeon ışıklar\n\n[chorus]\nYine sana", "cot": "full",
                "seed": None, "cfg_scale": None, "abc": None, "ode_steps": None, "quantization": None,
                "abc_sampling": {"temperature": 0.7, "top_p": 0.9, "top_k": 30, "repetition_penalty": 1.005,
                                 "penalty_window": 100, "min_tokens": 32, "max_tokens": 4096},
                "semantic_sampling": {"temperature": 1.0, "top_p": 0.95, "top_k": 100, "repetition_penalty": 1.2,
                                      "penalty_window": 50, "min_tokens": 200, "max_tokens": 9000},
                "prompt": {"description": "Gece", "styles": ["pop"], "custom_style": "", "voice": "female"}, **overrides}

    def test_health(self):
        status, health = call(self.api + "/health")
        self.assertEqual((status, health["mode"]), (200, "mock"))
        self.assertIn("fp8", health)

    def test_text_to_song(self):
        status, job = call(self.api + "/jobs", self.payload(ode_steps=16))
        self.assertEqual(status, 200)
        self.assertIsInstance(job["request"]["seed"], int)   # the server picks the random seed
        job = self.follow(job["id"])
        self.assertEqual(job["status"], "complete", job.get("error"))
        self.assertEqual(job["request"]["prompt"]["voice"], "female")
        self.assertTrue(job["result"]["abc"].startswith("X:1"))
        self.assertGreater(job["tokens"]["abc"], 0)

        status, headers, audio = request(f"{self.api}/jobs/{job['id']}/audio")
        self.assertEqual(status, 200)
        self.assertTrue(audio.startswith(b"RIFF"))
        self.assertEqual(headers["Cache-Control"], "no-store")
        # Seeking in the player asks for byte ranges.
        status, headers, part = request(f"{self.api}/jobs/{job['id']}/audio", headers={"Range": "bytes=0-99"})
        self.assertEqual((status, len(part)), (206, 100))
        self.assertTrue(headers["Content-Range"].startswith("bytes 0-99/"))

        status, jobs = call(self.api + "/jobs?limit=5")
        self.assertIn(job["id"], [item["id"] for item in jobs])
        self.assertEqual(call(f"{self.api}/stats")[0], 200)

    def test_cover(self):
        audio = tone_wav()
        status, _, body = request(self.api + "/uploads?filename=referans.wav", data=audio, method="POST",
                                  headers={"Content-Type": "application/octet-stream"})
        self.assertEqual(status, 200)
        upload_id = json.loads(body)["upload_id"]
        self.assertRegex(upload_id, r"^[0-9a-f]{32}$")
        # The same file gets the same id, so its analyses stay cached.
        _, _, again = request(self.api + "/uploads?filename=kopya.wav", data=audio, method="POST")
        self.assertEqual(json.loads(again)["upload_id"], upload_id)

        status, analysis = analyze(self.api, {
            "upload_id": upload_id, "start": 0, "duration": None, "bpm": None, "lyrics": "", "recognize_lyrics": True,
            "language": None, "transcriber": "sheetsage2"})
        self.assertEqual(status, 200)
        self.assertEqual(analysis["summary"]["lyrics_status"], "ok")
        self.assertIn("% verse", analysis["abc"])
        self.assertTrue(analysis["lyrics"].startswith("[verse]"))

        reference = {"file": "referans.wav", "start": 0, "duration": None, "mode": "free", "transcriber": "sheetsage2",
                     **{k: v for k, v in analysis["summary"].items() if k != "lines"}}
        status, job = call(self.api + "/jobs", self.payload(style="pop, Turkish", lyrics=analysis["lyrics"], cot="melody",
                                                            abc=analysis["abc"], reference=reference, prompt=None))
        self.assertEqual(status, 200)
        job = self.follow(job["id"])
        self.assertEqual(job["status"], "complete")
        self.assertEqual(job["result"]["abc"], analysis["abc"])   # an external score skips the plan stage
        self.assertNotIn("abc", job["tokens"])
        self.assertEqual(job["request"]["reference"]["mode"], "free")

    def test_cancel(self):
        _, job = call(self.api + "/jobs", self.payload())
        status, _ = call(f"{self.api}/jobs/{job['id']}/cancel", method="POST")
        self.assertEqual(status, 200)
        self.assertEqual(self.follow(job["id"])["status"], "cancelled")

    def test_server_errors_reach_the_site(self):
        status, data = call(self.api + "/jobs", {"lyrics": "x"})
        self.assertEqual(status, 422)
        self.assertIsInstance(data["detail"], list)
        status, _, body = request(self.api + "/uploads?filename=notlar.txt", data=b"x", method="POST")
        self.assertEqual(status, 415)
        self.assertIn("MP3", json.loads(body)["detail"])
        self.assertEqual(call(self.api + "/jobs/yok-boyle-is")[0], 404)


if __name__ == "__main__":
    unittest.main()
