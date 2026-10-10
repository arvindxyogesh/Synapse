"""Pure helpers in gpu_session.py (the rest needs a GPU)."""

from pathlib import Path

import gpu_session


def test_pip_cuda_home_finds_toolkit_next_to_vllm(tmp_path):
    env = tmp_path / "envs" / "vllm"
    nvcc = env / "lib" / "python3.12" / "site-packages" / "nvidia" / "cu13" / "bin" / "nvcc"
    nvcc.parent.mkdir(parents=True)
    nvcc.write_text("")
    (env / "bin").mkdir()
    assert gpu_session.pip_cuda_home(str(env / "bin" / "vllm")) == nvcc.parent.parent


def test_pip_cuda_home_none_without_toolkit_or_path(tmp_path):
    (tmp_path / "bin").mkdir()
    assert gpu_session.pip_cuda_home(str(tmp_path / "bin" / "vllm")) is None
    assert gpu_session.pip_cuda_home("vllm") is None


def test_vllm_env_prefers_existing_cuda_home(monkeypatch, tmp_path):
    monkeypatch.setenv("CUDA_HOME", "/opt/cuda-from-module")
    cfg = gpu_session.SessionConfig(mode="smoke", results_dir=tmp_path, gateway_python="python",
                                    gpu_label="test", gpu_price_per_hour=None, vllm_bin=str(tmp_path / "bin" / "vllm"))
    env = gpu_session._vllm_env(cfg)
    assert env["CUDA_HOME"] == "/opt/cuda-from-module"
    assert env["PATH"].split(":")[0] == str(Path(cfg.vllm_bin).parent)


def test_vllm_env_turns_off_flashinfer_sampler_and_usage_stats(tmp_path):
    cfg = gpu_session.SessionConfig(mode="smoke", results_dir=tmp_path, gateway_python="python",
                                    gpu_label="test", gpu_price_per_hour=None)
    env = gpu_session._vllm_env(cfg)
    assert env["VLLM_USE_FLASHINFER_SAMPLER"] == "0"
    assert env["VLLM_NO_USAGE_STATS"] == "1" and env["DO_NOT_TRACK"] == "1"


# -- server lifecycle: the full run's AWQ variant died because something other
# -- than its own vLLM answered its health check ------------------------------

import socket  # noqa: E402
import subprocess  # noqa: E402
import sys  # noqa: E402
import threading  # noqa: E402
import time  # noqa: E402
from http.server import BaseHTTPRequestHandler, HTTPServer  # noqa: E402

import pytest  # noqa: E402


def _free_port() -> int:
    with socket.socket() as s:
        s.bind(("127.0.0.1", 0))
        return s.getsockname()[1]


def test_wait_http_retries_through_a_garbled_reply_then_succeeds():
    port = _free_port()
    listener = socket.socket()
    listener.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
    listener.bind(("127.0.0.1", port))
    listener.listen()

    def garbage_then_http():
        conn, _ = listener.accept()  # 1st poll: echo the request back (-> BadStatusLine)
        conn.sendall(conn.recv(1024))
        conn.close()
        conn, _ = listener.accept()  # 2nd poll: a proper 200
        conn.recv(1024)
        conn.sendall(b"HTTP/1.1 200 OK\r\nContent-Length: 0\r\nConnection: close\r\n\r\n")
        conn.close()

    threading.Thread(target=garbage_then_http, daemon=True).start()
    gpu_session._wait_http(f"http://127.0.0.1:{port}/health", timeout_s=20)  # must not raise
    listener.close()


class _ModelsHandler(BaseHTTPRequestHandler):
    served = ["Qwen/Qwen2.5-7B-Instruct"]

    def do_GET(self):  # noqa: N802
        body = ('{"data": [' + ",".join(f'{{"id": "{m}"}}' for m in self.served) + "]}").encode()
        self.send_response(200)
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def log_message(self, *args):
        pass


def test_check_serves_rejects_a_different_model():
    server = HTTPServer(("127.0.0.1", 0), _ModelsHandler)
    threading.Thread(target=server.serve_forever, daemon=True).start()
    url = f"http://127.0.0.1:{server.server_address[1]}"
    try:
        gpu_session._check_serves(url, "Qwen/Qwen2.5-7B-Instruct")  # ok
        with pytest.raises(RuntimeError, match="expected"):
            gpu_session._check_serves(url, "Qwen/Qwen2.5-7B-Instruct-AWQ")  # the previous variant's server
    finally:
        server.shutdown()


def test_wait_port_free_refuses_when_port_stays_busy():
    with socket.socket() as busy:
        busy.bind(("127.0.0.1", 0))
        busy.listen()
        port = busy.getsockname()[1]
        assert gpu_session._port_free(port) is False
        with pytest.raises(RuntimeError, match="still in use"):
            gpu_session._wait_port_free(port, timeout_s=1)
    assert gpu_session._port_free(port) is True


def test_stop_kills_the_whole_process_group(tmp_path):
    # Parent starts a long-lived child (like vLLM's engine process), then both wait.
    pid_file = tmp_path / "child.pid"
    script = ("import subprocess, sys, time; "
              "c = subprocess.Popen([sys.executable, '-c', 'import time; time.sleep(120)']); "
              f"open({str(pid_file)!r}, 'w').write(str(c.pid)); time.sleep(120)")
    parent = subprocess.Popen([sys.executable, "-c", script], start_new_session=True)
    deadline = time.monotonic() + 10
    while not pid_file.exists() and time.monotonic() < deadline:
        time.sleep(0.05)
    child_pid = int(pid_file.read_text())

    gpu_session._stop([parent])

    assert parent.poll() is not None
    time.sleep(0.2)
    with pytest.raises(ProcessLookupError):
        import os
        os.kill(child_pid, 0)  # the child must be gone too, not orphaned
