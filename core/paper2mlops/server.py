"""本地 HTTP 服务器 — 交付 HTML 报告、接收用户确认事件。"""

import json
import os
import threading
from http.server import HTTPServer, SimpleHTTPRequestHandler
from pathlib import Path


class ReportHandler(SimpleHTTPRequestHandler):
    """处理报告页面请求和确认事件回传。"""

    report_dir: str = ""
    events_path: str = ""

    def do_POST(self):
        if self.path == "/api/confirm":
            # CSRF protection: only allow same-origin requests from localhost
            origin = self.headers.get("Origin", "")
            if origin and not origin.startswith(("http://127.0.0.1", "http://localhost")):
                self.send_response(403)
                self.end_headers()
                return
            content_length = int(self.headers.get("Content-Length", 0))
            body = self.rfile.read(content_length)
            event = json.loads(body)
            self._write_event(event)
            self.send_response(200)
            self.send_header("Content-Type", "application/json")
            self.end_headers()
            self.wfile.write(json.dumps({"status": "ok"}).encode())
        else:
            self.send_response(404)
            self.end_headers()

    def do_GET(self):
        # 默认返回最新报告
        if self.path == "/" or self.path == "/index.html":
            self._serve_latest_html()
        else:
            super().do_GET()

    def _serve_latest_html(self):
        """找到 report_dir 中最新的 HTML 文件并返回。"""
        html_files = sorted(
            Path(self.report_dir).glob("*.html"),
            key=lambda p: p.stat().st_mtime, reverse=True,
        )
        if html_files:
            self.send_response(200)
            self.send_header("Content-Type", "text/html; charset=utf-8")
            self.end_headers()
            self.wfile.write(html_files[0].read_bytes())
        else:
            self.send_response(200)
            self.send_header("Content-Type", "text/html; charset=utf-8")
            self.end_headers()
            self.wfile.write(b"<h2>No report yet</h2>")

    def _write_event(self, event: dict):
        """将事件追加写入 events.jsonl。"""
        os.makedirs(os.path.dirname(self.events_path), exist_ok=True)
        with open(self.events_path, "a", encoding="utf-8") as f:
            f.write(json.dumps(event, ensure_ascii=False) + "\n")

    def log_message(self, format, *args):
        pass  # 抑制日志，保持安静


def start_server(report_dir: str, state_dir: str, port: int = 65049) -> HTTPServer:
    """
    启动本地 HTTP 服务器。

    参数:
        report_dir: HTML 报告文件存放目录
        state_dir: 事件记录目录（events.jsonl）
        port: 监听端口，默认 65049

    返回:
        HTTPServer 实例
    """
    os.makedirs(report_dir, exist_ok=True)
    os.makedirs(state_dir, exist_ok=True)

    ReportHandler.report_dir = report_dir
    ReportHandler.events_path = os.path.join(state_dir, "events.jsonl")

    server = HTTPServer(("127.0.0.1", port), ReportHandler)

    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()

    return server


def check_confirmation(state_dir: str, timeout_seconds: int = 600) -> dict | None:
    """
    轮询检查用户是否已确认。
    读取 events.jsonl 中最新的确认事件。

    参数:
        state_dir: events.jsonl 所在目录
        timeout_seconds: 超时秒数，默认 10 分钟

    返回:
        确认事件 dict，或超时返回 None
    """
    import time
    events_path = os.path.join(state_dir, "events.jsonl")
    start = time.time()

    while time.time() - start < timeout_seconds:
        if os.path.isfile(events_path):
            with open(events_path, "r", encoding="utf-8") as f:
                for line in f:
                    event = json.loads(line.strip())
                    if event.get("action") == "confirm":
                        return event
                    if event.get("action") == "review_confirm":
                        return event
        time.sleep(2)

    return None


def clear_events(state_dir: str):
    """清空事件文件（进入下一阶段前调用）。"""
    events_path = os.path.join(state_dir, "events.jsonl")
    if os.path.isfile(events_path):
        os.remove(events_path)
