"""Tiny Ollama-compatible server used only for local integration testing."""

from __future__ import annotations

import json
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer


class Handler(BaseHTTPRequestHandler):
    def do_POST(self) -> None:  # noqa: N802 - BaseHTTPRequestHandler API
        if self.path != "/api/chat":
            self.send_error(404)
            return
        length = int(self.headers.get("Content-Length", "0"))
        request = json.loads(self.rfile.read(length))
        user_content = request["messages"][-1]["content"]
        words = json.loads(user_content.split("\n", 1)[1])
        evaluations = [
            {
                "word": item["word"],
                "abstractness": 0,
                "semantic_complexity": min(2, len(item["senses"]) // 3),
                "form_complexity": 0,
                "register": 0,
            }
            for item in words
        ]
        body = json.dumps(
            {"message": {"role": "assistant", "content": json.dumps(evaluations)}}
        ).encode("utf-8")
        self.send_response(200)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def log_message(self, format: str, *args: object) -> None:
        return


if __name__ == "__main__":
    ThreadingHTTPServer(("127.0.0.1", 18080), Handler).serve_forever()

