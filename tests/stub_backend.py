"""Minimal standalone OpenAI-compatible stub backend for smoke-testing the
orchestrator end to end without any real models. stdlib only.

Usage:
    python stub_backend.py --port 8091 --model-name ornith-1.0-35b

Run one instance per configured backend port (8090/8091/8092/8093) and the
orchestrator's health probes will see them as resident.
"""
from __future__ import annotations

import argparse
import json
import time
import uuid
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer


def _chat_completion(model: str, port: int) -> dict:
    content = f"stub response from port {port} model {model}"
    now = int(time.time())
    return {
        "id": f"chatcmpl-stub-{uuid.uuid4().hex[:24]}",
        "object": "chat.completion",
        "created": now,
        "model": model,
        "choices": [
            {
                "index": 0,
                "message": {"role": "assistant", "content": content},
                "finish_reason": "stop",
            }
        ],
        "usage": {"prompt_tokens": 0, "completion_tokens": 0, "total_tokens": 0},
    }


def _text_completion(model: str, port: int) -> dict:
    text = f"stub response from port {port} model {model}"
    now = int(time.time())
    return {
        "id": f"cmpl-stub-{uuid.uuid4().hex[:24]}",
        "object": "text_completion",
        "created": now,
        "model": model,
        "choices": [{"index": 0, "text": text, "finish_reason": "stop", "logprobs": None}],
        "usage": {"prompt_tokens": 0, "completion_tokens": 0, "total_tokens": 0},
    }


def _chat_sse_chunks(model: str, port: int) -> list[bytes]:
    cid = f"chatcmpl-stub-{uuid.uuid4().hex[:24]}"
    now = int(time.time())
    base = {"id": cid, "object": "chat.completion.chunk", "created": now, "model": model}
    content = f"stub response from port {port} model {model}"

    def sse(data: dict) -> bytes:
        return b"data: " + json.dumps(data, separators=(",", ":")).encode() + b"\n\n"

    chunks = [
        sse({**base, "choices": [{"index": 0, "delta": {"role": "assistant"}, "finish_reason": None}]}),
        sse({**base, "choices": [{"index": 0, "delta": {"content": content}, "finish_reason": None}]}),
        sse({**base, "choices": [{"index": 0, "delta": {}, "finish_reason": "stop"}]}),
        b"data: [DONE]\n\n",
    ]
    return chunks


def _completion_sse_chunks(model: str, port: int) -> list[bytes]:
    cid = f"cmpl-stub-{uuid.uuid4().hex[:24]}"
    now = int(time.time())
    base = {"id": cid, "object": "text_completion", "created": now, "model": model}
    text = f"stub response from port {port} model {model}"

    def sse(data: dict) -> bytes:
        return b"data: " + json.dumps(data, separators=(",", ":")).encode() + b"\n\n"

    chunks = [
        sse({**base, "choices": [{"index": 0, "text": text, "finish_reason": None, "logprobs": None}]}),
        sse({**base, "choices": [{"index": 0, "text": "", "finish_reason": "stop", "logprobs": None}]}),
        b"data: [DONE]\n\n",
    ]
    return chunks


def make_handler(model_name: str, port: int) -> type[BaseHTTPRequestHandler]:
    class Handler(BaseHTTPRequestHandler):
        def log_message(self, fmt: str, *args) -> None:
            pass

        def _send_json(self, payload: dict, status: int = 200) -> None:
            body = json.dumps(payload).encode()
            self.send_response(status)
            self.send_header("Content-Type", "application/json")
            self.send_header("Content-Length", str(len(body)))
            self.end_headers()
            self.wfile.write(body)

        def _send_sse(self, chunks: list[bytes]) -> None:
            self.send_response(200)
            self.send_header("Content-Type", "text/event-stream")
            self.send_header("Cache-Control", "no-cache")
            self.end_headers()
            for chunk in chunks:
                self.wfile.write(chunk)
                self.wfile.flush()

        def _read_json(self) -> dict:
            length = int(self.headers.get("Content-Length") or 0)
            if not length:
                return {}
            raw = self.rfile.read(length)
            return json.loads(raw) if raw else {}

        def do_GET(self) -> None:
            if self.path.rstrip("/") == "/v1/models":
                now = int(time.time())
                self._send_json(
                    {
                        "object": "list",
                        "data": [{"id": model_name, "object": "model", "created": now, "owned_by": "stub"}],
                    }
                )
            else:
                self._send_json({"error": {"message": "not found", "type": "not_found"}}, status=404)

        def do_POST(self) -> None:
            path = self.path.rstrip("/")
            try:
                body = self._read_json()
            except Exception:
                self._send_json({"error": {"message": "invalid JSON", "type": "invalid_request"}}, status=400)
                return

            stream = bool(body.get("stream"))

            if path == "/v1/chat/completions":
                if stream:
                    self._send_sse(_chat_sse_chunks(model_name, port))
                else:
                    self._send_json(_chat_completion(model_name, port))
            elif path == "/v1/completions":
                if stream:
                    self._send_sse(_completion_sse_chunks(model_name, port))
                else:
                    self._send_json(_text_completion(model_name, port))
            else:
                self._send_json({"error": {"message": "not found", "type": "not_found"}}, status=404)

    return Handler


def main() -> None:
    parser = argparse.ArgumentParser(description="stub OpenAI-compatible backend for smoke tests")
    parser.add_argument("--port", type=int, required=True)
    parser.add_argument("--model-name", type=str, required=True)
    parser.add_argument("--host", type=str, default="127.0.0.1")
    args = parser.parse_args()

    handler = make_handler(args.model_name, args.port)
    server = ThreadingHTTPServer((args.host, args.port), handler)
    print(f"stub_backend: serving model={args.model_name!r} on http://{args.host}:{args.port}")
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        pass
    finally:
        server.server_close()


if __name__ == "__main__":
    main()
