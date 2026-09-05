"""Minimal OpenAI-compatible HTTP server for tests: chat completions and embeddings.

Records every request body, can fail the first N requests with a chosen status (to
exercise retries) and answers with a fixed or computed reply. Embeddings are
deterministic hashes of the input so retrieval tests are reproducible.
"""

from __future__ import annotations

import hashlib
import json
import math
import threading
from collections.abc import Callable
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from typing import Any

Reply = str | Callable[[dict[str, Any]], str]


def fake_embedding(text: str, dim: int) -> list[float]:
    digest = hashlib.sha256(text.encode("utf-8")).digest()
    raw = [(digest[i % len(digest)] - 128) / 128 for i in range(dim)]
    norm = math.sqrt(sum(v * v for v in raw)) or 1.0
    return [v / norm for v in raw]


class FakeOpenAIServer:
    def __init__(
        self,
        reply: Reply = "ok",
        fail_times: int = 0,
        fail_status: int = 500,
        embedding_dim: int = 8,
    ) -> None:
        self.reply = reply
        self.fail_times = fail_times
        self.fail_status = fail_status
        self.embedding_dim = embedding_dim
        self.requests: list[dict[str, Any]] = []
        self.auth_headers: list[str | None] = []
        self._lock = threading.Lock()
        outer = self

        class Handler(BaseHTTPRequestHandler):
            def log_message(self, *_args: Any) -> None:  # silence
                pass

            def do_POST(self) -> None:
                length = int(self.headers.get("Content-Length", "0"))
                body = json.loads(self.rfile.read(length) or b"{}")
                with outer._lock:
                    outer.requests.append({"path": self.path, **body})
                    outer.auth_headers.append(self.headers.get("Authorization"))
                    must_fail = outer.fail_times > 0
                    if must_fail:
                        outer.fail_times -= 1
                if must_fail:
                    self._send(
                        outer.fail_status, {"error": {"message": "boom", "type": "server_error"}}
                    )
                    return
                if self.path.endswith("/chat/completions"):
                    text = outer.reply(body) if callable(outer.reply) else outer.reply
                    self._send(
                        200,
                        {
                            "id": "chatcmpl-fake",
                            "object": "chat.completion",
                            "created": 0,
                            "model": body.get("model", "fake"),
                            "choices": [
                                {
                                    "index": 0,
                                    "message": {"role": "assistant", "content": text},
                                    "finish_reason": "stop",
                                }
                            ],
                            "usage": {
                                "prompt_tokens": len(json.dumps(body)) // 4,
                                "completion_tokens": max(1, len(text) // 4),
                                "total_tokens": len(json.dumps(body)) // 4 + max(1, len(text) // 4),
                            },
                        },
                    )
                elif self.path.endswith("/embeddings"):
                    inputs = body.get("input")
                    texts = [inputs] if isinstance(inputs, str) else list(inputs or [])
                    self._send(
                        200,
                        {
                            "object": "list",
                            "data": [
                                {
                                    "object": "embedding",
                                    "index": i,
                                    "embedding": fake_embedding(t, outer.embedding_dim),
                                }
                                for i, t in enumerate(texts)
                            ],
                            "model": body.get("model", "fake-embed"),
                            "usage": {
                                "prompt_tokens": sum(len(t) for t in texts) // 4,
                                "total_tokens": 0,
                            },
                        },
                    )
                else:
                    self._send(404, {"error": {"message": f"unknown path {self.path}"}})

            def _send(self, status: int, payload: dict[str, Any]) -> None:
                data = json.dumps(payload, ensure_ascii=False).encode("utf-8")
                self.send_response(status)
                self.send_header("Content-Type", "application/json")
                self.send_header("Content-Length", str(len(data)))
                self.end_headers()
                self.wfile.write(data)

        self._server = ThreadingHTTPServer(("127.0.0.1", 0), Handler)
        self._thread = threading.Thread(target=self._server.serve_forever, daemon=True)

    @property
    def base_url(self) -> str:
        host, port = self._server.server_address[:2]
        return f"http://{host}:{port}/v1"

    def __enter__(self) -> FakeOpenAIServer:
        self._thread.start()
        return self

    def __exit__(self, *_exc: object) -> None:
        self._server.shutdown()
        self._server.server_close()


class HashEmbeddings:
    """Offline EmbeddingProvider for tests: deterministic hash vectors."""

    def __init__(self, dim: int = 8, provider: str = "fake", model: str = "hash") -> None:
        self.dim = dim
        self.provider = provider
        self.model = model
        self.calls: list[tuple[str, int]] = []

    @property
    def identity(self) -> Any:
        from twin.core.embeddings import EmbeddingIdentity

        return EmbeddingIdentity(self.provider, self.model, self.dim)

    def embed(self, texts: list[str], kind: str) -> list[list[float]]:
        self.calls.append((kind, len(texts)))
        return [fake_embedding(t, self.dim) for t in texts]
