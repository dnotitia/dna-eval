"""A scripted OpenAI-compatible server for parity tests (no GPU, no network).

The subject's reply is chosen by the user message ``CASE:<name>`` (see ``SUBJECT_REPLIES``).
A judge call is any request whose prompt contains ``JUDGE:<name>;`` (see ``JUDGE_REPLIES``,
extendable per server through ``judge_replies``); every other judge prompt gets a valid
verdict. A reply may be a ``sequence``: the n-th request for that name gets the n-th entry
(the last one repeats). All request bodies are recorded.

Used by the tests in this directory; keep it standard-library only.
"""
from __future__ import annotations

import json
import re
import threading
import time
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

VERDICT = '{"score": 8, "reason": "ok"}'

# What the model under evaluation sends back, per case.
SUBJECT_REPLIES: dict[str, dict] = {
    "plain": {"content": "답변입니다."},
    "surrounding_whitespace": {"content": "  답변입니다.\n\n"},
    "reasoning_content_field": {"content": "\n\n답변입니다.", "reasoning_content": "생각"},
    "reasoning_field": {"content": "답변입니다.", "reasoning": "생각"},
    "think_tag": {"content": "<think>\n생각\n</think>\n\n답변입니다."},
    "think_unclosed": {"content": "<think>\n생각만 하다가 잘림", "finish_reason": "length"},
    "reasoning_only": {"content": None, "reasoning_content": "생각만 하다가 잘림", "finish_reason": "length"},
    "empty": {"content": ""},
    "http_400": {"status": 400},
    "http_500": {"status": 500},
    "slow": {"content": "늦은 답변", "delay": 3.0},
    "http_500_then_answer": {"sequence": [{"status": 500}, {"content": "재시도 후 답변"}]},
    "empty_then_answer": {"sequence": [{"content": ""}, {"content": "재시도 후 답변"}]},
}

# What the judge sends back when its prompt carries the marker.
JUDGE_REPLIES: dict[str, dict] = {
    "judge_http_500": {"status": 500},
    "judge_unparsable": {"content": "점수는 8점입니다."},
    "judge_ok": {"content": VERDICT},
}


_JUDGE_MARKER = re.compile(r"JUDGE:(\w+);")


def _text(message: dict) -> str:
    content = message.get("content") or ""
    return content if isinstance(content, str) else "".join(p.get("text", "") for p in content)


class FakeOpenAI:
    """``with FakeOpenAI() as srv:`` serves on ``srv.base_url`` and records ``srv.requests``."""

    def __init__(self) -> None:
        self.requests: list[dict] = []
        self.judge_replies: dict[str, dict] = dict(JUDGE_REPLIES)
        self._seen: dict[str, int] = {}
        server = self

        class Handler(BaseHTTPRequestHandler):
            def log_message(self, *args):  # silence
                pass

            def do_GET(self):
                self._send(200, {"object": "list", "data": [{"id": "fake", "object": "model"}]})

            def do_POST(self):
                body = json.loads(self.rfile.read(int(self.headers["Content-Length"])))
                server.requests.append(body)
                prompt = _text(body["messages"][-1])
                if "Evaluation Steps" in prompt:
                    m = _JUDGE_MARKER.search(prompt)
                    name = m.group(1) if m and m.group(1) in server.judge_replies else "judge_ok"
                    reply = server.judge_replies[name]
                else:
                    name = prompt.removeprefix("CASE:")
                    reply = SUBJECT_REPLIES.get(name, SUBJECT_REPLIES["plain"])
                if "sequence" in reply:
                    n = server._seen.get(name, 0)
                    server._seen[name] = n + 1
                    reply = reply["sequence"][min(n, len(reply["sequence"]) - 1)]
                if reply.get("delay"):
                    time.sleep(reply["delay"])
                if "status" in reply:
                    return self._send(reply["status"], {"error": {"message": "scripted", "type": "fake"}})
                message = {"role": "assistant", "content": reply.get("content")}
                for key in ("reasoning_content", "reasoning"):
                    if key in reply:
                        message[key] = reply[key]
                self._send(200, {
                    "id": "fake", "object": "chat.completion", "created": int(time.time()),
                    "model": body.get("model"),
                    "choices": [{"index": 0, "message": message,
                                 "finish_reason": reply.get("finish_reason", "stop")}],
                    "usage": {"prompt_tokens": 1, "completion_tokens": 1, "total_tokens": 2},
                })

            def _send(self, code, obj):
                data = json.dumps(obj, ensure_ascii=False).encode()
                try:
                    self.send_response(code)
                    self.send_header("Content-Type", "application/json")
                    self.send_header("Content-Length", str(len(data)))
                    self.end_headers()
                    self.wfile.write(data)
                except (BrokenPipeError, ConnectionResetError):  # client gave up (timeout)
                    pass

        self._httpd = ThreadingHTTPServer(("127.0.0.1", 0), Handler)
        self._httpd.daemon_threads = True
        self.base_url = f"http://127.0.0.1:{self._httpd.server_address[1]}/v1"

    def subject_requests(self, case: str) -> list[dict]:
        return [b for b in self.requests if _text(b["messages"][-1]) == f"CASE:{case}"]

    def judge_prompts(self) -> list[str]:
        return [_text(b["messages"][-1]) for b in self.requests if "Evaluation Steps" in _text(b["messages"][-1])]

    def __enter__(self) -> "FakeOpenAI":
        threading.Thread(target=self._httpd.serve_forever, daemon=True).start()
        return self

    def __exit__(self, *exc) -> None:
        self._httpd.shutdown()
        self._httpd.server_close()
