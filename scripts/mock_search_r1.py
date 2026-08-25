#!/usr/bin/env python3
"""Tiny local Search-R1-compatible server for MemGen protocol smoke tests."""

from __future__ import annotations

import argparse
import json
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer


DOCUMENTS = [
    {
        "contents": (
            "Sunset Boulevard\n"
            "Andrew Lloyd Webber's musical Sunset Boulevard opened in the United States "
            "on 10 December 1993."
        )
    },
    {
        "contents": (
            "Henry Campbell-Bannerman\n"
            "Henry Campbell-Bannerman succeeded Arthur Balfour as Prime Minister of the "
            "United Kingdom in 1905."
        )
    },
    {
        "contents": (
            "Kiss You All Over\n"
            "Kiss You All Over was a number-one hit for the American band Exile in 1978."
        )
    },
]


class Handler(BaseHTTPRequestHandler):
    def do_POST(self):
        if self.path != "/retrieve":
            self.send_error(404)
            return
        try:
            length = int(self.headers.get("Content-Length", "0"))
            payload = json.loads(self.rfile.read(length))
            queries = payload["queries"]
            topk = int(payload.get("topk", 3))
            result = []
            for _query in queries:
                result.append(
                    [
                        {"document": document, "score": 1.0 / (index + 1)}
                        for index, document in enumerate(DOCUMENTS[:topk])
                    ]
                )
            body = json.dumps({"result": result}).encode()
            self.send_response(200)
            self.send_header("Content-Type", "application/json")
            self.send_header("Content-Length", str(len(body)))
            self.end_headers()
            self.wfile.write(body)
        except Exception as exc:
            self.send_error(400, str(exc))

    def log_message(self, format, *args):
        print(format % args, flush=True)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=8001)
    args = parser.parse_args()
    server = ThreadingHTTPServer((args.host, args.port), Handler)
    print(f"MOCK_SEARCH_R1_READY http://{args.host}:{args.port}/retrieve", flush=True)
    server.serve_forever()


if __name__ == "__main__":
    main()
