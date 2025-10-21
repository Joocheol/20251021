"""Simple HTTP server to serve the static dashboard.

Run this script and open the reported URL in your browser to view
``index.html`` along with the supporting assets.
"""
from __future__ import annotations

import argparse
import contextlib
import http.server
import socket
import socketserver
from functools import partial
from pathlib import Path
from typing import Iterator


def iter_addresses(server: socketserver.BaseServer) -> Iterator[str]:
    """Yield URLs where the server can be accessed."""
    host, port = server.server_address
    if host in {"", "0.0.0.0"}:
        # Try to detect a more specific host for local access.
        with contextlib.closing(socket.socket(socket.AF_INET, socket.SOCK_DGRAM)) as s:
            try:
                s.connect(("8.8.8.8", 80))
            except OSError:
                pass
            else:
                host = s.getsockname()[0]
        if host in {"", "0.0.0.0"}:
            host = "127.0.0.1"
    yield f"http://{host}:{port}/"


def main() -> None:
    parser = argparse.ArgumentParser(description="Serve the efficient frontier dashboard")
    parser.add_argument("--bind", default="0.0.0.0", help="Address to bind (default: %(default)s)")
    parser.add_argument("--port", type=int, default=8000, help="Port to listen on (default: %(default)s)")
    parser.add_argument(
        "--directory",
        default=Path(__file__).resolve().parent,
        type=Path,
        help="Directory to serve (default: repository root)",
    )
    args = parser.parse_args()

    handler = partial(http.server.SimpleHTTPRequestHandler, directory=str(args.directory))
    with socketserver.ThreadingTCPServer((args.bind, args.port), handler) as httpd:
        httpd.allow_reuse_address = True
        urls = ", ".join(iter_addresses(httpd))
        print(f"Serving {args.directory} at {urls}")
        try:
            httpd.serve_forever()
        except KeyboardInterrupt:
            print("\nShutting down server.")


if __name__ == "__main__":
    main()
