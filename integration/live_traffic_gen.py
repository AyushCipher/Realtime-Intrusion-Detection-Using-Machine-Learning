"""Generate real TCP traffic against a service you own, for the live-capture test.

Run by the `traffic-gen` service in docker-compose.live.yml, inside ingestion's
network namespace, so every connection below crosses the interface ingestion is
sniffing. Only point this at hosts you are authorised to test (here: the demo
stack's own dashboard-api container).

Three phases, each producing a distinct flow shape:
  normal  - a handful of ordinary HTTP GETs on fresh connections
  burst   - many rapid HTTP requests (high packet rate, DoS-like shape)
  probe   - connect() attempts across a port range, most refused (scan-like shape)
Stdlib only.
"""

from __future__ import annotations

import argparse
import socket
import time


def http_get(host: str, port: int, path: str = "/healthz", timeout: float = 3.0) -> None:
    try:
        with socket.create_connection((host, port), timeout=timeout) as s:
            s.sendall(f"GET {path} HTTP/1.1\r\nHost: {host}\r\nConnection: close\r\n\r\n".encode())
            while s.recv(4096):
                pass
    except OSError:
        pass


def probe(host: str, port: int, timeout: float = 0.3) -> None:
    try:
        socket.create_connection((host, port), timeout=timeout).close()
    except OSError:
        pass


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    p.add_argument("--target", required=True)
    p.add_argument("--port", type=int, default=8000)
    p.add_argument("--startup-delay", type=float, default=20.0, help="let the stack come up first")
    p.add_argument("--rounds", type=int, default=3)
    args = p.parse_args()

    print(f"traffic-gen: waiting {args.startup_delay}s, then targeting {args.target}:{args.port}", flush=True)
    time.sleep(args.startup_delay)

    for r in range(args.rounds):
        print(f"traffic-gen: round {r + 1}/{args.rounds}", flush=True)
        for _ in range(10):  # normal
            http_get(args.target, args.port)
            time.sleep(0.3)
        for _ in range(300):  # burst
            http_get(args.target, args.port)
        for port in range(7990, 8100):  # probe
            probe(args.target, port)
        time.sleep(8)
    print("traffic-gen: done", flush=True)


if __name__ == "__main__":
    main()
