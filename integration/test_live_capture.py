"""Verify the live-capture path end to end against a running live stack.

Start the stack with the live override first (see docker-compose.live.yml):

    docker compose -f docker-compose.yml -f docker-compose.live.yml up -d --build
    python integration/test_live_capture.py

Unlike test_e2e.py (which looks for the bundled demo pcap's 10.0.0.x flows),
this looks for alerts whose endpoints are NOT in the demo pcap. Such an alert
can only exist if ingestion sniffed a real packet on a live interface, built a
flow from it, published it to Kafka, ml scored it, and dashboard-api stored it.
The predicted attack classes are only reported, not asserted: the bundled model
is trained on a synthetic fixture (see ml/README.md).
"""

from __future__ import annotations

import argparse
import base64
import collections
import json
import sys
import time
import urllib.request

DEMO_PCAP_IPS_PREFIX = "10.0.0."


def get(url: str, user: str, password: str):
    req = urllib.request.Request(url)
    req.add_header("Authorization", "Basic " + base64.b64encode(f"{user}:{password}".encode()).decode())
    with urllib.request.urlopen(req, timeout=5.0) as resp:
        return json.loads(resp.read().decode())


def live_alerts(base_url: str, user: str, password: str) -> list:
    body = get(f"{base_url}/api/alerts?limit=500", user, password)
    return [
        a
        for a in body["alerts"]
        if not a["src_ip"].startswith(DEMO_PCAP_IPS_PREFIX) and not a["dst_ip"].startswith(DEMO_PCAP_IPS_PREFIX)
    ]


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    p.add_argument("--base-url", default="http://localhost:8000")
    p.add_argument("--username", default="analyst")
    p.add_argument("--password", default="changeme123")
    p.add_argument("--timeout", type=float, default=180.0, help="seconds to wait for live-capture alerts")
    p.add_argument("--min-alerts", type=int, default=1)
    args = p.parse_args()

    deadline = time.monotonic() + args.timeout
    alerts: list = []
    while time.monotonic() < deadline:
        try:
            alerts = live_alerts(args.base_url, args.username, args.password)
        except Exception as exc:  # noqa: BLE001 - stack may still be starting
            print(f"[..] dashboard-api not ready: {exc}")
            alerts = []
        if len(alerts) >= args.min_alerts:
            break
        time.sleep(5.0)

    if len(alerts) < args.min_alerts:
        print(f"[FAIL] saw {len(alerts)} live-capture alert(s) within {args.timeout:.0f}s (wanted >= {args.min_alerts})")
        return 1

    by_class = collections.Counter(a["stage2_predicted_class"] for a in alerts)
    escalated = sum(1 for a in alerts if a.get("escalated"))
    print(f"[ok] {len(alerts)} alert(s) from live-captured flows (none from the demo pcap)")
    print(f"[ok] predicted classes: {dict(by_class)}; escalated to Tier 2: {escalated}")
    for a in alerts[:5]:
        print(
            f"     {a['src_ip']}:{a.get('src_port')} -> {a['dst_ip']}:{a.get('dst_port')} "
            f"{a['stage2_predicted_class']} severity={a['severity']}"
        )
    print("\nLive capture check passed: real interface -> ingestion -> Kafka -> ml -> Kafka -> dashboard-api -> REST API.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
