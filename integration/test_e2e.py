"""End-to-end integration check for the full stack.

Verifies the real path this system exists for: pcap replay (ingestion) ->
flow features on Kafka -> scoring + alert (ml) -> alert on Kafka ->
dashboard-api consumes, stores, and serves it. Run against an already-
running `docker compose up` stack (see the top-level README):

    docker compose up -d --build
    python integration/test_e2e.py

It does not start or stop the stack itself -- keeping it a pure verifier
means it can also be pointed at a stack someone already has running, and a
failed run doesn't leave containers in a half-torn-down state to debug.

What it checks, in order:
  1. dashboard-api's REST API is reachable and authenticates.
  2. An alert whose src_ip matches one of the flows in the ingestion
     module's own bundled demo pcap (ingestion/tests/fixtures/sample_tcp.pcap,
     baked into the ingestion image -- see ingestion/Dockerfile) appears
     within the timeout. This proves the full chain end-to-end: it can only
     be present if ingestion actually extracted and published that flow,
     ml actually scored it, and dashboard-api actually consumed and stored
     it. The specific predicted attack category is intentionally NOT
     asserted on -- that depends on the demo model (trained on a synthetic
     fixture, see ml/README.md) and isn't part of the wiring being tested.
  3. The alert has a non-empty SHAP `explanation` (proves the ml module's
     explainability path ran, not just the bare classifier).
  4. The alert is reachable via `GET /api/alerts/{id}` (the same lookup the
     dashboard frontend's explainability panel uses).
"""

from __future__ import annotations

import argparse
import base64
import json
import sys
import time
import urllib.error
import urllib.request

# The two flows in ingestion/tests/fixtures/sample_tcp.pcap, baked into the
# ingestion Docker image as its demo replay source (see ingestion/Dockerfile
# and ingestion/tests/generate_fixtures.py for how this fixture was built).
KNOWN_DEMO_SRC_IPS = {"10.0.0.1", "10.0.0.5"}


def _get(url: str, username: str, password: str, timeout: float = 5.0):
    req = urllib.request.Request(url)
    token = base64.b64encode(f"{username}:{password}".encode()).decode()
    req.add_header("Authorization", f"Basic {token}")
    with urllib.request.urlopen(req, timeout=timeout) as resp:
        return json.loads(resp.read().decode())


def wait_for_healthz(base_url: str, timeout_s: float) -> None:
    deadline = time.monotonic() + timeout_s
    last_error = None
    while time.monotonic() < deadline:
        try:
            with urllib.request.urlopen(f"{base_url}/healthz", timeout=3.0) as resp:
                if resp.status == 200:
                    print(f"[ok] dashboard-api healthz reachable at {base_url}")
                    return
        except Exception as exc:  # noqa: BLE001
            last_error = exc
        time.sleep(2.0)
    raise TimeoutError(f"dashboard-api never became reachable at {base_url}: {last_error}")


def wait_for_demo_alert(base_url: str, username: str, password: str, timeout_s: float) -> dict:
    deadline = time.monotonic() + timeout_s
    last_seen_total = 0
    while time.monotonic() < deadline:
        try:
            body = _get(f"{base_url}/api/alerts?limit=100", username, password)
        except urllib.error.HTTPError as exc:
            if exc.code == 401:
                raise RuntimeError(
                    "dashboard-api rejected the given credentials (401) -- check "
                    "--username/--password match IDS_DASHBOARD_USERNAME/PASSWORD"
                ) from exc
            last_seen_total = -1
            time.sleep(2.0)
            continue

        last_seen_total = body["total"]
        for alert in body["alerts"]:
            if alert["src_ip"] in KNOWN_DEMO_SRC_IPS:
                return alert
        time.sleep(2.0)

    raise TimeoutError(
        f"no alert from the known demo pcap's flows ({KNOWN_DEMO_SRC_IPS}) appeared "
        f"within {timeout_s}s (saw {last_seen_total} alert(s) total, none matching)"
    )


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--base-url", default="http://localhost:8000")
    parser.add_argument("--username", default="analyst")
    parser.add_argument("--password", default="changeme123")
    parser.add_argument("--startup-timeout", type=float, default=60.0, help="seconds to wait for dashboard-api to come up")
    parser.add_argument("--alert-timeout", type=float, default=120.0, help="seconds to wait for the expected alert")
    args = parser.parse_args()

    try:
        wait_for_healthz(args.base_url, args.startup_timeout)

        print(f"[..] waiting up to {args.alert_timeout:.0f}s for an alert from the demo pcap's known flows")
        alert = wait_for_demo_alert(args.base_url, args.username, args.password, args.alert_timeout)
        print(
            f"[ok] found alert {alert['alert_id']} src_ip={alert['src_ip']} "
            f"predicted_class={alert['stage2_predicted_class']!r} severity={alert['severity']!r}"
        )

        if not alert["explanation"]:
            print("[FAIL] alert has no SHAP explanation attached")
            return 1
        print(f"[ok] alert carries a {len(alert['explanation'])}-feature SHAP explanation")

        fetched = _get(f"{args.base_url}/api/alerts/{alert['alert_id']}", args.username, args.password)
        if fetched["alert_id"] != alert["alert_id"]:
            print("[FAIL] GET /api/alerts/{id} did not return the same alert")
            return 1
        print("[ok] alert is individually retrievable via GET /api/alerts/{id}")

    except (TimeoutError, RuntimeError) as exc:
        print(f"[FAIL] {exc}")
        return 1

    print("\nEnd-to-end check passed: pcap replay -> ingestion -> Kafka -> ml -> Kafka -> dashboard-api -> REST API.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
