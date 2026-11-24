"""Verify Tier 2 end to end against a running stack with a REAL LLM.

Start the stack with a real LLM client first, e.g. the local open-weights model:

    docker compose -f docker-compose.yml -f docker-compose.realdata.yml up -d --build
    python integration/test_tier2_live.py

Passes only if (1) the ml service publishes escalated alerts, (2) tier2-reasoner
explains them, (3) dashboard-api stores the explanations, and (4) they came from
a real model, not the deterministic stub (model_version is not the stub's).
Reports latency, how many explanations name an ATT&CK technique, and what was
retrieved, so the quality of a live run is visible, not just its plumbing.
Correctness of the named technique cannot be judged here: live alerts carry no
ground-truth family (see tier2_reasoner/scripts/run_h3_real.py for that).
"""

from __future__ import annotations

import argparse
import base64
import json
import statistics
import sys
import time
import urllib.error
import urllib.request


def get(url: str, user: str, password: str):
    req = urllib.request.Request(url)
    req.add_header("Authorization", "Basic " + base64.b64encode(f"{user}:{password}".encode()).decode())
    with urllib.request.urlopen(req, timeout=10.0) as resp:
        return json.loads(resp.read().decode())


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--base-url", default="http://localhost:8000")
    ap.add_argument("--user", default="analyst")
    ap.add_argument("--password", default="changeme123")
    ap.add_argument("--want", type=int, default=5, help="real-LLM explanations required to pass")
    ap.add_argument("--timeout-s", type=float, default=900.0)
    args = ap.parse_args()

    deadline = time.time() + args.timeout_s
    explained = {}
    n_escalated = 0
    while time.time() < deadline and len(explained) < args.want:
        try:
            alerts = get(f"{args.base_url}/api/alerts?limit=500", args.user, args.password)["alerts"]
        except (urllib.error.URLError, OSError) as exc:
            print(f"[..] dashboard-api not ready: {exc}")
            time.sleep(10)
            continue
        escalated = [a for a in alerts if a.get("escalated")]
        n_escalated = len(escalated)
        for a in escalated:
            if a["alert_id"] in explained:
                continue
            try:
                explained[a["alert_id"]] = (a, get(f"{args.base_url}/api/alerts/{a['alert_id']}/explanation", args.user, args.password))
            except urllib.error.HTTPError as exc:
                if exc.code != 404:  # 404 = not explained yet
                    raise
        print(f"[..] {len(alerts)} alerts, {n_escalated} escalated, {len(explained)} explained")
        if len(explained) < args.want:
            time.sleep(15)

    if not explained:
        print(f"[FAIL] no explanations within {args.timeout_s:.0f}s ({n_escalated} escalated alerts seen)")
        return 1
    exps = [e for _, e in explained.values()]
    models = sorted({e["model_version"] for e in exps})
    real = [e for e in exps if "stub" not in e["model_version"].lower()]
    latencies = [e["llm_latency_ms"] for e in real]
    named = [e for e in real if e["suspected_technique_id"]]
    print(f"[ok] {len(exps)} explanations stored; model(s): {models}")
    if real:
        print(f"[ok] real-LLM explanations: {len(real)}; latency median {statistics.median(latencies)/1000:.1f}s, "
              f"max {max(latencies)/1000:.1f}s; technique named in {len(named)}/{len(real)}")
    for a, e in list(explained.values())[:5]:
        print(f"     predicted={a.get('stage2_predicted_class')!s:12} retrieved={e['retrieved_technique_ids']} "
              f"-> {e['suspected_technique_id'] or '(none)'}: {e['risk_explanation'][:90]!r}")
    if len(real) < args.want:
        print(f"[FAIL] wanted {args.want} real-LLM explanations, got {len(real)} (stub in use?)")
        return 1
    print("\nTier 2 live check passed: escalated alert -> tier2-reasoner (real LLM) -> Kafka -> dashboard-api -> REST API.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
