"""CLI: `python -m ids_dashboard` runs the dashboard API with uvicorn.

Configuration comes from environment variables (IDS_DASHBOARD_*, see
config.py and auth.py) rather than CLI flags, matching how a web service is
typically deployed.
"""

from __future__ import annotations

import logging
import os

import uvicorn

from .app import create_app


def main() -> int:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s: %(message)s")
    app = create_app()
    host = os.environ.get("IDS_DASHBOARD_HOST", "0.0.0.0")
    port = int(os.environ.get("IDS_DASHBOARD_PORT", "8000"))
    uvicorn.run(app, host=host, port=port)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
