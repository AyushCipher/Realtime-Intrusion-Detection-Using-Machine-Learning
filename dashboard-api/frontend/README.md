# IDS Alert Dashboard (frontend)

React + TypeScript + Vite dashboard for the `ids_dashboard` FastAPI backend
(`../src/ids_dashboard`). Consumes the backend's REST and WebSocket API only
-- no direct Kafka/DB access from the browser.

## Setup

```
npm install
cp .env.example .env   # set VITE_API_BASE_URL if the backend isn't on localhost:8000
npm run dev
```

Requires the backend running (see the top-level README's dashboard section
for how to start it with `--use-stub` or a real Kafka broker) and CORS on
the backend configured to allow this dev server's origin
(`IDS_DASHBOARD_CORS_ORIGINS`, default `*`).

## What's here

- `src/api.ts` -- REST client + WebSocket connection helper. Credentials are
  kept in memory only (component state), never written to
  localStorage/sessionStorage.
- `src/types.ts` -- the alert schema, mirrored from
  `ids_dashboard/schema.py`.
- `src/components/LoginForm.tsx` -- Basic-auth sign-in.
- `src/components/LiveAlertFeed.tsx` -- WebSocket-driven live feed (connects
  via a short-lived token from `POST /api/ws-token`, since browsers can't
  set an `Authorization` header on a WebSocket handshake).
- `src/components/TriageView.tsx` -- alerts grouped by severity with
  acknowledge/confirm/false-positive actions.
- `src/components/ExplainabilityPanel.tsx` -- renders an alert's SHAP
  feature-contribution chart and class probabilities.
- `src/components/SummaryView.tsx` -- alert-volume and false-positive-rate
  charts.

## Build

```
npm run build   # tsc -b && vite build
npm run lint    # oxlint
```

## Known limitation

The production build's JS bundle is ~580KB (mostly the `recharts`/`d3`
charting dependency used only by the Summary view). Not code-split; fine
for an internal analyst dashboard, worth revisiting with route-based
lazy-loading if this grows.
