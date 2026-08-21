// Thin REST/WebSocket client for the dashboard API (src/ids_dashboard).
// Credentials are held in memory only (see Auth.tsx) -- never written to
// localStorage/sessionStorage, so they don't persist across a tab reload.
// That is a deliberate, minimal trade-off consistent with this project's
// documented "Basic auth is a known-minimal gap" stance; see the README.

import type { Alert, AlertListResponse, SummaryResponse, TriageStatus } from "./types";

export const API_BASE_URL: string = (import.meta.env.VITE_API_BASE_URL as string) || "http://localhost:8000";
const WS_BASE_URL: string = API_BASE_URL.replace(/^http/, "ws");

export interface Credentials {
  username: string;
  password: string;
}

export class ApiError extends Error {
  status: number;
  constructor(status: number, message: string) {
    super(message);
    this.status = status;
  }
}

function authHeader(creds: Credentials): string {
  return "Basic " + btoa(`${creds.username}:${creds.password}`);
}

async function request<T>(path: string, creds: Credentials, init: RequestInit = {}): Promise<T> {
  const resp = await fetch(`${API_BASE_URL}${path}`, {
    ...init,
    headers: {
      ...(init.headers || {}),
      Authorization: authHeader(creds),
      ...(init.body ? { "Content-Type": "application/json" } : {}),
    },
  });
  if (!resp.ok) {
    const body = await resp.text().catch(() => "");
    throw new ApiError(resp.status, body || resp.statusText);
  }
  return (await resp.json()) as T;
}

/** Verifies credentials by hitting a lightweight authenticated endpoint. */
export async function verifyCredentials(creds: Credentials): Promise<boolean> {
  try {
    await request<SummaryResponse>("/api/alerts/summary", creds);
    return true;
  } catch (err) {
    if (err instanceof ApiError && err.status === 401) return false;
    throw err;
  }
}

export interface AlertFilters {
  severity?: string;
  attack_type?: string;
  start_time?: number;
  end_time?: number;
  triage_status?: string;
  limit?: number;
  offset?: number;
}

export function listAlerts(creds: Credentials, filters: AlertFilters = {}): Promise<AlertListResponse> {
  const params = new URLSearchParams();
  Object.entries(filters).forEach(([key, value]) => {
    if (value !== undefined && value !== "") params.set(key, String(value));
  });
  const qs = params.toString();
  return request<AlertListResponse>(`/api/alerts${qs ? `?${qs}` : ""}`, creds);
}

export function getSummary(creds: Credentials, startTime?: number, endTime?: number): Promise<SummaryResponse> {
  const params = new URLSearchParams();
  if (startTime !== undefined) params.set("start_time", String(startTime));
  if (endTime !== undefined) params.set("end_time", String(endTime));
  const qs = params.toString();
  return request<SummaryResponse>(`/api/alerts/summary${qs ? `?${qs}` : ""}`, creds);
}

export function setTriage(
  creds: Credentials,
  alertId: string,
  status: TriageStatus,
  note?: string,
): Promise<Alert> {
  return request<Alert>(`/api/alerts/${encodeURIComponent(alertId)}/triage`, creds, {
    method: "PATCH",
    body: JSON.stringify({ status, note }),
  });
}

async function fetchWsToken(creds: Credentials): Promise<string> {
  const resp = await request<{ token: string; expires_in: number }>("/api/ws-token", creds, { method: "POST" });
  return resp.token;
}

/**
 * Opens the live alert WebSocket. Fetches a short-lived token via the
 * Basic-auth-protected REST endpoint first, since browsers cannot attach an
 * Authorization header to a WebSocket handshake -- see routes_ws.py.
 */
export async function connectAlertStream(
  creds: Credentials,
  onAlert: (alert: Alert) => void,
  onStatusChange: (status: "connecting" | "open" | "closed" | "error") => void,
): Promise<() => void> {
  const token = await fetchWsToken(creds);
  onStatusChange("connecting");
  const ws = new WebSocket(`${WS_BASE_URL}/ws/alerts?token=${encodeURIComponent(token)}`);

  ws.onopen = () => onStatusChange("open");
  ws.onclose = () => onStatusChange("closed");
  ws.onerror = () => onStatusChange("error");
  ws.onmessage = (event) => {
    try {
      onAlert(JSON.parse(event.data) as Alert);
    } catch {
      // ignore malformed frames rather than crashing the live feed
    }
  };

  return () => ws.close();
}
