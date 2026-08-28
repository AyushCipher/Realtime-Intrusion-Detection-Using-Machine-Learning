import { useCallback, useEffect, useState } from "react";
import { listAlerts, setTriage, type Credentials } from "../api";
import type { Alert, TriageStatus } from "../types";
import { SEVERITY_ORDER } from "../types";
import AlertRow from "./AlertRow";

const TRIAGE_ACTIONS: { status: TriageStatus; label: string }[] = [
  { status: "acknowledged", label: "Acknowledge" },
  { status: "confirmed", label: "Confirm" },
  { status: "false_positive", label: "Mark false positive" },
];

export default function TriageView({ creds, onSelect }: { creds: Credentials; onSelect: (a: Alert) => void }) {
  const [alerts, setAlerts] = useState<Alert[]>([]);
  const [severityFilter, setSeverityFilter] = useState<string>("");
  const [triageFilter, setTriageFilter] = useState<string>("");
  const [escalatedOnly, setEscalatedOnly] = useState(false);
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState<string | null>(null);

  const refresh = useCallback(async () => {
    setLoading(true);
    setError(null);
    try {
      const resp = await listAlerts(creds, {
        severity: severityFilter || undefined,
        triage_status: triageFilter || undefined,
        escalated: escalatedOnly ? true : undefined,
        limit: 200,
      });
      setAlerts(resp.alerts);
    } catch {
      setError("Failed to load alerts.");
    } finally {
      setLoading(false);
    }
  }, [creds, severityFilter, triageFilter, escalatedOnly]);

  useEffect(() => {
    refresh();
  }, [refresh]);

  async function handleTriage(alertId: string, status: TriageStatus) {
    const updated = await setTriage(creds, alertId, status);
    setAlerts((prev) => prev.map((a) => (a.alert_id === alertId ? updated : a)));
  }

  const bySeverity = SEVERITY_ORDER.map((sev) => ({
    severity: sev,
    items: alerts.filter((a) => a.severity === sev),
  })).filter((group) => group.items.length > 0);

  return (
    <div className="triage-view">
      <div className="triage-toolbar">
        <h2>Triage</h2>
        <select value={severityFilter} onChange={(e) => setSeverityFilter(e.target.value)}>
          <option value="">All severities</option>
          {SEVERITY_ORDER.map((s) => (
            <option key={s} value={s}>
              {s}
            </option>
          ))}
        </select>
        <select value={triageFilter} onChange={(e) => setTriageFilter(e.target.value)}>
          <option value="">All statuses</option>
          <option value="new">New</option>
          <option value="acknowledged">Acknowledged</option>
          <option value="confirmed">Confirmed</option>
          <option value="false_positive">False positive</option>
        </select>
        <label className="escalated-filter-label">
          <input type="checkbox" checked={escalatedOnly} onChange={(e) => setEscalatedOnly(e.target.checked)} />
          Escalated only
        </label>
        <button onClick={refresh} disabled={loading}>
          {loading ? "Refreshing..." : "Refresh"}
        </button>
      </div>

      {error && <div className="error-banner">{error}</div>}

      {bySeverity.length === 0 && !loading ? (
        <p className="empty-state">No alerts match these filters.</p>
      ) : (
        bySeverity.map((group) => (
          <section key={group.severity} className="triage-group">
            <h3>
              <span className={`severity-dot severity-dot-${group.severity}`} />
              {group.severity} ({group.items.length})
            </h3>
            <div className="alert-list">
              {group.items.map((alert) => (
                <div key={alert.alert_id} className="triage-row">
                  <AlertRow alert={alert} onSelect={onSelect} />
                  <div className="triage-actions">
                    {TRIAGE_ACTIONS.map((action) => (
                      <button
                        key={action.status}
                        disabled={alert.triage_status === action.status}
                        onClick={() => handleTriage(alert.alert_id, action.status)}
                      >
                        {action.label}
                      </button>
                    ))}
                  </div>
                </div>
              ))}
            </div>
          </section>
        ))
      )}
    </div>
  );
}
