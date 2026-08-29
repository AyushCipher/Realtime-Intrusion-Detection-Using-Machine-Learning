import type { Alert } from "../types";
import SeverityBadge from "./SeverityBadge";

export default function AlertRow({ alert, onSelect }: { alert: Alert; onSelect: (a: Alert) => void }) {
  // Live WebSocket-pushed alerts carry ml's raw Kafka event shape, which
  // has no triage_status field at all -- that's a dashboard-api/DB-only
  // concept, added by the store on insert (DEFAULT 'new'), not part of
  // ml's alert schema. Only alerts fetched via REST (already round-tripped
  // through the DB) are guaranteed to have it. A freshly-pushed alert with
  // no triage_status yet has, by definition, never been triaged -- "new".
  const triageStatus = alert.triage_status ?? "new";
  return (
    <button className="alert-row" onClick={() => onSelect(alert)}>
      <SeverityBadge severity={alert.severity} />
      <span className="alert-row-class">{alert.stage2_predicted_class}</span>
      <span className="alert-row-flow">
        {alert.src_ip}:{alert.src_port} &rarr; {alert.dst_ip}:{alert.dst_port}
      </span>
      <span className="alert-row-confidence">{(alert.stage2_confidence * 100).toFixed(0)}%</span>
      <span className="alert-row-time">{new Date(alert.scored_at * 1000).toLocaleTimeString()}</span>
      <span className="alert-row-escalated">{alert.escalated && <span className="escalated-badge">ESC</span>}</span>
      <span className={`alert-row-triage triage-${triageStatus}`}>{triageStatus.replace("_", " ")}</span>
    </button>
  );
}
