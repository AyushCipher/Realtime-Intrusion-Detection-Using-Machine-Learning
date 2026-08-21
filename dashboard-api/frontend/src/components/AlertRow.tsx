import type { Alert } from "../types";
import SeverityBadge from "./SeverityBadge";

export default function AlertRow({ alert, onSelect }: { alert: Alert; onSelect: (a: Alert) => void }) {
  return (
    <button className="alert-row" onClick={() => onSelect(alert)}>
      <SeverityBadge severity={alert.severity} />
      <span className="alert-row-class">{alert.stage2_predicted_class}</span>
      <span className="alert-row-flow">
        {alert.src_ip}:{alert.src_port} &rarr; {alert.dst_ip}:{alert.dst_port}
      </span>
      <span className="alert-row-confidence">{(alert.stage2_confidence * 100).toFixed(0)}%</span>
      <span className="alert-row-time">{new Date(alert.scored_at * 1000).toLocaleTimeString()}</span>
      <span className={`alert-row-triage triage-${alert.triage_status}`}>{alert.triage_status.replace("_", " ")}</span>
    </button>
  );
}
