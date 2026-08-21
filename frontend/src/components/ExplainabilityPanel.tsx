import { Bar, BarChart, CartesianGrid, Cell, ResponsiveContainer, Tooltip, XAxis, YAxis } from "recharts";
import type { Alert } from "../types";
import SeverityBadge from "./SeverityBadge";

export default function ExplainabilityPanel({ alert, onClose }: { alert: Alert | null; onClose: () => void }) {
  if (!alert) return null;

  const chartData = [...alert.explanation].sort((a, b) => Math.abs(b.shap_value) - Math.abs(a.shap_value));

  return (
    <aside className="explain-panel">
      <div className="explain-header">
        <h2>Alert detail</h2>
        <button className="close-btn" onClick={onClose} aria-label="Close">
          ×
        </button>
      </div>

      <div className="explain-meta">
        <SeverityBadge severity={alert.severity} />
        <span className="explain-class">{alert.stage2_predicted_class}</span>
      </div>

      <dl className="explain-fields">
        <dt>Flow</dt>
        <dd>
          {alert.src_ip}:{alert.src_port} &rarr; {alert.dst_ip}:{alert.dst_port} (proto {alert.protocol})
        </dd>
        <dt>Scored at</dt>
        <dd>{new Date(alert.scored_at * 1000).toLocaleString()}</dd>
        <dt>Stage 1 anomaly score</dt>
        <dd>
          {alert.stage1_anomaly_score.toFixed(3)} {alert.stage1_flagged ? "(flagged)" : "(not flagged)"}
        </dd>
        <dt>Stage 2 confidence</dt>
        <dd>{(alert.stage2_confidence * 100).toFixed(1)}%</dd>
        <dt>Model version</dt>
        <dd>{alert.model_version}</dd>
      </dl>

      <h3>Why this prediction (SHAP)</h3>
      {chartData.length === 0 ? (
        <p className="explain-empty">No explanation attached to this alert (stage 2 did not run).</p>
      ) : (
        <div className="explain-chart">
          <ResponsiveContainer width="100%" height={Math.max(160, chartData.length * 36)}>
            <BarChart data={chartData} layout="vertical" margin={{ top: 4, right: 24, bottom: 4, left: 24 }}>
              <CartesianGrid strokeDasharray="3 3" horizontal={false} />
              <XAxis type="number" tick={{ fontSize: 11 }} />
              <YAxis type="category" dataKey="feature" width={150} tick={{ fontSize: 11 }} />
              <Tooltip formatter={(value) => [Number(value).toFixed(4), "SHAP value"]} />
              <Bar dataKey="shap_value">
                {chartData.map((entry, idx) => (
                  <Cell key={idx} fill={entry.shap_value >= 0 ? "#e03131" : "#2f9e44"} />
                ))}
              </Bar>
            </BarChart>
          </ResponsiveContainer>
          <p className="explain-legend">
            <span className="legend-swatch" style={{ background: "#e03131" }} /> pushes toward {alert.stage2_predicted_class}
            &nbsp;&nbsp;
            <span className="legend-swatch" style={{ background: "#2f9e44" }} /> pushes toward BENIGN
          </p>
        </div>
      )}

      <h3>Class probabilities</h3>
      <ul className="prob-list">
        {Object.entries(alert.stage2_class_probabilities)
          .sort(([, a], [, b]) => b - a)
          .map(([cls, p]) => (
            <li key={cls}>
              <span>{cls}</span>
              <span>{(p * 100).toFixed(1)}%</span>
            </li>
          ))}
      </ul>
    </aside>
  );
}
