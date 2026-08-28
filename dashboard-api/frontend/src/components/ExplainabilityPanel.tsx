import { useEffect, useState } from "react";
import { Bar, BarChart, CartesianGrid, Cell, ResponsiveContainer, Tooltip, XAxis, YAxis } from "recharts";
import { getExplanation, type Credentials } from "../api";
import type { Alert, Tier2Explanation } from "../types";
import SeverityBadge from "./SeverityBadge";

export default function ExplainabilityPanel({
  alert,
  creds,
  liveExplanation,
  onClose,
}: {
  alert: Alert | null;
  creds: Credentials;
  liveExplanation?: Tier2Explanation;
  onClose: () => void;
}) {
  const [fetchedExplanation, setFetchedExplanation] = useState<Tier2Explanation | null>(null);
  const [explanationLoading, setExplanationLoading] = useState(false);

  useEffect(() => {
    setFetchedExplanation(null);
    if (!alert?.escalated) return;

    let cancelled = false;
    setExplanationLoading(true);
    getExplanation(creds, alert.alert_id)
      .then((result) => {
        if (!cancelled) setFetchedExplanation(result);
      })
      .finally(() => {
        if (!cancelled) setExplanationLoading(false);
      });

    return () => {
      cancelled = true;
    };
  }, [alert?.alert_id, alert?.escalated, creds]);

  if (!alert) return null;

  // A live WebSocket push (if one has arrived for this alert) always wins
  // over the REST fetch, since it can only be more recent.
  const tier2Explanation = liveExplanation ?? fetchedExplanation;

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
        {alert.escalated && <span className="escalated-badge">ESCALATED</span>}
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
        <dt>Unknown mass (open-set)</dt>
        <dd>
          {alert.unknown_mass.toFixed(3)}
          {alert.escalation_trigger && ` (${alert.escalation_trigger} gate)`}
        </dd>
        <dt>Model version</dt>
        <dd>{alert.model_version}</dd>
      </dl>

      {alert.escalated && (
        <>
          <h3>Tier 2 analysis (LLM/RAG)</h3>
          {explanationLoading && !tier2Explanation ? (
            <p className="explain-empty">Loading...</p>
          ) : tier2Explanation ? (
            <div className="tier2-explanation">
              <dl className="explain-fields">
                <dt>Suspected technique</dt>
                <dd>
                  {tier2Explanation.suspected_technique_id
                    ? `${tier2Explanation.suspected_technique_id} (${tier2Explanation.suspected_technique_name})`
                    : "None identified"}
                </dd>
                <dt>Risk explanation</dt>
                <dd>{tier2Explanation.risk_explanation}</dd>
                <dt>Recommended action</dt>
                <dd>{tier2Explanation.recommended_action}</dd>
                <dt>Grounding</dt>
                <dd>
                  {tier2Explanation.rag_enabled
                    ? `RAG-grounded (${tier2Explanation.retrieved_technique_ids.join(", ") || "none retrieved"})`
                    : "Bare LLM (RAG disabled)"}
                </dd>
                <dt>LLM latency</dt>
                <dd>{(tier2Explanation.llm_latency_ms / 1000).toFixed(1)}s</dd>
              </dl>
            </div>
          ) : (
            <p className="explain-empty">
              Escalated -- Tier 2 analysis pending. This can take anywhere from several seconds to a minute or more
              (real, measured latency -- see the project README), not an error.
            </p>
          )}
        </>
      )}

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
