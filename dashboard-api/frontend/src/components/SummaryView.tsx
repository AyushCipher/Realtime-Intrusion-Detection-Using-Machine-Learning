import { useEffect, useState } from "react";
import { Bar, BarChart, CartesianGrid, Cell, Line, LineChart, ResponsiveContainer, Tooltip, XAxis, YAxis } from "recharts";
import { getSummary, type Credentials } from "../api";
import type { SummaryResponse } from "../types";
import { severityColor } from "../severity";

function StatTile({ label, value, sub }: { label: string; value: string; sub?: string }) {
  return (
    <div className="stat-tile">
      <span className="stat-label">{label}</span>
      <span className="stat-value">{value}</span>
      {sub && <span className="stat-sub">{sub}</span>}
    </div>
  );
}

export default function SummaryView({ creds }: { creds: Credentials }) {
  const [summary, setSummary] = useState<SummaryResponse | null>(null);
  const [error, setError] = useState<string | null>(null);

  useEffect(() => {
    getSummary(creds)
      .then(setSummary)
      .catch(() => setError("Failed to load summary."));
  }, [creds]);

  if (error) return <div className="error-banner">{error}</div>;
  if (!summary) return <p className="empty-state">Loading summary...</p>;

  const severityData = Object.entries(summary.by_severity).map(([severity, count]) => ({ severity, count }));
  const attackTypeData = Object.entries(summary.by_attack_type)
    .sort(([, a], [, b]) => b - a)
    .map(([type, count]) => ({ type, count }));

  const stage1 = summary.stage1_proxy_false_positive;
  const reviewed = summary.analyst_reviewed_false_positive;

  return (
    <div className="summary-view">
      <h2>Alert volume &amp; false-positive summary</h2>

      <div className="stat-grid">
        <StatTile label="Total alerts" value={String(summary.total_alerts)} />
        <StatTile
          label="Stage-1 proxy FP rate"
          value={stage1.available && stage1.rate !== null ? `${(stage1.rate * 100).toFixed(1)}%` : "n/a"}
          sub={
            stage1.available
              ? `${stage1.benign_count} of ${stage1.total_count} alerts`
              : "requires alert_on_stage1_flag_only"
          }
        />
        <StatTile
          label="Analyst-reviewed FP rate"
          value={reviewed.rate !== null ? `${(reviewed.rate * 100).toFixed(1)}%` : "n/a"}
          sub={`${reviewed.reviewed_count} of ${reviewed.total_count} alerts reviewed`}
        />
      </div>

      <div className="chart-row">
        <div className="chart-card">
          <h3>By severity</h3>
          <ResponsiveContainer width="100%" height={220}>
            <BarChart data={severityData}>
              <CartesianGrid strokeDasharray="3 3" />
              <XAxis dataKey="severity" tick={{ fontSize: 12 }} />
              <YAxis allowDecimals={false} tick={{ fontSize: 12 }} />
              <Tooltip />
              <Bar dataKey="count">
                {severityData.map((entry, idx) => (
                  <Cell key={idx} fill={severityColor(entry.severity)} />
                ))}
              </Bar>
            </BarChart>
          </ResponsiveContainer>
        </div>

        <div className="chart-card">
          <h3>By attack type</h3>
          <ResponsiveContainer width="100%" height={220}>
            <BarChart data={attackTypeData} layout="vertical" margin={{ top: 4, right: 16, bottom: 4, left: 24 }}>
              <CartesianGrid strokeDasharray="3 3" horizontal={false} />
              <XAxis type="number" allowDecimals={false} tick={{ fontSize: 12 }} />
              <YAxis type="category" dataKey="type" width={110} tick={{ fontSize: 11 }} />
              <Tooltip />
              <Bar dataKey="count" fill="#3b82f6" />
            </BarChart>
          </ResponsiveContainer>
        </div>
      </div>

      <div className="chart-card">
        <h3>Volume by day</h3>
        <ResponsiveContainer width="100%" height={220}>
          <LineChart data={summary.volume_by_day}>
            <CartesianGrid strokeDasharray="3 3" />
            <XAxis dataKey="date" tick={{ fontSize: 12 }} />
            <YAxis allowDecimals={false} tick={{ fontSize: 12 }} />
            <Tooltip />
            <Line type="monotone" dataKey="count" stroke="#3b82f6" strokeWidth={2} dot={false} />
          </LineChart>
        </ResponsiveContainer>
      </div>
    </div>
  );
}
