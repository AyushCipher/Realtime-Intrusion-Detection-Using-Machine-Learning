// Mirrors src/ids_dashboard/schema.py's ALERT_EVENT_FIELDS plus the
// store-added triage fields returned by the REST API. Keep in sync with
// that schema -- this is the frontend's copy of the same documented
// contract, not a code dependency on the backend.

export type Severity = "info" | "low" | "medium" | "high" | "critical";

export type TriageStatus = "new" | "acknowledged" | "confirmed" | "false_positive";

export interface FeatureContribution {
  feature: string;
  value: number;
  shap_value: number;
}

export interface Alert {
  alert_id: string;
  flow_id: string;
  src_ip: string;
  src_port: number;
  dst_ip: string;
  dst_port: number;
  protocol: number;
  flow_start_time: number;
  scored_at: number;
  stage1_anomaly_score: number;
  stage1_flagged: boolean;
  stage2_predicted_class: string;
  stage2_confidence: number;
  stage2_class_probabilities: Record<string, number>;
  severity: Severity;
  explanation: FeatureContribution[];
  model_version: string;
  schema_version: number;
  received_at: number;
  triage_status: TriageStatus;
  triage_note: string | null;
  triage_updated_at: number | null;
}

export interface AlertListResponse {
  alerts: Alert[];
  total: number;
  limit: number;
  offset: number;
}

export interface FalsePositiveStat {
  rate: number | null;
  total_count: number;
  [key: string]: number | boolean | null;
}

export interface SummaryResponse {
  total_alerts: number;
  by_severity: Record<string, number>;
  by_attack_type: Record<string, number>;
  volume_by_day: { date: string; count: number }[];
  stage1_proxy_false_positive: FalsePositiveStat & { available: boolean; benign_count: number };
  analyst_reviewed_false_positive: FalsePositiveStat & { reviewed_count: number; false_positive_count: number };
}

export const SEVERITY_ORDER: Severity[] = ["critical", "high", "medium", "low", "info"];
