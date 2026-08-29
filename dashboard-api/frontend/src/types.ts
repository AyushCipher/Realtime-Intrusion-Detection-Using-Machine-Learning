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
  // Open-set escalation fields -- see ml/README.md's "Open-set escalation"
  // section. unknown_mass is 0.0 and escalation_trigger is "" unless the
  // detector was built with a gate.
  unknown_mass: number;
  escalated: boolean;
  escalation_trigger: "" | "openset" | "softmax";
  model_version: string;
  schema_version: number;
  received_at: number;
  // Store-added, not part of ml's own alert schema -- always present on a
  // REST-fetched alert (round-tripped through the DB), but absent on an
  // alert delivered live over the WebSocket (that's ml's raw Kafka event
  // shape, broadcast as received -- see ingest_service.py's
  // `_handle_alert`). Treat a missing triage_status as "new": a
  // freshly-pushed alert has, by definition, never been triaged yet.
  triage_status?: TriageStatus;
  triage_note?: string | null;
  triage_updated_at?: number | null;
}

// Mirrors tier2_reasoner/src/ids_tier2/schema.py's EXPLANATION_EVENT_FIELDS.
// Only ever present for alerts where escalated is true, and only once
// tier2_reasoner has actually processed it -- Tier 2's real measured
// latency is 5-40+ seconds per alert (see ml/README.md's Latency section),
// so "escalated but no explanation yet" is an expected, normal state.
export interface Tier2Explanation {
  explanation_id: string;
  alert_id: string;
  flow_id: string;
  generated_at: number;
  suspected_technique_id: string;
  suspected_technique_name: string;
  risk_explanation: string;
  recommended_action: string;
  retrieved_technique_ids: string[];
  rag_enabled: boolean;
  llm_latency_ms: number;
  model_version: string;
  schema_version: number;
}

// The live WebSocket carries two message shapes down one connection --
// see api.ts's connectAlertStream. Alert broadcasts are unmarked (the
// original, unchanged shape); explanation broadcasts carry this marker so
// the client can tell them apart without a breaking change to the alert
// message format.
export interface Tier2ExplanationBroadcast extends Tier2Explanation {
  __type: "explanation";
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
