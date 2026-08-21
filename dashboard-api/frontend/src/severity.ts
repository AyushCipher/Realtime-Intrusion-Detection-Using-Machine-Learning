import type { Severity } from "./types";

export const SEVERITY_COLORS: Record<Severity, string> = {
  critical: "#e03131",
  high: "#e8590c",
  medium: "#f0b429",
  low: "#3b82f6",
  info: "#6b7280",
};

export function severityColor(severity: string): string {
  return SEVERITY_COLORS[severity as Severity] ?? "#6b7280";
}
