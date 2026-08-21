import { severityColor } from "../severity";

export default function SeverityBadge({ severity }: { severity: string }) {
  return (
    <span className="severity-badge" style={{ backgroundColor: severityColor(severity) }}>
      {severity}
    </span>
  );
}
