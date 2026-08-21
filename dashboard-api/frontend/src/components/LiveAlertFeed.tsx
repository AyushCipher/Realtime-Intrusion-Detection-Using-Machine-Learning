import { useEffect, useRef, useState } from "react";
import { connectAlertStream, type Credentials } from "../api";
import type { Alert } from "../types";
import AlertRow from "./AlertRow";

const MAX_FEED_SIZE = 200;

export default function LiveAlertFeed({ creds, onSelect }: { creds: Credentials; onSelect: (a: Alert) => void }) {
  const [alerts, setAlerts] = useState<Alert[]>([]);
  const [status, setStatus] = useState<"connecting" | "open" | "closed" | "error">("connecting");
  const closeRef = useRef<(() => void) | null>(null);

  useEffect(() => {
    let cancelled = false;

    connectAlertStream(
      creds,
      (alert) => {
        if (cancelled) return;
        setAlerts((prev) => {
          if (prev.some((a) => a.alert_id === alert.alert_id)) return prev;
          return [alert, ...prev].slice(0, MAX_FEED_SIZE);
        });
      },
      setStatus,
    ).then((close) => {
      if (cancelled) close();
      else closeRef.current = close;
    });

    return () => {
      cancelled = true;
      closeRef.current?.();
    };
  }, [creds]);

  return (
    <div className="live-feed">
      <div className="live-feed-header">
        <h2>Live alert feed</h2>
        <span className={`ws-status ws-status-${status}`}>{status}</span>
      </div>
      {alerts.length === 0 ? (
        <p className="empty-state">Waiting for alerts...</p>
      ) : (
        <div className="alert-list">
          {alerts.map((alert) => (
            <AlertRow key={alert.alert_id} alert={alert} onSelect={onSelect} />
          ))}
        </div>
      )}
    </div>
  );
}
