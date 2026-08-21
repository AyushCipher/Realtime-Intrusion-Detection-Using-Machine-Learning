"""Persistent alert storage: history/filter queries, and analyst triage state.

SQLite (stdlib `sqlite3`) rather than an in-memory list, so alert history and
triage decisions survive a dashboard restart -- a live system's "historical
alert query" requirement doesn't hold up if a restart empties it. A single
file-backed connection with a lock around every statement is enough for a
dashboard's read/write volume; this is not meant to scale to a
multi-process deployment (see the README's known-limitations section).
"""

from __future__ import annotations

import json
import sqlite3
import threading
import time
from typing import Any, Dict, List, Optional

TRIAGE_STATUSES = ("new", "acknowledged", "confirmed", "false_positive")

_SCHEMA = """
CREATE TABLE IF NOT EXISTS alerts (
    alert_id TEXT PRIMARY KEY,
    flow_id TEXT NOT NULL,
    src_ip TEXT NOT NULL,
    src_port INTEGER NOT NULL,
    dst_ip TEXT NOT NULL,
    dst_port INTEGER NOT NULL,
    protocol INTEGER NOT NULL,
    flow_start_time REAL NOT NULL,
    scored_at REAL NOT NULL,
    stage1_anomaly_score REAL NOT NULL,
    stage1_flagged INTEGER NOT NULL,
    stage2_predicted_class TEXT NOT NULL,
    stage2_confidence REAL NOT NULL,
    stage2_class_probabilities TEXT NOT NULL,
    severity TEXT NOT NULL,
    explanation TEXT NOT NULL,
    model_version TEXT NOT NULL,
    schema_version INTEGER NOT NULL,
    received_at REAL NOT NULL,
    triage_status TEXT NOT NULL DEFAULT 'new',
    triage_note TEXT,
    triage_updated_at REAL
);
CREATE INDEX IF NOT EXISTS idx_alerts_scored_at ON alerts(scored_at);
CREATE INDEX IF NOT EXISTS idx_alerts_severity ON alerts(severity);
CREATE INDEX IF NOT EXISTS idx_alerts_class ON alerts(stage2_predicted_class);
CREATE INDEX IF NOT EXISTS idx_alerts_triage ON alerts(triage_status);
"""

_JSON_FIELDS = ("stage2_class_probabilities", "explanation")


def _row_to_dict(row: sqlite3.Row) -> Dict[str, Any]:
    d = dict(row)
    d["stage1_flagged"] = bool(d["stage1_flagged"])
    for field in _JSON_FIELDS:
        d[field] = json.loads(d[field])
    return d


class AlertStore:
    def __init__(self, db_path: str = ":memory:") -> None:
        self._conn = sqlite3.connect(db_path, check_same_thread=False)
        self._conn.row_factory = sqlite3.Row
        self._lock = threading.Lock()
        with self._lock:
            self._conn.executescript(_SCHEMA)
            self._conn.commit()

    def insert_alert(self, alert: Dict[str, Any]) -> bool:
        """Inserts one alert (schema.ALERT_EVENT_FIELDS shape). Returns False
        without erroring if alert_id already exists (redelivery is expected
        on a Kafka consumer restart)."""
        with self._lock:
            cursor = self._conn.execute(
                """
                INSERT OR IGNORE INTO alerts (
                    alert_id, flow_id, src_ip, src_port, dst_ip, dst_port, protocol,
                    flow_start_time, scored_at, stage1_anomaly_score, stage1_flagged,
                    stage2_predicted_class, stage2_confidence, stage2_class_probabilities,
                    severity, explanation, model_version, schema_version, received_at,
                    triage_status
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, 'new')
                """,
                (
                    alert["alert_id"],
                    alert["flow_id"],
                    alert["src_ip"],
                    alert["src_port"],
                    alert["dst_ip"],
                    alert["dst_port"],
                    alert["protocol"],
                    alert["flow_start_time"],
                    alert["scored_at"],
                    alert["stage1_anomaly_score"],
                    int(alert["stage1_flagged"]),
                    alert["stage2_predicted_class"],
                    alert["stage2_confidence"],
                    json.dumps(alert["stage2_class_probabilities"]),
                    alert["severity"],
                    json.dumps(alert["explanation"]),
                    alert["model_version"],
                    alert["schema_version"],
                    time.time(),
                ),
            )
            self._conn.commit()
            return cursor.rowcount > 0

    def get_alert(self, alert_id: str) -> Optional[Dict[str, Any]]:
        with self._lock:
            row = self._conn.execute("SELECT * FROM alerts WHERE alert_id = ?", (alert_id,)).fetchone()
        return _row_to_dict(row) if row is not None else None

    def _where_clause(
        self,
        severity: Optional[str],
        attack_type: Optional[str],
        start_time: Optional[float],
        end_time: Optional[float],
        triage_status: Optional[str],
    ):
        clauses: List[str] = []
        params: List[Any] = []
        if severity is not None:
            clauses.append("severity = ?")
            params.append(severity)
        if attack_type is not None:
            clauses.append("stage2_predicted_class = ?")
            params.append(attack_type)
        if start_time is not None:
            clauses.append("scored_at >= ?")
            params.append(start_time)
        if end_time is not None:
            clauses.append("scored_at <= ?")
            params.append(end_time)
        if triage_status is not None:
            clauses.append("triage_status = ?")
            params.append(triage_status)
        where = f"WHERE {' AND '.join(clauses)}" if clauses else ""
        return where, params

    def list_alerts(
        self,
        severity: Optional[str] = None,
        attack_type: Optional[str] = None,
        start_time: Optional[float] = None,
        end_time: Optional[float] = None,
        triage_status: Optional[str] = None,
        limit: int = 50,
        offset: int = 0,
    ) -> List[Dict[str, Any]]:
        where, params = self._where_clause(severity, attack_type, start_time, end_time, triage_status)
        query = f"SELECT * FROM alerts {where} ORDER BY scored_at DESC LIMIT ? OFFSET ?"
        with self._lock:
            rows = self._conn.execute(query, (*params, limit, offset)).fetchall()
        return [_row_to_dict(r) for r in rows]

    def count_alerts(
        self,
        severity: Optional[str] = None,
        attack_type: Optional[str] = None,
        start_time: Optional[float] = None,
        end_time: Optional[float] = None,
        triage_status: Optional[str] = None,
    ) -> int:
        where, params = self._where_clause(severity, attack_type, start_time, end_time, triage_status)
        query = f"SELECT COUNT(*) FROM alerts {where}"
        with self._lock:
            (count,) = self._conn.execute(query, params).fetchone()
        return count

    def set_triage(self, alert_id: str, status: str, note: Optional[str] = None) -> bool:
        if status not in TRIAGE_STATUSES:
            raise ValueError(f"unknown triage status: {status!r}")
        with self._lock:
            cursor = self._conn.execute(
                "UPDATE alerts SET triage_status = ?, triage_note = ?, triage_updated_at = ? WHERE alert_id = ?",
                (status, note, time.time(), alert_id),
            )
            self._conn.commit()
            return cursor.rowcount > 0

    def summary(self, start_time: Optional[float] = None, end_time: Optional[float] = None) -> Dict[str, Any]:
        """Alert-volume and false-positive-rate summary over an optional time range.

        Two distinct false-positive signals are reported, since neither
        alone is trustworthy -- see the README's known-limitations section:

        - stage1_proxy: among *received* alerts, the fraction stage 2
          resolved back to BENIGN (only present at all if the ML module was
          run with `alert_on_stage1_flag_only=True`; otherwise no such
          alerts ever arrive and this reads as unavailable, not zero).
        - analyst_reviewed: among alerts a human has triaged
          (confirmed/false_positive), the fraction marked false_positive.
          Coverage (how many alerts have been reviewed at all) is reported
          alongside so a low-coverage rate isn't mistaken for a reliable one.
        """
        where, params = self._where_clause(None, None, start_time, end_time, None)

        with self._lock:
            (total,) = self._conn.execute(f"SELECT COUNT(*) FROM alerts {where}", params).fetchone()

            by_severity = {
                row["severity"]: row["n"]
                for row in self._conn.execute(
                    f"SELECT severity, COUNT(*) as n FROM alerts {where} GROUP BY severity", params
                ).fetchall()
            }
            by_class = {
                row["stage2_predicted_class"]: row["n"]
                for row in self._conn.execute(
                    f"SELECT stage2_predicted_class, COUNT(*) as n FROM alerts {where} GROUP BY stage2_predicted_class",
                    params,
                ).fetchall()
            }

            benign_count = by_class.get("BENIGN", 0)
            stage1_proxy_available = benign_count > 0

            reviewed_where_sql = where + (" AND " if where else "WHERE ") + "triage_status IN ('confirmed', 'false_positive')"
            (reviewed_count,) = self._conn.execute(f"SELECT COUNT(*) FROM alerts {reviewed_where_sql}", params).fetchone()
            fp_where_sql = where + (" AND " if where else "WHERE ") + "triage_status = 'false_positive'"
            (reviewed_fp_count,) = self._conn.execute(f"SELECT COUNT(*) FROM alerts {fp_where_sql}", params).fetchone()

            volume_by_day = [
                {"date": row["day"], "count": row["n"]}
                for row in self._conn.execute(
                    f"""
                    SELECT date(scored_at, 'unixepoch') as day, COUNT(*) as n
                    FROM alerts {where}
                    GROUP BY day ORDER BY day
                    """,
                    params,
                ).fetchall()
            ]

        return {
            "total_alerts": total,
            "by_severity": by_severity,
            "by_attack_type": by_class,
            "volume_by_day": volume_by_day,
            "stage1_proxy_false_positive": {
                "available": stage1_proxy_available,
                "benign_count": benign_count,
                "total_count": total,
                "rate": (benign_count / total) if stage1_proxy_available and total > 0 else None,
            },
            "analyst_reviewed_false_positive": {
                "reviewed_count": reviewed_count,
                "false_positive_count": reviewed_fp_count,
                "total_count": total,
                "rate": (reviewed_fp_count / reviewed_count) if reviewed_count > 0 else None,
            },
        }

    def close(self) -> None:
        with self._lock:
            self._conn.close()
