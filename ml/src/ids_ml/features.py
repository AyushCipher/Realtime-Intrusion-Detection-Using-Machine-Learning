"""The canonical numeric feature vector used by both training and inference.

Training data (CICIDS2017/CIC-IDS2018 CSVs) and live inference data (flow
events off the Kafka topic) come from two different pipelines with two
different column-naming conventions. `CANONICAL_FEATURE_COLUMNS` is the
common ground: `data.py` maps CICIDS CSV columns onto it, and
`event_to_feature_vector` maps live flow events onto it, so the model always
sees the same feature space regardless of source.
"""

from __future__ import annotations

from typing import Any, Dict, List

import numpy as np

# Deliberately restricted to features the ingestion module's flow extractor
# actually computes (see ids_ingestion/features.py) intersected with what's
# available in the standard CICIDS2017/CIC-IDS2018 CSVs. Note cwr_flag_count
# maps to CICFlowMeter's "CWE Flag Count" column -- a long-documented typo
# in the original tool (it counts the CWR flag, not "CWE") -- see data.py.
CANONICAL_FEATURE_COLUMNS: List[str] = [
    "flow_duration",
    "total_fwd_packets",
    "total_bwd_packets",
    "total_fwd_bytes",
    "total_bwd_bytes",
    "fwd_packet_length_min",
    "fwd_packet_length_max",
    "fwd_packet_length_mean",
    "fwd_packet_length_std",
    "bwd_packet_length_min",
    "bwd_packet_length_max",
    "bwd_packet_length_mean",
    "bwd_packet_length_std",
    "flow_bytes_per_sec",
    "flow_packets_per_sec",
    "flow_iat_mean",
    "flow_iat_std",
    "flow_iat_min",
    "flow_iat_max",
    "fwd_iat_mean",
    "fwd_iat_std",
    "fwd_iat_min",
    "fwd_iat_max",
    "bwd_iat_mean",
    "bwd_iat_std",
    "bwd_iat_min",
    "bwd_iat_max",
    "syn_flag_count",
    "ack_flag_count",
    "fin_flag_count",
    "rst_flag_count",
    "psh_flag_count",
    "urg_flag_count",
    "ece_flag_count",
    "cwr_flag_count",
]


def event_to_feature_vector(event: Dict[str, Any]) -> np.ndarray:
    """Extract the canonical feature vector from one live flow event."""
    return np.array([float(event.get(name, 0.0) or 0.0) for name in CANONICAL_FEATURE_COLUMNS], dtype=float)


def events_to_matrix(events: List[Dict[str, Any]]) -> np.ndarray:
    if not events:
        return np.empty((0, len(CANONICAL_FEATURE_COLUMNS)), dtype=float)
    return np.vstack([event_to_feature_vector(e) for e in events])
