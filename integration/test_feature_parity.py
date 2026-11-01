"""Train/serve parity: ingestion's live feature dict must cover the ML feature vector.

The ingestion module (`ids_ingestion.features`) emits a flat dict per closed
flow, and the ML module (`ids_ml.features`) turns that dict into the 35-column
vector the models were trained on. If a column is renamed or dropped on either
side, `event_to_feature_vector` silently fills 0.0 (it uses `event.get(name, 0.0)`),
so the model would score garbage without any error. This test fails loudly instead.
"""

import math
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent
for src in (REPO / "ingestion" / "src", REPO / "ml" / "src"):
    if str(src) not in sys.path:
        sys.path.insert(0, str(src))

from ids_ingestion.features import FlowFeatureExtractor, WindowConfig  # noqa: E402
from ids_ingestion.replay import PcapReplay  # noqa: E402
from ids_ml.features import CANONICAL_FEATURE_COLUMNS, event_to_feature_vector  # noqa: E402

FIXTURE = REPO / "ingestion" / "tests" / "fixtures" / "sample_tcp.pcap"


def _closed_flows():
    extractor = FlowFeatureExtractor(WindowConfig(active_timeout=120.0, idle_timeout=60.0))
    flows = []
    for pkt in PcapReplay(str(FIXTURE), speed=0).packets():
        flows.extend(extractor.process_packet(pkt))
    flows.extend(extractor.flush())
    return flows


def test_ingestion_emits_every_canonical_column():
    flows = _closed_flows()
    assert flows, "fixture produced no flows"
    for flow in flows:
        missing = [c for c in CANONICAL_FEATURE_COLUMNS if c not in flow]
        assert not missing, f"ingestion is missing ML feature columns: {missing}"


def test_feature_vector_matches_ingestion_values_in_canonical_order():
    for flow in _closed_flows():
        vector = event_to_feature_vector(flow)
        assert vector.shape == (len(CANONICAL_FEATURE_COLUMNS),)
        assert all(math.isfinite(v) for v in vector)
        expected = [float(flow[c]) for c in CANONICAL_FEATURE_COLUMNS]
        assert list(vector) == expected


def test_canonical_columns_are_unique():
    assert len(set(CANONICAL_FEATURE_COLUMNS)) == len(CANONICAL_FEATURE_COLUMNS) == 35
