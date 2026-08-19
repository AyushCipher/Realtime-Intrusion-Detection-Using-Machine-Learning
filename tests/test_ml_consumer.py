"""End-to-end test: flow events in -> two-stage scoring -> alerts out.

Uses StubFlowEventSource / StubAlertProducer (no live Kafka broker needed)
to exercise scoring_service.py, mirroring how ids_ingestion's own tests stub
Kafka -- consistent with both modules being independently testable via
their documented topic contracts only.
"""

from pathlib import Path
from typing import Any, Dict

import pandas as pd

from ids_ml.alert_producer import BufferedAlertProducer, StubAlertProducer
from ids_ml.data import load_and_map
from ids_ml.explainability import ShapExplainer
from ids_ml.features import CANONICAL_FEATURE_COLUMNS
from ids_ml.flow_consumer import StubFlowEventSource
from ids_ml.pipeline import TwoStageDetector
from ids_ml.schema import validate_alert_event
from ids_ml.scoring_service import ScoringService, ScoringServiceConfig
from ids_ml.split import time_based_split
from ids_ml.stage1_iforest import AnomalyPreFilter, Stage1Config
from ids_ml.stage2_xgboost import AttackClassifier, Stage2Config

FIXTURE_PATH = Path(__file__).parent / "fixtures" / "synthetic_cicids_sample.csv"


def _row_to_flow_event(row: pd.Series, flow_id: str) -> Dict[str, Any]:
    event = {name: float(row[name]) for name in CANONICAL_FEATURE_COLUMNS}
    # cast integer-typed fields per schema.FLOW_EVENT_FIELDS
    for int_field in (
        "total_fwd_packets", "total_bwd_packets", "total_fwd_bytes", "total_bwd_bytes",
        "syn_flag_count", "ack_flag_count", "fin_flag_count", "rst_flag_count",
        "psh_flag_count", "urg_flag_count", "ece_flag_count", "cwr_flag_count",
    ):
        event[int_field] = int(row[int_field])
    event.update(
        flow_id=flow_id,
        src_ip=str(row.get("src_ip", "10.0.0.1")),
        src_port=12345,
        dst_ip="10.0.0.9",
        dst_port=443,
        protocol=6,
        flow_start_time=0.0,
        flow_end_time=float(row["flow_duration"]),
        close_reason="fin",
        schema_version=1,
    )
    return event


def _fit_detector_and_explainer():
    df = load_and_map(FIXTURE_PATH)
    train, _val, test = time_based_split(df, train_frac=0.7, val_frac=0.15)
    X_train = train[CANONICAL_FEATURE_COLUMNS].to_numpy()

    stage1 = AnomalyPreFilter(Stage1Config(n_estimators=100, contamination=0.25, random_state=0)).fit(X_train)
    stage2 = AttackClassifier(Stage2Config(n_estimators=100, random_state=0)).fit(
        X_train, train["attack_category"].tolist()
    )
    detector = TwoStageDetector(stage1, stage2)
    explainer = ShapExplainer(stage2)
    return detector, explainer, test


def test_scoring_service_publishes_only_non_benign_alerts_by_default():
    detector, explainer, test_df = _fit_detector_and_explainer()
    events = [_row_to_flow_event(row, f"flow-{i}") for i, (_, row) in enumerate(test_df.iterrows())]

    source = StubFlowEventSource(events)
    inner_producer = StubAlertProducer()
    service = ScoringService(source, detector, inner_producer, explainer=explainer)

    for event in source:
        service.process_event(event)

    assert service.processed == len(events)
    assert service.alerts_published == len(inner_producer.published)
    for alert in inner_producer.published:
        validate_alert_event(alert)
        assert alert["stage2_predicted_class"] != "BENIGN"
        assert alert["explanation"]  # SHAP contributions attached


def test_alert_on_stage1_flag_only_publishes_more_alerts():
    detector, explainer, test_df = _fit_detector_and_explainer()
    events = [_row_to_flow_event(row, f"flow-{i}") for i, (_, row) in enumerate(test_df.iterrows())]

    strict_producer = StubAlertProducer()
    strict_service = ScoringService(StubFlowEventSource(events), detector, strict_producer, explainer=explainer)
    for event in events:
        strict_service.process_event(event)

    loose_producer = StubAlertProducer()
    loose_service = ScoringService(
        StubFlowEventSource(events), detector, loose_producer, explainer=explainer,
        config=ScoringServiceConfig(alert_on_stage1_flag_only=True),
    )
    for event in events:
        loose_service.process_event(event)

    assert len(loose_producer.published) >= len(strict_producer.published)


def test_scoring_service_works_through_buffered_alert_producer():
    detector, explainer, test_df = _fit_detector_and_explainer()
    events = [_row_to_flow_event(row, f"flow-{i}") for i, (_, row) in enumerate(test_df.iterrows())][:20]

    stub = StubAlertProducer()
    buffered = BufferedAlertProducer(stub, queue_size=100, retry_backoff_s=0.01)
    service = ScoringService(StubFlowEventSource(events), detector, buffered, explainer=explainer)

    service.run()  # drains the stub source, then closes source + buffered producer

    assert stub.closed
    stats = buffered.stats()
    assert stats["published"] == service.alerts_published
    assert stats["dropped"] == 0
