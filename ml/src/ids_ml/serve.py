"""CLI: run the live scoring service, consuming flow events and publishing alerts.

    python -m ids_ml.serve --model-dir models --bootstrap-servers localhost:9092

`--use-stub` runs against in-memory stubs instead of a real Kafka broker --
useful for smoke-testing a trained model without standing up Kafka.
"""

from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path

from .alert_producer import BufferedAlertProducer, KafkaAlertProducer, StubAlertProducer
from .explainability import ShapExplainer
from .flow_consumer import KafkaFlowEventSource, StubFlowEventSource
from .pipeline import TwoStageDetector
from .schema import ALERT_TOPIC, FLOW_TOPIC
from .scoring_service import ScoringService, ScoringServiceConfig
from .stage1_iforest import AnomalyPreFilter
from .stage2_xgboost import AttackClassifier

logger = logging.getLogger("ids_ml.serve")


def parse_args(argv=None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run the ids_ml live scoring service")
    parser.add_argument("--model-dir", default="models")
    parser.add_argument("--bootstrap-servers", default="localhost:9092")
    parser.add_argument("--flow-topic", default=FLOW_TOPIC)
    parser.add_argument("--alert-topic", default=ALERT_TOPIC)
    parser.add_argument("--use-stub", action="store_true", help="Use in-memory stubs instead of Kafka")
    parser.add_argument("--explain-top-k", type=int, default=5)
    parser.add_argument(
        "--alert-on-stage1-flag-only",
        action="store_true",
        help="Also publish alerts stage 1 flagged that stage 2 resolved back to BENIGN",
    )
    parser.add_argument("--drop-policy", choices=["drop_oldest", "block"], default="drop_oldest")
    parser.add_argument("-v", "--verbose", action="store_true")
    return parser.parse_args(argv)


def main(argv=None) -> int:
    args = parse_args(argv)
    logging.basicConfig(
        level=logging.DEBUG if args.verbose else logging.INFO,
        format="%(asctime)s %(levelname)s %(name)s: %(message)s",
    )

    model_dir = Path(args.model_dir)
    stage1 = AnomalyPreFilter.load(model_dir / "stage1_iforest.joblib")
    stage2 = AttackClassifier.load(model_dir / "stage2_xgboost.joblib")
    detector = TwoStageDetector(stage1, stage2)
    explainer = ShapExplainer(stage2)

    bootstrap_servers = [s.strip() for s in args.bootstrap_servers.split(",")]

    if args.use_stub:
        source = StubFlowEventSource([])
        inner_producer = StubAlertProducer()
    else:
        source = KafkaFlowEventSource(bootstrap_servers, topic=args.flow_topic)
        inner_producer = KafkaAlertProducer(bootstrap_servers, topic=args.alert_topic)

    producer = BufferedAlertProducer(inner_producer, drop_policy=args.drop_policy)
    config = ScoringServiceConfig(
        explain_top_k=args.explain_top_k, alert_on_stage1_flag_only=args.alert_on_stage1_flag_only
    )
    service = ScoringService(source, detector, producer, explainer=explainer, config=config)

    try:
        service.run()
    except KeyboardInterrupt:
        logger.info("Interrupted; shutting down")
        source.close()
        producer.close()

    logger.info("Processed %d flows, published %d alerts", service.processed, service.alerts_published)
    return 0


if __name__ == "__main__":
    sys.exit(main())
