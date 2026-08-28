"""CLI: run the live scoring service, consuming flow events and publishing alerts.

    python -m ids_ml.serve --model-dir models --bootstrap-servers localhost:9092

`--use-stub` runs against in-memory stubs instead of a real Kafka broker --
useful for smoke-testing a trained model without standing up Kafka.

If `--model-dir` contains gate artifacts saved by `ids_ml.train --gate ...`
(`gate_type.txt` + `escalation_gate.joblib`, plus `openset_gate.joblib` for
the openset case), they're loaded automatically and `TwoStageDetector` runs
the full three-way open-set router; otherwise it falls back to the
original closed-set behavior unchanged. There is no `--gate` flag here --
the gate type is a property of the trained model artifacts, not a serving-
time choice, so it's read from what `train.py` produced rather than
re-specified.
"""

from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path

from .alert_producer import BufferedAlertProducer, KafkaAlertProducer, StubAlertProducer
from .conformal_gate import ConformalGate
from .explainability import ShapExplainer
from .flow_consumer import KafkaFlowEventSource, StubFlowEventSource
from .openset_head import OpenMaxHead
from .pipeline import TwoStageDetector
from .schema import ALERT_TOPIC, FLOW_TOPIC
from .scoring_service import ScoringService, ScoringServiceConfig
from .softmax_gate import SoftmaxGate
from .stage1_iforest import AnomalyPreFilter
from .stage2_xgboost import AttackClassifier
from .train import ESCALATION_GATE_FILENAME, GATE_TYPE_FILENAME, OPENSET_GATE_FILENAME

logger = logging.getLogger("ids_ml.serve")


def load_gate(model_dir: Path, stage2: AttackClassifier):
    """Returns (gate, escalation_gate, escalation_trigger_name), all None/""
    if `model_dir` has no gate artifacts (the original closed-set-only
    behavior)."""
    gate_type_path = model_dir / GATE_TYPE_FILENAME
    if not gate_type_path.exists():
        return None, None, ""

    gate_type = gate_type_path.read_text().strip()
    if gate_type == "openset":
        gate = OpenMaxHead.load(model_dir / OPENSET_GATE_FILENAME, stage2)
    elif gate_type == "softmax":
        gate = SoftmaxGate(stage2)
    else:
        raise ValueError(f"unknown gate type in {gate_type_path}: {gate_type!r}")

    escalation_gate = ConformalGate.load(model_dir / ESCALATION_GATE_FILENAME)
    return gate, escalation_gate, gate_type


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

    gate, escalation_gate, escalation_trigger_name = load_gate(model_dir, stage2)
    if gate is not None:
        logger.info(
            "Loaded %s gate: escalation threshold=%.4f (budget=%.2f)",
            escalation_trigger_name,
            escalation_gate.threshold,
            escalation_gate.budget,
        )
    detector = TwoStageDetector(
        stage1, stage2, gate=gate, escalation_gate=escalation_gate, escalation_trigger_name=escalation_trigger_name
    )
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
