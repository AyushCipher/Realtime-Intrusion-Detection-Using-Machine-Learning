"""CLI: run the live Tier 2 reasoning service, consuming escalated alerts
and publishing explanations.

    python -m ids_tier2.serve --bootstrap-servers localhost:9092

`--use-stub` runs against in-memory stubs instead of a real Kafka broker.
`--llm` selects the LLM client ("stub", the default -- no API key needed;
"anthropic", which requires `pip install anthropic` and
`ANTHROPIC_API_KEY` set; or "gemini", which requires `pip install
google-genai` and `GEMINI_API_KEY` set -- a lower-friction option since
Google AI Studio keys work on the free tier immediately with no payment
method). `--no-rag` disables retrieval (the H3 ablation switch: does
grounding the LLM in retrieved ATT&CK context actually improve
explanation quality, vs. a bare LLM call).

Real clients (`anthropic`/`gemini`) are wrapped in `retry.RetryingLLMClient`
by default -- built after live testing showed transient 503s, hard 429
rate limits, and ConnectError failures all occur in practice (see
README's "Latency (live-verified -- and a reliability problem)" section).
`--min-interval-s` paces requests proactively to stay under a known
per-minute limit instead of only reacting after hitting it; `--max-retries`
and `--base-backoff-s` control reactive retry. `--no-retry` disables the
wrapper entirely (e.g. for isolating whether a failure is retry logic vs.
the underlying client).
"""

from __future__ import annotations

import argparse
import logging
import sys

from .alert_consumer import KafkaEscalatedAlertSource, StubEscalatedAlertSource
from .explanation_producer import KafkaExplanationProducer, StubExplanationProducer
from .llm_client import AnthropicLLMClient, GeminiLLMClient, LLMClient, StubLLMClient
from .reasoner import Tier2Reasoner
from .retry import RetryingLLMClient
from .schema import ALERT_TOPIC, EXPLANATION_TOPIC
from .service import Tier2Service

logger = logging.getLogger("ids_tier2.serve")


def parse_args(argv=None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run the ids_tier2 live reasoning service")
    parser.add_argument("--bootstrap-servers", default="localhost:9092")
    parser.add_argument("--alert-topic", default=ALERT_TOPIC)
    parser.add_argument("--explanation-topic", default=EXPLANATION_TOPIC)
    parser.add_argument("--use-stub", action="store_true", help="Use in-memory stubs instead of Kafka")
    parser.add_argument("--llm", choices=["stub", "anthropic", "gemini"], default="stub")
    parser.add_argument("--anthropic-model", default="claude-sonnet-5")
    parser.add_argument("--gemini-model", default="gemini-3.6-flash")
    parser.add_argument("--no-rag", action="store_true", help="Disable retrieval (bare-LLM ablation)")
    parser.add_argument("--top-k", type=int, default=3, help="Retrieved technique count when RAG is enabled")
    parser.add_argument("--no-retry", action="store_true", help="Disable RetryingLLMClient (real clients only)")
    parser.add_argument("--max-retries", type=int, default=4)
    parser.add_argument("--base-backoff-s", type=float, default=15.0)
    parser.add_argument(
        "--min-interval-s",
        type=float,
        default=12.0,
        help="Minimum seconds between LLM calls, paced proactively -- default 12s matches the 5 requests/minute "
        "free-tier limit observed live for Gemini (see README); set to 0 to disable pacing",
    )
    parser.add_argument("-v", "--verbose", action="store_true")
    return parser.parse_args(argv)


def build_llm_client(args: argparse.Namespace) -> LLMClient:
    if args.llm == "anthropic":
        real_client: LLMClient = AnthropicLLMClient(model=args.anthropic_model)
    elif args.llm == "gemini":
        real_client = GeminiLLMClient(model=args.gemini_model)
    else:
        return StubLLMClient()

    if args.no_retry:
        return real_client
    return RetryingLLMClient(
        inner=real_client,
        max_retries=args.max_retries,
        base_backoff_s=args.base_backoff_s,
        min_interval_s=args.min_interval_s,
    )


def main(argv=None) -> int:
    args = parse_args(argv)
    logging.basicConfig(
        level=logging.DEBUG if args.verbose else logging.INFO,
        format="%(asctime)s %(levelname)s %(name)s: %(message)s",
    )

    llm_client = build_llm_client(args)

    reasoner = Tier2Reasoner(llm_client, use_rag=not args.no_rag, top_k=args.top_k)

    bootstrap_servers = [s.strip() for s in args.bootstrap_servers.split(",")]
    if args.use_stub:
        source = StubEscalatedAlertSource([])
        producer = StubExplanationProducer()
    else:
        source = KafkaEscalatedAlertSource(bootstrap_servers, topic=args.alert_topic)
        producer = KafkaExplanationProducer(bootstrap_servers, topic=args.explanation_topic)

    service = Tier2Service(source, reasoner, producer)

    try:
        service.run()
    except KeyboardInterrupt:
        logger.info("Interrupted; shutting down")
        source.close()
        producer.close()

    logger.info("Processed %d escalated alerts, published %d explanations", service.processed, service.explanations_published)
    return 0


if __name__ == "__main__":
    sys.exit(main())
