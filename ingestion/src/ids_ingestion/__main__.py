"""Command-line entrypoint: `python -m ids_ingestion ...`."""

from __future__ import annotations

import argparse
import logging
import sys

from .config import PipelineConfig
from .pipeline import IngestionPipeline


def parse_args(argv=None) -> PipelineConfig:
    parser = argparse.ArgumentParser(description="IDS ingestion & streaming pipeline")
    source = parser.add_mutually_exclusive_group(required=True)
    source.add_argument("--interface", help="Network interface for live capture")
    source.add_argument("--pcap", help="Pcap file path to replay")

    parser.add_argument("--bpf-filter", default="tcp or udp", help="BPF filter for live capture")
    parser.add_argument("--replay-speed", type=float, default=1.0,
                         help="Pcap replay speed multiplier (0 = instant playback)")
    parser.add_argument("--active-timeout", type=float, default=120.0)
    parser.add_argument("--idle-timeout", type=float, default=60.0)
    parser.add_argument("--bootstrap-servers", default="localhost:9092",
                         help="Comma-separated Kafka bootstrap servers")
    parser.add_argument("--topic", default=None, help="Kafka topic (defaults to schema.DEFAULT_TOPIC)")
    parser.add_argument("--use-stub-producer", action="store_true",
                         help="Publish to an in-memory stub instead of Kafka")
    parser.add_argument("--drop-policy", choices=["drop_oldest", "block"], default="drop_oldest")
    parser.add_argument("-v", "--verbose", action="store_true")

    args = parser.parse_args(argv)

    kwargs = dict(
        interface=args.interface,
        bpf_filter=args.bpf_filter,
        pcap_path=args.pcap,
        replay_speed=args.replay_speed,
        active_timeout_s=args.active_timeout,
        idle_timeout_s=args.idle_timeout,
        bootstrap_servers=[s.strip() for s in args.bootstrap_servers.split(",")],
        use_stub_producer=args.use_stub_producer,
        drop_policy=args.drop_policy,
    )
    if args.topic:
        kwargs["topic"] = args.topic

    logging.basicConfig(
        level=logging.DEBUG if args.verbose else logging.INFO,
        format="%(asctime)s %(levelname)s %(name)s: %(message)s",
    )

    return PipelineConfig(**kwargs)


def main(argv=None) -> int:
    config = parse_args(argv)
    pipeline = IngestionPipeline.from_config(config)
    logger = logging.getLogger("ids_ingestion.cli")
    try:
        pipeline.run()
    except KeyboardInterrupt:
        logger.info("Interrupted; shutting down")
        pipeline.stop()
    logger.info(
        "Processed %d packets, published %d flow events",
        pipeline.processed_packets,
        pipeline.published_flows,
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())
