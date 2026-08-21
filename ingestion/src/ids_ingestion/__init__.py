"""Ingestion and streaming layer for a real-time network intrusion detection system.

This package captures live or replayed network traffic, groups packets into
flows, extracts sliding-window flow features, and publishes those features as
events to a downstream stream (Kafka by default). It does not implement any
detection logic itself -- see the project README for module boundaries.
"""

__version__ = "0.1.0"
