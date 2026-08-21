"""ML detection layer for a real-time network intrusion detection system.

Consumes flow-feature events published by the ingestion module's Kafka
topic, scores them with a two-stage detector (Isolation Forest pre-filter +
XGBoost classifier), and publishes alerts to a downstream topic for the
dashboard/API module. This package does not implement ingestion, a
dashboard, or an API -- see the project README for module boundaries.

This module intentionally does not import the `ids_ingestion` package: the
only coupling between the two is the documented Kafka topic contract in
`ids_ml.schema`, duplicated here from the ingestion module's contract so
each module stays independently buildable and testable.
"""

__version__ = "0.1.0"
