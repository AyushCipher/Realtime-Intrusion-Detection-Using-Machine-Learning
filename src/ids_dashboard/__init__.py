"""Alert dashboard and API layer for a real-time network intrusion detection system.

Consumes scored alerts published by the ML module's Kafka topic, stores them
for historical query/filter, pushes live alerts to connected clients over a
WebSocket, and serves a React dashboard. This package does not implement
ingestion or any detection model -- see the project README for module
boundaries.

Like `ids_ml` relative to `ids_ingestion`, this package has no code
dependency on `ids_ml`: the only coupling is the documented Kafka topic
contract duplicated in `schema.py`.
"""

__version__ = "0.1.0"
