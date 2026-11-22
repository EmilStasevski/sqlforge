"""Kafka `ask.events` -> ClickHouse `sqlforge.ask_events` (batched inserts).

Env: KAFKA_BROKERS (localhost:9092), KAFKA_TOPIC (ask.events), CLICKHOUSE_HOST/PORT/USER/PASSWORD.
Run: uv run python -m sqlforge.serve.consumer
"""

from __future__ import annotations

import datetime as dt
import json
import logging
import os
import time

DDL = """
CREATE TABLE IF NOT EXISTS sqlforge.ask_events (
    ts DateTime64(3, 'UTC'),
    question String,
    db LowCardinality(String),
    cache LowCardinality(String),
    status UInt16,
    outcome LowCardinality(String),
    sql String,
    latency_ms Float64
) ENGINE = MergeTree ORDER BY ts
"""
COLUMNS = ["ts", "question", "db", "cache", "status", "outcome", "sql", "latency_ms"]


def parse_event(raw: bytes) -> list | None:
    """Kafka message value -> ClickHouse row (None if malformed)."""
    try:
        e = json.loads(raw)
        ts = dt.datetime.fromisoformat(e["ts"].replace("Z", "+00:00"))
        return [
            ts,
            e.get("question", ""),
            e.get("db", ""),
            e.get("cache", ""),
            int(e.get("status", 0)),
            e.get("outcome") or "",
            e.get("sql") or "",
            float(e.get("latency_ms", 0.0)),
        ]
    except (ValueError, KeyError, TypeError):
        return None


def main() -> None:
    import clickhouse_connect
    from confluent_kafka import Consumer

    logging.basicConfig(level=logging.INFO)
    ch = None
    for _ in range(60):  # wait for ClickHouse to come up
        try:
            ch = clickhouse_connect.get_client(
                host=os.environ.get("CLICKHOUSE_HOST", "localhost"),
                port=int(os.environ.get("CLICKHOUSE_PORT", "8123")),
                username=os.environ.get("CLICKHOUSE_USER", "sqlforge"),
                password=os.environ.get("CLICKHOUSE_PASSWORD", "sqlforge"),
            )
            break
        except Exception:  # noqa: BLE001
            time.sleep(2)
    if ch is None:
        raise SystemExit("ClickHouse not reachable")
    ch.command("CREATE DATABASE IF NOT EXISTS sqlforge")
    ch.command(DDL)
    consumer = Consumer(
        {
            "bootstrap.servers": os.environ.get("KAFKA_BROKERS", "localhost:9092"),
            "group.id": "clickhouse-sink",
            "auto.offset.reset": "earliest",
            "enable.auto.commit": False,
        }
    )
    consumer.subscribe([os.environ.get("KAFKA_TOPIC", "ask.events")])
    batch: list[list] = []
    last = time.time()
    while True:
        msg = consumer.poll(0.5)
        if msg is not None and not msg.error():
            row = parse_event(msg.value())
            if row:
                batch.append(row)
        if batch and (len(batch) >= 500 or time.time() - last > 1.0):
            ch.insert("sqlforge.ask_events", batch, column_names=COLUMNS)
            consumer.commit(asynchronous=False)  # at-least-once: commit after the insert
            logging.info("inserted %d events", len(batch))
            batch, last = [], time.time()


if __name__ == "__main__":
    main()
