"""consumer.py — long-running RabbitMQ worker.

Responsibilities
----------------
* On start-up, seed the schema and base data (idempotent) so a clean machine
  comes up populated.
* Connect to RabbitMQ via ``RABBITMQ_URL`` and declare the durable topology
  (direct exchange ``tasks`` → queue ``tasks_q`` bound on routing key ``tasks``).
* Apply backpressure with ``basic_qos(prefetch_count=1)`` — one message at a
  time, so a slow task never lets the queue overwhelm the worker.
* For each message: open a database transaction, route by ``kind`` through a
  task map, and **commit only on success**.  Acknowledge *after* the commit;
  on any error roll back and ``basic_nack(requeue=False)`` so a poison message
  is dropped instead of looping forever.

The pure logic (topology, dispatch, message handling, start-up load) is factored
into small functions so it can be unit-tested without a live broker.
"""

import json
import logging
import os

import pika

from etl.incremental_scraper import handle_scrape_new_data
from etl.query_data import handle_recompute_analytics

from db.db import get_conn
from db.load_data import load_data

EXCHANGE = "tasks"
QUEUE = "tasks_q"
ROUTING_KEY = "tasks"
PREFETCH = 1

# Route each task kind to its handler. Handlers share the signature
# ``handler(conn, payload) -> dict`` and run inside the per-message transaction.
TASK_MAP = {
    "scrape_new_data": handle_scrape_new_data,
    "recompute_analytics": handle_recompute_analytics,
}

logging.basicConfig(level=logging.INFO, format="%(asctime)s [worker] %(message)s")
logger = logging.getLogger("worker")


def declare_topology(channel) -> None:
    """Idempotently declare the durable exchange, queue, and binding.

    :param channel: An open pika channel.
    :returns: None
    :rtype: None
    """
    channel.exchange_declare(exchange=EXCHANGE, exchange_type="direct", durable=True)
    channel.queue_declare(queue=QUEUE, durable=True)
    channel.queue_bind(exchange=EXCHANGE, queue=QUEUE, routing_key=ROUTING_KEY)


def dispatch(kind: str, payload: dict) -> dict:
    """Run one task inside a fresh per-message transaction.

    Opens a connection (which commits on clean exit and rolls back on error),
    looks up the handler for ``kind``, and executes it.

    :param kind: The task kind from the message body.
    :type kind: str
    :param payload: The task payload.
    :type payload: dict
    :raises ValueError: If ``kind`` has no registered handler.
    :returns: The handler's summary dict.
    :rtype: dict
    """
    handler = TASK_MAP.get(kind)
    if handler is None:
        raise ValueError(f"unknown task kind: {kind!r}")
    with get_conn() as conn:
        return handler(conn, payload)


def on_message(channel, method, _properties, body) -> None:
    """Handle one delivery: route, commit, and ack — or roll back and nack.

    :param channel: The pika channel the message arrived on.
    :param method: Delivery metadata (carries ``delivery_tag``).
    :param _properties: AMQP message properties (unused).
    :param body: Raw message bytes.
    :returns: None
    :rtype: None
    """
    try:
        message = json.loads(body)
        kind = message.get("kind")
        payload = message.get("payload") or {}
        result = dispatch(kind, payload)
        channel.basic_ack(delivery_tag=method.delivery_tag)
        logger.info("task %s ok: %s", kind, result)
    except Exception as exc:  # pylint: disable=broad-exception-caught
        # Drop poison/malformed messages instead of requeuing forever.
        channel.basic_nack(delivery_tag=method.delivery_tag, requeue=False)
        logger.error("task failed, nacked (requeue=false): %s", exc)


def initial_load() -> None:
    """Seed the schema, base data, and cached analytics on start-up.

    Wrapped defensively: if the data file is missing the worker still comes up
    and serves the queue rather than crash-looping.

    :returns: None
    :rtype: None
    """
    try:
        total = load_data()
        with get_conn() as conn:
            handle_recompute_analytics(conn, {})
        logger.info("initial load complete: %s rows", total)
    except Exception as exc:  # pylint: disable=broad-exception-caught
        logger.warning("initial load skipped: %s", exc)


def build_channel(url: str):
    """Open a blocking RabbitMQ connection and return ``(connection, channel)``.

    :param url: AMQP URL (``RABBITMQ_URL``).
    :type url: str
    :returns: The open connection and channel pair.
    :rtype: tuple
    """
    connection = pika.BlockingConnection(pika.URLParameters(url))
    return connection, connection.channel()


def main() -> None:
    """Run the worker: initial load, then consume forever with prefetch=1.

    :returns: None
    :rtype: None
    """
    initial_load()
    _connection, channel = build_channel(os.environ["RABBITMQ_URL"])
    declare_topology(channel)
    channel.basic_qos(prefetch_count=PREFETCH)
    channel.basic_consume(queue=QUEUE, on_message_callback=on_message)
    logger.info("waiting for tasks on %s (prefetch=%s)", QUEUE, PREFETCH)
    channel.start_consuming()


if __name__ == "__main__":  # pragma: no cover
    main()
