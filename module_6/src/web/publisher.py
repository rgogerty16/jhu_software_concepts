"""publisher.py — RabbitMQ producer used by the Flask web tier.

The web app never does slow or data-modifying work in a request. Instead each
button publishes a task to RabbitMQ and returns immediately (HTTP 202), so the
page stays fast and stateless while the worker pool does the work.

Topology (declared idempotently on every publish so the app is safe to start in
any order relative to the broker):

* durable **direct** exchange ``tasks``
* durable queue ``tasks_q``
* binding on routing key ``tasks``

Messages are persistent (``delivery_mode=2``) so they survive a broker restart.
Failures are raised, never swallowed, so the Flask endpoint can surface 503.
"""

import json
import os
from datetime import datetime, timezone

import pika

EXCHANGE = "tasks"
QUEUE = "tasks_q"
ROUTING_KEY = "tasks"


def _open_channel():
    """Open a connection + channel and declare the durable topology.

    :returns: The open ``(connection, channel)`` pair. The caller must close
        the connection.
    :rtype: tuple
    """
    params = pika.URLParameters(os.environ["RABBITMQ_URL"])
    conn = pika.BlockingConnection(params)
    channel = conn.channel()
    channel.exchange_declare(exchange=EXCHANGE, exchange_type="direct", durable=True)
    channel.queue_declare(queue=QUEUE, durable=True)
    channel.queue_bind(exchange=EXCHANGE, queue=QUEUE, routing_key=ROUTING_KEY)
    return conn, channel


def publish_task(kind: str, payload: dict | None = None,
                 headers: dict | None = None) -> None:
    """Publish a persistent task message to the ``tasks`` exchange.

    :param kind: Task kind, e.g. ``"scrape_new_data"`` or ``"recompute_analytics"``.
    :type kind: str
    :param payload: Optional task payload; defaults to ``{}``.
    :type payload: dict or None
    :param headers: Optional AMQP headers.
    :type headers: dict or None
    :raises Exception: Propagates any broker/publish failure so the caller can
        return HTTP 503.
    :returns: None
    :rtype: None
    """
    body = json.dumps(
        {
            "kind": kind,
            "ts": datetime.now(timezone.utc).isoformat(),
            "payload": payload or {},
        },
        separators=(",", ":"),
    ).encode("utf-8")

    conn, channel = _open_channel()
    try:
        channel.basic_publish(
            exchange=EXCHANGE,
            routing_key=ROUTING_KEY,
            body=body,
            properties=pika.BasicProperties(delivery_mode=2, headers=headers or {}),
            mandatory=False,
        )
    finally:
        conn.close()
