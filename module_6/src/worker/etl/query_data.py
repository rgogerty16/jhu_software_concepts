"""query_data.py — analytics recompute for the worker's recompute_analytics task.

Computes a small set of scalar metrics over the ``applicants`` table and upserts
them into ``analysis_summary``, the cache the web UI renders.  Running this off
the web tier means the page load never pays for the aggregate scan; the cached
numbers (and their ``updated_at`` timestamp) only change after the worker
processes a ``recompute_analytics`` message.

SQL-injection defence: the aggregate binds all constants through ``%s``
placeholders and the table name is composed with :class:`psycopg.sql.Identifier`.
"""

from psycopg import sql

_TABLE = sql.Identifier("applicants")

# One pass over the table produces every cached metric.
_AGGREGATE = sql.SQL("""
    SELECT
        COUNT(*)::float                                        AS total_applicants,
        COUNT(DISTINCT llm_generated_university)::float        AS distinct_universities,
        COUNT(DISTINCT llm_generated_program)::float           AS distinct_programs,
        ROUND(AVG(CASE WHEN gpa BETWEEN %s AND %s THEN gpa END)::numeric, 3)::float
                                                               AS avg_gpa,
        ROUND(
            100.0 * SUM(CASE WHEN status = %s THEN 1 ELSE 0 END)
            / NULLIF(COUNT(*), 0), 2
        )::float                                               AS pct_accepted
    FROM {tbl}
""").format(tbl=_TABLE)

_UPSERT = sql.SQL("""
    INSERT INTO analysis_summary (metric, value, updated_at)
    VALUES (%(metric)s, %(value)s, now())
    ON CONFLICT (metric) DO UPDATE
        SET value = EXCLUDED.value, updated_at = now()
""")

_METRIC_NAMES = (
    "total_applicants",
    "distinct_universities",
    "distinct_programs",
    "avg_gpa",
    "pct_accepted",
)


def recompute_summary(cur) -> dict:
    """Recompute every cached metric and upsert it into ``analysis_summary``.

    :param cur: An open psycopg cursor.
    :returns: A dict mapping each metric name to its freshly computed value.
    :rtype: dict
    """
    cur.execute(_AGGREGATE, (0, 4.0, "Accepted"))
    row = cur.fetchone()
    metrics = dict(zip(_METRIC_NAMES, row))
    for metric, value in metrics.items():
        cur.execute(_UPSERT, {"metric": metric, "value": value})
    return metrics


def handle_recompute_analytics(conn, payload: dict | None = None) -> dict:
    """Recompute the cached analytics within the caller's transaction.

    :param conn: An open psycopg connection supplied by the consumer.
    :param payload: Unused; accepted for a uniform task-handler signature.
    :type payload: dict or None
    :returns: A summary dict — ``{"metrics": {...}}``.
    :rtype: dict
    """
    _ = payload
    with conn.cursor() as cur:
        metrics = recompute_summary(cur)
    return {"metrics": metrics}
