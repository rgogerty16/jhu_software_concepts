"""query_data.py — read-side SQL for the web analysis page.

Two kinds of results feed the page:

* **Live queries** (Q1–Q11) computed on each request — these reflect new rows
  the instant the worker's ``scrape_new_data`` task commits them.
* **Cached summary** read from ``analysis_summary`` — scalar metrics plus the
  timestamp of the last worker ``recompute_analytics`` run, so the page can show
  "cached vs live" and make the async recompute visible.

SQL-injection defence mirrors Module 5: statement shape is locked with
``sql.SQL`` / ``sql.Identifier`` before any value is bound through ``%s``.
"""

from psycopg import sql

from db.db import get_conn

_TABLE = sql.Identifier("applicants")
_MAX_LIMIT = 100
_DEFAULT_LIMIT = 5


def _live_queries(cur, limit: int) -> dict:
    """Run the Q1–Q11 analysis queries and return them as a dict.

    :param cur: An open psycopg cursor.
    :param limit: Row cap for the multi-row queries (Q10, Q11).
    :type limit: int
    :returns: Dict of live query results.
    :rtype: dict
    """
    results = {}
    one = sql.Literal(1)

    stmt = sql.SQL("SELECT COUNT(*) FROM {tbl} LIMIT {lim}").format(tbl=_TABLE, lim=one)
    cur.execute(stmt)
    results["total_entries"] = cur.fetchone()[0]

    stmt = sql.SQL("SELECT COUNT(*) FROM {tbl} WHERE term = %s LIMIT {lim}").format(
        tbl=_TABLE, lim=one)
    cur.execute(stmt, ("Fall 2026",))
    results["q1_fall_2026_count"] = cur.fetchone()[0]

    stmt = sql.SQL("""
        SELECT ROUND(100.0 * SUM(CASE WHEN us_or_international = %s THEN 1 ELSE 0 END)
               / NULLIF(COUNT(*), 0), 2)
        FROM {tbl} LIMIT {lim}
    """).format(tbl=_TABLE, lim=one)
    cur.execute(stmt, ("International",))
    val = cur.fetchone()[0]
    results["q2_pct_international"] = float(val) if val is not None else 0.0

    stmt = sql.SQL("""
        SELECT
            ROUND(AVG(CASE WHEN gpa    BETWEEN %s AND %s THEN gpa    END)::numeric, 2),
            ROUND(AVG(CASE WHEN gre    BETWEEN %s AND %s THEN gre    END)::numeric, 2),
            ROUND(AVG(CASE WHEN gre_v  BETWEEN %s AND %s THEN gre_v  END)::numeric, 2),
            ROUND(AVG(CASE WHEN gre_aw BETWEEN %s AND %s THEN gre_aw END)::numeric, 2)
        FROM {tbl} LIMIT {lim}
    """).format(tbl=_TABLE, lim=one)
    cur.execute(stmt, (0, 4.0, 130, 340, 130, 170, 0, 6.0))
    row = cur.fetchone()
    results["q3_avg_gpa"] = float(row[0]) if row[0] is not None else None
    results["q3_avg_gre"] = float(row[1]) if row[1] is not None else None
    results["q3_avg_gre_v"] = float(row[2]) if row[2] is not None else None
    results["q3_avg_gre_aw"] = float(row[3]) if row[3] is not None else None

    stmt = sql.SQL("""
        SELECT ROUND(100.0 * SUM(CASE WHEN status = %s THEN 1 ELSE 0 END)
               / NULLIF(COUNT(*), 0), 2)
        FROM {tbl} WHERE term = %s LIMIT {lim}
    """).format(tbl=_TABLE, lim=one)
    cur.execute(stmt, ("Accepted", "Fall 2026"))
    val = cur.fetchone()[0]
    results["q5_pct_accepted_fall2026"] = float(val) if val is not None else 0.0

    stmt = sql.SQL("""
        SELECT llm_generated_program,
               COUNT(*) AS total,
               SUM(CASE WHEN status = %s THEN 1 ELSE 0 END) AS accepted,
               ROUND(100.0 * SUM(CASE WHEN status = %s THEN 1 ELSE 0 END)
                     / COUNT(*), 1) AS rate
        FROM {tbl}
        WHERE llm_generated_program IS NOT NULL
        GROUP BY llm_generated_program
        HAVING COUNT(*) >= %s
        ORDER BY rate DESC
        LIMIT {lim}
    """).format(tbl=_TABLE, lim=sql.Literal(limit))
    cur.execute(stmt, ("Accepted", "Accepted", 10))
    results["q10_top_acceptance_programs"] = [
        {"program": r[0], "total": r[1], "accepted": r[2], "rate_pct": float(r[3])}
        for r in cur.fetchall()
    ]

    stmt = sql.SQL("""
        SELECT status, COUNT(*) AS n, ROUND(AVG(gpa)::numeric, 3) AS avg_gpa
        FROM {tbl}
        WHERE gpa IS NOT NULL AND status IN (%s, %s, %s)
        GROUP BY status
        ORDER BY avg_gpa DESC
        LIMIT {lim}
    """).format(tbl=_TABLE, lim=sql.Literal(limit))
    cur.execute(stmt, ("Accepted", "Rejected", "Waitlisted"))
    results["q11_gpa_by_status"] = [
        {"status": r[0], "n": r[1], "avg_gpa": float(r[2])} for r in cur.fetchall()
    ]
    return results


def _cached_summary(cur) -> tuple[dict, object]:
    """Read the worker-maintained ``analysis_summary`` cache.

    :param cur: An open psycopg cursor.
    :returns: ``(metrics, last_updated)`` — a metric→value dict and the newest
        ``updated_at`` timestamp (or None when the cache is empty).
    :rtype: tuple
    """
    cur.execute(sql.SQL("SELECT metric, value, updated_at FROM analysis_summary"))
    rows = cur.fetchall()
    metrics = {metric: value for (metric, value, _updated) in rows}
    stamps = [updated for (_metric, _value, updated) in rows if updated is not None]
    return metrics, (max(stamps) if stamps else None)


def run_queries(database_url: str | None = None, limit: int = _DEFAULT_LIMIT) -> dict:
    """Execute all page queries and return a single results dict.

    :param database_url: Optional Postgres URL override.
    :type database_url: str or None
    :param limit: Row cap for multi-row queries, clamped to ``[1, _MAX_LIMIT]``.
    :type limit: int
    :returns: Combined live + cached results for the template.
    :rtype: dict
    """
    limit = max(1, min(int(limit), _MAX_LIMIT))
    with get_conn(database_url) as conn:
        with conn.cursor() as cur:
            results = _live_queries(cur, limit)
            results["summary"], results["summary_updated"] = _cached_summary(cur)
    return results
