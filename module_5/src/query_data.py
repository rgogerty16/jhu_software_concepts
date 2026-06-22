"""
query_data.py — SQL analysis of the Grad Café applicants table.

SQL Injection Defence
---------------------
Every query is constructed with psycopg.sql composition:

* ``sql.SQL(...)``  — locks the query *shape* before any runtime data is bound.
* ``sql.Identifier(...)`` — safely double-quotes table/column names so they
  can never be interpreted as SQL keywords or injection payloads.
* ``sql.Literal(...)`` — embeds constant values (like LIMIT) directly in the
  SQL text at construction time, not at execution time.
* ``%s`` placeholders + ``cur.execute(stmt, params)`` — binds all runtime
  values through the driver's parameterisation layer.

Constructing the statement object (``stmt = sql.SQL(...).format(...)``)
is always done *before* calling ``cur.execute(stmt, params)``.  This
separation makes it structurally impossible for runtime parameters to
alter the query shape.

Every query includes a LIMIT clause.  Multi-row queries (Q10, Q11)
accept a configurable *limit* parameter clamped to 1–{max_limit}.

Usage::

    python query_data.py
"""

from psycopg import sql

from db import get_conn

# Hard ceiling on rows returned by any multi-row query.
_MAX_LIMIT = 100
# Default used when the caller does not supply one.
_DEFAULT_LIMIT = 5


def run_queries(database_url: str | None = None, limit: int = _DEFAULT_LIMIT) -> dict:
    """Execute all analysis queries and return results as a dict.

    :param database_url: Optional Postgres connection string.  When ``None``,
        ``db.py`` resolves credentials from the environment.  Tests inject the
        URL of a throwaway database to keep production data untouched.
    :type database_url: str or None
    :param limit: Maximum rows returned by multi-row queries (Q10, Q11).
        Clamped to ``[1, _MAX_LIMIT]``.  Defaults to ``_DEFAULT_LIMIT``.
    :type limit: int
    :returns: Dict of query results keyed by question name.
    :rtype: dict
    """
    # Clamp the caller-supplied limit to the safe range before any query runs.
    limit = max(1, min(int(limit), _MAX_LIMIT))

    results = {}

    with get_conn(database_url) as conn:
        with conn.cursor() as cur:

            # ------------------------------------------------------------------
            # Total row count — page intro ("Analysis of X entries")
            # ------------------------------------------------------------------
            stmt = sql.SQL(
                "SELECT COUNT(*) FROM {tbl} LIMIT {lim}"
            ).format(
                tbl=sql.Identifier("applicants"),
                lim=sql.Literal(1),
            )
            cur.execute(stmt)
            results["total_entries"] = cur.fetchone()[0]

            # ------------------------------------------------------------------
            # Q1 — How many entries are for Fall 2026?
            # ------------------------------------------------------------------
            stmt = sql.SQL(
                "SELECT COUNT(*) FROM {tbl} WHERE term = %s LIMIT {lim}"
            ).format(
                tbl=sql.Identifier("applicants"),
                lim=sql.Literal(1),
            )
            cur.execute(stmt, ("Fall 2026",))
            results["q1_fall_2026_count"] = cur.fetchone()[0]

            # ------------------------------------------------------------------
            # Q2 — Percentage of entries from International students
            # ------------------------------------------------------------------
            stmt = sql.SQL("""
                SELECT
                    ROUND(
                        100.0
                        * SUM(CASE WHEN us_or_international = %s THEN 1 ELSE 0 END)
                        / COUNT(*),
                        2
                    )
                FROM {tbl}
                LIMIT {lim}
            """).format(
                tbl=sql.Identifier("applicants"),
                lim=sql.Literal(1),
            )
            cur.execute(stmt, ("International",))
            val = cur.fetchone()[0]
            results["q2_pct_international"] = float(val) if val is not None else 0.0

            # ------------------------------------------------------------------
            # Q3 — Average GPA, GRE, GRE V, GRE AW (outlier-filtered)
            #
            # GPA valid range: 0–4.0  (Grad Café allows free-text entry)
            # GRE valid range: 130–340 (covers both section and combined scores)
            # GRE V valid range: 130–170
            # GRE AW valid range: 0–6.0
            # ------------------------------------------------------------------
            stmt = sql.SQL("""
                SELECT
                    ROUND(AVG(CASE WHEN gpa    BETWEEN %s AND %s THEN gpa    END)::numeric, 2),
                    ROUND(AVG(CASE WHEN gre    BETWEEN %s AND %s THEN gre    END)::numeric, 2),
                    ROUND(AVG(CASE WHEN gre_v  BETWEEN %s AND %s THEN gre_v  END)::numeric, 2),
                    ROUND(AVG(CASE WHEN gre_aw BETWEEN %s AND %s THEN gre_aw END)::numeric, 2)
                FROM {tbl}
                LIMIT {lim}
            """).format(
                tbl=sql.Identifier("applicants"),
                lim=sql.Literal(1),
            )
            cur.execute(stmt, (0, 4.0, 130, 340, 130, 170, 0, 6.0))
            row = cur.fetchone()
            results["q3_avg_gpa"]    = float(row[0]) if row[0] else None
            results["q3_avg_gre"]    = float(row[1]) if row[1] else None
            results["q3_avg_gre_v"]  = float(row[2]) if row[2] else None
            results["q3_avg_gre_aw"] = float(row[3]) if row[3] else None

            # ------------------------------------------------------------------
            # Q4 — Average GPA of American students in Fall 2026
            # ------------------------------------------------------------------
            stmt = sql.SQL("""
                SELECT ROUND(AVG(gpa)::numeric, 2)
                FROM {tbl}
                WHERE us_or_international = %s
                  AND term = %s
                  AND gpa BETWEEN %s AND %s
                LIMIT {lim}
            """).format(
                tbl=sql.Identifier("applicants"),
                lim=sql.Literal(1),
            )
            cur.execute(stmt, ("American", "Fall 2026", 0, 4.0))
            val = cur.fetchone()[0]
            results["q4_avg_gpa_american_fall2026"] = float(val) if val else None

            # ------------------------------------------------------------------
            # Q5 — Percentage of Fall 2026 entries that are Acceptances
            # ------------------------------------------------------------------
            stmt = sql.SQL("""
                SELECT
                    ROUND(
                        100.0
                        * SUM(CASE WHEN status = %s THEN 1 ELSE 0 END)
                        / COUNT(*),
                        2
                    )
                FROM {tbl}
                WHERE term = %s
                LIMIT {lim}
            """).format(
                tbl=sql.Identifier("applicants"),
                lim=sql.Literal(1),
            )
            cur.execute(stmt, ("Accepted", "Fall 2026"))
            val = cur.fetchone()[0]
            results["q5_pct_accepted_fall2026"] = float(val) if val is not None else 0.0

            # ------------------------------------------------------------------
            # Q6 — Average GPA of Fall 2026 Acceptances
            # ------------------------------------------------------------------
            stmt = sql.SQL("""
                SELECT ROUND(AVG(gpa)::numeric, 2)
                FROM {tbl}
                WHERE term = %s
                  AND status = %s
                  AND gpa BETWEEN %s AND %s
                LIMIT {lim}
            """).format(
                tbl=sql.Identifier("applicants"),
                lim=sql.Literal(1),
            )
            cur.execute(stmt, ("Fall 2026", "Accepted", 0, 4.0))
            val = cur.fetchone()[0]
            results["q6_avg_gpa_accepted_fall2026"] = float(val) if val else None

            # ------------------------------------------------------------------
            # Q7 — JHU Master's Computer Science entries (LLM-normalised fields)
            #
            # The LLM step from Module 2 standardises university names, so
            # "JHU", "Johns Hopkins", "John Hopkins" all map to a single value.
            # ------------------------------------------------------------------
            stmt = sql.SQL("""
                SELECT COUNT(*)
                FROM {tbl}
                WHERE LOWER(llm_generated_university) LIKE %s
                  AND LOWER(degree) = %s
                  AND LOWER(llm_generated_program) LIKE %s
                LIMIT {lim}
            """).format(
                tbl=sql.Identifier("applicants"),
                lim=sql.Literal(1),
            )
            cur.execute(stmt, ("%johns hopkins%", "masters", "%computer science%"))
            results["q7_jhu_masters_cs"] = cur.fetchone()[0]

            # ------------------------------------------------------------------
            # Q8 — 2026 PhD CS acceptances at top-4 schools — raw scraped fields
            #
            # The raw `program` field contains only the program name, not the
            # university name, so this query returns 0.  Contrast with Q9.
            # ------------------------------------------------------------------
            stmt = sql.SQL("""
                SELECT COUNT(*)
                FROM {tbl}
                WHERE term LIKE %s
                  AND status = %s
                  AND LOWER(degree) = %s
                  AND LOWER(program) LIKE %s
                  AND (
                       LOWER(program) LIKE %s
                    OR LOWER(program) LIKE %s
                    OR LOWER(program) LIKE %s
                    OR LOWER(program) LIKE %s
                  )
                LIMIT {lim}
            """).format(
                tbl=sql.Identifier("applicants"),
                lim=sql.Literal(1),
            )
            cur.execute(
                stmt,
                ("%2026%", "Accepted", "phd", "%computer science%",
                 "%georgetown%", "%mit%", "%stanford%", "%carnegie mellon%"),
            )
            results["q8_raw_top4_phd_cs_2026"] = cur.fetchone()[0]

            # ------------------------------------------------------------------
            # Q9 — Same as Q8, using LLM-normalised university field
            # ------------------------------------------------------------------
            stmt = sql.SQL("""
                SELECT COUNT(*)
                FROM {tbl}
                WHERE term LIKE %s
                  AND status = %s
                  AND LOWER(degree) = %s
                  AND LOWER(llm_generated_program) LIKE %s
                  AND (
                       LOWER(llm_generated_university) LIKE %s
                    OR LOWER(llm_generated_university) LIKE %s
                    OR LOWER(llm_generated_university) LIKE %s
                    OR LOWER(llm_generated_university) LIKE %s
                    OR LOWER(llm_generated_university) LIKE %s
                  )
                LIMIT {lim}
            """).format(
                tbl=sql.Identifier("applicants"),
                lim=sql.Literal(1),
            )
            cur.execute(
                stmt,
                ("%2026%", "Accepted", "phd", "%computer science%",
                 "%georgetown%", "%massachusetts institute%", "% mit%",
                 "%stanford%", "%carnegie mellon%"),
            )
            results["q9_llm_top4_phd_cs_2026"] = cur.fetchone()[0]

            # ------------------------------------------------------------------
            # Q10 (original) — Programs with highest acceptance rate
            #   Minimum 10 applicants to filter statistical noise.
            #   LIMIT is the caller-supplied value (default 5, max 100).
            # ------------------------------------------------------------------
            stmt = sql.SQL("""
                SELECT
                    llm_generated_program,
                    COUNT(*) AS total,
                    SUM(CASE WHEN status = %s THEN 1 ELSE 0 END) AS accepted,
                    ROUND(
                        100.0
                        * SUM(CASE WHEN status = %s THEN 1 ELSE 0 END)
                        / COUNT(*),
                        1
                    ) AS acceptance_rate_pct
                FROM {tbl}
                WHERE llm_generated_program IS NOT NULL
                GROUP BY llm_generated_program
                HAVING COUNT(*) >= %s
                ORDER BY acceptance_rate_pct DESC
                LIMIT {lim}
            """).format(
                tbl=sql.Identifier("applicants"),
                lim=sql.Literal(limit),
            )
            cur.execute(stmt, ("Accepted", "Accepted", 10))
            results["q10_top_acceptance_programs"] = [
                {"program": r[0], "total": r[1], "accepted": r[2], "rate_pct": float(r[3])}
                for r in cur.fetchall()
            ]

            # ------------------------------------------------------------------
            # Q11 (original) — Average GPA by admission outcome
            #   Returns at most 3–4 rows in practice; LIMIT caps at 100.
            # ------------------------------------------------------------------
            stmt = sql.SQL("""
                SELECT
                    status,
                    COUNT(*) AS n,
                    ROUND(AVG(gpa)::numeric, 3) AS avg_gpa
                FROM {tbl}
                WHERE gpa IS NOT NULL
                  AND status IN (%s, %s, %s)
                GROUP BY status
                ORDER BY avg_gpa DESC
                LIMIT {lim}
            """).format(
                tbl=sql.Identifier("applicants"),
                lim=sql.Literal(limit),
            )
            cur.execute(stmt, ("Accepted", "Rejected", "Waitlisted"))
            results["q11_gpa_by_status"] = [
                {"status": r[0], "n": r[1], "avg_gpa": float(r[2])}
                for r in cur.fetchall()
            ]

    return results


def print_results(r: dict) -> None:
    """Print all query results to stdout in a readable format.

    :param r: Results dict as returned by :func:`run_queries`.
    :type r: dict
    :returns: None
    :rtype: None
    """
    print("=" * 60)
    print("Grad Café SQL Analysis Results")
    print("=" * 60)

    print(f"\nQ1  Fall 2026 entries:                    {r['q1_fall_2026_count']:,}")
    print(f"Q2  International student %:               {r['q2_pct_international']:.2f}%")
    print(f"\nQ3  Average metrics (among reporters):")
    print(f"      GPA:    {r['q3_avg_gpa']}")
    print(f"      GRE:    {r['q3_avg_gre']}")
    print(f"      GRE V:  {r['q3_avg_gre_v']}")
    print(f"      GRE AW: {r['q3_avg_gre_aw']}")
    print(f"\nQ4  Avg GPA — American students, Fall 2026: {r['q4_avg_gpa_american_fall2026']}")
    print(f"Q5  Acceptance rate — Fall 2026:            {r['q5_pct_accepted_fall2026']:.2f}%")
    print(f"Q6  Avg GPA — Fall 2026 acceptances:        {r['q6_avg_gpa_accepted_fall2026']}")
    print(f"\nQ7  JHU Master's CS entries:               {r['q7_jhu_masters_cs']}")
    print(f"\nQ8  Top-4 PhD CS 2026 acceptances (raw):  {r['q8_raw_top4_phd_cs_2026']}")
    print("    (0 because raw `program` field has no university name — see Q9)")
    print(f"Q9  Top-4 PhD CS 2026 acceptances (LLM):  {r['q9_llm_top4_phd_cs_2026']}")

    print("\nQ10 Top 5 programs by acceptance rate (min 10 applicants):")
    for row in r["q10_top_acceptance_programs"]:
        print(f"      {row['program']:<40} {row['rate_pct']}%  ({row['accepted']}/{row['total']})")

    print("\nQ11 Average GPA by admission outcome:")
    for row in r["q11_gpa_by_status"]:
        print(f"      {row['status']:<12} avg GPA {row['avg_gpa']}  (n={row['n']:,})")

    print("=" * 60)


if __name__ == "__main__":  # pragma: no cover
    results = run_queries()
    print_results(results)
