"""
query_data.py — SQL analysis of the Grad Café applicants table.

Answers all required questions and two additional original questions.
Run directly to print results to the console.

Usage:
    python query_data.py
"""

from db import get_conn


def run_queries() -> dict:
    """Execute all analysis queries and return results as a dict."""
    results = {}

    with get_conn() as conn:
        with conn.cursor() as cur:

            # ------------------------------------------------------------------
            # Q1: How many entries are for Fall 2026?
            # ------------------------------------------------------------------
            cur.execute("SELECT COUNT(*) FROM applicants WHERE term = 'Fall 2026';")
            results["q1_fall_2026_count"] = cur.fetchone()[0]

            # ------------------------------------------------------------------
            # Q2: Percentage of entries from International students (not American or Other)
            # ------------------------------------------------------------------
            cur.execute("""
                SELECT
                    ROUND(
                        100.0 * SUM(CASE WHEN us_or_international = 'International' THEN 1 ELSE 0 END)
                        / COUNT(*),
                        2
                    )
                FROM applicants;
            """)
            results["q2_pct_international"] = float(cur.fetchone()[0])

            # ------------------------------------------------------------------
            # Q3: Average GPA, GRE, GRE V, GRE AW of applicants who provide each
            # GRE AW valid range: 0–6 (Grad Café has data-entry outliers like 99.99)
            # GRE (quant/verbal section) valid range: 130–170; combined: 260–340
            # We keep both section and combined scores by accepting 130–340.
            # GPA valid range: 0–4.0 (a handful of entries exceed this via typos)
            # ------------------------------------------------------------------
            cur.execute("""
                SELECT
                    ROUND(AVG(CASE WHEN gpa   BETWEEN 0   AND 4.0  THEN gpa   END)::numeric, 2),
                    ROUND(AVG(CASE WHEN gre   BETWEEN 130 AND 340  THEN gre   END)::numeric, 2),
                    ROUND(AVG(CASE WHEN gre_v BETWEEN 130 AND 170  THEN gre_v END)::numeric, 2),
                    ROUND(AVG(CASE WHEN gre_aw BETWEEN 0  AND 6.0  THEN gre_aw END)::numeric, 2)
                FROM applicants;
            """)
            row = cur.fetchone()
            results["q3_avg_gpa"]    = float(row[0]) if row[0] else None
            results["q3_avg_gre"]    = float(row[1]) if row[1] else None
            results["q3_avg_gre_v"]  = float(row[2]) if row[2] else None
            results["q3_avg_gre_aw"] = float(row[3]) if row[3] else None

            # ------------------------------------------------------------------
            # Q4: Average GPA of American students in Fall 2026
            # ------------------------------------------------------------------
            cur.execute("""
                SELECT ROUND(AVG(gpa)::numeric, 2)
                FROM applicants
                WHERE us_or_international = 'American'
                  AND term = 'Fall 2026'
                  AND gpa BETWEEN 0 AND 4.0;
            """)
            val = cur.fetchone()[0]
            results["q4_avg_gpa_american_fall2026"] = float(val) if val else None

            # ------------------------------------------------------------------
            # Q5: Percentage of Fall 2026 entries that are Acceptances
            # ------------------------------------------------------------------
            cur.execute("""
                SELECT
                    ROUND(
                        100.0 * SUM(CASE WHEN status = 'Accepted' THEN 1 ELSE 0 END)
                        / COUNT(*),
                        2
                    )
                FROM applicants
                WHERE term = 'Fall 2026';
            """)
            results["q5_pct_accepted_fall2026"] = float(cur.fetchone()[0])

            # ------------------------------------------------------------------
            # Q6: Average GPA of Fall 2026 Acceptances
            # ------------------------------------------------------------------
            cur.execute("""
                SELECT ROUND(AVG(gpa)::numeric, 2)
                FROM applicants
                WHERE term = 'Fall 2026'
                  AND status = 'Accepted'
                  AND gpa BETWEEN 0 AND 4.0;
            """)
            val = cur.fetchone()[0]
            results["q6_avg_gpa_accepted_fall2026"] = float(val) if val else None

            # ------------------------------------------------------------------
            # Q7: Entries applied to JHU for a Master's in Computer Science
            # Uses the LLM-generated university field for standardized matching.
            # LLM maps "JHU", "Johns Hopkins", "John Hopkins" → "Johns Hopkins University"
            # ------------------------------------------------------------------
            cur.execute("""
                SELECT COUNT(*)
                FROM applicants
                WHERE LOWER(llm_generated_university) LIKE '%johns hopkins%'
                  AND LOWER(degree) = 'masters'
                  AND LOWER(llm_generated_program) LIKE '%computer science%';
            """)
            results["q7_jhu_masters_cs"] = cur.fetchone()[0]

            # ------------------------------------------------------------------
            # Q8: 2026 Acceptances from Georgetown, MIT, Stanford, or CMU for PhD in CS
            #     using raw (scraped) fields
            # ------------------------------------------------------------------
            cur.execute("""
                SELECT COUNT(*)
                FROM applicants
                WHERE term LIKE '%2026%'
                  AND status = 'Accepted'
                  AND LOWER(degree) = 'phd'
                  AND LOWER(program) LIKE '%computer science%'
                  AND (
                       LOWER(program) LIKE '%georgetown%'
                    OR LOWER(program) LIKE '%mit%'
                    OR LOWER(program) LIKE '%stanford%'
                    OR LOWER(program) LIKE '%carnegie mellon%'
                  );
            """)
            results["q8_raw_top4_phd_cs_2026"] = cur.fetchone()[0]

            # ------------------------------------------------------------------
            # Q9: Same as Q8 but using LLM-generated university field
            # ------------------------------------------------------------------
            cur.execute("""
                SELECT COUNT(*)
                FROM applicants
                WHERE term LIKE '%2026%'
                  AND status = 'Accepted'
                  AND LOWER(degree) = 'phd'
                  AND LOWER(llm_generated_program) LIKE '%computer science%'
                  AND (
                       LOWER(llm_generated_university) LIKE '%georgetown%'
                    OR LOWER(llm_generated_university) LIKE '%massachusetts institute%'
                    OR LOWER(llm_generated_university) LIKE '% mit%'
                    OR LOWER(llm_generated_university) LIKE '%stanford%'
                    OR LOWER(llm_generated_university) LIKE '%carnegie mellon%'
                  );
            """)
            results["q9_llm_top4_phd_cs_2026"] = cur.fetchone()[0]

            # ------------------------------------------------------------------
            # Q10 (original): Which program has the highest acceptance rate?
            #     (minimum 10 applicants to filter noise)
            # ------------------------------------------------------------------
            cur.execute("""
                SELECT
                    llm_generated_program,
                    COUNT(*) AS total,
                    SUM(CASE WHEN status = 'Accepted' THEN 1 ELSE 0 END) AS accepted,
                    ROUND(
                        100.0 * SUM(CASE WHEN status = 'Accepted' THEN 1 ELSE 0 END) / COUNT(*),
                        1
                    ) AS acceptance_rate_pct
                FROM applicants
                WHERE llm_generated_program IS NOT NULL
                GROUP BY llm_generated_program
                HAVING COUNT(*) >= 10
                ORDER BY acceptance_rate_pct DESC
                LIMIT 5;
            """)
            results["q10_top_acceptance_programs"] = [
                {"program": r[0], "total": r[1], "accepted": r[2], "rate_pct": float(r[3])}
                for r in cur.fetchall()
            ]

            # ------------------------------------------------------------------
            # Q11 (original): For applicants who reported a GPA, how does average
            #     GPA differ between Accepted, Rejected, and Waitlisted outcomes?
            # ------------------------------------------------------------------
            cur.execute("""
                SELECT
                    status,
                    COUNT(*) AS n,
                    ROUND(AVG(gpa)::numeric, 3) AS avg_gpa
                FROM applicants
                WHERE gpa IS NOT NULL
                  AND status IN ('Accepted', 'Rejected', 'Waitlisted')
                GROUP BY status
                ORDER BY avg_gpa DESC;
            """)
            results["q11_gpa_by_status"] = [
                {"status": r[0], "n": r[1], "avg_gpa": float(r[2])}
                for r in cur.fetchall()
            ]

    return results


def print_results(r: dict) -> None:
    """Print all query results to stdout in a readable format."""
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
    print(f"    (0 because raw `program` field has no university name — see Q9)")
    print(f"Q9  Top-4 PhD CS 2026 acceptances (LLM):  {r['q9_llm_top4_phd_cs_2026']}")

    print(f"\nQ10 Top 5 programs by acceptance rate (min 10 applicants):")
    for row in r["q10_top_acceptance_programs"]:
        print(f"      {row['program']:<40} {row['rate_pct']}%  ({row['accepted']}/{row['total']})")

    print(f"\nQ11 Average GPA by admission outcome:")
    for row in r["q11_gpa_by_status"]:
        print(f"      {row['status']:<12} avg GPA {row['avg_gpa']}  (n={row['n']:,})")

    print("=" * 60)


if __name__ == "__main__":
    results = run_queries()
    print_results(results)
