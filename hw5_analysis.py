"""
LLM Zoomcamp 2026 — HW5 Part 2: SQLite exporter + analysis for Q4, Q5, Q6

Run AFTER hw5_monitoring.py works and you've answered Q1-Q3.
This swaps the console exporter for a SQLite exporter, then re-runs
the query multiple times so you can analyze the data with pandas.
"""

import os
import time
import sqlite3
from dotenv import load_dotenv
from opentelemetry import trace
from opentelemetry.sdk.trace import TracerProvider
from opentelemetry.sdk.trace.export import SimpleSpanProcessor, SpanExporter, SpanExportResult

load_dotenv()

# ---------------------------------------------------------------------------
# Tracer setup with SQLite exporter (replaces ConsoleSpanExporter)
# ---------------------------------------------------------------------------
class SQLiteSpanExporter(SpanExporter):
    def __init__(self, db_path="traces.db"):
        self.conn = sqlite3.connect(db_path)
        self.conn.execute("""
            CREATE TABLE IF NOT EXISTS spans (
                name TEXT,
                start_time INTEGER,
                end_time INTEGER,
                input_tokens INTEGER,
                output_tokens INTEGER,
                cost REAL
            )
        """)
        self.conn.commit()

    def export(self, spans):
        for span in spans:
            attrs = dict(span.attributes or {})
            self.conn.execute(
                "INSERT INTO spans VALUES (?, ?, ?, ?, ?, ?)",
                (
                    span.name,
                    span.start_time,
                    span.end_time,
                    attrs.get("input_tokens"),
                    attrs.get("output_tokens"),
                    attrs.get("cost"),
                ),
            )
        self.conn.commit()
        return SpanExportResult.SUCCESS

    def shutdown(self):
        self.conn.close()

    def force_flush(self):
        return True


provider = TracerProvider()
provider.add_span_processor(SimpleSpanProcessor(SQLiteSpanExporter("traces.db")))
trace.set_tracer_provider(provider)

tracer = trace.get_tracer("llm-zoomcamp")

os.environ.setdefault("OPENAI_API_KEY", os.getenv("GROQ_API_KEY", ""))
os.environ.setdefault("OPENAI_BASE_URL", "https://api.groq.com/openai/v1")

# Re-import your RAGTraced class here, or copy it from hw5_monitoring.py
# (import after tracer setup, same as before)
from hw5_monitoring import traced_rag, QUERY  # noqa: E402

# ---------------------------------------------------------------------------
# Q4 + Q5 + Q6: run the query 4 times total, then analyze
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    N_RUNS = 4
    for i in range(N_RUNS):
        print(f"\n--- Run {i+1}/{N_RUNS} ---")
        if i > 0:
            print("Waiting 60s to stay under Groq's free-tier rate limit...")
            time.sleep(60)
        traced_rag.rag(QUERY)

    # ---- Analysis with pandas ----
    import pandas as pd

    conn = sqlite3.connect("traces.db")
    df = pd.read_sql("SELECT * FROM spans", conn)

    print("\n\n=== Q4: span names in the table ===")
    print(df["name"].unique())

    print("\n=== Q5: total duration per span type (excluding 'rag') ===")
    df["duration_ms"] = (df["end_time"] - df["start_time"]) / 1_000_000  # ns -> ms
    by_span = (
        df[df["name"] != "rag"]
        .groupby("name")["duration_ms"]
        .sum()
        .sort_values(ascending=False)
    )
    print(by_span)

    print("\n=== Q6: input token variation across llm runs ===")
    llm_tokens = df["name"] == "llm"
    llm_tokens = df[llm_tokens]["input_tokens"].dropna()
    print(llm_tokens.tolist())

    lines = []
    lines.append("=== Q4: span names in the table ===")
    lines.append(str(df["name"].unique()))
    lines.append("")
    lines.append("=== Q5: total duration per span type (excluding 'rag') ===")
    lines.append(str(by_span))
    lines.append("")
    lines.append("=== Q6: input token variation across llm runs ===")
    lines.append(str(llm_tokens.tolist()))
    if len(llm_tokens) > 1:
        pct_variation = (llm_tokens.max() - llm_tokens.min()) / llm_tokens.mean() * 100
        summary = f"Variation: {pct_variation:.1f}% (min={llm_tokens.min()}, max={llm_tokens.max()}, mean={llm_tokens.mean():.0f})"
        print(summary)
        lines.append(summary)

    with open("analysis_output.txt", "w") as f:
        f.write("\n".join(lines))
    print("\n--- Full results also saved to analysis_output.txt ---")
