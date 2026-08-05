"""
LLM Zoomcamp 2026 — Homework 5: Monitoring
Adapted for Groq, matching the REAL rag_helper.py (RAGBase) structure.

IMPORTANT: rag_helper.py's RAGBase.llm() uses OpenAI's newer *Responses API*
(`client.responses.create(...)`), which Groq does NOT support — Groq only
supports the older *Chat Completions API* (`client.chat.completions.create`).
So RAGTraced below overrides `llm()` and `rag()` fully, using
chat.completions.create with equivalent behavior, instead of just wrapping
RAGBase's versions.

SETUP:
    mkdir llm-zoomcamp-hw5 && cd llm-zoomcamp-hw5
    uv init
    uv add minsearch openai python-dotenv opentelemetry-api opentelemetry-sdk

    PREFIX=https://raw.githubusercontent.com/DataTalksClub/llm-zoomcamp/main/cohorts/2026/05-monitoring
    wget $PREFIX/rag_helper.py
    wget $PREFIX/starter.py

.env file:
    GROQ_API_KEY=gsk_...
"""

import os
from dotenv import load_dotenv
from openai import OpenAI

from opentelemetry import trace
from opentelemetry.sdk.trace import TracerProvider
from opentelemetry.sdk.trace.export import SimpleSpanProcessor, SpanExporter, SpanExportResult

from rag_helper import RAGBase, INSTRUCTIONS, PROMPT_TEMPLATE

load_dotenv()


# ---------------------------------------------------------------------------
# A file-writing exporter instead of ConsoleSpanExporter — writes clean,
# readable span info to spans_output.txt so you don't have to scroll/copy
# from the terminal.
# ---------------------------------------------------------------------------
class FileSpanExporter(SpanExporter):
    def __init__(self, path="spans_output.txt"):
        self.path = path
        open(self.path, "w").close()  # clear file at start

    def export(self, spans):
        with open(self.path, "a") as f:
            for span in spans:
                f.write(f"SPAN NAME: {span.name}\n")
                f.write(f"  start_time: {span.start_time}\n")
                f.write(f"  end_time:   {span.end_time}\n")
                f.write(f"  duration_ns: {span.end_time - span.start_time}\n")
                f.write(f"  duration_ms: {(span.end_time - span.start_time) / 1_000_000:.2f}\n")
                attrs = dict(span.attributes or {})
                f.write(f"  attributes: {attrs}\n")
                f.write("\n")
        return SpanExportResult.SUCCESS

    def shutdown(self):
        pass

    def force_flush(self, timeout_millis=30000):
        return True


# ---------------------------------------------------------------------------
# STEP 0 — Tracer setup
# ---------------------------------------------------------------------------
provider = TracerProvider()
provider.add_span_processor(SimpleSpanProcessor(FileSpanExporter("spans_output.txt")))
trace.set_tracer_provider(provider)

tracer = trace.get_tracer("llm-zoomcamp")

# ---------------------------------------------------------------------------
# STEP 1 — Groq client (OpenAI-compatible, Chat Completions API only)
# ---------------------------------------------------------------------------
groq_client = OpenAI(
    api_key=os.getenv("GROQ_API_KEY"),
    base_url="https://api.groq.com/openai/v1",
)

MODEL = "qwen/qwen3.6-27b"

# ---------------------------------------------------------------------------
# STEP 2 — Your search index
# ---------------------------------------------------------------------------
# You need whatever `index` object starter.py builds (likely a minsearch.Index
# over the FAQ documents from Module 1). If starter.py already builds one
# called `index`, import it directly instead of rebuilding it here:
#
from starter import index  # noqa: E402  (adjust if starter.py names it differently)


# ---------------------------------------------------------------------------
# STEP 3 — RAGTraced: instrumented subclass, using Chat Completions (not Responses API)
# ---------------------------------------------------------------------------
class RAGTraced(RAGBase):

    def search(self, query, num_results=2):
        with tracer.start_as_current_span("search") as span:
            results = self.index.search(query, num_results=num_results)
            span.set_attribute("num_results", len(results))
            return results

    def llm(self, prompt):
        with tracer.start_as_current_span("llm") as span:
            response = self.llm_client.chat.completions.create(
                model=self.model,
                messages=[
                    {"role": "system", "content": self.instructions},
                    {"role": "user", "content": prompt},
                ],
            )
            usage = response.usage
            # Groq/Chat Completions naming: prompt_tokens / completion_tokens
            span.set_attribute("input_tokens", usage.prompt_tokens)
            span.set_attribute("output_tokens", usage.completion_tokens)
            return response

    def rag(self, query):
        with tracer.start_as_current_span("rag") as span:
            search_results = self.search(query)
            prompt = self.build_prompt(query, search_results)
            response = self.llm(prompt)
            # Chat Completions shape: response.choices[0].message.content
            # (RAGBase.rag() expects response.output_text, which is Responses-API-only,
            # so we override rag() fully here rather than calling super().rag())
            return response.choices[0].message.content


traced_rag = RAGTraced(
    index=index,
    llm_client=groq_client,
    instructions=INSTRUCTIONS,
    prompt_template=PROMPT_TEMPLATE,
    model=MODEL,
)

QUERY = "How does the agentic loop keep calling the model until it stops?"

if __name__ == "__main__":
    answer = traced_rag.rag(QUERY)
    print("\n\nANSWER:", answer)
    print("\n--- Span data saved to spans_output.txt — open that file (not the terminal) ---")
    print("Q1: count the span blocks printed (rag / search / llm = 3)")
    print("Q2: find 'input_tokens' inside the llm span's attributes")
    print("Q3: check start_time/end_time on the llm span to estimate duration")
