"""Chapter 16: tracing fundamentals (no LLM needed).

A trace is a tree of spans for one request. Each span has a type, inputs, outputs,
attributes, timing and a status. This chapter builds traces by hand with the decorator
and the context manager, records an error, masks a secret, and queries traces back.
"""
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import mlflow
from mlflow.entities import SpanType

from shared import banner, last_trace, setup

ORDERS = {"A-100": {"status": "shipped", "eta_days": 2}, "A-200": {"status": "processing", "eta_days": 5}}


@mlflow.trace(span_type=SpanType.TOOL)
def lookup_order(order_id: str) -> dict:
    time.sleep(0.05)
    if order_id not in ORDERS:
        raise KeyError(f"unknown order {order_id}")
    return ORDERS[order_id]


@mlflow.trace(span_type=SpanType.PARSER)
def extract_order_id(message: str) -> str:
    return next(word.strip("?.,") for word in message.split() if word.startswith("A-"))


@mlflow.trace(name="order_status_bot", span_type=SpanType.AGENT)
def handle(message: str, user: str, session: str, api_key: str = "sk-demo-secret") -> str:
    # Trace-level context: filterable tags plus the reserved user/session metadata keys.
    mlflow.update_current_trace(
        tags={"channel": "chat"},
        metadata={"mlflow.trace.user": user, "mlflow.trace.session": session},
    )
    order_id = extract_order_id(message)
    # A manual span for a block of code that is not its own function.
    with mlflow.start_span(name="policy_check", span_type=SpanType.CHAIN) as span:
        span.set_inputs({"order_id": order_id})
        allowed = order_id.startswith("A-")
        span.set_attributes({"rule": "prefix-A", "allowed": allowed})
        span.set_outputs({"allowed": allowed})
    order = lookup_order(order_id)
    return f"Order {order_id} is {order['status']}, arriving in about {order['eta_days']} days."


def mask_secrets(span):
    """Span processor: runs on every span before export. Never store credentials."""
    inputs = span.inputs
    if isinstance(inputs, dict) and "api_key" in inputs:
        span.set_inputs({**inputs, "api_key": "***"})


def main():
    experiment = setup("book-16-tracing")
    mlflow.tracing.configure(span_processors=[mask_secrets])

    banner("Produce a few traces")
    requests = [
        ("Where is A-100?", "alice", "s-1"),
        ("And A-200?", "alice", "s-1"),
        ("Status of A-999 please", "bob", "s-2"),
    ]
    for message, user, session in requests:
        try:
            print(handle(message, user, session))
        except KeyError as error:
            print(f"failed: {error}  (the trace is still logged, with status ERROR)")
    banner("Inspect one trace")
    trace = last_trace()
    trace_id = trace.info.trace_id
    print(f"trace {trace.info.trace_id} status={trace.info.state} took {trace.info.execution_duration}ms")
    for span in trace.data.spans:
        depth = 0
        parent = span.parent_id
        while parent:
            depth += 1
            parent = next(s.parent_id for s in trace.data.spans if s.span_id == parent)
        print(f"{'  ' * depth}{span.name:18s} {span.span_type:8s} {span.status.status_code.value}")
    root = trace.data.spans[0]
    print(f"root inputs (api_key masked): {root.inputs}")
    error_span = next(s for s in trace.data.spans if s.name == "lookup_order")
    print(f"error event on lookup_order: {[e.name for e in error_span.events]}")

    banner("Tag after the fact, then search")
    mlflow.set_trace_tag(trace_id, "needs_review", "true")
    queries = [
        "trace.status = 'ERROR'",
        "tags.needs_review = 'true'",
        "metadata.`mlflow.trace.user` = 'alice'",
        "span.name = 'lookup_order'",
    ]
    for query in queries:
        found = mlflow.search_traces(locations=[experiment.experiment_id], filter_string=query, return_type="list")
        print(f"{query:42s} -> {len(found)} traces")

    df = mlflow.search_traces(locations=[experiment.experiment_id], max_results=5)
    print(f"\nas a DataFrame: columns include {[c for c in df.columns][:8]}")


if __name__ == "__main__":
    main()
