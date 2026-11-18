# 16 · Tracing fundamentals

**You will learn:** the trace data model and how to instrument any Python code without an LLM.

## Data model

```
Trace tr-...  (status, duration, tags, metadata, token usage, assessments)
 └─ order_status_bot   AGENT     inputs/outputs, attributes, events, status
     ├─ extract_order_id PARSER
     ├─ policy_check     CHAIN   (manual span)
     └─ lookup_order     TOOL    ERROR + exception event
```

Span types include `AGENT`, `CHAIN`, `TOOL`, `RETRIEVER`, `CHAT_MODEL`, `LLM`, `EMBEDDING`,
`PARSER`, `RERANKER` and `UNKNOWN`. The UI uses them for icons and specialised views, for
example rendering retrieved documents.

## Key APIs

```python
@mlflow.trace(span_type=SpanType.TOOL)              # decorate a function
def lookup(order_id): ...

with mlflow.start_span(name="policy", span_type="CHAIN") as span:   # any block of code
    span.set_inputs({...}); span.set_attributes({...}); span.set_outputs({...})

mlflow.update_current_trace(tags={...}, metadata={"mlflow.trace.user": u, "mlflow.trace.session": s})
mlflow.tracing.configure(span_processors=[mask_secrets])   # redact before export
mlflow.get_last_active_trace_id(); mlflow.flush_trace_async_logging(); mlflow.get_trace(id)
mlflow.search_traces(locations=[exp_id], filter_string="trace.status = 'ERROR'")
mlflow.set_trace_tag(trace_id, "needs_review", "true")
```

Filters: `trace.status`, `trace.timestamp_ms`, `trace.execution_time_ms`, `tags.x`,
`` metadata.`mlflow.trace.user` ``, `span.name`, `span.type`, `feedback.<name>`, `expectation.<name>`.

## Run it

`./book.sh run 16`, then open the experiment's **Traces** tab. Filter by user and open the
failed trace.

## Production notes

* Export is asynchronous, so it adds little latency. Flush before reading back in tests.
* `pip install mlflow-tracing` is a small, tracing-only SDK for services.
* Traces are OpenTelemetry-compatible. MLflow can export to an OTLP collector and accept OTel spans.
* Sample high-volume traffic (`MLFLOW_TRACE_SAMPLING_RATIO`) and always redact secrets and PII.

## Exercises

1. Trace an `async def` function and a generator.
2. Add a span processor that truncates any input string longer than 500 characters.
