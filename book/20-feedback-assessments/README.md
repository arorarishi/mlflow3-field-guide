# 20 · Human feedback and assessments

**You will learn:** how to attach user feedback, reviewer ground truth and automated
judgments to traces, and how to turn them into the next evaluation set.

## Assessment types

| | `log_feedback` | `log_expectation` |
|---|---|---|
| Meaning | how good was this output? | what should the output have been? |
| Values | bool, number, string, dict, list | any JSON value |
| Typical source | `HUMAN` (end user, reviewer), `LLM_JUDGE`, `CODE` | `HUMAN` |

Each assessment has a **source** (`AssessmentSource(source_type, source_id)`), an optional
rationale and metadata, and can target a single span. `override_feedback` replaces a
judgment while keeping the original, marked invalid, for audit.

## Key APIs

```python
mlflow.log_feedback(trace_id=..., name="user_thumbs_up", value=False,
                    source=AssessmentSource(source_type="HUMAN", source_id="user-42"), rationale="...")
mlflow.log_expectation(trace_id=..., name="expected_response", value="I don't know.")
mlflow.override_feedback(trace_id=..., assessment_id=..., value=False, rationale="...")
mlflow.search_traces(filter_string="feedback.user_thumbs_up = 'false'")
dataset.merge_records(traces_df)        # traces with expectations become eval records
```

## The improvement loop

```
production traces ─▶ user thumbs-down ─▶ reviewer adds expectation
        ▲                                         │
        └── ship v2 ◀── evaluate v1 vs v2 ◀── regression dataset
```

In your web app, return the trace ID with each response (`mlflow.get_active_trace_id()`
inside the request), and let the UI post thumbs up or down to an endpoint that calls `log_feedback`.

## Run it

`./book.sh run 20`. In the UI, open a trace and check the **Assessments** panel.

## Exercises

1. Evaluate prompt `v2` (chapter 19) on `book-feedback-regressions`.
2. Log span-level feedback on the `retrieve` span (`span_id=...`).
