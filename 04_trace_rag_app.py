"""LLM tracing: autolog captures the LLM calls, decorators capture the app's own steps."""
import mlflow

from common import GENAI_EXPERIMENT, PROVIDER, TRACKING_URI

QUESTIONS = [
    ("How long do refunds take?", "session-alice"),
    ("Does shipping cost anything on a $70 order?", "session-alice"),
    ("Is accidental damage covered by the warranty?", "session-bob"),
    ("Do you have a store in Paris?", "session-bob"),
]


def main():
    mlflow.set_tracking_uri(TRACKING_URI)
    mlflow.set_experiment(GENAI_EXPERIMENT)
    mlflow.openai.autolog()
    from support_bot import answer

    for question, session in QUESTIONS:
        with mlflow.start_span(name="request", span_type="CHAIN") as span:
            span.set_inputs({"question": question})
            mlflow.update_current_trace(
                tags={"provider": PROVIDER},
                metadata={"mlflow.trace.session": session, "mlflow.trace.user": session[8:]},
            )
            reply = answer(question)
            span.set_outputs({"answer": reply})
        print(f"Q: {question}\nA: {reply}\n")
    mlflow.flush_trace_async_logging()


if __name__ == "__main__":
    main()
