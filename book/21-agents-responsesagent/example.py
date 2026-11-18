"""Chapter 21: packaging an agent as an MLflow model.

ResponsesAgent gives an agent a standard request/response schema (compatible with the
OpenAI Responses API), streaming, and automatic tracing. Logged from code, the agent can
be versioned, registered, evaluated and served like any other MLflow model.
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import mlflow

from shared import banner, setup

HERE = Path(__file__).parent
BOOK = HERE.parent
REQUEST = {"input": [{"role": "user", "content": "How long is the hardware warranty?"}]}


def main():
    setup("book-21-agents")

    banner("Log the agent from code")
    with mlflow.start_run(run_name="support-agent"):
        info = mlflow.pyfunc.log_model(
            name="agent",
            python_model=str(HERE / "agent_code.py"),
            # Local modules the agent imports. They are copied into the model.
            code_paths=[str(BOOK / "shared.py"), str(BOOK / "support_bot.py")],
            input_example=REQUEST,
            registered_model_name="book-support-agent",
        )
    print(f"logged {info.model_uri} as book-support-agent v{info.registered_model_version}")
    print(f"signature inputs: {str(info.signature.inputs)[:100]}...")

    banner("Load and call it (each call is traced)")
    agent = mlflow.pyfunc.load_model(info.model_uri)
    response = agent.predict(REQUEST)
    print(response["output"][0]["content"][0]["text"])

    banner("Stream it")
    events = list(agent.predict_stream({"input": [{"role": "user", "content": "What does the Pro plan cost per month?"}]}))
    deltas = "".join(e.get("delta", "") for e in events if e["type"] == "response.output_text.delta")
    print(f"{len(events)} events: {deltas.strip()}")

    banner("Same payload works over REST")
    payload = '{"input": [{"role": "user", "content": "Do you have a store in Paris?"}]}'
    print(mlflow.models.validate_serving_input(info.model_uri, payload))
    print("Serve it with: mlflow models serve -m models:/book-support-agent/latest -p 5101 --env-manager local")


if __name__ == "__main__":
    main()
