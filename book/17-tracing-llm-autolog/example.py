"""Chapter 17: tracing real LLM calls with autolog.

`mlflow.openai.autolog()` records every OpenAI-client call as a CHAT_MODEL or EMBEDDING
span with messages, parameters, token usage and latency. Any OpenAI-compatible server
works, so the same code traces Ollama, OpenAI or DeepInfra. Your own functions add the
surrounding structure: an agent span with a tool span between two model calls.

Needs a real model: LLM_PROVIDER=ollama (default URL http://127.0.0.1:11434/v1).
"""
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import mlflow
from mlflow.entities import SpanType

from shared import LLM, PROVIDER, banner, last_trace, llm_client, require_real_llm, setup

TOOLS = [
    {
        "type": "function",
        "function": {
            "name": "get_price",
            "description": "Get the monthly price in USD of an Acme plan.",
            "parameters": {
                "type": "object",
                "properties": {"plan": {"type": "string", "enum": ["basic", "pro"]}},
                "required": ["plan"],
            },
        },
    }
]
PRICES = {"basic": 8, "pro": 20}


@mlflow.trace(span_type=SpanType.TOOL)
def get_price(plan: str) -> dict:
    return {"plan": plan, "usd_per_month": PRICES.get(plan.lower())}


@mlflow.trace(name="pricing_agent", span_type=SpanType.AGENT)
def pricing_agent(question: str) -> str:
    client = llm_client()
    messages = [
        {"role": "system", "content": "Answer pricing questions. Use the get_price tool for prices."},
        {"role": "user", "content": question},
    ]
    first = client.chat.completions.create(model=LLM["chat_model"], messages=messages, tools=TOOLS, temperature=0)
    message = first.choices[0].message
    if not message.tool_calls:
        return message.content
    messages.append(message.model_dump(exclude_none=True))
    for call in message.tool_calls:
        # Models sometimes send malformed arguments. Return the error to the model, as a
        # tool result, instead of crashing. The failed tool span stays in the trace.
        try:
            result = get_price(**json.loads(call.function.arguments))
        except (TypeError, ValueError) as error:
            result = {"error": f"bad arguments {call.function.arguments}: {error}"}
        messages.append({"role": "tool", "tool_call_id": call.id, "content": json.dumps(result)})
    final = client.chat.completions.create(model=LLM["chat_model"], messages=messages, temperature=0)
    return final.choices[0].message.content


def main():
    require_real_llm("17-tracing-llm-autolog")
    experiment = setup("book-17-llm-tracing")
    mlflow.openai.autolog()
    client = llm_client()

    banner(f"1. A single autologged chat call ({PROVIDER}: {LLM['chat_model']})")
    reply = client.chat.completions.create(
        model=LLM["chat_model"],
        temperature=0,
        messages=[{"role": "user", "content": "In one sentence, what is experiment tracking?"}],
    )
    print(reply.choices[0].message.content)
    trace = last_trace()
    print(f"spans={[s.span_type for s in trace.data.spans]} tokens={trace.info.token_usage}")

    banner("2. Streaming is traced too (the span closes when the stream ends)")
    stream = client.chat.completions.create(
        model=LLM["chat_model"], stream=True, temperature=0,
        messages=[{"role": "user", "content": "Count from 1 to 5, separated by commas."}],
    )
    print("".join(chunk.choices[0].delta.content or "" for chunk in stream if chunk.choices))

    banner("3. An agent: model -> tool -> model, in one trace")
    print(pricing_agent("How much is the pro plan per month?"))
    trace = last_trace()
    for span in trace.data.spans:
        print(f"  {span.name:16s} {span.span_type:10s} {(span.end_time_ns - span.start_time_ns) / 1e6:8.0f} ms")
    print("Small models often skip the tool or send bad arguments. The trace shows exactly which happened.")

    mlflow.flush_trace_async_logging()
    found = mlflow.search_traces(locations=[experiment.experiment_id], max_results=10, return_type="list")
    print(f"\n{len(found)} traces in {experiment.name}. Open the Traces tab to browse them.")


if __name__ == "__main__":
    main()
