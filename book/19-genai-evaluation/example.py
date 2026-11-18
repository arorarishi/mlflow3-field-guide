"""Chapter 19: evaluating GenAI apps.

mlflow.genai.evaluate runs your app over a dataset, traces every call, and applies
scorers to each row. Scorers can be plain code (cheap, deterministic) or LLM judges
(for qualities code cannot check). Results land in an evaluation run you can compare.

Offline with the fake LLM: code scorers only. With a real provider: judges are added.
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import mlflow
from mlflow.entities import Feedback
from mlflow.genai.judges import make_judge
from mlflow.genai.scorers import Correctness, Guidelines, RegexMatch, ResponseLength, scorer

from shared import PROVIDER, banner, judge_model, setup
from support_bot import answer

RECORDS = [
    ("How many days do I have to request a refund?", ["30"], "Refunds are accepted within 30 days."),
    ("How many business days until a refund reaches me?", ["5"], "Refunds arrive in 5 business days."),
    ("How much does standard shipping cost?", ["$5"], "Standard shipping costs $5."),
    ("How long is the hardware warranty?", ["2"], "The warranty lasts 2 years."),
    ("When does a password reset link expire?", ["15"], "Reset links expire after 15 minutes."),
    ("What does the Pro plan cost per month?", ["$20"], "The Pro plan costs $20 per month."),
    ("Do you have a store in Paris?", ["don't know"], "I don't know."),
    ("Who is the CEO of Acme?", ["don't know"], "I don't know."),
]


# ---- code scorers -------------------------------------------------------------------
@scorer
def has_expected_answer(outputs: str, expectations: dict) -> bool:
    text = outputs.lower().replace("’", "'")
    return all(term.lower() in text for term in expectations["must_contain"])


@scorer
def used_retrieval(trace) -> Feedback:
    """Scorers can read the whole trace, e.g. to check what retrieval returned."""
    spans = trace.search_spans(span_type="RETRIEVER")
    docs = spans[0].outputs if spans else []
    return Feedback(value=bool(docs), rationale=f"retriever returned {len(docs or [])} documents")


def build_scorers():
    scorers = [
        has_expected_answer,
        used_retrieval,
        ResponseLength(max_length=40, unit="words"),  # built-in, deterministic
        RegexMatch(name="no_markdown", pattern=r"^[^#*`]*$"),  # built-in, deterministic
    ]
    model = judge_model()
    if model:
        scorers += [
            Correctness(model=model),  # compares outputs with expectations.expected_response
            Guidelines(name="polite", guidelines="The response must be polite and professional.", model=model),
            make_judge(
                name="no_speculation",
                instructions=(
                    "Does the answer in {{ outputs }} avoid inventing facts that are not implied "
                    "by the question {{ inputs }}? Answer yes or no."
                ),
                model=model,
                feedback_value_type=str,
            ),
        ]
    return scorers


def main():
    experiment = setup("book-19-genai-eval")

    banner("An evaluation dataset stored in MLflow")
    name = "book-support-golden"
    try:
        dataset = mlflow.genai.datasets.get_dataset(name=name)
    except Exception:  # noqa: BLE001 - not found on the first run
        dataset = mlflow.genai.datasets.create_dataset(name=name, experiment_id=experiment.experiment_id)
    dataset.merge_records([
        {"inputs": {"question": q}, "expectations": {"must_contain": terms, "expected_response": ref}}
        for q, terms, ref in RECORDS
    ])
    print(f"dataset {dataset.name}: {len(dataset.to_df())} records (re-merging the same inputs updates them)")

    scorers = build_scorers()
    print(f"scorers: {[s.name for s in scorers]}" + ("" if PROVIDER != "fake" else "  (judges skipped: fake LLM)"))

    results = {}
    for version in ("v1", "v2"):
        banner(f"Evaluate prompt {version}")
        with mlflow.start_run(run_name=f"eval-prompt-{version}") as run:
            mlflow.log_param("prompt_version", version)
            result = mlflow.genai.evaluate(
                data=dataset,
                predict_fn=lambda question, v=version: answer(question, prompt_version=v),
                scorers=scorers,
            )
        results[version] = result.metrics
        for key, value in sorted(result.metrics.items()):
            if key.endswith("/mean"):
                print(f"  {key:32s} {value:.2f}")
        print(f"  run: {run.info.run_id}")

    banner("Decision")
    key = "has_expected_answer/mean"
    winner = max(results, key=lambda v: results[v][key])
    print(f"{key}: v1={results['v1'][key]:.2f} v2={results['v2'][key]:.2f} -> ship {winner}")
    print("Open the evaluation runs and compare them row by row in the UI.")

    banner("Score traces you already have (no re-running the app)")
    traces = mlflow.search_traces(locations=[experiment.experiment_id], max_results=5)
    rescored = mlflow.genai.evaluate(data=traces, scorers=[ResponseLength(max_length=40, unit="words")])
    print(f"re-scored {len(traces)} existing traces: {rescored.metrics}")


if __name__ == "__main__":
    main()
