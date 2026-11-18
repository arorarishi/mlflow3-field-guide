"""Chapter 22: versioning a GenAI app.

For an app whose "model" is code plus prompts plus config, MLflow 3 uses a LoggedModel as
the version record. set_active_model() links every trace produced afterwards to that
version, so you can compare versions by their traces and evaluation results.
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import mlflow
from mlflow.genai.scorers import scorer

from shared import LLM, banner, setup
from support_bot import PROMPTS, answer

QUESTIONS = [
    ("How much does standard shipping cost?", "$5"),
    ("What does the Pro plan cost per month?", "$20"),
    ("Do you have a store in Paris?", "don't know"),
    ("Who is the CEO of Acme?", "don't know"),
]


@scorer
def correct(outputs: str, expectations: dict) -> bool:
    return expectations["contains"].lower() in outputs.lower().replace("’", "'")


def main():
    experiment = setup("book-22-app-versioning")
    data = [{"inputs": {"question": q}, "expectations": {"contains": c}} for q, c in QUESTIONS]

    for version in ("v1", "v2"):
        banner(f"App version support-bot-{version}")
        active = mlflow.set_active_model(name=f"support-bot-{version}")
        # Record what defines this version: prompt, model and (in a real repo) the git commit.
        mlflow.log_model_params({"prompt_version": version, "llm": LLM["chat_model"], "prompt": PROMPTS[version][:80]}, model_id=active.model_id)

        # "Production" traffic: linked to the version automatically.
        for question, _ in QUESTIONS:
            answer(question, prompt_version=version)

        # Evaluation results are linked to the version through model_id.
        result = mlflow.genai.evaluate(
            data=data,
            predict_fn=lambda question, v=version: answer(question, prompt_version=v),
            scorers=[correct],
            model_id=active.model_id,
        )
        print(f"{active.model_id}: correct/mean = {result.metrics['correct/mean']:.2f}")
    mlflow.clear_active_model()
    mlflow.flush_trace_async_logging()

    banner("Compare versions")
    models = mlflow.search_logged_models(
        experiment_ids=[experiment.experiment_id],
        filter_string="name LIKE 'support-bot-%'",
        output_format="list",
    )
    for model in sorted(models, key=lambda m: m.creation_timestamp)[-2:]:
        traces = mlflow.search_traces(locations=[experiment.experiment_id], model_id=model.model_id, return_type="list")
        metrics = {m.key: round(m.value, 2) for m in model.metrics}
        print(f"{model.name:16s} prompt={model.params.get('prompt_version')} traces={len(traces):2d} metrics={metrics}")
    print("In the UI: Models tab of the experiment -> each version with its traces and eval runs.")


if __name__ == "__main__":
    main()
