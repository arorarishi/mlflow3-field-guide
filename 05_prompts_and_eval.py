"""Prompt registry plus evaluation: compare two prompt versions, promote only on evidence."""
import mlflow
from mlflow.genai.scorers import scorer

from common import GENAI_EXPERIMENT, PROMPT_NAME, TRACKING_URI

PROMPT_V1 = "You are a helpful assistant for Acme.\n\nBackground:\n{{context}}"
PROMPT_V2 = (
    "You are Acme's support agent. Answer in one or two short sentences, using only the "
    "context below. Always state the exact number, price or duration from the context. "
    'If the context does not contain the answer, reply exactly: "I don\'t know."\n\n'
    "Context:\n{{context}}"
)

EVAL_DATA = [
    {"inputs": {"question": "How many days do I have to request a refund?"}, "expectations": {"must_contain": ["30"]}},
    {"inputs": {"question": "How long does a refund take to arrive?"}, "expectations": {"must_contain": ["5"]}},
    {"inputs": {"question": "How much does standard shipping cost?"}, "expectations": {"must_contain": ["$5"]}},
    {"inputs": {"question": "How long is the hardware warranty?"}, "expectations": {"must_contain": ["2"]}},
    {"inputs": {"question": "How long is a password reset link valid?"}, "expectations": {"must_contain": ["15"]}},
    {"inputs": {"question": "What does the Pro plan cost per month?"}, "expectations": {"must_contain": ["$20"]}},
    {"inputs": {"question": "How many seats come with the Pro plan?"}, "expectations": {"must_contain": ["5"]}},
    {"inputs": {"question": "Do you have a store in Paris?"}, "expectations": {"must_contain": ["don't know"]}},
    {"inputs": {"question": "Who is the CEO of Acme?"}, "expectations": {"must_contain": ["don't know"]}},
]


@scorer
def has_expected_answer(outputs, expectations) -> bool:
    normalized = outputs.lower().replace("’", "'")
    return all(term.lower() in normalized for term in expectations["must_contain"])


@scorer
def is_concise(outputs) -> bool:
    return len(outputs.split()) <= 40


def evaluate_prompt(version):
    from support_bot import answer

    prompt = mlflow.genai.load_prompt(f"prompts:/{PROMPT_NAME}/{version}")

    def predict(question):
        return answer(question, prompt=prompt)

    with mlflow.start_run(run_name=f"eval-prompt-v{version}"):
        mlflow.log_param("prompt_version", version)
        result = mlflow.genai.evaluate(
            data=EVAL_DATA, predict_fn=predict, scorers=[has_expected_answer, is_concise]
        )
    return result.metrics


def main():
    mlflow.set_tracking_uri(TRACKING_URI)
    mlflow.set_experiment(GENAI_EXPERIMENT)
    mlflow.openai.autolog()

    v1 = mlflow.genai.register_prompt(PROMPT_NAME, PROMPT_V1, commit_message="Initial prompt")
    v2 = mlflow.genai.register_prompt(
        PROMPT_NAME, PROMPT_V2, commit_message="Ground answers, require exact values, allow refusal"
    )
    mlflow.genai.set_prompt_alias(PROMPT_NAME, "production", v1.version)

    baseline = evaluate_prompt(v1.version)
    candidate = evaluate_prompt(v2.version)
    print(f"v{v1.version} metrics: {baseline}")
    print(f"v{v2.version} metrics: {candidate}")

    key = "has_expected_answer/mean"
    if candidate[key] > baseline[key]:
        mlflow.genai.set_prompt_alias(PROMPT_NAME, "production", v2.version)
        print(f"PROMOTED prompt v{v2.version} to @production")
    else:
        print(f"NOT promoted: @production stays on v{v1.version}")

    production = mlflow.genai.load_prompt(f"prompts:/{PROMPT_NAME}@production")
    print(f"App now loads prompts:/{PROMPT_NAME}@production -> v{production.version}")


if __name__ == "__main__":
    main()
