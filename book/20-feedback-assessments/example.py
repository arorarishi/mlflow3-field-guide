"""Chapter 20: human feedback, expectations and assessments.

Assessments attach judgments to a trace (or one span). Feedback is "how good was this?",
from a user, a reviewer, code or an LLM judge. An expectation is the ground truth: what
the answer should have been. Labeled traces become your next evaluation dataset.
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import mlflow
from mlflow.entities import AssessmentSource, AssessmentSourceType

from shared import banner, setup
from support_bot import answer

TRAFFIC = [  # (question, did the user click thumbs up?)
    ("How much does standard shipping cost?", True),
    ("Do you have a store in Paris?", False),
    ("What does the Pro plan cost per month?", True),
    ("Can I pay with crypto?", False),
]


@mlflow.trace(name="support_request")
def handle(question: str) -> str:
    """The production entry point. Its inputs become the dataset inputs later."""
    return answer(question, prompt_version="v1")


def main():
    experiment = setup("book-20-feedback")

    banner("Production-like traffic, with thumbs up/down from end users")
    trace_ids = []
    for question, thumbs_up in TRAFFIC:
        reply = handle(question)
        trace_id = mlflow.get_last_active_trace_id()
        # Traces export asynchronously. Wait until this one is stored before labeling it.
        mlflow.flush_trace_async_logging()
        trace_ids.append(trace_id)
        mlflow.log_feedback(
            trace_id=trace_id,
            name="user_thumbs_up",
            value=thumbs_up,
            source=AssessmentSource(source_type=AssessmentSourceType.HUMAN, source_id="end-user-42"),
            rationale=None if thumbs_up else "Not what I asked.",
        )
        print(f"{'+1' if thumbs_up else '-1'}  {question} -> {reply}")

    banner("A reviewer adds ground truth to the thumbs-down traces")
    reviewer = AssessmentSource(source_type=AssessmentSourceType.HUMAN, source_id="reviewer@acme")
    for trace_id, (question, thumbs_up) in zip(trace_ids, TRAFFIC):
        if not thumbs_up:
            mlflow.log_expectation(trace_id=trace_id, name="expected_response", value="I don't know.", source=reviewer)
            print(f"expected_response for {question!r}: \"I don't know.\"")
    bad = trace_ids[-1]

    banner("An automated check, then a human overrides it")
    auto = mlflow.log_feedback(
        trace_id=bad,
        name="grounded",
        value=True,
        source=AssessmentSource(source_type=AssessmentSourceType.CODE, source_id="overlap-check"),
        rationale="Answer shares words with the retrieved context.",
    )
    mlflow.override_feedback(
        trace_id=bad,
        assessment_id=auto.assessment_id,
        value=False,
        rationale="The answer is about a different topic. The bot should have refused.",
        source=reviewer,
    )

    banner("Read the assessments on that trace")
    for assessment in mlflow.get_trace(bad).info.assessments:
        value = assessment.feedback.value if assessment.feedback else assessment.expectation.value
        state = "valid" if assessment.valid is not False else "overridden"
        print(f"{assessment.name:18s} {str(value):14s} by {assessment.source.source_type}:{assessment.source.source_id} ({state})")

    banner("Find traces by feedback")
    for query in ("feedback.user_thumbs_up = 'false'", "feedback.user_thumbs_up = 'true'"):
        found = mlflow.search_traces(locations=[experiment.experiment_id], filter_string=query, return_type="list")
        print(f"{query:38s} -> {len(found)} traces")

    banner("Turn labeled traces into an evaluation dataset")
    name = "book-feedback-regressions"
    try:
        dataset = mlflow.genai.datasets.get_dataset(name=name)
    except Exception:  # noqa: BLE001 - created on the first run
        dataset = mlflow.genai.datasets.create_dataset(name=name, experiment_id=experiment.experiment_id)
    labeled = mlflow.search_traces(locations=[experiment.experiment_id], filter_string="feedback.user_thumbs_up = 'false'")
    dataset.merge_records(labeled)
    print(f"{name}: {len(dataset.to_df())} records. Expectations came along with the traces.")
    print(dataset.to_df()[["inputs", "expectations"]].head(3).to_string())


if __name__ == "__main__":
    main()
