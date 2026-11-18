"""A small RAG support bot, traced with MLflow. Retrieval uses real embeddings."""
import math

import mlflow
from mlflow.entities import Document, SpanType
from openai import OpenAI

from common import LLM

KNOWLEDGE_BASE = {
    "refunds": "Acme refunds purchases within 30 days. Refunds reach the original payment method in 5 business days.",
    "shipping": "Standard shipping takes 4 days and costs $5. Orders above $50 ship free.",
    "warranty": "Acme hardware has a 2 year warranty. Accidental damage is not covered.",
    "password": "Reset a password from Settings > Security. Reset links expire after 15 minutes.",
    "plans": "The Pro plan costs $20 per month and includes 5 seats. Extra seats cost $4 each.",
}

client = OpenAI(base_url=LLM["base_url"], api_key=LLM["api_key"], timeout=180)
_doc_vectors = {}


def _embed(texts):
    response = client.embeddings.create(model=LLM["embed_model"], input=texts)
    return [item.embedding for item in response.data]


def _cosine(a, b):
    dot = sum(x * y for x, y in zip(a, b))
    return dot / (math.sqrt(sum(x * x for x in a)) * math.sqrt(sum(y * y for y in b)))


@mlflow.trace(span_type=SpanType.RETRIEVER)
def retrieve(question: str, k: int = 2) -> list[Document]:
    if not _doc_vectors:
        _doc_vectors.update(zip(KNOWLEDGE_BASE, _embed(list(KNOWLEDGE_BASE.values()))))
    query = _embed([question])[0]
    scored = sorted(
        ((_cosine(query, vector), doc_id) for doc_id, vector in _doc_vectors.items()),
        reverse=True,
    )
    return [
        Document(id=doc_id, page_content=KNOWLEDGE_BASE[doc_id], metadata={"score": round(score, 4)})
        for score, doc_id in scored[:k]
    ]


@mlflow.trace(name="support_bot", span_type=SpanType.AGENT)
def answer(question: str, prompt=None, system: str | None = None) -> str:
    """Answer from retrieved context. `prompt` is a registered MLflow prompt."""
    documents = retrieve(question)
    context = "\n".join(document.page_content for document in documents)
    if prompt is not None:
        system = prompt.format(context=context)
    elif system is None:
        system = f"Answer using only this context:\n{context}"
    response = client.chat.completions.create(
        model=LLM["chat_model"],
        temperature=0,
        messages=[{"role": "system", "content": system}, {"role": "user", "content": question}],
    )
    return response.choices[0].message.content
