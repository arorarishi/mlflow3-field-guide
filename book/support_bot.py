"""A tiny traced RAG bot used by chapters 19 to 22. Retrieval is keyword overlap, so it
runs offline. Generation uses shared.chat(): the fake model or a real one."""
import mlflow
from mlflow.entities import Document, SpanType

from shared import KNOWLEDGE_BASE, _words, chat

PROMPTS = {
    "v1": "You are a helpful assistant for Acme.\n\nBackground:\n{context}",
    "v2": (
        "You are Acme's support agent. Answer in one or two short sentences using only the "
        "context. State exact numbers. If the context does not contain the answer, reply "
        "exactly: \"I don't know.\"\n\nContext:\n{context}"
    ),
}


@mlflow.trace(span_type=SpanType.RETRIEVER)
def retrieve(question: str, k: int = 2) -> list[Document]:
    scored = sorted(
        ((len(_words(text) & _words(question)), doc_id) for doc_id, text in KNOWLEDGE_BASE.items()),
        reverse=True,
    )
    return [
        Document(id=doc_id, page_content=KNOWLEDGE_BASE[doc_id], metadata={"overlap": score})
        for score, doc_id in scored[:k]
        if score > 0
    ]


@mlflow.trace(name="support_bot", span_type=SpanType.AGENT)
def answer(question: str, prompt_version: str = "v2") -> str:
    documents = retrieve(question)
    if prompt_version == "v1":
        # The naive version stuffs the whole knowledge base into the prompt.
        context = "\n".join(KNOWLEDGE_BASE.values())
    else:
        context = "\n".join(d.page_content for d in documents) or "(no relevant documents)"
    system = PROMPTS[prompt_version].format(context=context)
    return chat([{"role": "system", "content": system}, {"role": "user", "content": question}])
