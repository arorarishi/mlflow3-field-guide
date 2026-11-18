"""Chapter 18: the Prompt Registry.

Prompts are versioned like models: each register_prompt call makes a new immutable
version with a commit message, and aliases (@production) are movable pointers your app
loads. Templates use {{variable}} placeholders. Runs offline with the fake LLM.
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import mlflow

from shared import KNOWLEDGE_BASE, LLM, banner, chat, setup

NAME = "book-support-answer"
CHAT_NAME = "book-support-chat"


def main():
    setup("book-18-prompts")

    banner("Register two versions of a text prompt")
    v1 = mlflow.genai.register_prompt(
        name=NAME,
        template="Answer the question using this context:\n{{context}}",
        commit_message="first draft",
        tags={"author": "book-reader"},
    )
    v2 = mlflow.genai.register_prompt(
        name=NAME,
        template=(
            "You are Acme support. Use only the context. Quote exact numbers. "
            "If the context lacks the answer, say \"I don't know.\"\n\nContext:\n{{context}}"
        ),
        commit_message="ground answers and allow refusal",
        # Recommended model settings travel with the prompt version.
        model_config={"model_name": LLM["chat_model"], "temperature": 0.0, "max_tokens": 120},
    )
    print(f"v{v1.version}: {v1.commit_message!r}   v{v2.version}: {v2.commit_message!r}")
    print(f"variables in v{v2.version}: {v2.variables}")

    banner("Aliases decide what the app loads")
    mlflow.genai.set_prompt_alias(NAME, alias="production", version=v1.version)
    mlflow.genai.set_prompt_alias(NAME, alias="staging", version=v2.version)
    for ref in (f"prompts:/{NAME}@production", f"prompts:/{NAME}@staging", f"prompts:/{NAME}/{v1.version}", f"prompts:/{NAME}@latest"):
        print(f"{ref:40s} -> v{mlflow.genai.load_prompt(ref).version}")

    banner("Use a prompt")
    prompt = mlflow.genai.load_prompt(f"prompts:/{NAME}@staging")
    system = prompt.format(context=KNOWLEDGE_BASE["shipping"])
    print(system)
    print("model config:", prompt.model_config)
    print("reply:", chat([{"role": "system", "content": system}, {"role": "user", "content": "How much does shipping cost?"}]))

    banner("Chat-style prompts are a list of messages")
    chat_prompt = mlflow.genai.register_prompt(
        name=CHAT_NAME,
        template=[
            {"role": "system", "content": "You are Acme support. Context:\n{{context}}"},
            {"role": "user", "content": "{{question}}"},
        ],
        commit_message="chat template",
    )
    messages = chat_prompt.format(context=KNOWLEDGE_BASE["warranty"], question="How long is the warranty?")
    print(messages)
    print("reply:", chat(messages))

    banner("Release: move @production to v2 (no code change in the app)")
    mlflow.genai.set_prompt_alias(NAME, alias="production", version=v2.version)
    print(f"@production -> v{mlflow.genai.load_prompt(f'prompts:/{NAME}@production').version}")

    banner("Find prompts")
    for found in mlflow.genai.search_prompts(filter_string=f"name LIKE 'book-%'"):
        print(f"- {found.name}")
    print("The UI's Prompts page shows each version and a diff between versions.")


if __name__ == "__main__":
    main()
