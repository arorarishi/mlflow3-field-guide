# Offline to Online: Evaluating and Shipping LLM Apps with MLflow 3

The table of contents, setup and suggested reading paths are in the
[main README](../README.md#contents).

Each chapter folder contains:

* `README.md` with concepts, APIs, pitfalls and exercises
* `example.py`, which you run with `./book.sh run NN` from the repository root
* supporting files where the chapter needs them, such as `MLproject`, `gate.py` or `model_code.py`

Shared code:

```
shared.py        tracking URI, datasets, LLM provider switch, fake LLM
support_bot.py   small traced RAG bot used by chapters 19–22
build_pdf.py     builds Offline-to-Online.pdf (./book.sh pdf)
.data/           the book server's database and artifacts (./book.sh reset deletes it)
```
