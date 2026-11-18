# 15 · A CI promotion gate

**You will learn:** how to make model promotion an automated, auditable decision, with
an exit code a CI system acts on.

## The gate

`gate.py` loads `@challenger` and `@champion`, scores both on a held-out test set, and
promotes only if the challenger:

1. clears an **absolute bar** (`f1 >= 0.9`), so a mediocre first model cannot ship, and
2. **beats the champion** by `--min-gain`.

It writes the decision to version tags (`gate`, `gate_f1`, `rollback_to`), prints a
markdown summary (also written to `$GITHUB_STEP_SUMMARY` on GitHub Actions), and exits with
0 for promoted and 1 for not promoted.

```
dummy           v1 -> gate FAIL (f1 0.77 < 0.9)        exit 1
logreg C=1      v2 -> gate PASS -> @champion            exit 0
logreg C=0.001  v3 -> gate FAIL (worse than champion)   exit 1
```

## CI wiring

`github-workflow.yml` is a template: train, tag `@challenger`, run the gate. Give CI a
service account that has EDIT on the registered model (chapter 24) and the tracking URI
as secrets.

## Run it

`./book.sh run 15`, or the gate alone:
`.venv/bin/python book/15-ci-promotion-gate/gate.py --model book-gated-classifier`.

## Pitfalls

* Reusing one test set for every promotion slowly overfits to it. Refresh it, or keep a
  locked final holdout.
* A single-number gate hides regressions in slices. Add per-segment checks for the groups
  you care about.
* Compare on the **same data** and the **same code path** (pyfunc), as the gate does.

## Exercises

1. Add a latency check: fail if the challenger is more than 2 times slower than the champion on 1,000 rows.
2. Use `mlflow.validate_evaluation_results` (chapter 11) inside the gate.
