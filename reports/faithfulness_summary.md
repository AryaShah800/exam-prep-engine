# Answer faithfulness: completed sample

Measured on 14 November 2026 using `python -m exam_prep evaluate-answers --embedder local`.

| Metric | Result |
| --- | --- |
| Sample | 40 queries, 10 per subject |
| Mean RAGAS faithfulness | 94.1% |
| Every statement judged supported | 72.5% |
| Passed citation validation | 100% |
| Passed on first generation attempt | 82.5% |

The sample comes from the AI-written golden_v1 set. Retrieval uses hybrid BM25 and BAAI/bge-base-en-v1.5 local embeddings, k=8. Generation and the RAGAS judge both use gemini-3.5-flash-lite; shared model family may bias the judge. Citation validity checks source identifiers and sentence coverage; it does not establish factual support. Faithfulness is model-judged, not human-validated.

The full report (`answer_eval_golden_v1.md`), answer records (`data/eval/answers.jsonl`) and MLflow run remain local because they contain text derived from private university papers. This public summary contains aggregate measurements only. The run resumed the previous 31 cached answers and completed the remaining nine.
