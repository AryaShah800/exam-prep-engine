# Retrieval evaluation — golden_v1

- Run: 2026-11-12 23:26, backend `memory` (exact cosine + Okapi BM25), embeddings `BAAI/bge-base-en-v1.5`, RRF k=60, subject filter on every query.
- Golden set: 111 queries over 4 subjects, 439 relevant (query, question) pairs; written by: claude.
- Relevance was judged from question text against the full question bank of each subject (not from topic tags).

## All queries

| Setup | Recall@5 | Hit@5 | MRR@10 | nDCG@10 |
| --- | --- | --- | --- | --- |
| bm25 | 78.1% | 98.2% | 0.946 | 0.877 |
| dense | 79.5% | 100.0% | 0.941 | 0.878 |
| hybrid | 83.0% | 99.1% | 0.955 | 0.906 |
| bm25 (no topic names) | 74.6% | 96.4% | 0.929 | 0.835 |
| dense (no topic names) | 77.9% | 100.0% | 0.928 | 0.851 |
| hybrid (no topic names) | 79.3% | 98.2% | 0.937 | 0.873 |
| hybrid, external weight 1.0 | 82.2% | 99.1% | 0.957 | 0.902 |
| hybrid, external weight 0.9 | 81.3% | 99.1% | 0.952 | 0.898 |

## Is hybrid better? (paired bootstrap, 95% CI)

| Comparison | Metric | Mean difference | 95% CI |
| --- | --- | --- | --- |
| hybrid − bm25 | recall@5 | +0.049 | [+0.025, +0.074] (significant) |
| hybrid − bm25 | mrr@10 | +0.009 | [-0.017, +0.036] (not significant) |
| hybrid − dense | recall@5 | +0.035 | [-0.000, +0.069] (not significant) |
| hybrid − dense | mrr@10 | +0.014 | [-0.022, +0.046] (not significant) |

## By query type

| Query type | n | bm25 R@5 / MRR | dense R@5 / MRR | hybrid R@5 / MRR |
| --- | --- | --- | --- | --- |
| paraphrase | 16 | 71.4% / 0.825 | 90.1% / 0.953 | 84.9% / 0.816 |
| specific | 43 | 91.1% / 0.965 | 90.3% / 0.917 | 94.1% / 0.965 |
| topic | 52 | 69.4% / 0.968 | 67.3% / 0.958 | 73.3% / 0.990 |

## By subject

| Subject | n | bm25 R@5 / MRR | dense R@5 / MRR | hybrid R@5 / MRR |
| --- | --- | --- | --- | --- |
| AI | 25 | 73.7% / 0.960 | 74.9% / 0.893 | 81.9% / 0.953 |
| DC | 28 | 80.8% / 0.948 | 83.3% / 0.976 | 86.6% / 0.973 |
| MLDL | 31 | 87.4% / 0.984 | 82.3% / 0.930 | 90.7% / 0.984 |
| WT | 27 | 68.6% / 0.889 | 76.6% / 0.963 | 71.7% / 0.907 |

## By author

| Author | n | bm25 R@5 / MRR | dense R@5 / MRR | hybrid R@5 / MRR |
| --- | --- | --- | --- | --- |
| claude | 111 | 78.1% / 0.946 | 79.5% / 0.941 | 83.0% / 0.955 |

## Hardest queries for hybrid

| Query | Subject | Relevant | Recall@5 | MRR@10 |
| --- | --- | --- | --- | --- |
| two people sending messages to each other on a web page | WT | 1 | 0% | 0.14 |
| ADO.NET CRUD operations | WT | 7 | 14% | 1.00 |
| important questions on A* search | AI | 6 | 33% | 1.00 |
| razor pages | WT | 3 | 33% | 1.00 |
| important questions on state management | WT | 12 | 42% | 1.00 |
| uninformed search algorithms | AI | 7 | 43% | 1.00 |
| microservice architecture questions | DC | 7 | 43% | 1.00 |
| master page | WT | 9 | 44% | 1.00 |

## Caveats

- Recall@5 is capped below 100% for queries with more than 5 relevant questions; Hit@5 and MRR are not.
- Queries written after reading the question bank tend to reuse its wording, which favours keyword search. The `paraphrase` rows avoid that wording on purpose.
- One semester of DDU papers (2025) plus other universities' papers; numbers will move as papers are added.
- The external-weight rows were not used to pick the default weight from this same set (that would overfit); they show sensitivity only.
- Queries marked `claude` were written and judged by an AI (Claude), not by students. Say so wherever these numbers are quoted, until a human-written set is added.
