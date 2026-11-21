# Retrieval evaluation — sample_paraphrase

- Run: 2026-11-21 16:23, backend `memory` (exact cosine + Okapi BM25), embeddings `BAAI/bge-base-en-v1.5`, RRF k=60, subject filter on every query.
- Golden set: 48 queries over 3 subjects, 240 relevant (query, question) pairs; written by: human-synthetic.
- Relevance was judged from question text against the full question bank of each subject (not from topic tags).

## All queries

| Setup | Recall@5 | Hit@5 | MRR@10 | nDCG@10 |
| --- | --- | --- | --- | --- |
| bm25 | 20.8% | 54.2% | 0.397 | 0.245 |
| dense | 45.4% | 91.7% | 0.714 | 0.609 |
| hybrid | 41.7% | 87.5% | 0.736 | 0.597 |
| bm25 (no topic names) | 18.3% | 54.2% | 0.416 | 0.223 |
| dense (no topic names) | 45.0% | 93.8% | 0.724 | 0.556 |
| hybrid (no topic names) | 38.8% | 87.5% | 0.666 | 0.536 |
| hybrid, external weight 1.0 | 41.7% | 87.5% | 0.736 | 0.597 |
| hybrid, external weight 0.9 | 41.7% | 87.5% | 0.736 | 0.597 |

## Is hybrid better? (paired bootstrap, 95% CI)

| Comparison | Metric | Mean difference | 95% CI |
| --- | --- | --- | --- |
| hybrid − bm25 | recall@5 | +0.208 | [+0.142, +0.275] (significant) |
| hybrid − bm25 | mrr@10 | +0.339 | [+0.236, +0.437] (significant) |
| hybrid − dense | recall@5 | -0.037 | [-0.108, +0.029] (not significant) |
| hybrid − dense | mrr@10 | +0.022 | [-0.075, +0.118] (not significant) |

## By query type

| Query type | n | bm25 R@5 / MRR | dense R@5 / MRR | hybrid R@5 / MRR |
| --- | --- | --- | --- | --- |
| paraphrase | 48 | 20.8% / 0.397 | 45.4% / 0.714 | 41.7% / 0.736 |

## By subject

| Subject | n | bm25 R@5 / MRR | dense R@5 / MRR | hybrid R@5 / MRR |
| --- | --- | --- | --- | --- |
| DB | 16 | 23.8% / 0.495 | 45.0% / 0.729 | 40.0% / 0.841 |
| ML | 16 | 11.2% / 0.210 | 46.2% / 0.721 | 38.8% / 0.700 |
| OS | 16 | 27.5% / 0.484 | 45.0% / 0.693 | 46.2% / 0.667 |

## By author

| Author | n | bm25 R@5 / MRR | dense R@5 / MRR | hybrid R@5 / MRR |
| --- | --- | --- | --- | --- |
| human-synthetic | 48 | 20.8% / 0.397 | 45.4% / 0.714 | 41.7% / 0.736 |

## Hardest queries for hybrid

| Query | Subject | Relevant | Recall@5 | MRR@10 |
| --- | --- | --- | --- | --- |
| programs stuck forever each holding something the other needs | OS | 5 | 0% | 0.00 |
| swapping out blocks and the system spending its time swapping | OS | 5 | 0% | 0.00 |
| predicting a number from features with a straight line fit | ML | 5 | 0% | 0.00 |
| squashing a score into a probability for yes or no outcomes | ML | 5 | 0% | 0.00 |
| correct interleaving of concurrent operations on stored data | DB | 5 | 0% | 0.12 |
| what happens when a needed block of a program is not in ram | OS | 5 | 0% | 0.17 |
| finding the widest street between two classes of points | ML | 5 | 20% | 0.20 |
| making sure two programs do not touch shared data at the same moment | OS | 5 | 20% | 0.25 |

## Caveats

- Recall@5 is capped below 100% for queries with more than 5 relevant questions; Hit@5 and MRR are not.
- Queries written after reading the question bank tend to reuse its wording, which favours keyword search. The `paraphrase` rows avoid that wording on purpose.
- One semester of DDU papers (2025) plus other universities' papers; numbers will move as papers are added.
- The external-weight rows were not used to pick the default weight from this same set (that would overfit); they show sensitivity only.
