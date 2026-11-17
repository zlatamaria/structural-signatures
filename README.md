# Structural Signatures of Correct Mathematical Reasoning Traces

**Maria Marina and Alexander Panchenko · MathNLP 2026 · Budapest**

[Paper](https://aclanthology.org/2026.mathnlp-main.15/) · [Figure 1 notebook](notebooks/figure1.ipynb) · [Reproduction audit](docs/REPRODUCTION.md) · [BibTeX](paper.bib)

Why can six traces beat seven votes? On MATH500 question 315, seven traces end in the wrong answer, `1`. Their recurring checkpoints form a tree. Six traces end in the correct answer, `41`, and connect their checkpoints along several routes. The strongest connected component has four edges beyond a spanning tree. The structural selector chooses `41`.

![Six shortcut graphs for two repaired majority answers and one incorrect switch. Purple edges complete cycles.](docs/assets/figure1.png)

*Figure 1. Left and middle pairs show majority errors repaired by the structural selector. The right pair shows the selector switching to a wrong answer one trace later. Each panel reports answer frequency and excess edges. The notebook reconstructs the graph from saved precedence counts and verifies all 72 arrows against the published artwork.* [Vector figure](docs/assets/figure1.pdf)

## Reproduce the results on CPU

From the repository root, install the package and run the check. Both test input tables and the archived predictions are included in the checkout.

```bash
python -m venv .venv
source .venv/bin/activate
python -m pip install -e .
python scripts/check_results.py
```

The check reconstructs each answer graph, selects answers at budgets 5–99, compares predictions to both archived runs, and recomputes Table 1 from development-set counts. The following command saves per-answer scores, per-question choices and accuracy by budget.

```bash
structural-signatures replay --preset paper_w5 --budgets 5-99 --expected-ids 500 \
  --answers data/math500/answers.csv.gz --precedence data/math500/precedence.csv.gz \
  --out runs/test-w5
```

| Published test result | Structural | Majority | Net gain |
|---|---:|---:|---:|
| 67 traces | 314 / 500 | 304 / 500 | +10 questions |
| 80 traces | 318 / 500 | 308 / 500 | +10 questions |
| 99 traces | 318 / 500 | 313 / 500 | +5 questions |

The peak gain is **2.0 percentage points**. At 99 traces the structural selector repairs 17 majority errors and introduces 12 errors, reaching **63.6%** against **62.6%**. The five-prefix run matches the published net gain and both accuracy curves at every budget. The case-study notebook and parts of the results prose use the three-prefix run. [Reproduction audit](docs/REPRODUCTION.md)

![Net gain, changes in question-level contribution and matched test accuracy across 95 sampling budgets.](docs/assets/performance.png)

## Rebuild a repair from the original traces

The example includes all 99 input rows for question 315 and the original directional scorer values needed through budget 25. Running both commands reconstructs segmentation, checkpoints, precedence, graph scores and the final decision on CPU.

```bash
structural-signatures build --traces data/example/q315_traces.csv.gz \
  --scores data/example/q315_scores.csv.gz --budgets 5-25 --out runs/q315-graphs
structural-signatures replay --preset historical_w3 --budgets 5-25 --expected-ids 1 \
  --answers runs/q315-graphs/answers.csv.gz --precedence runs/q315-graphs/precedence.csv.gz \
  --out runs/q315-w3
```

If you have an original v8 SQLite shard, supply `--legacy-cache /path/to/shard.sqlite` in place of `--scores`. For fresh model scores, install `python -m pip install -e '.[model]'` and use `--model-cache runs/model-cache --device cuda`. The build command writes the two input tables for `replay` and a mapping from original row positions to retained trace positions. Choose a fresh output directory per run. [Input formats and data provenance](data/README.md)

## Method

1. **Find checkpoints.** Within a single answer and the current prefix, score every cross-trace pair of reasoning units in both directions with Qwen3-Reranker-0.6B. Use the smaller score. Merge at the answer-local 90th percentile with complete-link cohesion and at most one unit per trace. Keep communities supported by five or more traces.
2. **Build directed shortcuts.** Project each trace onto its ordered checkpoints. Keep a directed pair when at least three traces contain both checkpoints, at least 80% agree on that order, and the same direction occurs both adjacently and with a checkpoint in between.
3. **Score each answer.** For each weakly connected component, count the distinct edges beyond a spanning tree, $|E|-|V|+1$. Divide the largest such count by the number of traces with that answer in the current prefix. Average over the trailing five available prefixes for `paper_w5`, or three for `historical_w3`.
4. **Select.** Rank answers with at least two observations by smoothed score, current count, first retained position, then answer string. If every candidate is a singleton, all are eligible. With zero structural evidence, this ordering returns the matched majority.

## Figures, checks and citation

```bash
python -m pip install -e '.[notebook,test]'
jupyter lab notebooks/figure1.ipynb
python scripts/plot_performance.py
python -m pytest -q
```

The Figure 1 notebook exports the six-panel PNG and PDF. The performance plotting script rebuilds all three panels from archived question-level predictions and reconciles the transition counts at every budget.

Maria Marina and Alexander Panchenko. 2026. *Structural Signatures of Correct Mathematical Reasoning Traces.* The Proceedings of the 4th Workshop on Mathematical Natural Language Processing, pp. 194–200. Association for Computational Linguistics. [DOI](https://doi.org/10.18653/2026.mathnlp-main.15) · [BibTeX](paper.bib) · [Citation metadata](CITATION.cff)

The experiment builds on self-consistency, the MATH benchmark and process-supervision evaluation, and Qwen’s reranker. [Wang et al.](https://arxiv.org/abs/2203.11171) · [Hendrycks et al.](https://arxiv.org/abs/2103.03874) · [Lightman et al.](https://arxiv.org/abs/2305.20050) · [Zhang et al.](https://arxiv.org/abs/2506.05176)
