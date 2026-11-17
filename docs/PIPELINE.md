# Run the pipeline

Commands below run from the repository root. Use a new `--out` directory for each run. A completed directory contains its input fingerprints and effective settings in `run.json`. A model cache can be shared between runs.

| Starting material | First step | Stages you execute |
|---|---|---|
| Problem text and gold final answers, one question per JSONL row | 1 | Sample, grade, segment, score, cluster, build graphs, select and evaluate |
| Already graded completions, one trace per CSV row | 2 | All structural stages, with fresh reranker scores |
| Graded completions and a matching directional-score cache | 2, frozen command | All structural stages using existing pair scores |
| Saved answer and precedence tables | 3 | Filter edges, compute component excess, smooth, select and evaluate |
| Saved per-question predictions or development counts | 4 | Figures, ablations and comparisons with the paper |

## 1. Sample and grade from public MATH500 questions

For a new sampling experiment, the input is a JSONL file with the question text in `problem` and the gold final answer in `answer`. The public test file contains 500 questions. The problem text, gold answer and `unique_id` of every row match the archived experiment in the same order. The sampler uses the zero-based row number as `id`. [MATH500 test file](https://huggingface.co/datasets/HuggingFaceH4/MATH-500/blob/main/test.jsonl)

```bash
python -m venv .venv
source .venv/bin/activate
python -m pip install -e '.[generation,model]'
mkdir -p external
curl -fL https://huggingface.co/datasets/HuggingFaceH4/MATH-500/resolve/6e4ed1a2a79af7d8630a6b768ec859cb5af4d3be/test.jsonl \
  -o external/math500-test.jsonl
python scripts/sample_math500.py --questions external/math500-test.jsonl \
  --generator-model meta-llama/Llama-3.1-8B-Instruct \
  --generator-revision 0e9e39f249a16976918f6564b8830bc894c89659 \
  --samples 99 --temperature 0.9 --top-p 1.0 --max-new-tokens 2048 \
  --seed 42 --batch-size 4 --device cuda --out runs/sampled
```

Use a Hugging Face account with access to the Llama weights, or pass an existing model snapshot directory as `--generator-model`. The optional `--limit` runs only the first N questions. For a short end-to-end run, use `--limit 1 --samples 5` here, `--budgets 5` in steps 2 and 3, and `--expected-ids 1` in step 3. [Generator model card](https://huggingface.co/meta-llama/Llama-3.1-8B-Instruct)

| Sampling or grading setting | Meaning |
|---|---|
| `--generator-model`, `--generator-revision`, `--device` | Causal chat model, model revision, and device. This model writes the solutions. |
| `--samples 99`, `--temperature 0.9`, `--top-p 1.0` | Number of independent draws per question, temperature and nucleus probability. The sampler disables top-k truncation. |
| `--max-new-tokens 2048`, `--seed 42`, `--batch-size 4` | Maximum tokens generated per draw, RNG seed and draws generated together. Batch size affects memory and can change the sampled text. |
| `--questions`, `--limit`, `--out` | Source JSONL, optional first-N subset and destination. |

The sampler uses the generation prompt in Appendix A.2. It takes the last `\boxed{…}` expression, matches nested braces, and uses Math-Verify to compare the extracted answer with the gold answer in LaTeX math mode. A missing or incomplete box gives an empty answer. It writes all draws in order to `runs/sampled/traces.csv.gz`. The paper used a fixed set of 49,500 draws. This command generates a new sample with Hugging Face Transformers, so individual votes, graphs and accuracies can change. The bundled graph tables and frozen example use the original draws. [Paper, Appendix A](https://aclanthology.org/2026.mathnlp-main.15.pdf) · [Math-Verify](https://github.com/huggingface/Math-Verify)

## 2. Build checkpoints and precedence from traces

The structural pipeline starts with `id, completion, final_answer_new, row_hit` in a plain or gzipped CSV. Preserve the original draw order within each question. Gold labels serve only to grade the later decisions. If you already have the original expanded `math500_test_expanded.csv`, pass its path to `--traces` below to use the original draws. The source manifest gives its SHA-256. [Trace schema and real example](../data/README.md#sampled-traces-and-directional-scores) · [Source manifest](../data/source_manifest.json)

```bash
structural-signatures build --traces runs/sampled/traces.csv.gz \
  --model-cache runs/pair-cache --reranker-model Qwen/Qwen3-Reranker-0.6B \
  --reranker-revision e61197ed45024b0ed8a2d74b80b4d909f1255473 \
  --reranker-max-length 256 --reranker-batch-size 64 --device cuda \
  --max-unit-chars 900 --max-units-per-trace 96 --minimum-unit-position 0 \
  --score-mode raw_min --edge-quantile 0.90 --cohesion-mode min \
  --community-min-support 5 --transition-min-support 3 --consistency 0.80 \
  --budgets 5-99 --out runs/graphs
```

| Stage | Arguments and paper values | Operation and observable output |
|---|---|---|
| Retain and segment | `--budgets 5-99`, `--max-unit-chars 900`, `--max-units-per-trace 96`, `--minimum-unit-position 0` | Drop rows with an empty normalized answer. Cut off the final-answer suffix and split using sentence-ending punctuation. Split very long units and evenly subsample units in very long traces. The position cutoff is relative to the trace, from 0 to 1. `trace_order.csv.gz` maps original to retained positions. |
| Score units | `--reranker-model`, `--reranker-revision`, `--reranker-max-length 256`, `--reranker-batch-size 64`, `--device`, `--model-cache` | Compare every pair of units from different traces with the same answer. Score both orders with the Appendix A.1 checkpoint prompt. Length caps the ordered tokenized pair. Batch size controls pairs per forward pass. Scalar scores are saved in SQLite. |
| Combine directions | `--score-mode raw_min` | Take the smaller directional output. The original alternatives are `raw_mean`, `sigmoid_min` and `sigmoid_mean`. The sigmoid variants apply a sigmoid to each direction before combining. |
| Form checkpoints | `--edge-quantile 0.90`, `--cohesion-mode min`, `--community-min-support 5` | Compute the cutoff separately for each question, candidate answer and prefix. A merge must satisfy the cutoff for the weakest cross-community pair. `q10` uses their 10th percentile. A checkpoint contains at most one unit per trace and must reach the chosen distinct-trace count. Inspect `edge_threshold` and `n_local_communities` in `answers.csv.gz`. |
| Count orders and shortcuts | `--transition-min-support 3`, `--consistency 0.80` | Save a precedence row for each pair occurring together in enough traces. The dominant order must reach the consistency cutoff and occur at least once adjacently and once with an intermediate checkpoint to count as a shortcut. Inspect the support, directional counts and `shortcut_survives` in `precedence.csv.gz`. |

The budget `k` refers to the first `min(k, number of retained traces)` traces of each question. Candidate frequencies sum to that retained count. Raw unit pairs are scored once through the largest requested prefix. Percentiles and checkpoints are rebuilt from the observations available at *each* prefix.

### Select another reranker

`--reranker-model` accepts a Hugging Face Hub ID for a Sentence Transformers CrossEncoder that returns one scalar per ordered pair. The paper model defaults to its archived commit. Another model defaults to its own Hub revision. Supply `--reranker-revision` to pin that model. The run records the resolved snapshot. Fresh caches are isolated by model, snapshot, prompt, token limit, device, precision and package versions. Changing the reranker can change checkpoint membership and every downstream score. [CrossEncoder API](https://sbert.net/docs/package_reference/cross_encoder/model.html)

```bash
structural-signatures build --traces data/example/q315_traces.csv.gz \
  --model-cache runs/pair-cache --reranker-model Qwen/Qwen3-Reranker-4B \
  --device cuda --budgets 5-25 --out runs/q315-alternate
```

### Use frozen directional scores on CPU

```bash
structural-signatures build --traces data/example/q315_traces.csv.gz \
  --scores data/example/q315_scores.csv.gz --budgets 5-25 --out runs/q315-graphs
structural-signatures replay --answers runs/q315-graphs/answers.csv.gz \
  --precedence runs/q315-graphs/precedence.csv.gz --preset historical_w3 \
  --budgets 5-25 --expected-ids 1 --out runs/q315-w3
```

The 13,709 saved keys cover the unit pairs for question 315 through prefix 25 under the paper’s segmentation. A read-only original v8 shard can be supplied with `--legacy-cache PATH.sqlite`. Each of `--scores`, `--legacy-cache` and `--model-cache` selects one source of directional scores. A new reranker, token limit or a missing pair requires fresh scoring. The two `q315_expected_*` files independently check the resulting graph tables. [Data dictionary](../data/README.md)

## 3. Score each candidate and select answers

```bash
structural-signatures replay --answers runs/graphs/answers.csv.gz \
  --precedence runs/graphs/precedence.csv.gz --budgets 5-99 --expected-ids 500 \
  --score-mode raw_min --edge-quantile 0.90 --cohesion-mode min \
  --community-min-support 5 --transition-min-support 3 --consistency 0.80 \
  --metric largest_component_excess_edges_per_trace \
  --window 5 --minimum-answer-frequency 2 --out runs/selection
```

To replay the **original full test set**, substitute `data/math500/answers.csv.gz` and `data/math500/precedence.csv.gz` as the inputs. The run writes `answer_scores.csv.gz`, `predictions.csv.gz`, `accuracy.csv` and `run.json`.

| Argument | Meaning |
|---|---|
| `--preset paper_w5` or `--preset historical_w3` | Start with the published five-prefix selector or the three-prefix Figure 1 selector. Explicit flags override a preset. |
| `--score-mode`, `--edge-quantile`, `--cohesion-mode`, `--community-min-support`, `--transition-min-support` | Select the configuration actually present in **both** input tables. For another checkpoint configuration, run `build` with those settings and use the same values in `replay`. |
| `--consistency 0.80` | Minimum dominant-order fraction. Replay can tighten this cutoff. For a cutoff below the one recorded in an input table, use a build with the desired cutoff. |
| `--metric largest_component_excess_edges_per_trace` | Count distinct undirected edges minus vertices plus one in each weakly connected component. Use the largest excess across components and divide by the candidate’s current vote count. An empty graph or tree scores zero. `largest_component_excess_edges` uses the same excess before division. |
| `--window 5` | Average the chosen metric over the most recent available prefixes for this answer. A newly observed answer uses the available history. Start at budget 5 to reproduce the paper’s smoothing history. A window of 1 gives the current-prefix score. |
| `--minimum-answer-frequency 2` | Candidates need this many votes to be eligible. If none qualifies, all are eligible. Order them by descending score, descending votes, earliest retained position, then answer string. Majority uses the same vote and tie-breaking rules. |
| `--budgets 5-99`, `--expected-ids 500` | Inclusive consecutive sizes and the required number of questions at every size. For the bundled one-question example use `5-25` and `1`. |

## 4. Evaluate and visualize

```bash
python scripts/plot_performance.py --predictions runs/selection/predictions.csv.gz \
  --out runs/selection/performance.png
python scripts/check_results.py
python scripts/check_table1.py
```

- **Test metrics.** For each budget, `accuracy` and `majority_accuracy` are fractions over the same questions. A repair changes a wrong majority to a correct answer. A harm changes a correct majority to a wrong answer. `net_repairs = repairs − harms`, and the gain in percentage points is `100 × net_repairs / n_ids`. The performance plot shows the net count, signed changes between consecutive budgets and both accuracies.
- **Archive checks.** `check_results.py` compares the two archived windows against fresh graph replays on 500 questions. `check_table1.py` recomputes development means at `k=5–99` and `k=61–99`. Both scripts use the fixed paper references shipped in `data/reference/`. [Results and cohort sizes](REPRODUCTION.md)
- **Figure 1.** Install `python -m pip install -e '.[notebook]'` and open `jupyter lab notebooks/figure1.ipynb`. Run all cells with `layout_mode="published"` to recover the paper’s positions, or `"spring"` for a new layout of the same graph. The notebook reads the six saved cases, checks the arrows and their colors, and exports `docs/assets/figure1.png` and `figure1.pdf`. [Figure notebook](../notebooks/figure1.ipynb)
- **Additional checks.** `python scripts/check_trace_counts.py --traces PATH --answers data/math500/answers.csv.gz --budgets 5-99` compares the original expanded input with all published candidate frequencies, positions and labels. `python scripts/check_scorer.py --cache runs/probe-cache --reranker-model Qwen/Qwen3-Reranker-0.6B --device cpu` compares four pairs in both directions with the historical score cache. A different model makes this a comparison against the original Qwen outputs. [Scorer probe](../data/reference/scorer_probe.json)
