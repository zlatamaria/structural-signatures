# Data dictionary

A **question** has many sampled completions. At each requested budget `k`, traces are grouped by their extracted answer. One **candidate row** describes one answer for one question at one budget. One **precedence row** describes one ordered checkpoint pair within that candidate. One **prediction row** compares the selected answer and majority vote for the entire question at that budget.

## What is saved

| File under `data/` | Rows or records | What has already been computed |
|---|---:|---|
| `example/q315_traces.csv.gz` | 99 | Original completions, extracted answers and grading labels for question 315. There are 89 retained answers. |
| `example/q315_scores.csv.gz` | 13,709 | Cached directional model outputs needed through prefix 25 of that question. |
| `example/q315_expected_answers.csv.gz`, `example/q315_expected_precedence.csv.gz` | 186 / 112 | Independent archived intermediate tables for prefixes 5–25. These are check targets. |
| `math500/answers.csv.gz`, `math500/precedence.csv.gz` | 894,903 / 2,537,658 | Candidate counts and supported checkpoint pairs, with the columns needed to evaluate all 500 test questions at budgets 5–99. |
| `math500/run.json`, `source_manifest.json` | One each | Projection settings, original source file identities and run provenance. |
| `reference/w3_predictions.csv.gz`, `reference/w5_predictions.csv.gz` | 47,500 each | Historical question-level choices for the three- and five-prefix selectors. |
| `reference/w3_accuracy.csv`, `reference/w5_accuracy.csv` | 95 each | Budget-level summaries recomputed from the corresponding decisions. |
| `figure1/panels.json`, `figure1/answers.csv`, `figure1/precedence.csv` | 6 / 6 / 205 | Cases in display order and the candidate and precedence rows used to rebuild them. |
| `figure1/layout.csv`, `figure1/printed_edges.csv` | 51 / 72 | Node positions and directed arrows independently read from the publication artwork. |
| `reference/figure2_from_pdf.csv` | 95 | Net gain and correct counts independently read from the published plot. |
| `reference/dev_curves.csv.gz`, `reference/table1_printed.csv`, `reference/dev_sources.json` | 1,520 / 14 / 6 | Per-budget ablation counts including matched baselines, printed Table 1 values and their sources. |
| `reference/trace_audit.json`, `reference/scorer_probe.json` | One each | Input coverage and a four-pair comparison of fresh and historical scores. |

The checkout contains trace text for question 315 and graph inputs for all 500 questions. Starting from the public questions generates fresh trace text for the full test set. The paper’s original full expanded CSV is identified by its hash in the manifest. [From-scratch and replay commands](../docs/PIPELINE.md)

## Sampled traces and directional scores

`sample_math500.py` reads one JSON object per question from the public test file. `problem` is the prompt input and `answer` is the gold LaTeX answer for grading. `unique_id` identifies the source MATH problem, `subject` and `level` describe its topic and difficulty, and `solution` is the reference derivation. The generator only sees the problem and the fixed instruction.

| Field in `q315_traces.csv.gz` or a new `traces.csv.gz` | Meaning |
|---|---|
| `id` | Question identifier, constant across its draws. The public sampler uses zero-based question order. |
| `completion` | Full generated reasoning and final-answer text. CSV quoting preserves embedded newlines. |
| `final_answer_new` | Extracted answer as a string, including mathematical notation. Empty entries are dropped before prefix construction. Whitespace is normalized when grouping candidates. |
| `row_hit` | Boolean grade for this draw. A candidate’s archived correctness label is true if any retained draw for that answer is graded correct across the full question. Correctness is used for evaluation. |

For example, the first stored draw for `315` begins “Step 1: The geometric sequence must have a common ratio.” Its extracted answer is `31` and `row_hit` is false. Keep row order when loading the file. Use strings for answers so that distinct spellings survive a join.

| Field in `q315_scores.csv.gz` | Meaning |
|---|---|
| `pair_key` | SHA-256 key over model name, prompt hash, token limit and the two unit texts in lexicographic order. |
| `canonical_left_to_right`, `canonical_right_to_left` | Scorer outputs for each order. The first archived row contains `0.125` and `2.125`. With `raw_min` the pair’s similarity is `0.125`. |

New score databases live under `--model-cache`. Their `identity` table contains a JSON scorer fingerprint. Each `directional_logits` row holds the three fields above plus `model_name`, `prompt_hash`, `max_length` and the actual unit texts `canonical_left` and `canonical_right`. Pair keys identify the text pair within a scorer. New databases also distinguish resolved model snapshots and environments.

## Candidate and precedence tables

The two full-test tables, both Figure 1 case tables, and both `q315_expected_*` tables carry the following shared fields. Join on `id, sample_size, answer` within a single configuration. Checkpoint IDs are local to that key and can change at the next budget.

| Shared field | Meaning |
|---|---|
| `id`, `sample_size`, `answer` | Question ID, requested number of retained traces, and the candidate’s normalized answer string. When fewer than `sample_size` valid draws exist, the last available prefix is reused. |
| `score_mode` | How the two directional scores are combined. The paper uses `raw_min`. |
| `edge_quantile` | Within-candidate quantile used as the similarity threshold, `0.90` in the paper. |
| `cohesion_mode` | Statistic for accepting a community merge. `min` requires the weakest cross-community match to pass. |
| `community_min_support`, `transition_min_support` | Minimum distinct traces per retained checkpoint (5) and traces containing a pair for a saved precedence row (3). |

| Candidate field | Meaning |
|---|---|
| `raw_count` | Number of retained traces voting for this answer in this prefix. |
| `first_trace_position` | Zero-based index of its first retained trace. Used to resolve ties. |
| `is_correct_answer` | Full-question any-hit label for this answer. |
| `edge_threshold`, `n_local_communities` | Actual similarity cutoff and number of retained checkpoints. Saved in the question 315 expected file and fresh builds. The cutoff is infinite when there are no candidate unit pairs. |
| `retained_budget` | Actual number of retained traces of this question in the prefix. Written by `build`. In compact tables, sum `raw_count` over the question and budget to recover it. |

| Precedence field | Meaning |
|---|---|
| `from_community`, `to_community` | Source and target checkpoint IDs in the observed dominant order. In an exact order tie the implementation orients the pair by its canonical checkpoint order. |
| `cooccurrence_support` | Number of traces for this answer that contain both checkpoints in either order. |
| `ordering_consistency` | Fraction of co-occurrences in the dominant direction. |
| `adjacent_forward_count`, `nonadjacent_forward_count` | In the dominant direction, how many traces put the two checkpoints consecutively or with at least one checkpoint between them. |
| `precedence_min_consistency` | Dominant-order cutoff used by the source run. A supported row can fall below this value. |
| `forward_count`, `reverse_count` | Numbers of co-occurrences in the dominant and opposite directions. They sum to `cooccurrence_support`. |
| `support_fraction`, `direction_strength` | Co-occurrence support divided by the answer’s vote count, and `max(0, 2 × ordering_consistency − 1)`. |
| `mean_forward_gap` | Average difference in checkpoint-sequence positions over dominant-direction co-occurrences. Adjacent checkpoints have gap 1. |
| `shortcut_survives` | True if the order meets the consistency threshold and both forward realization counts are positive. |

The last four rows of the precedence dictionary describe extra diagnostics in the question 315 expected table and fresh builds. The compact test and Figure 1 files store the preceding fields needed to reconstruct membership. An answer with no saved precedence rows still competes with a structural score of zero.

**Follow question 315 at budget 25.** The candidate row for `41` has `raw_count=6`, `first_trace_position=11`, `is_correct_answer=True`, `edge_threshold=2.5` and seven retained checkpoints. Checkpoints `0 → 1` occur together in five traces. All five agree on the order, two are adjacent and three are expanded. This is a shortcut. The pair `0 → 2` also has five co-occurrences and unanimous order, but all five are expanded, so it contributes no edge. In the strongest component, four excess edges give the answer a current score of `4/6`. The three-prefix selection records a repair over the seven votes for `1`.

A new build also writes `trace_order.csv.gz`. Each row has `id`, `input_position` (zero-based draw position within its question) and `retained_position` (zero-based after filtering, blank for a dropped draw).

## Candidate scores, question decisions and budget summaries

`replay` writes `answer_scores.csv.gz` with the shared fields, `raw_count`, `first_trace_position` and `is_correct_answer`. `excess_edges` is the maximum of `E − V + 1` over the candidate’s connected components with directions ignored. `largest_component_excess_edges_per_trace` is that value divided by `raw_count`. The optional unnormalized metric column `largest_component_excess_edges` equals `excess_edges`. The column ending in `__rolling_mean_wN` is the actual selection score averaged over up to N available prefixes for the candidate.

`reference/w3_predictions.csv.gz` and `reference/w5_predictions.csv.gz` each have one row per `id, sample_size`. They also carry the five shared graph settings. New `predictions.csv.gz` files add `retained_budget`.

| Decision fields | Meaning |
|---|---|
| `minimum_answer_frequency`, `metric` | Vote threshold for eligibility and the name of the score including its rolling window. If no answer reaches the threshold, all candidates are eligible. |
| `predicted_answer`, `predicted_score`, `selected_raw_count`, `prediction_correct` | Structural choice, its smoothed score, its current votes and its correctness indicator (0 or 1). |
| `majority_answer`, `majority_score`, `majority_raw_count`, `majority_correct` | Matched majority choice, its score under the structural metric, its votes and its correctness indicator. |
| `repair`, `harm`, `net_repair` | A repair is a correct structural choice over a wrong majority. A harm is the reverse. The signed contribution is `repair − harm`. |
| `changed_from_majority`, `all_scores_zero` | Whether the answer strings differ and whether the best eligible score is numerically zero. |
| `metric_ties`, `score_margin` | Number of eligible candidates numerically close to the highest score, and the gap to the highest score outside that tie. The margin is zero if all eligible scores tie. Exact numeric scores determine selection. |
| `n_candidate_answers`, `n_eligible_answers` | Number of distinct observed answers and number in the selection pool. |

`reference/w3_accuracy.csv`, `reference/w5_accuracy.csv` and a new `accuracy.csv` have one row per `sample_size` and configuration. They include the five graph fields, `minimum_answer_frequency` and `metric`.

| Summary fields | Meaning |
|---|---|
| `n_ids`, `correct`, `majority_correct` | Questions evaluated, count correct under structural selection and count correct under majority. |
| `repairs`, `harms`, `net_repairs`, `changed_answers` | Counts of repaired errors, introduced errors, repairs minus harms, and changes in the selected answer string. |
| `zero_score_fallbacks`, `total_metric_ties`, `mean_score_margin` | Questions with a numerically zero top score, sum of the `metric_ties` counts, and average diagnostic margin. Each question contributes at least one candidate to `total_metric_ties`. |
| `accuracy`, `majority_accuracy`, `gain` | Structural and majority correctness fractions and their difference. Multiply `gain` by 100 for percentage points. |
| `change_rate`, `zero_score_rate` | Fractions of questions with a changed answer or with a zero top score. |

At budget 99 the five-prefix summary has `n_ids=500`, `correct=318`, `majority_correct=313`, `repairs=17` and `harms=12`. In `reference/figure2_from_pdf.csv` the fields are `sample_size, net_repairs, correct, majority_correct`. They use the same units and were decoded independently from the three plotted quantities.

## Figure 1 and Table 1

- **`figure1/panels.json`** is a list in panel order. `case` names the example, `role` identifies the structural, majority, previous or current selection, `id, sample_size, answer` link to the inputs, `raw_count` gives the votes and `correct` is the answer label. `figure1/answers.csv` and `figure1/precedence.csv` are compact table slices with the schemas above.
- **`figure1/layout.csv`** gives the one-based `panel`, local checkpoint `node`, and normalized panel coordinates `x, y`. **`figure1/printed_edges.csv`** gives `panel, from_community, to_community, highlighted`. A true `highlighted` value means a purple excess edge. The remaining arrows form the spanning tree shown in that panel. Another valid tree could highlight different edges while giving the same excess.
- **`reference/dev_curves.csv.gz`** holds 16 curves at 95 budgets. `row` names the ablation, `reference` names its matched baseline row, `source` identifies the archive, and `window` is the effective smoothing width. `sample_size, n_ids, correct, majority_correct` give the budget, cohort size and two correct counts. Each value in `reference/dev_sources.json` maps its source name to the original `path` and `sha256`.
- **`reference/table1_printed.csv`** uses `row` to identify each of the 14 printed settings. `full_gain` and `late_gain` are mean accuracy gains in percentage points for budgets 5–99 and 61–99. `full_delta` and `late_delta` subtract the matched reference gain over the same range. The two additional curves in the development file are subset baselines. [Cohorts and the printed rolling-window discrepancy](../docs/REPRODUCTION.md#development-ablations-in-table-1)

## Metadata and audit records

| File | Fields and interpretation |
|---|---|
| `math500/run.json` | `config` holds the five graph filters, `candidate_rows` and `precedence_rows` are projected row counts, `inputs` gives the source filenames and SHA-256 digests, and `python` and `packages` record the projection environment. |
| `source_manifest.json` | `repository_base_commit`, `paper` and `original_project` identify the sources. `original_files_sha256` fingerprints code, figures and original tables. `expanded_test_csv` identifies the complete original trace input. The URL and SHA-256 in `public_test_questions` identify the input JSONL for fresh sampling, with the fields and row count checked against the archive. `cache_sample` records the shard, scorer and pair coverage for question 315. `test_parts` gives the merged shards. `original_test_configs` records the source paths, filters, windows, evaluated metrics and population for both selectors. `projection_source` explains which exports were used. |
| `reference/trace_audit.json` | `raw_traces`, `retained_traces` and `rejected_answers` count input and retained draws. `questions` and `budgets` are the coverage dimensions. `matched_candidate_rows` and `matched_question_prefixes` count verified groups. `short_prefixes` counts requested groups with fewer valid traces than requested, `questions_short_at_last_budget` counts affected questions at 99, and `minimum_retained_traces` is the smallest per-question count. `conflicting_answer_labels` counts question–answer pairs with inconsistent draw-level grades. |
| `reference/scorer_probe.json` | `identity` describes the model, resolved snapshot, instruction hash, input length, prediction API, device, dtype, package versions and cache schema. `keys` gives the four pair keys in order. `archive` and `fresh_cpu` list the two canonical directions for each key. `max_abs_error` is the largest absolute difference. `cache_new` and `cache_hits` count fresh entries and subsequent hits. |
| New `run.json` | `command`, `inputs`, `python`, `packages` and `n_ids` identify the step, files, environment and population. A sampling run adds generator ID and revision, RNG and decoding settings, dtype and `generated_rows`. A build adds `preset`, resolved `config`, `requested_budgets`, `segmentation` and `scorer`. A replay also records `answer_rows` and `question_prefixes` along with its effective selection settings. |

## Inspect or regenerate files

Read a small slice without expanding the entire archive. Use `keep_default_na=False` for string keys.

```python
import pandas as pd

rows = pd.read_csv("data/example/q315_expected_precedence.csv.gz",
                   dtype={"id": str, "answer": str}, keep_default_na=False)
print(rows.query("id == '315' and sample_size == 25 and answer == '41'").head())
```

To project the original merged CSVs, regenerate the Figure 1 case slices, or extract the vector references from the ACL PDF, run the corresponding command from the repository root.

```bash
python scripts/prepare_tables.py --answers /path/to/answer_metrics.csv \
  --precedence /path/to/precedence_details.csv --out runs/projected
python scripts/extract_figure1.py
python -m pip install pdfplumber
python scripts/read_published_figures.py /path/to/2026.mathnlp-main.15.pdf --out runs/printed
```
