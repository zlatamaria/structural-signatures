# Data and input formats

## MATH500 test inputs

The two gzip CSVs in `math500/` contain every row required to replay the fixed paper configuration. They are column projections of the merged v8 test run. Both preserve the spelling of answer strings. The input contains 894,903 candidate rows and 2,537,658 supported precedence rows across the 500 questions and requested budgets 5–99. Counts and input fingerprints are recorded in the [projection metadata](math500/run.json) and [source manifest](source_manifest.json).

| Field | Meaning |
|---|---|
| `id`, `sample_size`, `answer` | Question, requested prefix size, canonical answer string. Together they identify a candidate in one configuration. |
| `score_mode`, `edge_quantile`, `cohesion_mode`, `community_min_support`, `transition_min_support` | Stored graph-construction settings. Both tables carry these fields. |
| `raw_count`, `first_trace_position`, `is_correct_answer` | Candidate frequency, zero-based first retained position, and any-hit label for the answer over all retained traces of the question. |
| `from_community`, `to_community`, `cooccurrence_support` | Checkpoint IDs in dominant order and number of supporting traces. IDs are local to one question, answer and prefix. |
| `ordering_consistency`, `adjacent_forward_count`, `nonadjacent_forward_count` | Fraction in the dominant direction and counts of its direct and expanded realizations. |
| `precedence_min_consistency` | Threshold recorded in the source run. The replay validates the requested threshold against it. |

An answer may have zero precedence rows. The answer table defines the evaluation population. The sum of `raw_count` over candidates is the retained budget. Of the 49,500 original generations, 47,180 had a retained answer. All 500 IDs and all 95 requested budgets remain in evaluation. The stored projections support both smoothing windows. Different checkpoint quantiles or community supports require upstream data.

## Expanded traces and directional scores

`build` accepts plain or gzipped CSV. It reads `id`, `completion`, `final_answer_new`, and `row_hit`. Preserve input order within each question. The first two fields identify the question and provide generated reasoning. The last two provide an already extracted answer and its evaluation label. The recovered helper collapses whitespace and drops missing or empty answers before assigning retained positions. Generate or grade new answers separately.

`example/q315_traces.csv.gz` contains those four fields for all 99 original rows of question 315. The full question preserves the historical any-hit evaluation labels. Its frozen score table covers the first 25 retained traces. Each row contains a legacy SHA-256 pair key and the two original scalar scores in canonical text order. The key encodes the scorer name, prompt hash, maximum length and both unit strings. `q315_expected_*.csv.gz` are independent rows from the merged archive used by the tests. The original test questions and model documentation are available separately. [MATH-500](https://huggingface.co/datasets/HuggingFaceH4/MATH-500) · [Llama 3.1](https://huggingface.co/meta-llama/Llama-3.1-8B-Instruct) · [Qwen3 reranker](https://huggingface.co/Qwen/Qwen3-Reranker-0.6B)

## Figures and independent references

- `figure1/panels.json` records the historical choices. The two case CSVs are small slices of the test inputs. The layout and printed-edge CSVs come independently from the published PDF. The notebook checks all directed endpoints and highlighted counts.
- `reference/w{3,5}_predictions.csv.gz` are filtered historical predictions for the target metric. Each has 47,500 records. The corresponding accuracy CSVs are summaries from replay. `figure2_from_pdf.csv` contains the independently decoded plot coordinates.
- `reference/dev_curves.csv.gz` contains per-budget counts for each Table 1 setting and its matched reference. `dev_sources.json` identifies the six source tables. `table1_printed.csv` transcribes the published values in percentage points.
- `reference/scorer_probe.json` contains fresh and cached predictions for the four-pair CPU check. `reference/trace_audit.json` records the comparison to the original 49,500-row expanded test CSV.

## Regenerate the projections

Point the first command at the merged answer and precedence CSVs. The second command re-extracts the six panels from the distributed tables. Use the last command with the ACL edition to decode the printed arrows, positions and performance curves.

```bash
python scripts/prepare_tables.py --answers /path/to/answer_metrics.csv \
  --precedence /path/to/precedence_details.csv --out runs/projected
python scripts/extract_figure1.py
python -m pip install pdfplumber
python scripts/read_published_figures.py /path/to/2026.mathnlp-main.15.pdf --out runs/printed
```
