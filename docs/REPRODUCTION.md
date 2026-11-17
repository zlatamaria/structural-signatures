# Reproduction audit

The October 2026 ACL edition is the reference for printed values. The test and development comparisons below use archived outputs from the authors’ checkpoint project. Model generation, answer extraction and grading were upstream of the saved expanded CSV. [Published paper](https://aclanthology.org/2026.mathnlp-main.15.pdf) · [Source fingerprints](../data/source_manifest.json)

## Test decisions and Figure 2

Each run has one prediction for each of 500 questions at 95 requested budgets. A fresh CPU replay reconstructed the maximum component excess from 2,537,658 precedence records and joined it to 894,903 observed candidate records. Fifty questions have no stored support-three precedence row and remain in the denominator. Both windows reproduced all **47,500 question-level decisions apiece**, including majority predictions, frequencies, labels, repairs, harms and tie diagnostics. Floating scores and margins agree within an absolute tolerance of $10^{-12}$.

| Evidence | Three-prefix mean | Five-prefix mean | Published value or origin |
|---|---:|---:|---|
| Peak above majority | +10 at 66 | +10 at 67 and 80 | +10 questions |
| Net at 99 | +5 | +5 | +5 |
| All three plotted values agree with Figure 2 | 48 / 95 budgets | **95 / 95 budgets** | Vector paths, p. 197 |
| Net from 85 to 86 | +4 → −1 | +5 → +1 | Prose describes a drop of five |
| Zero structural scores at 45 and 60 | 227 → 173 | 224 → 170 | Prose gives 227 → 173 |
| Questions with tied structural scores at 45 and 60 | 206 → 158 | 203 → 155 | Prose gives 206 → 158 |

The two selectors choose different answers on 245 question-prefixes. At 99, each has 17 repairs and 12 harms. The five-prefix curve matches the article’s methods section. For the three-prefix transition from 85 to 86, three lost repairs come from majority catching up and one from the structural selector losing its correction. Direct repair-to-harm transitions contribute −2. The five-prefix run contains one such event, which appears in the figure legend although the caption describes six transition types. [Saved prediction sources](../data/source_manifest.json) · [Decoded published curves](../data/reference/figure2_from_pdf.csv) · [Replay checker](../scripts/check_results.py)

## Figure 1

The recovered `05_diagnose_net_repair_gaps.ipynb` reads the c080/w3 test predictions. Its case-selection, graph-construction and compact-plot cells selected these exact question IDs, budgets and answers. The reconstruction repeated its selection from the saved predictions and component statistics. The early search considered budgets 15–35 and the late search 70–99. The original notebook favored readable repair graphs and sorted candidates by score margin and excess. The drift search compared adjacent choices on the same question.

| Case | Question | Budget | Answer | Frequency | Excess |
|---|---:|---:|---|---:|---:|
| Early repair, structural | 315 | 25 | 41 | 6 | 4 |
| Early repair, majority | 315 | 25 | 1 | 7 | 0 |
| Late repair, structural | 80 | 78 | 501 | 8 | 9 |
| Late repair, majority | 80 | 78 | 499 | 12 | 7 |
| Before drift, correct | 435 | 40 | (−∞, 2) ∪ (3, ∞) | 12 | 3 |
| After drift, incorrect | 435 | 41 | (−∞, 2] ∪ [3, ∞) | 10 | 4 |

The notebook starts with the 205 saved precedence rows for these candidates and constructs 72 directed shortcut edges. All 72 endpoint pairs and directions match the printed arrows. The 27 purple edges complete cycles. The drawing uses the published node coordinates and original purple-edge choices. The code independently verifies each spanning tree and its cycle rank. [Executed notebook](../notebooks/figure1.ipynb) · [PDF coordinate extraction](../scripts/read_published_figures.py)

## Development ablations in Table 1

All **14 rows**, including the reference, are recoverable to the printed three decimal places from archived per-budget correct counts. Means are accuracy gains over matched majority, in percentage points. The full range has 95 budgets and the late range has 39. The source configurations explain two apparent inconsistencies.

| Table 1 rows | Questions | Archived window | Reference for printed deltas |
|---|---:|---:|---|
| Reference, precedence support, ordering consistency | 400 | 5 | +0.158 full, +0.737 late |
| Rolling variants | 400 | 1, 3, 7, 10 | The five-prefix reference above |
| Community support 2 and 3 | 100 | 3 | Support 5 on the matched 100, −1.432 full and −2.333 late |
| Similarity quantiles .85 and .95 | 350 | 3 | Quantile .90 on the matched 350, +0.144 full and +0.696 late |

The row labeled “5” in the rolling-window block gives **+0.150 and +0.673**, the archived **three-prefix** values. Its printed reference label is “3”. The reference row and caption use **five prefixes**. For the community and quantile sweeps, the paper’s statement that every variant uses the same questions differs from the saved cohort sizes. Their printed deltas agree with the appropriate subset baselines. [Recompute all rows](../scripts/check_table1.py) · [Printed values](../data/reference/table1_printed.csv) · [Six development sources](../data/reference/dev_sources.json)

## Raw traces and scorer checks

- The original expanded test CSV has 49,500 generations, exactly 99 per question. Dropping 2,320 rows with missing or empty extracted answers leaves 47,180 retained traces. The smallest question has 51 retained traces and 397 questions have fewer than 99. Requested and retained budgets differ in 2,320 question-prefix groups.
- Recounting from the expanded CSV in input order reproduces **all 894,903 candidate rows** for frequency, first retained position and correctness label. One normalized answer on question 420 has inconsistent per-trace labels. The historical evaluator uses the logical OR of those labels over the full question. Correctness is used only when assessing decisions.
- For question 315, 13,709 distinct cached directional pairs cover all 14,558 cross-trace unit pairs through budget 25. The rebuilt pipeline agrees with the source on all 186 candidate rows, including thresholds and community counts, and on all 112 supported precedence rows. The unmodified original evaluator also reproduced the 21 decisions from these rebuilt tables. The example includes 89 retained traces for evaluation labels.
- Four fresh CPU pair evaluations used the snapshot found on the server, `e61197ed45024b0ed8a2d74b80b4d909f1255473`, with the recovered prompt and CrossEncoder call. Three of eight directional values match exactly. The largest absolute difference is **0.1875**. Re-reading the new cache returned all eight values exactly. Historical cache keys omit dependency versions and model revision. This probe establishes the behavior of the current CPU stack on this sample. Exact scorer parity across all historical pairs is unverified.

The full test replay starts at the saved graph inputs. The trace-level replay starts at expanded completions and frozen scores for one question. Trace generation, regrading and fresh model scoring across the full corpus remain outside the executed checks. [Trace audit](../data/reference/trace_audit.json) · [Scorer environment and values](../data/reference/scorer_probe.json) · [Fresh scorer check](../scripts/check_scorer.py)

## Source lineage and implementation

The server merge manifest identifies three test shards containing 200, 150 and 150 questions. The evaluation metadata points to their merged `answer_metrics.csv` and `precedence_details.csv`. The repository projects those files directly. An older compact precedence export spells some numeric answers differently, including `2117.0` where the matching answer is `2117`. At the paper configuration, 116 hard-graph keys for questions 158 and 160 fail an exact join in that export. The merged sources preserve the candidate keys and reproduce all archived decisions without spelling corrections. The expanded test comparison independently validates split and order. [Manifest and source checksums](../data/source_manifest.json)

| Stage | Recovered implementation | Repository entry point |
|---|---|---|
| Segmentation and answer preparation | `checkpoint_route_convergence_punctuation.py` and v8 `prepare_question` | `traces.prepare` |
| Directional scoring | v8 `CachedDirectionalReranker`, `PairScoreStore`, `score_from_directional` | `scoring.FrozenScores` and `scoring.ModelScores` |
| Prefixes and communities | Optimized v8 incremental pair state, v8 `build_constrained_communities` | `traces.build_question` |
| Checkpoint order | Optimized v8 `prepare_structural_sequence_context` and `precedence_metrics_from_context` | `build` precedence output |
| Topology | Consistency evaluator’s maximum component excess, restricted to the published metric | `graphs`, `replay.load_scores` |
| Answer selection and metrics | Consistency evaluator’s exact ranking and accuracy aggregation | `replay.select`, `replay` CLI |

The four original modules are byte-preserved in `src/structural_signatures/legacy/`. The public entry points keep input order, current-prefix quantiles, trace-exclusivity and tie order. The scorer caches both directions, and new model caches include the snapshot, prompt, device, precision and package versions. The scorer’s historical scalar meaning is defined by the library call. The code performs no additional sigmoid or token-margin conversion. [Legacy sources](../src/structural_signatures/legacy) · [Data schemas](../data/README.md)
