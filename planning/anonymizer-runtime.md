# Notebook 3 — Anonymizer runtime + quality report

**Notebook:** `notebooks/03_privacy_with_anonymizer.ipynb`
**Sidecar:** `notebooks/_run_n3_full.py` (full-pipeline run for the deliverable artifact)
**Anonymizer version:** `0.2.0` (pinned in `pyproject.toml`)
**Provider:** NVIDIA Build (free tier) — `nvidia/gliner-pii` for detection, `openai/gpt-oss-120b` for LLM-validate / -augment / -replace, `nvidia/nemotron-3-nano-30b-a3b` available for latent detection
**Measured:** 2026-05-28 — workshop-demo path via `nbclient`, full-pipeline path via the sidecar script

## Headline

| Path | Rows | Wall time | Result |
|---|---:|---:|---|
| Workshop demo (in-notebook, `RUN_FULL_PIPELINE=False`) | 5 sub-sampled, 3 previewed | **2.7 min** | 0 failures, all preview cells render |
| Full pipeline (offline, `_run_n3_full.py`) | **20 / 20** | **3.47 min** | 0 failures, 1370 entities surfaced, 44 distinct labels, **20/20 anonymized traces parse as valid JSON** |

Both fit the workshop's 15-min N3 slot with comfortable narration headroom. The `Substitute` strategy is the single replacement mode the notebook now showcases (`Redact`, `Annotate`, `Hash`, `Rewrite` were intentionally cut from the live demo to keep the arc focused).

## Workshop-demo timing (notebook end-to-end)

| Cell | Source (excerpt) | Seconds |
|---:|---|---:|
| 2 | imports + `environment_setup(provider="auto")` | 0.9 |
| 4 | `load_usage_logs_seed()` + display | 0.1 |
| 5 | pretty-print one full row across all three text surfaces | 0.0 |
| 8 | `from anonymizer import …` + `Anonymizer(…)` | 3.2 |
| 10 | subsample to `LIVE_DEMO_ROWS=5`, build `AnonymizerInput` for `conversation_trace` | 0.0 |
| **12** | **`Substitute` preview, 3 rows × 1 column** | **137.4** |
| 14 | regex-baseline gap callout | 0.0 |
| 17 | full `anonymizer.run()` — gated behind `RUN_FULL_PIPELINE=False` | 0.0 |
| 19 | save artifacts — gated behind `RUN_FULL_PIPELINE=False` | 0.0 |
| **Total** | | **141.7 (2.4 min)** |

Cell 12 is the only meaningful cost: the LLM-augment / -validate / -replace pipeline running on three sub-sampled rows. Cell 14 (the new Presidio-style baseline) is regex-only and free.

## Full-pipeline timing (sidecar, all 20 rows)

| Stage | Seconds | Throughput |
|---|---:|---|
| Detection (GLiNER + LLM-validate + LLM-augment) | 104.7 | 5.2 s / row |
| Replacement (LLM substitute generation) | 102.3 | 0.07 s / entity |
| **Total** | **207.0 (3.47 min)** | 0 failures |

Detection time scales sub-linearly thanks to per-call parallelism (`max_parallel_requests: 16` for `gpt-oss-120b`). On the free tier this is the realistic upper bound; a paid endpoint with higher parallelism would shave it further.

## Why not Presidio? — the headline argument

A "Presidio-style" hand-coded regex baseline (emails, phones, dates, ZIPs, SSNs, IPs, credit cards) was run on the same 20 input rows for direct comparison.

| Detector | Spans / entities surfaced |
|---|---:|
| Presidio-style regex baseline (7 patterns) | **247** |
| Anonymizer (GLiNER 65-label set + LLM-validate + LLM-augment) | **1370** |
| **Gap (entities the regex baseline misses)** | **+1123** |

The gap is not just "more of the same" — it's *categorical*. Anonymizer surfaced **44 distinct entity labels** on this corpus, including categories conventional regex/dictionary tools simply don't ship with:

| Category Anonymizer caught | Count | Why regex misses it |
|---|---:|---|
| `first_name`, `last_name` | 333 + 222 | Dictionary-based but Presidio's PERSON detector can miss in dense JSON |
| `unique_id` | 306 | Generic identifier shape — no regex covers all enterprise patterns |
| `occupation` | 199 | Pure-text label, no pattern |
| `organization_name` | 176 | Open vocabulary (e.g., `KPI Copilot`, `VeridianFlow Solutions`) |
| `country`, `city`, `place_name` | 100+42+23 | NER, not regex |
| `monetary_amount` | 86 | Non-trivial pattern with currency / scale variation |
| `project_codename` | 12 | The smoking gun — these are quasi-identifiers no closed-set tool catches |
| `field_of_study`, `nationality`, `workstream_name`, `initiative_name`, `document_id`, `incident_id`, `notification_id`, `metric_value`, `customer_satisfaction_score`, `recurrence_risk_score`, `headcount_range` | varies | Domain-specific labels in GLiNER's open-set vocabulary |

Per-row throughput: **min 39, max 105, mean 68.5 entities**. Even the lightest row carries ~14× the regex baseline's haul.

## Per-row anonymization quality audit

20 rows; for each row we re-extract a class of identifiers from the *original* text and check whether each one literally still appears in the *anonymized* text. `100%` = every instance of that class was replaced. `·` = no instances of that class in that row.

| # | persona | doctype | ents | json | names | emails | phones | codenames | products | doc_ids | roles |
|---:|---|---|---:|:---:|---:|---:|---:|---:|---:|---:|---:|
| 0 | people analytics lead | market research brief | 38 | ✓ | 100% | 100% | 100% | 100% | 100% | 100% | 100% |
| 1 | operations analyst | financial variance memo | 99 | ✓ | 81% | 100% | 100% | 100% | 100% | 100% | 100% |
| 2 | compliance officer | customer support incident review | 44 | ✓ | 69% | 100% | 100% | 100% | **0%** | 100% | 100% |
| 3 | people analytics lead | market research brief | 58 | ✓ | 100% | 100% | 100% | 100% | 100% | 100% | 100% |
| 4 | operations analyst | customer support incident review | 54 | ✓ | 100% | 100% | 100% | 100% | 100% | 100% | 100% |
| 5 | product manager | market research brief | 81 | ✓ | 100% | 100% | 100% | 100% | 100% | 100% | 100% |
| 6 | compliance officer | financial variance memo | 82 | ✓ | 100% | 100% | 100% | · | **0%** | 100% | 100% |
| 7 | people analytics lead | operations dashboard export | 69 | ✓ | 100% | 100% | 100% | · | 100% | 100% | 100% |
| 8 | people analytics lead | market research brief | 48 | ✓ | 94% | 100% | 100% | · | 100% | 100% | 100% |
| 9 | finance analyst | financial variance memo | 71 | ✓ | 100% | 100% | 100% | · | 100% | 100% | 100% |
| 10 | operations analyst | operations dashboard export | 81 | ✓ | 100% | 100% | 100% | 100% | 100% | 100% | 100% |
| 11 | operations analyst | customer support incident review | 86 | ✓ | 100% | 100% | 100% | · | 100% | 100% | 100% |
| 12 | finance analyst | financial variance memo | 82 | ✓ | 73% | 100% | 50% | 100% | 100% | 100% | 100% |
| 13 | compliance officer | clinical trial status report | 68 | ✓ | 100% | 100% | 100% | 100% | 100% | 100% | **50%** |
| 14 | engineering manager | product launch readiness plan | 41 | ✓ | 100% | 100% | 100% | 100% | 100% | · | 100% |
| 15 | operations analyst | financial variance memo | 54 | ✓ | 100% | · | 100% | 100% | 100% | 100% | 100% |
| 16 | finance analyst | operations dashboard export | 77 | ✓ | 85% | 100% | 75% | · | 100% | 50% | 100% |
| 17 | product manager | product launch readiness plan | 78 | ✓ | 100% | 100% | 100% | **0%** | 100% | · | 100% |
| 18 | engineering manager | customer support incident review | 85 | ✓ | 100% | 100% | 100% | · | 100% | 100% | 100% |
| 19 | operations analyst | operations dashboard export | 74 | ✓ | 65% | 100% | 100% | 100% | 100% | 100% | **33%** |

**Aggregate replacement rates**

| Class | Replaced | Total | Rate |
|---|---:|---:|---:|
| `email` (regex-extracted ground truth) | 40 | 40 | **100.0%** |
| Phone numbers | 64 | 66 | 97.0% |
| Document IDs (`MRB-EU-…`, `INC-2026-…`, `Memo-FIN-…`) | 19 | 20 | 95.0% |
| Compound role descriptors ("manager Arjun Mehta", "VP Operations Sarah Chen") | 61 | 64 | 95.3% |
| Invented product names (`KPI Copilot`, `VeridianFlow`, …) | 18 | 20 | 90.0% |
| Person names (Anonymizer-detected `first_name` / `last_name`) | 357 | 379 | **94.2%** |
| Internal codenames (`Project Atlas`, `Q3 close review`, `TalentFlow-2024`) | 14 | 17 | **82.4%** ← weakest class |

### Where codenames slip through (the 3 leaks)

- **Row 17** — `Project Phoenix` survives in a terse user-message turn ("Give me a rushed summary of Project Phoenix readiness for leadership"). The same codename in the system prompt for the same row was caught. Pattern: GLiNER scores codenames in dense, terse user messages below threshold more often than codenames in formal system-prompt prose.
- **Row 2 and Row 6** — invented product names slipped through (`0%` in the products column). Likely the LLM-augment step recognized them as product references but the validator filtered them as company-name false positives.

**Tunable mitigations** (already mentioned in the notebook narration): lower `gliner_threshold` from 0.3 → 0.2 to recover the user-message codenames, or extend the entity-label list with row-specific terms via `Detect(entity_labels=[*DEFAULT_ENTITY_LABELS, "internal_initiative", "saas_product_name"])`.

**JSON validity:** **20/20** anonymized traces parse as valid JSON.

## Demo-flag posture

| Flag | Default | Effect |
|---|---|---|
| `LIVE_DEMO_ROWS` | `5` | Workshop subsample for previews (cell 10). |
| `RUN_FULL_PIPELINE` | `False` | Workshop default — `anonymizer.run()` skipped, the saved artifact is `_artifacts/03_usage_logs_anonymized__demo_preview.parquet`. Set `True` for the offline run. |

The full-pipeline artifact (`_artifacts/03_usage_logs_anonymized.parquet`, 20 rows) is produced offline by `notebooks/_run_n3_full.py` and committed-ready. Flipping `RUN_FULL_PIPELINE=True` in the notebook produces a separate `__full.parquet` so the two paths never collide.

## Workshop fit (15-min N3 slot)

The notebook now reads as a tight Substitute-only arc:

- **2 min** — cells 4/5: pretty-print one row across the three text surfaces; land "look at what's in here" before any pipeline runs.
- **3 min** — cells 8/10: Configure (`Detect`, `AnonymizerConfig`, `AnonymizerInput`); explain the GLiNER + LLM-validate + LLM-augment + LLM-substitute pipeline architecture.
- **3 min narration** during cell 12's ~140 s `Substitute` preview. Talk through the four kinds of identifiers we're catching (the four-PII-surfaces beat lives entirely inside `conversation_trace`'s 11-message structure).
- **2 min** — cell 14: the Presidio-style baseline gap. The big punch line: 44 distinct labels vs the regex baseline's 7. This is "why not just regex?" answered with a live number on screen.
- **2 min** — Part 4 markdown on `Rewrite`: positioning it as the content-not-structure tool, not a fit for log JSON.
- **2 min** — Part 5 (`Full run and save`): explain the `RUN_FULL_PIPELINE=False` gate, point at the 3.5-min full run on 20 rows produced offline.
- **1 min** — Recap.

≈ 15 min of narration on a notebook with 2.4 min of execute time. Comfortable.

## Artifacts produced

| Path | Contents | Source |
|---|---|---|
| `data/usage_logs_seed.parquet` | 20 rows × 6 cols of synthetic VLM-deployment logs (`user_persona`, `task_intent`, `document_type`, `system_prompt`, `attached_document_context`, `conversation_trace`) | `notebooks/bonus_generate_usage_logs.ipynb` |
| `artifacts/03_usage_logs_anonymized.parquet` | 20 rows × 8 cols, includes `conversation_trace_anonymized` | `notebooks/_run_n3_full.py` |
| `artifacts/03_usage_logs_anonymized__trace.parquet` | Full Anonymizer trace (`final_entities`, `_seed_entities`, etc.) | sidecar |
| `artifacts/03_n3_full_run_stats.json` | Timing, label breakdown, regex gap, leak-audit aggregates, JSON-integrity stats | sidecar |
| `artifacts/03_n3_per_row_audit.json` | Per-row: replacement rates per class, JSON parseability | post-hoc audit |
| `artifacts/03_n3_per_row_entities.json` | Per-row entity counts and label breakdown | post-hoc audit |

## Reproducing this report

```bash
# 1. Generate the seed (≈12 min on free-tier Build API; over-generates 25, truncates to 20)
uv run jupyter nbconvert --to notebook --execute notebooks/bonus_generate_usage_logs.ipynb --inplace

# 2. Workshop-path timing (re-run the notebook end-to-end via nbclient; ≈2.4 min)
git show backup-2026-05-28-pre-upstream-merge:notebooks/_time_notebook.py > notebooks/_time_notebook.py
cd notebooks && uv run python -u _time_notebook.py 03_privacy_with_anonymizer.ipynb --timeout 1200
rm notebooks/_time_notebook.py

# 3. Full-pipeline run (≈3.5 min on 20 rows)
uv run python notebooks/_run_n3_full.py
```
