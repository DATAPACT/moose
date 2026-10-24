# DPV annotation on DPVCallBench

Evaluation harness and first results for Moose's DPV annotation, the two-step
retrieval approach and the Laya "System 1" model, on
[DPVCallBench](../DPVCallBench) v0.1.

All numbers below are on the **dev split** (150 records, 344 gold mentions). The test
split is untouched, as the benchmark's protocol requires, until the approaches are
frozen. GLiNER's threshold and Laya's temperature were fitted on **train**.

## How to run

```bash
# LLM baselines through a local Moose API (shared models, MOOSE_API_KEY from .env)
python run_moose.py --register                                  # once: 21 labels as a Moose schema
python run_moose.py --schema dpvcallbench_labels --split dev    # closed label set
python run_moose.py --schema dpv --split dev                    # open DPV (all concepts)
python run_moose.py --schema dpv_privacy --split dev            # open DPV, two-step (284 concepts)

# Local models (separate environment: research/.venv-laya with laya, gliner, sentence-transformers)
../.venv-laya/bin/python laya_link.py --pool labels --k 21      # Laya on gold spans
../.venv-laya/bin/python laya_link.py --pool dpv --k 5          # retrieval over all DPV + Laya
../.venv-laya/bin/python local_pipeline.py --split dev          # GLiNER, GLiNER + Laya
../.venv-laya/bin/python laya_rank.py --definition-chars 140    # Laya ranks the two-step candidates
../.venv-laya/bin/python fit_system1.py                         # System 1 + LLM: fit the consult threshold
../.venv-laya/bin/python make_colab_notebook.py                 # notebook to fine-tune Laya on a GPU
```

- `bench.py`: loading, scoring with the benchmark's own `scripts/evaluate.py`, per-label
  scores and error lists, mapping DPV concepts up to the benchmark's 18 concepts.
- Each run writes its predictions and a `report.json` under `runs/`.
- Moose outputs are cached per record (`raw.jsonl`), so a rerun doesn't pay again.

## End-to-end results (dev)

F1 scores. "Strict" needs exact span offsets, "overlap" any overlap. "Span + concept"
needs the exact span and the right DPV concept. "Triple" also needs the right label.

| System | Strict span | Overlap span | Span + concept | Triple | Cost / record | Time / record |
|---|---|---|---|---|---|---|
| Moose, open DPV (9,481 concepts in the prompt) | 0.432 | 0.755 | 0.369¹ | 0.363¹ | $0.026 (98k prompt tokens) | ~10 s |
| **Moose, two-step `dpv_privacy` (284 concepts, retrieved per mention)** | **0.823** | 0.885 | 0.632¹ | 0.480² | $0.0014 (1,100 tokens) | ~2 s |
| Moose, closed set (21 labels as a user schema) | 0.735 | **0.984** | 0.730 | 0.730 | $0.001 (581 tokens) | ~1–2 s |
| GLiNER (`gliner_multi_pii-v1`), zero-shot, 21 labels | **0.793** | 0.885 | **0.755** | **0.755** | local | 0.06 s |
| GLiNER spans + Laya choosing the label | 0.773 | 0.867 | 0.575 | 0.575 | local | 0.18 s |

LLM: `openai/gpt-oss-120b` through OpenRouter, temperature 0. The dev runs cost $4.10
in total.

¹ After mapping finer DPV concepts up to the benchmark's concepts (for example
`EmailAddressPersonal` to `EmailAddress`). Without that mapping it is 0.335 / 0.329. The
protocol asks for both, because the mapping is added by the adapter.

² Low only because three labels share `dpv-pd:Identifier` and two share
`AccountIdentifier`, so an open-DPV system cannot recover the label from the concept.
For open DPV, compare span + concept and hierarchical F1 (two-step: 0.752; full DPV
prompt: 0.407).

### Per label (triple F1, dev)

| Label | Gold | Moose closed set | GLiNER | Moose open DPV |
|---|---|---|---|---|
| age | 51 | 0.53 | 0.91 | 0.06 |
| birth_date | 51 | 0.92 | 0.99 | 0.76 |
| customer_id | 51 | 0.43 | 0.83 | 0.00 |
| ticket_id | 51 | 0.86 | 0.91 | 0.02 |
| health_issue | 54 | 0.98 | 0.54 | 0.67 |
| physical_address | 22 | 0.05 | 0.65 | 0.45 |
| postal_code | 22 | 1.00 | 0.05 | 0.47 |
| phone | 21 | 1.00 | 1.00 | 1.00 |
| payment_token | 10 | 1.00 | 0.00 | 0.00 |
| 10 other labels | 1–2 each | | | |

The dev split is dominated by five labels with about 50 mentions each. Ten labels have
only one or two, so their scores are anecdotal.

## Laya on gold spans (dev)

This measures only the concept choice: Laya gets the correct span and picks among
candidates.

| Candidates | Gold among candidates | Accuracy | Calibration error (ECE), raw → fitted temperature |
|---|---|---|---|
| All 21 labels | 100% | 0.744 | 0.189 → 0.133 (T = 3.28) |
| Top 5 of the 21 labels, by embedding retrieval | 84.9% | 0.628 | 0.075 → 0.036 (T = 1.15) |
| Top 5 of all DPV concepts (8,648 with a definition) | 47.4% | 0.265 | 0.272 → 0.199 (T = 1.68) |
| Top 10 of all DPV concepts | 49.4% | 0.265 | 0.296 → 0.118 (T = 1.85) |

Embeddings: `BAAI/bge-small-en-v1.5`. Laya: `convaiinnovations/laya` (English,
ModernBERT-large, 421M parameters) on Apple MPS, about 0.05 s per mention. No input was
truncated.

## Two-step open DPV (`dpv_privacy`)

The two-step mode is implemented in Moose: `candidate_mode: "two_step"` in a schema's
registry entry.

1. The LLM finds mentions and describes each in a few words, without seeing any type
   list. The prompt also states span rules: the value without cue words, split
   addresses from postal codes, no category-only words.
2. BM25 retrieves 8 candidates per mention from the concept labels and definitions, and
   adds the direct parents and children of the top 3. The LLM then chooses one, or
   none.

| Dev, open DPV | Strict span | Span + concept¹ | Hierarchical | Gold among candidates³ | Cost |
|---|---|---|---|---|---|
| `dpv`, all 9,481 concepts in one prompt | 0.432 | 0.369 | 0.407 | (all) | $3.94 |
| `dpv_privacy`, BM25 top 8 | 0.823 | 0.623 | 0.750 | 214 / 276 (78%) | $0.19 |
| `dpv_privacy`, BM25 top 8 + hierarchy | 0.823 | 0.632 | 0.752 | **274 / 276 (99%)** | $0.21 |

³ Among the spans that match a gold span exactly.

- **Retrieving from the raw text doesn't work.** Values such as `USR 661 204` share no
  words with "Identifier". From the transcript alone (`candidate_recall.py`), an
  embedding model finds only 50% of gold concepts in its top 10, and 72% in its top 80.
  Lexical matching finds 23%. The LLM's short description in step 1 bridges that gap.
- **Choosing is solved once the gold concept is a candidate.** Of the 64 remaining
  concept disagreements, 62 are choices among candidates (2 concepts were not
  retrieved), and 57 follow two conventions of the benchmark:
  - For conditions such as "anxiety" the model prefers `MedicalHealth` ("information
    about health, medical conditions or health care") over the gold `Health`
    ("information about health").
  - For payment tokens it picks `PaymentCard`; the benchmark deliberately uses
    `AccountIdentifier`.

  Both are defensible under DPV's own definitions. They are worth raising in the
  dataset's independent review rather than tuning the prompt towards.
- **Span quality improved too.** Strict span F1 is 0.823, the best of all systems,
  because the step-1 prompt states the span rules (value only, postal code separate).
  The remaining span losses are mostly mentions not found (overlap span F1 0.885).

## All three DPV schemas, System 1 + LLM against the LLM deciding (dev)

DPVCallBench's gold concepts sit in `dpv` (all 18), `dpv_pd` (the 15 personal-data ones,
not the 3 roles) and `dpv_privacy` (all 18). Each schema ran on dev through the local
API, two-step, with zero-shot Laya, in two ways:
- `system1+llm`: Laya decides and consults the LLM below 0.6;
- `llm`: the LLM decides among the same candidates.

| Schema | Linker | Strict span F1 | Span + concept F1¹ | Hierarchical F1 | Right concept² | Mean confidence | ECE | Cost |
|---|---|---|---|---|---|---|---|---|
| `dpv_privacy` (264) | **system1+llm** | 0.816 | **0.662** | **0.759** | **0.811** | 0.799 | **0.070** | $0.16 |
| `dpv_privacy` | llm | 0.816 | 0.620 | 0.740 | 0.760 | 0.640 | 0.232 | $0.21 |
| `dpv_pd` (406) | system1+llm | 0.788 | 0.602 | 0.714 | 0.764 | 0.731 | 0.091 | $0.16 |
| `dpv_pd` | llm | 0.788 | 0.622 | 0.731 | 0.790 | 0.537 | 0.267 | $0.20 |
| `dpv` (8,488) | system1+llm | 0.644 | 0.374 | 0.439 | 0.581 | 0.864 | 0.320 | $0.27 |
| `dpv` | llm | 0.649 | 0.501 | 0.583 | 0.772 | 0.482 | 0.357 | $0.34 |
| `dpv`, one step (all IDs in one prompt) | – | 0.432 | 0.369 | 0.407 | – | – | – | $3.94 |

¹ After mapping finer concepts up to the benchmark's. Dropping the spans that only name a kind of
data ("email addresses"; 0–34 per run, found by a heuristic) changes it by at most 0.014.
² On predicted spans that match a gold span exactly (267–303 per run).

- **System 1 + LLM gives calibrated probabilities where Laya knows the pool.** On
  `dpv_privacy` it is also the most accurate. On `dpv_pd`, the LLM picks the right concept
  slightly more often (0.79 against 0.76), but only System 1's confidence is near
  calibrated (ECE 0.09 against 0.27).
- **On the full `dpv`, zero-shot Laya is not good enough.** It is right 58% of the time but
  86% sure on average (0.864), among about 8,500 concepts that are mostly irrelevant to a
  call. The LLM choosing among the same candidates is much better (0.77). Even that
  two-step LLM beats one-step `dpv` (span + concept F1 0.50 against 0.37) at 1/11 of the
  cost.
- **Laya's probability of the LLM's choice is far too low** (ECE 0.23–0.36). Moose now
  reports it with `"calibrated": false`.
- Next: fit the consult threshold per schema (`fit_system1.py`, which needs Laya's
  scores alone and with the LLM's suggestion on these runs' mentions). For `dpv`, the
  default should probably be the LLM until a fine-tuned Laya, or a higher threshold, makes
  System 1 trustworthy there.

## Final results on the test split (run once, settings frozen on dev)

All runs use `openai/gpt-oss-120b`, temperature 0, on the 150 test records (268 gold
mentions). Settings were frozen on dev first:
- span rules shared by all prompts (value only, qualifiers kept, spoken values whole, no
  leading articles);
- `dpv_privacy` guidance naming ticket and reference IDs, plus its address rule;
- consult thresholds as fitted.

Laya variants are scored offline (`system1_eval.py`) on the same step-1 mentions and
candidates as the "LLM decides" run, so they differ only in who chooses.

| System | Strict span F1 | Span + concept F1¹ | Right concept² | Confidence ECE² |
|---|---|---|---|---|
| Direct LLM, closed prompt (21 labels + definitions + rules) | 0.888 | 0.818 | 0.921 | – |
| Moose, the 21 labels as a schema (one step) | 0.824 | 0.780 | – | – |
| **Moose `dpv_privacy`, two-step, LLM decides** | **0.900** | **0.855** | **0.950** | 0.029³ |
| Moose `dpv_privacy`, fine-tuned Laya alone | 0.900 | 0.794 | 0.882 | 0.179 |
| Moose `dpv_privacy`, zero-shot Laya, LLM below 0.6 | 0.900 | 0.790 | 0.878 | 0.108 |
| Moose `dpv_pd`, LLM decides | 0.789 | 0.733 | 0.930 | 0.054³ |
| Moose `dpv_pd`, fine-tuned Laya alone | 0.789 | 0.715 | 0.906 | 0.198 |
| Moose `dpv_pd`, zero-shot Laya, LLM below 0.9 | 0.789 | 0.685 | 0.869 | 0.157 |
| Moose `dpv`, two-step, LLM decides | 0.649 | 0.519 (0.592 values only) | 0.800 | 0.18³ |

¹ After mapping finer DPV concepts up to the benchmark's. ² On predicted spans matching a
gold span. ³ The LLM's stated confidence (reported as not calibrated); on dev it was
0.20–0.36.

- **Moose's two-step `dpv_privacy` with the LLM deciding beats the direct closed prompt**
  (0.855 against 0.818), although the direct prompt knows the benchmark's 21 answers.
- **The fine-tuned Laya's dev lead did not hold.** On dev it was right 96.8% of the time,
  against 76% for the LLM; on test 88.2%, against 95.0%. Its gap is mostly one label,
  `financial_account_identifier` (0 of 14 right; the LLM 14 of 14), which sits in a single
  template group missing from train. Its calibration did not transfer either (ECE 0.078
  on dev, 0.179 on test).
- **Calibration is not stable across template families for any approach.** The
  zero-shot Laya + LLM probabilities stay the closest to calibrated on both splits (0.07
  dev, 0.11 test). The LLM's stated confidence swings from 0.23 (dev) to 0.03 (test),
  because its accuracy does.
- **A benchmark convention costs every system about 16% of test recall.** In "send the
  bill to my mother, Lina", gold annotates only "mother" (`third_party`) and not the name;
  all systems annotate the name instead (44 mentions). Worth raising in the dataset's
  review.
- Cost on test: about $0.81 for the four Moose runs. The direct baseline uses about 760
  prompt tokens per record (OpenRouter did not report its cost).

## Direct LLM baseline and the fine-tuned Laya (dev)

**Direct LLM, closed set** (`direct_llm.py`, the notebook's prompt). One call per record
lists the 21 benchmark labels with their DPV ids and definitions, plus the benchmark's own
rules. It scores strict span F1 0.973, span + concept F1 0.970, and gets the concept right
on 99.7% of matched spans. It knows the answer space and the conventions in advance, so it
is the ceiling for this benchmark.

**Moose with the same 21 labels** (`dpvcallbench_labels`, one step, now with definitions)
finds nearly every mention (overlap span F1 0.999) and almost always picks the right
label. But strict span F1 is only 0.778: its one-step prompt lacks the value-only span
rules that the two-step mention prompt and the direct prompt have (no cue words, no
"years old", postal code separate). That is the gap to close there.

**Fine-tuned Laya v1** (Colab T4, notebook v1, trained on 736 train mentions with the old
candidate pool), scored on the current dev mentions:

| Right concept (matched spans) | LLM alone | Zero-shot Laya | Fine-tuned Laya |
|---|---|---|---|
| `dpv_privacy` | 0.760 | 0.756 | **0.953** |
| `dpv_pd` | 0.790 | 0.768 | **0.955** |
| `dpv` | **0.772** | 0.553 | 0.679 |

- It no longer needs the LLM for step 2: consulting the LLM adds nothing.
- Its top-1 probability is under-confident and barely ranks right above wrong (ECE
  0.22–0.26, AUROC 0.33–0.55), and no single temperature fixes that. Training spreads the
  target over every correct candidate (a concept and its finer children), so the model
  splits probability among close relatives.
- Using as confidence the probability of the chosen concept plus its direct DPV parents
  and children gives, on `dpv_privacy`, ECE 0.066, Brier 0.051 and AUROC 0.76; on
  `dpv_pd`, ECE 0.118.
- Its remaining errors are confident ones, mostly payment tokens labelled `PaymentCard`
  (the benchmark uses `AccountIdentifier`).
- Caveat: it learned DPVCallBench's conventions from its train split. Dev's templates are
  disjoint from train's, but all of it is synthetic call transcripts.

## Per-schema thresholds and defaults (dev, current code)

`fit_system1.py --schema … --run moose_<schema>_llm_…_{split} --splits dev` scores each
mention of the LLM-decides run with Laya twice, alone and with the LLM's choice. That
gives what `system1+llm` decides at any threshold.

| Schema | LLM alone | Laya alone | System 1 + LLM | LLM consulted | ECE | Default |
|---|---|---|---|---|---|---|
| `dpv_privacy` | 0.760 | 0.756 | 0.811 at 0.6 (0.844 at 0.65) | 21% | 0.07 | system1+llm, validated |
| `dpv_pd` | 0.790 | 0.768 | 0.805 at 0.9 | 87% | 0.13 | system1+llm, not validated |
| `dpv` | 0.772 | 0.553 | 0.636 at 0.8 | 31% | 0.27 | llm, two-step |

Accuracy is on predicted spans that match a gold span. `dpv_privacy` keeps 0.6, the
threshold chosen on train and dev together (1,093 mentions), rather than the dev-only
optimum. On `dpv_pd`, System 1 + LLM is ahead of the LLM only when it consults the LLM
for most mentions. On `dpv`, zero-shot Laya is not usable: its errors are confident, so
consulting the LLM does not catch them.

## System 1 decides, the LLM assists (current design)

Moose now makes every type decision with Laya (`moose.linkers`). Laya scores the
retrieved candidates. Where its top probability is below 0.6, the LLM's choice is added
to Laya's input as "An LLM suggested: …", and Laya decides again. The reported
probability is always Laya's. `fit_system1.py` reproduces this. Scores are on mentions
whose span matches gold, so they estimate P(concept right | span right).

Two changes made Laya competitive on its own:
- **Candidates must be categories.** The privacy pool had included 20 DPV properties and
  documentation pages (`hasDataProcessor`, `primer`, …). Laya kept choosing the property
  `hasDataProcessor` over the category `DataProcessor`. Once they were removed, zero-shot
  Laya's train accuracy rose from 0.55 to 0.80.
- **The LLM's view enters as evidence, not as a separate decision.**

| Train + dev pooled (1,093 mentions) | Accuracy | Mean confidence | ECE | Brier | AUROC | LLM consulted |
|---|---|---|---|---|---|---|
| LLM alone, its stated confidence | 0.771 | 0.97 | 0.20 | 0.19 | 0.69 | 100% |
| Laya alone (`system1`) | 0.790 | 0.781 | 0.082 | 0.115 | **0.898** | 0% |
| **Laya, consulting the LLM below 0.6 (`system1+llm`)** | **0.801** | 0.801 | **0.030** | 0.125 | 0.808 | 16% |
| Laya, always given the LLM's choice | 0.783 | 0.784 | 0.075 | 0.155 | 0.701 | 100% |

| Same, per split | train accuracy / ECE | dev accuracy / ECE |
|---|---|---|
| LLM alone | 0.772 / – | 0.768 / – |
| Laya alone | 0.799 / 0.091 | 0.764 / 0.104 |
| Laya, consulting below 0.6 | 0.796 / 0.049 | 0.819 / 0.063 |

- **Consulting below 0.6 is the best pooled setting.** On both splits it is as accurate as
  the better of Laya and the LLM, or more so, and its probabilities are the best
  calibrated. The threshold was chosen on train and dev together; test will confirm it.
- **The suggestion makes Laya's probability rank less well (AUROC 0.90 → 0.81).** Laya
  leans towards the LLM's answer, right or wrong. For ranking items by risk, `system1`
  alone gives the sharper signal.
- **Laya's remaining weakness is names.** Zero-shot, it often prefers `Identifier` to
  `Name`. This is the first thing fine-tuning should fix (Colab notebook in `colab/`).
- **Tables use the same mechanism,** with column descriptions instead of mentions. There
  is no labelled table data yet, so table probabilities are reported as not calibrated.

## Laya ranking the two-step candidates

`laya_rank.py` takes the candidates the two-step run retrieved for each mention (9.1 on
average, at most 16). Laya gives each a probability, using the transcript, the mention
and its step-1 description, with each candidate shown as its label plus the start of
its definition. Scored on the 276 mentions that match a gold span exactly, where the
gold concept is among the candidates for 99.3%.

| Candidates shown as | Gold ranked 1st | In top 3 | MRR | Raw ECE |
|---|---|---|---|---|
| label only | 0.572 | 0.957 | 0.769 | 0.372 |
| label + 60 characters of definition | 0.754 | 0.949 | 0.860 | **0.051** |
| label + 140 characters of definition | 0.754 | 0.949 | 0.860 | 0.102 |
| label + 60 characters, candidates shuffled | 0.460 | 0.909 | 0.688 | 0.249 |
| *LLM's own choice (step 2), for reference* | *0.768* | | | |
| *first retrieved candidate (BM25 order)* | *0.529* | | | |

Laya decides the mentions it is most confident about; the LLM's choice is kept for
the rest (140 characters of definition):

| Share decided by Laya | 0% (LLM only) | 25% | 50% | 75% | 100% |
|---|---|---|---|---|---|
| Accuracy | 0.768 | 0.772 | 0.837 | **0.848** | 0.754 |
| Laya's own accuracy on that share | | 0.957 | 0.964 | 0.884 | 0.754 |

- **Laya's ranking is close to the LLM's choice once definitions are shown.** The
  definitions matter: with labels only it drops to 0.572. The gold concept is in Laya's
  top 3 for 95% of mentions, so Laya could also shortlist for the LLM.
- **It depends on the retrieval order.** With the candidates shuffled, top-1 falls to
  0.460. Order is available in practice, but Laya is using it as a strong cue, not
  judging the definitions alone.
- **Laya and the LLM differ on two concept pairs:**
  - Laya labels 41 birth dates as `Age`; the LLM gets them right.
  - Laya picks the benchmark's `Health` over `MedicalHealth` on 40 health mentions; the
    LLM differs.

  So the combined 0.848 owes much to Laya matching the benchmark's `Health` convention
  (see the two-step section). The share was also picked on dev, so confirm it on
  held-out data before trusting it.
- **Its confidence is useful for routing.** Its most confident half is 96% correct, and
  raw calibration is already fair (ECE 0.05–0.10).
- **Temperature fitted on half of dev doesn't carry over.** The two record folds give
  temperatures of 6.4 and 0.5, because the dev split's template groups put whole
  concepts in one fold. Fit the temperature on train instead. That needs a two-step run
  on train, which would cost about $1.

## Findings

1. **The open-DPV prompt is the problem, not the LLM.** Moose's `dpv` schema puts all
   9,481 concepts in every prompt: 98k tokens, about 10 s and $0.026 per short
   transcript. On the same records, the 21-label prompt finds almost every mention
   (overlap F1 0.98) and picks the right concept for all but 2 exactly matched spans.
   Narrowing the candidates before the LLM call should recover most of that accuracy at
   about 1/150 of the cost. This is the two-step design: retrieve candidates, then choose.
2. **The LLM's remaining errors are span-boundary conventions.** About 100 of its 344
   predictions overlap the gold span without matching it exactly:
   - It includes cue words (`Customer ID USR 661 204` instead of `USR 661 204`) and
     "years old".
   - It keeps the postal code inside the address.

   The gold data's conventions are written in the annotation guidelines. Adding them to
   the schema's instructions is the cheapest improvement left, and the dev split may be
   used for that.
3. **GLiNER is a strong local span finder.** Zero-shot it reaches triple F1 0.755 at
   0.06 s per record on a laptop. It is better than the LLM on boundaries for IDs and
   ages, but weaker on health issues, on postal codes (folded into addresses) and on
   rare labels.
4. **Zero-shot Laya is not yet good enough to choose DPV concepts.**
   - With all 21 options and the gold span it is right 74% of the time, against about
     99% for the LLM.
   - Its main error is labelling health issues as `age` (44 of 344 mentions).
   - Replacing GLiNER's own labels with Laya's choice lowers F1 from 0.755 to 0.575.

   This matches Laya's model card, which reports near-chance zero-shot accuracy on typed
   decisions and recommends fine-tuning. Temperature scaling does work as described: it
   cuts calibration error by 30–60%.
5. **Retrieval over the whole DPV vocabulary is the weak link for the open setting.**
   Only 47–49% of gold concepts are in the top 5–10 from all 8,648 concepts. Most of
   DPV (purposes, legal bases, technical measures, locations) is irrelevant to a
   mention. Restricting the pool to personal-data categories and entity roles (about
   450 concepts, as in Moose's `dpv_pd` schema) should raise this sharply.

## Suggested next steps

1. ~~**Two-step open-DPV mode in Moose.**~~ Done: see the two-step section above.
2. **Fine-tune Laya** on the train split (997 mentions), with candidates retrieved as at
   inference time and a "none" option for wrong spans. Then refit its temperature and
   compare on dev. It is the only route to a fast, calibrated local linker.
3. **Confidence routing (hybrid):** Laya ranks the two-step candidates and decides the
   confident mentions, and only the rest go to the LLM's step 2. The first test (above)
   reaches 0.848 with Laya deciding 75%. Next: fit the temperature and routing threshold
   on train, fine-tune Laya on the train mentions (birth date vs age), and confirm on
   test. Report F1 against the share of
   mentions sent to the LLM.
4. Freeze choices on dev, then report once on test.
