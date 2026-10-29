# Personal-data detection on TAB

Moose's `dpv_privacy` on [TAB](https://github.com/NorskRegnesentral/text-anonymization-benchmark),
the Text Anonymization Benchmark (MIT): European Court of Human Rights cases, with every
mention that identifies the person to protect marked DIRECT (a name, a case number) or QUASI
(a date, a place, an employer, an amount: identifying together with other facts).

Moose annotates each document as it ships; the spans it gives a personal-data type
(`dpv-pd:*`) are the spans it would mask. Roles such as `dpv:Applicant` or
`dpv:DataController` are not masked; the `all` variant masks every annotated span. Scoring is
TAB's own `evaluation.py`: entity recall on DIRECT and QUASI identifiers (an entity counts only
if all its mentions are masked) and token precision.

## How to run

```bash
# Data under data/: echr_dev.json, echr_test.json and evaluation.py from the TAB repository
# The scorer needs spacy and en_core_web_md in research/.venv-laya
PYTHONPATH=../../src ../.venv-laya/bin/python tab.py --split dev --docs 15
# moose.anonymise (docs/design/anonymisation.md), protecting the applicant
PYTHONPATH=../../src ../.venv-laya/bin/python tab.py --mode anonymise --split dev --docs 15
# The confidence of its decisions: Laya's scores on a train and a test run, then the fit
# (echr_train.json under data/ too; 90 train cases cost $1.59)
PYTHONPATH=../../src ../.venv-laya/bin/python tab.py --mode anonymise --split train --docs 90 --max-cost 2
PYTHONPATH=../../src ../.venv-laya/bin/python laya_mask.py runs/anonymise_train_openai_gpt-oss-120b --split train
PYTHONPATH=../../src ../.venv-laya/bin/python laya_mask.py runs/anonymise_test_openai_gpt-oss-120b --split test
PYTHONPATH=../../src ../.venv-laya/bin/python fit_mask_calibration.py --write
```

- Each run writes `runs/<split>_<model>[_<tag>]/report.json` and caches Moose's output per
  document (`raw.jsonl`), so a rerun only annotates the documents still missing.
- `--max-cost` (USD, $0.50 by default) stops starting new documents once spent. A court case
  costs about $0.008 with `gpt-oss-120b` (6-8 LLM calls).
- Keep `--concurrency` low (2-3): each document already runs its chunks in parallel.

## Results

Dev split, 15 random documents (seed 0), `gpt-oss-120b`, $0.11.

| Masked | DIRECT recall | QUASI recall | Token precision | Mention precision |
|---|---|---|---|---|
| personal-data spans (`dpv-pd:*`) | 0.90 | 0.36 | 0.81 | 0.80 |
| every annotated span | 0.91 | 0.38 | 0.69 | 0.72 |

Gold mentions of each TAB type (first annotator, DIRECT and QUASI) that Moose found, and the
types it gave them:

| TAB type | Found | Moose's types |
|---|---|---|
| PERSON | 94/100 | Name 55, Identifier 17, School 8 |
| CODE | 71/71 | Identifier 52, DeviceApplications 13 |
| ORG | 8/65 | DataProcessor 4, DataController 2 (roles, not masked) |
| DATETIME | 38/297 | BirthDate 25 |
| LOC | 7/50 | BirthPlace 2, Locality 2, City 2 |
| MISC | 12/47 | PhysicalTrait 7 |
| DEM | 10/45 | Nationality 6 |
| QUANTITY | 7/35 | Financial 4 |

Names and case numbers, what `dpv_privacy` is for, are found. Most QUASI identifiers are
outside its scope: TAB masks every event date, place, amount, organisation and injury that
could single out the applicant, while `dpv_privacy` looks for personal data and privacy roles.
Low QUASI recall is that difference, not an error; TAB measures anonymisation. Some types
are odd (case numbers as `DeviceApplications`, names as `School`) but still personal data.

### Test split

30 random test documents (seed 0), run once with all three fixes below, and once more after
the window fix of 2 (800 → 400 characters), developed on dev. None failed; $0.24 per run.

| Personal-data spans masked | DIRECT recall | QUASI recall | Token precision | Names typed `dpv:Adult` |
|---|---|---|---|---|
| 800-character window | 0.76 | 0.32 | 0.80 | 60 |
| 400-character window | 0.85 | 0.35 | 0.79 | 5 |

With every annotated span masked: DIRECT 0.87, QUASI 0.36, token precision 0.71. Names are
found (PERSON 228/243, typed Name 199 times) and so are case numbers (CODE 77/82).

## What TAB found in Moose

All three are fixed in the library and covered by tests. On the first 3 documents (1, 2) and
on the 13 documents both later runs completed (3), for personal-data spans:

| Change | DIRECT recall | QUASI recall | Token precision | Cost per document |
|---|---|---|---|---|
| before (3 documents) | 0 | 0 | | $0.003 |
| 1. mentions found per chunk | 0.19 | 0.04 | 0.73 | $0.031 |
| 2. Laya sees the text around a mention | 0.56 | 0.10 | 0.86 | $0.031 |
| (13 documents) with 1 and 2 | 0.89 | 0.19 | 0.85 | $0.026 |
| 3. mentions given as text, not offsets | 0.93 | 0.36 | 0.79 | $0.007 |

1. **Long texts got no annotations.** The mention step asked for every mention of the whole
   text in one call; on a court case (7,000-8,000 characters) `gpt-oss-120b` returned none,
   and it already dropped mentions at 2,500. Mentions are now found per chunk of at most 1,500
   characters, cut at paragraph breaks (`moose.ner.MENTION_CHUNK_CHARS`).
2. **Laya never saw mentions in long texts.** Laya reads 512 tokens with the text first, so
   in a long text the mention was cut off and every mention of a document got the same
   scores: all names became `dpv:Adult`. Laya now sees 400 characters around a mention in a
   longer text (`moose.linkers.LAYA_CONTEXT_CHARS`). 800 fit in its tokens too, but Laya was
   trained and calibrated on short calls, and with that much context it still typed many
   names `dpv:Adult` or Identifier; rescoring the 87 person names of the dev run (no LLM
   calls) gave Name 16 / Adult 21 with 800 characters, 66 / 3 with 400 (200 and 300 were in
   between). DPVCallBench's texts (at most 146 characters) are given to Laya as before, so
   its calibration is unchanged.
3. **The LLM spent its reasoning counting characters.** The mention step asked for character
   offsets, which Moose then corrected by searching for the mention's text anyway.
   `gpt-oss-120b` spent 6,000-15,000 reasoning tokens per chunk on them, and on some chunks
   reasoned to its 40,000-token limit and answered nothing (2 of 15 documents failed). It
   now lists each mention's text in order of appearance, a repeated value each time, and
   Moose places each at the next occurrence. Long texts cost about a quarter as much, none
   failed, and repeated names and case numbers are now all found. On 40 DPVCallBench dev
   records the mention step finds the same spans (strict recall 0.972 both, precision 0.986
   → 1.000) at half the cost.
