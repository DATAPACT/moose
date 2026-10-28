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
```

- Each run writes `runs/<split>_<model>[_<tag>]/report.json` and caches Moose's output per
  document (`raw.jsonl`), so a rerun only annotates the documents still missing.
- `--max-cost` (USD, $0.50 by default) stops starting new documents once spent. A court case
  costs about $0.03 with `gpt-oss-120b` (6-8 LLM calls).
- Keep `--concurrency` low (2-3): each document already runs its chunks in parallel.

## Results

Dev split, 13 of 15 random documents (seed 0; 2 failed, see below), `gpt-oss-120b`, $0.33.

| Masked | DIRECT recall | QUASI recall | Token precision | Mention precision |
|---|---|---|---|---|
| personal-data spans (`dpv-pd:*`) | 0.89 | 0.19 | 0.85 | 0.87 |
| every annotated span | 0.91 | 0.22 | 0.62 | 0.66 |

Gold mentions of each TAB type (first annotator, DIRECT and QUASI) that Moose found, and the
types it gave them:

| TAB type | Found | Moose's types |
|---|---|---|
| PERSON | 65/65 | Name 28, Identifier 27, `dpv:Adult` 7 |
| CODE | 18/69 | Identifier 15 |
| ORG | 22/64 | DataController 10, DataProcessor 5 (roles, not masked) |
| DATETIME | 26/255 | BirthDate 17 |
| MISC | 4/47 | |
| LOC | 6/42 | Locality 5 |
| DEM | 5/39 | Nationality 4 |
| QUANTITY | 3/34 | |

Names and case numbers, what `dpv_privacy` is for, are found. Most QUASI identifiers are
outside its scope: TAB masks every event date, place, amount, organisation and injury that
could single out the applicant, while `dpv_privacy` looks for personal data and privacy roles.
Low QUASI recall is that difference, not an error; TAB measures anonymisation.

## What TAB found in Moose

Both are fixed in the library and covered by tests:

1. **Long texts got no annotations.** The mention step asked for every mention of the whole
   text in one call; on a court case (7,000-8,000 characters) `gpt-oss-120b` returned none,
   and it already dropped mentions at 2,500. Mentions are now found per chunk of at most 1,500
   characters, cut at paragraph breaks (`moose.ner.MENTION_CHUNK_CHARS`).
2. **Laya never saw mentions in long texts.** Laya reads 512 tokens with the text first, so
   in a long text the mention was cut off and every mention of a document got the same
   scores: all names became `dpv:Adult`. Laya now sees 800 characters around a mention in a
   longer text (`moose.linkers.LAYA_CONTEXT_CHARS`). On the first 3 documents, DIRECT recall
   of personal-data spans went 0.19 → 0.56, token precision 0.73 → 0.86.

Texts under 800 characters, such as DPVCallBench's calls (at most 121), are annotated as
before, so its calibration still holds.

## Open problem

`gpt-oss-120b` spends 6,000-15,000 reasoning tokens per chunk, mostly counting character
offsets that Moose then repairs by searching for the mention's text anyway. On some chunks it
reasons until its 40,000-token limit and answers nothing; Moose does not retry an empty
answer, so the document fails (2 of 15 here). Asking for the mention's text without offsets
would likely cut the cost of long texts and remove these failures.
