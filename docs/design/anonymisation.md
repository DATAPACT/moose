# Design: an anonymisation mode

Status: proposal, not implemented. Nothing below has been run yet.

## Why

`dpv_privacy` answers "which personal data and privacy roles does this text mention?". On
TAB (`research/tab_eval`), which asks "what must be masked so that the person cannot be
re-identified?", it finds names and case numbers (DIRECT recall 0.85 on test) but few
quasi identifiers (0.35): event dates, places, organisations, amounts and descriptions that
identify someone only in combination. That is by design, so anonymisation needs its own mode
rather than a wider `dpv_privacy`.

## What TAB says about the task

From the first annotator of TAB's dev split (127 court cases):

| Type | Mentions | DIRECT | QUASI | NO_MASK | Not masked, for example |
|---|---|---|---|---|---|
| DATETIME | 2615 | 0% | 94% | 6% | "several years later", "five years" |
| ORG | 1917 | 0% | 27% | 73% | "Turkish Government", "Istanbul State Security Court" |
| PERSON | 1222 | 20% | 65% | 15% | judges, other parties |
| DEM | 568 | 0% | 52% | 48% | "Turkish", "naval police" |
| LOC | 492 | 0% | 61% | 39% | "Istanbul" |
| CODE | 463 | 31% | 63% | 6% | the application number, law articles |
| MISC | 259 | 1% | 59% | 40% | names of benefits |
| QUANTITY | 232 | 0% | 91% | 9% | "six charges" |

Two consequences:
- **The decision depends on context, not on type.** Courts and governments stay, the
  applicant's employer goes; a country stays, a village goes. A type list cannot say which,
  so the mode needs a masking decision per entity.
- **The decision is per entity, made over the whole document.** TAB's guidelines decide on
  the first mention and apply it to every mention of the same entity, and a quasi identifier
  is one that re-identifies together with the others in the text.

## Proposal

A separate mode, using the two-step pipeline that exists:

1. **Find mentions** (as now: per chunk of 1,500 characters, text in order of appearance),
   with guidance for anonymisation: every person, code, place, organisation, demographic
   attribute, date or time, quantity and other specific detail. The category is one of TAB's
   eight, not free text, so the next step can group by it.
2. **Decide what to mask, once per document.** One LLM call gets the distinct entities (the
   same text counted once, with the number of mentions and one line of context) and the
   person or people to protect, and answers DIRECT, QUASI or NO_MASK for each, following
   TAB's definitions: specific, publicly knowable attributes are QUASI; very general ones
   (a country, "several years") and public bodies acting as such are NO_MASK. The answer
   applies to every mention of that text. A dev case has about 60 annotated mentions (7,768
   in 127 cases), fewer distinct ones, so this is one call of a few thousand tokens.
3. **Probability from Laya**, as everywhere in Moose: a yes/no verdict "Must this be masked
   to protect the applicant?" on the 400-character window. Until it is calibrated on TAB's
   train split (about 1,000 cases, never used for testing), results carry
   `"calibrated": false`.
4. **Optionally, DPV types** for masked spans that are personal data, from the existing
   `dpv_privacy` linker, and TAB's confidential categories (health, beliefs, politics,
   ethnicity, sex life) mapped to DPV's special categories. Off by default: it adds calls.
5. **Output**: the spans to mask, each with `identifier_type`, `category`, `confidence` and
   the entity it belongs to, and the masked text, with one placeholder per entity
   (`[PERSON_1]`, `[DATE_3]`) so the text stays readable.

### API

```python
result = moose.anonymise(text, protect="the applicant")   # default: every private individual
result["masked_text"]   # "... [PERSON_1] was born in [DATE_1] in [LOC_2] ..."
result["entities"]      # [{"text", "start", "end", "category", "identifier_type", "entity", "confidence", ...}]
```

The REST API would get the same as a job type. `protect` matters: TAB protects one person
per case, a customer-service transcript every private individual in it.

## Evaluation and cost

On TAB with its own scorer (`research/tab_eval/tab.py`, a new `--mode anonymise`):
DIRECT and QUASI recall, token precision, compared with `dpv_privacy` on the same documents.
Expected cost with `gpt-oss-120b`: about $0.01 per court case (mention calls as now, plus one
decision call), so $0.15 for 15 dev cases and $0.30 for the 30 test cases. Calibrating Laya
on the train split needs no LLM calls beyond annotating it once (about $10 for all 1,000
cases, or $2 for 200).

Steps, each checked before the next:
1. Mention guidance and fixed categories; measure mention recall per TAB type on 15 dev
   cases (~$0.10). Target: most dates, places and amounts found.
2. The masking decision; DIRECT/QUASI recall and token precision on the same cases (~$0.15).
3. `masked_text`, the API and the docs; unit tests.
4. Calibration of Laya's verdict on 200 train cases (~$2), then the test split once (~$0.30).

## Decisions needed

1. **A new mode, `moose.anonymise()`, rather than a new schema.** Its output (identifier
   types, entities, masked text) is not a type annotation, so it does not fit `annotate_text`.
   Recommended.
2. **Default `protect`:** every private individual (recommended), or require it.
3. **Ship before calibration** with `"calibrated": false` (recommended, as tables do), or
   only once calibrated.
4. **Scope:** English first; TAB has no other language.
