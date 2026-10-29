# Design: an anonymisation mode

Status: implemented: `Annotator.anonymise`, `POST /anonymise`, documented in
`docs/library.md` and the README. With Laya installed, each decision has a calibrated
`confidence` (step 4); without it, `"confidence": null`.

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
   Done on 90 train cases ($1.59; train cases are longer than test ones), checked on the
   cached test run (no cost): see Step 4 results.

## Step 1 results

TAB dev, the same 15 cases as `dpv_privacy`, every mention found masked (an upper bound on
recall; deciding what not to mask is step 2). $0.15, none failed.

| | DIRECT recall | QUASI recall | Token precision |
|---|---|---|---|
| `dpv_privacy` (personal-data spans) | 0.91 | 0.37 | 0.81 |
| anonymisation mentions, all masked | 0.93 | 0.95 | 0.52 |

DIRECT and QUASI mentions of each TAB type found:

| Type | Anonymisation | `dpv_privacy` | Category given |
|---|---|---|---|
| DATETIME | 296/297 | 38/297 | DATETIME 286, QUANTITY 10 |
| PERSON | 97/100 | 95/100 | PERSON 97 |
| CODE | 71/71 | 70/71 | CODE 71 |
| ORG | 61/65 | 8/65 | ORG 47, LOC 9 |
| LOC | 49/50 | 7/50 | LOC 48 |
| QUANTITY | 29/35 | 7/35 | QUANTITY 29 |
| DEM | 31/45 | 10/45 | DEM 26 |
| MISC | 21/47 | 12/47 | MISC 9, QUANTITY 6, CODE 4 |

Of the mentions TAB does not mask, 225 of 286 are found too, which is what costs precision:
step 2 has to leave them unmasked. The weak spots are MISC (injuries and other descriptions)
and DEM (job titles, ranks).

## Step 2 results

The same 15 dev cases, protecting "the applicant", masking what the decision says to mask.
The mentions are step 1's; only the decision was rerun for the second rules ($0.07).

| | DIRECT recall | QUASI recall | Token precision | Mention precision |
|---|---|---|---|---|
| `dpv_privacy` (personal-data spans) | 0.91 | 0.37 | 0.81 | 0.80 |
| step 1, every mention masked | 0.93 | 0.95 | 0.52 | 0.52 |
| step 2, first rules | 0.93 | 0.60 | 0.90 | 0.87 |
| step 2, rules as TAB masks | 0.93 | 0.80 | 0.77 | 0.74 |

The first rules left public bodies, officials and judges unmasked. TAB's annotators masked
them: officials' and judges' names, local courts and councils (they give the place), other
applications' numbers, cities, nationalities and injuries; only the state, its national
bodies, international institutions and legal references stayed. The rules now say so. Cost:
about $0.014 per case, $0.010 to find mentions and $0.004 to decide.

Not tuned further on these 15 cases.

### Test split

The 30 random test cases `dpv_privacy` was tested on (seed 0), run once with the rules
above, protecting the applicant; none failed, $0.43.

| | DIRECT recall | QUASI recall | Token precision | Mention precision |
|---|---|---|---|---|
| `dpv_privacy` (personal-data spans) | 0.85 | 0.35 | 0.79 | 0.80 |
| `moose.anonymise`, masked | 0.98 | 0.75 | 0.82 | 0.78 |
| `moose.anonymise`, every mention found masked | 0.98 | 0.94 | 0.48 | 0.47 |
| `moose.anonymise`, masked, every name masked (below) | 0.98 | 0.91 | 0.74 | 0.70 |

Mentions found per type: DATETIME 525/526, PERSON 242/243, CODE 82/82, QUANTITY 34/35, LOC
77/80, ORG 100/105, DEM 64/81, MISC 21/37. The decision, not the search, is where QUASI
recall is lost; MISC (injuries and other descriptions) is the weak spot of the search.

One case failed at first: on an ordinary paragraph the LLM reasoned until its 40,000-token
limit and answered nothing. The mention step now splits such a chunk in two and tries each
half (`moose.ner.MIN_SPLIT_CHARS`), for every two-step schema, not only here.

### Every name masked

The decision left the names of judges, government agents and lawyers unmasked, taking them
for the international courts and national bodies the rules leave: of the names left, TAB
masks 259 of 339 in 90 train cases and 64 of 67 in the test cases above. The rules now say
that a person's name is never NO_MASK, whatever their role. Only the decision was rerun, on
the same mentions (`tab.py --decide-from`): $0.07 on dev, $0.14 on test.

| | DIRECT recall | QUASI recall | Token precision | Mention precision | Token F1 |
|---|---|---|---|---|---|
| dev 15, before | 0.93 | 0.80 | 0.77 | 0.74 | |
| dev 15, every name masked | 0.93 | 0.91 | 0.76 | 0.73 | 0.82 |
| test 30, before | 0.98 | 0.75 | 0.82 | 0.78 | 0.805 |
| test 30, every name masked | 0.98 | 0.91 | 0.74 | 0.70 | 0.819 |

More of what TAB masks is masked, at some cost in precision: masking a name TAB leaves (a
judge whose name the annotator kept) costs tokens. For anonymisation, missing an identifier
is the worse error.

## Step 4 results

Laya is asked, for each entity's first mention (400 characters around it): "Must the mention
be masked to protect the applicant?" (`moose.anonymise.laya_mask_inputs`). On TAB, with the
LLM's decisions of the runs above (`research/tab_eval/laya_mask.py`):

| | Train, 90 cases | Test, 30 cases |
|---|---|---|
| entities | 6,782 | 1,839 |
| the LLM's decision right | 0.80 | 0.82 |
| Laya's P(mask) against TAB, AUC | 0.65 | 0.70 |
| Laya's probability that the decision is right, mean | 0.53 | 0.56 |
| the same, ECE | 0.27 | 0.26 |

Laya alone is a weak judge of masking: it was trained on short calls and personal-data
types, not on re-identification. Telling it the LLM's decision changed little (test AUC
0.71). So its answer is combined with what is known about the decision: a logistic
regression, fitted on train, on the entity's category and decision (masked or left) and the
log-odds of Laya's probability (`research/tab_eval/fit_mask_calibration.py`, weights in
`src/moose/data/anonymise/laya_calibration.json`). On test:

| Confidence | AUC (right vs wrong) | ECE | Mean |
|---|---|---|---|
| Laya alone | 0.685 | 0.263 | 0.56 |
| category and decision | 0.790 | 0.041 | 0.80 |
| category, decision and Laya (shipped) | 0.795 | 0.039 | 0.80 |

The confidence is calibrated (ECE 0.039 ≤ 0.10), so results carry `"calibrated": true`.
With every name masked (above), the same weights on the new test decisions: 0.83 right,
mean confidence 0.81, ECE 0.045, AUC 0.87; they were not refitted. Most
of what it knows comes from the category and decision, not from Laya (weight 0.16 on its
log-odds). Mean confidence and share right on test, per category and decision:
masked dates 0.93 (95% right) and names 0.92 (98%); masked organisations 0.49 (48%), job
titles and other demographic details 0.39 (57%) and descriptions 0.37 (55%, 11 entities);
names left unmasked 0.23 (4%: they should have been masked). Those low ones are what to
review first. Within a group it is coarser: organisations left unmasked get 0.80 and are 95%
right, places left 0.66 and 84% right. The calibration holds for court cases protecting one person; other texts and
`protect` values were not checked.

## Decisions

All four taken as recommended.

1. **A new mode, `moose.anonymise()`, rather than a new schema.** Its output (identifier
   types, entities, masked text) is not a type annotation, so it does not fit `annotate_text`.
   Recommended.
2. **Default `protect`:** every private individual (recommended), or require it.
3. **Ship before calibration** with `"calibrated": false` (recommended, as tables do), or
   only once calibrated.
4. **Scope:** English first; TAB has no other language.
