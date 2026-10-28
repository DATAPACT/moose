# schema.org table annotation on SOTAB

Moose's `schemaorg_cta_v1` (a class per column) and `schemaorg_cpa_v1` (the property relating
the main column to each other column) on [SOTAB v2](https://webdatacommons.org/structureddata/sotab/v2/),
the WDC Schema.org Table Annotation Benchmark (CC BY 4.0). Both schemas run in one step: the
LLM chooses, and the confidence is Laya's yes/no probability that the choice is right.

## How to run

```bash
# Data: the schema.org CTA/CPA validation and test zips, unzipped under data/ (about 160 MB)
cd data && for f in CTA_validation CTA_test CPA_validation CPA_test; do
  curl -LO https://data.dws.informatik.uni-mannheim.de/structureddata/sotab/v2/${f}_schemaorg.zip && unzip -q ${f}_schemaorg.zip -d ${f}_schemaorg
done && cd ..

# Laya environment: research/.venv-laya with moose[laya]; src on the path
PYTHONPATH=../../src ../.venv-laya/bin/python sotab.py --task cta --split validation --tables 50
PYTHONPATH=../../src ../.venv-laya/bin/python sotab.py --task cpa --split validation --tables 50
```

- Each run writes `runs/<task>_<split>_<model>[_<tag>]/report.json` and caches Moose's output per
  table (`raw.jsonl`), so a rerun doesn't pay again.
- `--max-cost` (USD, $0.50 by default) stops starting new tables once spent: about $0.002 per
  CTA table and $0.008 per CPA table with `gpt-oss-120b`.
- `--tables-from <raw.jsonl>` annotates the same tables as an earlier run.
- Validation is the development split; test is for checking frozen changes.

## Label mapping

SOTAB's labels and Moose's only partly overlap; columns without a Moose label are left out
and reported as coverage.
- **CTA:** a SOTAB class is the same Moose class. `Book/name` (the names of books) is
  `schema:Book`, since Moose types a column by the entities it holds; `MusicArtistAT` accepts
  `MusicGroup` or `Person`. Descriptions and literal properties (`telephone`, `price`, ...) have
  no Moose class.
- **CPA:** a SOTAB property is the same schema.org property, when `curated_v1` has it.

## Results

LLM `openai/gpt-oss-120b`, 5 rows per table, cell values cut at 150 characters. Accuracy is
the share of scored columns whose label is exactly right; ECE and AUROC are for Laya's
probability against that.

| Run | Columns | Accuracy | Laya ECE | Laya AUROC | Cost |
|---|---|---|---|---|---|
| CTA validation, before (150 tables) | 381 | 0.528 | 0.26 | 0.63 | $0.31 |
| CTA validation, after (same tables) | 400 | 0.578 | 0.22 | 0.65 | $0.31 |
| CTA test, after (50 tables) | 99 | 0.677 | 0.19 | 0.54 | $0.10 |
| CPA validation, before (150 tables) | 516 | 0.421 | 0.44 | 0.51 | $1.00 |
| CPA validation, after (same tables) | 710 | 0.448 | 0.43 | 0.49 | $1.13 |
| CPA test, after (50 tables) | 176 | 0.375 | 0.49 | 0.49 | $0.32 |
| CPA validation, table shortlists (50 of the same tables) | 204 | **0.627** | 0.26 | 0.46 | $0.20 |
| CPA test, table shortlists (the same 50 test tables) | 176 | **0.602** | 0.25 | 0.47 | $0.19 |

On the columns the old label sets could already score, validation went from 0.528 to 0.575
(CTA) and from 0.421 to 0.436 (CPA). The changes:
- CTA: a column of names of things gets those things' class, not `schema:Text`
  (`Event/name` 4 → 12 of 13).
- CPA: properties of value classes the subject can have (`PostalAddress`, `GeoCoordinates`,
  `Offer`, ...) pass the subject-class filter (`addressCountry` 0 → 8 of 13, `postalCode` 1 → 7
  of 16).
- `curated_v1` keeps the properties of `schema:Thing` and those value classes (`name`, `url`,
  `description`, `price`, `streetAddress`, `ratingValue`, ...): CPA coverage on validation went
  from 70% to 96%.

Then CPA's candidate selection was redesigned (last row). It used to shortlist each target
column alone, from the subject and that column only, and columns of the same coarse type
shared the first one's shortlist; it often dropped the right property, and `moose:OTHER`
was the most common answer. Now one call shortlists every target column, seeing the whole
table and the subject's class, and one call decides all of them, each with its shortlist. On
the same 50 tables: accuracy 0.451 → 0.627, `moose:OTHER` answers 58 → 4, cost $0.37 → $0.20
(3 LLM calls per table). On the same 50 test tables it holds: 0.375 → 0.602.

Laya's probabilities are not calibrated on tables (ECE 0.19–0.49, AUROC about 0.5 for CPA),
so table results keep `"calibrated": false`.

### Can they be calibrated?

Not usefully. Fitted on the validation runs of the current pipeline (CTA 400 columns, CPA 204)
and checked on the test runs (CTA 99, CPA 176), with no new LLM calls:

| Probability on test | CTA ECE | CTA AUROC | CPA ECE | CPA AUROC |
|---|---|---|---|---|
| Laya's yes/no, as reported | 0.189 | 0.54 | 0.254 | 0.47 |
| the same, isotonic / Platt | 0.115 / 0.116 | 0.53 / 0.54 | 0.053 / 0.033 | 0.52 / 0.53 |
| CTA: chosen type's probability in Laya's ranking, Platt | 0.143 | 0.49 | | |
| CTA: Laya's top type agrees with the LLM's, Platt | 0.088 | 0.51 | | |
| a constant: validation accuracy | 0.099 | 0.50 | 0.025 | 0.50 |

No signal tells right from wrong answers on test (AUROC 0.47–0.54, chance is 0.5). Mapping
the probabilities lowers ECE only by pulling every value to the average accuracy, which a
constant does as well. Shipping such a mapping would make tables look calibrated while their
confidence carries no information, so none is shipped. Telling right from wrong on tables
needs a better signal: Laya fine-tuned on SOTAB's training split, for example.

Showing the other columns while labelling one target did not help: with SOTAB's numbered
columns the LLM lost track of the target and answered `moose:OTHER` for nearly everything.
Labelling all targets together, as the redesign does, works. Too few start/end date pairs
are in these 50 tables (two) to say whether that confusion is gone.
