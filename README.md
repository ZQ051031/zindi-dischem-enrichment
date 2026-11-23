# Dis-Chem evidence-first enrichment

Python **3.10+**, open-source, offline/CPU baseline for the Zindi **Dis-Chem
Product Data Enrichment Challenge**. This is a conservative starter, not a
trained model or a claim about leaderboard performance. **No real challenge data
is included or was available during development.**

## Quick start

```bash
python3 -m pip install --user -r requirements.txt
python3 -m pytest -q
python3 -m dischem_enrich make-synthetic --out data/synthetic
python3 -m dischem_enrich run --data-dir data/synthetic --out data/synthetic/submission.csv
python3 -m dischem_enrich validate-submission --data-dir data/synthetic --pred data/synthetic/submission.csv
python3 -m dischem_enrich evaluate --pred data/synthetic/submission.csv --ref data/synthetic/ground_truth.csv
```

The core pipeline uses the standard library. `openpyxl` is needed only for the
optional image mapping reader; pytest is a development dependency. Seeds are 42
by default; extraction and evaluation are deterministic. No network requests,
model downloads or GPU allocations occur in the default workflow.

## Real data, step by step

1. Join the competition and download the permitted datasets yourself into
   `data/raw/`. Do not commit them. Expect `Test_Masked.csv` (1,157 records),
   `unlabelled_data.csv` (13,864 records), `submission_format.json`,
   `SampleSubmission.csv`, `template_input.csv`, `template.csv`, and optionally
   `image_mapping.xlsx` plus `product_images.zip`. Read the provided
   `product_enrichment_guardrail.pdf` and `columns.docx` before adapting rules.
2. Inspect the **actual** schema and source headers. The official
   `submission_format.json` (`required_columns` + `field_formats`) is loaded
   directly. Its targets are `online_name`, `short_description`,
   `long_description`, `ingredient_information`, `storage_instructions`,
   `variant_attributes`, `product_specifications`, `weight`, `dimensions` and
   `nutritional_information`. The schema controls columns/order and empty
   shapes; the sample must agree with its columns/order.

   **Dis-Chem profile.** When the input has the official columns (`Product Name`,
   `Short Description`, `Product Description`, `Nutritional Information`,
   `Specifications`), `run` auto-selects the rule-only adapter in
   `dischem_enrich/dischem.py` (`--profile dischem|generic|auto`). It reads only
   the same record's five content cells: names and descriptions are copied
   (whitespace normalised only), `Storage:` blocks become storage instructions,
   explicit `Ingredients:` declarations are split, and specification/nutrition
   values come only from unmasked HTML table rows / nutrition JSON values.
   Masked parts of a cell are skipped individually; catalogue context
   (`T_WEIGHT_UNIT`, `VOLUME`, ...) and `unlabelled_data.csv` are never used as
   values; units are not converted or added. Every emitted leaf must have an
   exact column/offset/quote citation or the run fails.
3. Supported schema forms: a dictionary of column-to-empty values; a dictionary
   of typed specs; a list of `{name/column/field, empty_value/empty/default/type}`
   specs; or `columns`, `fields`, `submission_columns`, `output_fields` wrappers.
   A string column list can use a sibling `empty_values`/`empty_shapes` map.
   Object type specs require `properties`/`keys` or explicit empties. Unknown or
   malformed contracts fail, not silently guessed structured fields. An absent
   `article_id` is prepended. Empty objects retain their defined keys recursively.
4. By default **every non-ID, non-control column** is considered as a source.
   Existing output-like source columns are allowed if supplied in the input;
   templates/answers are never read by `run`. Use an optional JSON configuration
   to limit source columns and map actual output concepts to source labels:

   ```json
   {
     "source_columns": ["supplier_notes", "sap_name", "web_description"],
     "field_aliases": {
       "online_name": ["sap_name", "product name"],
       "actual_ingredient_field": ["ingredients", "composition"],
       "size": ["pack size", "net content"]
     }
   }
   ```

   Object properties use their property name as the slot (`size` above). A label
   may appear as a column name or `Label: value` section. Unknown concepts abstain.
   Do not assume synthetic aliases match real category-specific requirements.
5. Run and evaluate the 14 labelled examples first (joined by ID, never row order):

   ```bash
   python3 -m dischem_enrich run --data-dir data/raw --input template_input.csv --out data/template_predictions.csv
   python3 -m dischem_enrich evaluate --pred data/template_predictions.csv --ref data/raw/template.csv
   ```

   Add `--config path/to/config.json` to `run` when needed. Review the evidence
   sidecar and abstentions against the supplied guardrail. These tiny examples
   are not enough to estimate generalisation.
6. Run the held-out test and validate before uploading:

   ```bash
   python3 -m dischem_enrich run --data-dir data/raw --out data/submission.csv
   python3 -m dischem_enrich validate-submission --data-dir data/raw --pred data/submission.csv
   ```

   IDs are strings and copied exactly into `article_id`. CSV lists/objects are
   JSON cells, not Python repr. Excluded targets are literal blank CSV cells,
   which decode to the schema's empty shape. Other unsupported structured targets
   are serialised as their schema empty list/object.

**Deadline:** submissions close **13 Dec 2026 at 23:29 UTC**, according to the
scraped timeline (not the evaluation phase date range). Limit **2/day, 20 total**.
Confirm the current official page before submitting.

## Design and modules

| Module | Responsibility |
|---|---|
| `data.py` | Lossless CSV strings, masks, control fields, own-record source cross-reference; XLSX one-to-many image mapping; convention-only vocabulary/observed parenthetical expansions/name patterns |
| `schema.py` | Dynamic contract loading, empty shapes, JSON cell encoding, recursive leaves |
| `extraction.py` | Labelled ingredients, usage, warnings, features/claims, categories/variants and size/flavour/shade/dimensions; source conflict abstention; grounded identity name assembly |
| `grounding.py` | Citation span/quote checks, token/number/unit checks, negation protection, whitespace/unit casing |
| `pipeline.py` | Record isolation, shape-driven population, dedupe, exclusions, submission writer/validator and audit JSONL |
| `plugins.py` | Optional local OpenAI-compatible extractor and local sentence-transformer similarity |
| `evaluation.py` | Pure-Python TF-IDF char cosine, score buckets, pooled structured matching/F1, hallucination proxy |
| `synthetic.py` | Independent seed-controlled mixed-category fixture generator and held-out truth |
| `__main__.py` | Four CLI commands |

### Grounding and limitations

See [RULES.md](RULES.md) for the executable policy. Facts come only from the
**same record** or its mapped images, never another catalogue product. Masked
cells are unavailable. Conflicting slots blank rather than using source order.
Every emitted atomic fact has source column, start/end character span and exact
quote in `<submission stem>.evidence.jsonl`; discarded slots include reasons.
`article_id` and object property keys are copied contract metadata, not generated
facts. Catalogue conventions can be inspected through `data.conventions` but are
not evidence and are deliberately not used to fill missing product attributes.

No medical knowledge, web search, speculative claims, unit conversions or `N/A`
placeholders. Names use Brand + Product + explicitly evidenced Variant + Size
when separate identity columns permit it; an already-complete source name is
preserved, not aggressively title-cased. Descriptions are extractive, and never
invented from a category. Ingredients are comma-separated only when explicitly
identified; complementary source lists may conservatively conflict. Lists of
unknown item structure emit strings, since an empty list alone cannot specify
an object item schema. More complex real contracts may need explicit adapters.
The baseline does not infer nutritional tables or understand arbitrary prose,
and cannot detect all unlabelled semantic contradictions. Source citations are
not a proof that supplied text or OCR is correct.

### Optional, explicitly enabled backends

For OCR, provide an installed **trusted Python** module with
`ocr_image(path: pathlib.Path) -> str`, then use `run --ocr-plugin module:ocr_image`.
Only images matched by `image_mapping.xlsx` are passed; unpack the archive into
`data/raw/product_images/` yourself. Mapping IDs must be stored as text in XLSX
(numeric IDs are rejected to avoid silent zero loss). Paths outside that image
directory are rejected. OCR text appears under `ocr:<filename>` evidence columns.

An open model served locally with an OpenAI-compatible chat endpoint may propose
extracts via `--llm-endpoint http://127.0.0.1:8000/v1/chat/completions --model NAME`.
Only loopback endpoints are allowed, not paid hosted services. Outputs require
quoted evidence with offsets and are independently gated. Temperature 0, seed
42; at most **one retry per failed call**, including malformed responses; global
model-call time budget 9,000 seconds, request timeout 30 seconds. No model/server
is installed automatically. Local model reproducibility and GPU memory depend
on your backend. Optional similarity: install open-source `sentence-transformers`
and pass `evaluate --sentence-model /path/to/already-downloaded/open/model`;
weights are loaded locally only.

**Resource responsibility:** the default tiny deterministic pipeline needs no
GPU or paid inference. The challenge caps the **entire run at 3 hours / US$20 /
one GPU with at most 24 GB VRAM**. The plugin deadline is not a full process/GPU
monitor. You must choose compliant open weights, enforce memory allocation and
measure end-to-end time including OCR; do not enable a backend whose runtime or
licence you cannot verify. Only supplied product datasets are allowed; open
pretrained weights are allowed, no additional product datasets.

## Offline evaluation

This is a **proxy**, not the actual LLM judge. Text uses pairwise TF-IDF character
3–5-gram cosine rounded to the nearest `{0, .25, .5, .75, 1}` bucket. Structured
values match canonically (case/spacing/punctuation/unit casing), keeping object
key paths but ignoring list indices. Each reference value matches once. Extras
are duplicates, neutral packaging elaborations (acceptable), or FP; a Python
compatibility callback supports richer local judging. This intentionally narrow
matcher cannot replicate semantic LLM decisions. Counts are pooled per field,
`F1=2TP/(2TP+FP+FN)`. Text is averaged per referenced product. Empty reference
fields are excluded for that product, and fields never referenced are excluded
from the overall equal-weight mean. Hallucination reporting separately counts
emitted structured values absent from held-out truth **including empty refs**;
it is a canonical mismatch proxy, not a claim of factual hallucination.

## Synthetic benchmark

**SYNTHETIC — not real Zindi scores.** Seed 42, 60 held-out test records plus 14
separate labelled examples. The seven structured names `ingredients`,
`directions`, `warnings`, `features`, `specifications`, `categories`, `variants`
are **ASSUMED placeholder names**, not the real unknown contract. Health, baby,
beauty, food and household records use varied label/prose grammars, missing/masked
cells, multiple columns, contradictory ingredient/size sources, exclusions and
correctly blank answers. Some prose intentionally defeats the baseline; truth
is authored from the generator's product facts, not by calling the extractor.
The generator has no dependency on extraction rules, but this small handcrafted
catalogue is still a development fixture, not a robust independent benchmark.

Measured on Python 3.10.12 with the commands above:

| Field | Proxy score |
|---|---:|
| online_name | 1.000000 |
| short_description | 0.759259 |
| long_description | 0.759259 |
| ingredients | 0.805970 |
| directions | 0.839506 |
| warnings | 0.805970 |
| features | 0.839506 |
| specifications | 1.000000 |
| categories | 1.000000 |
| variants | 1.000000 |
| **Overall** | **0.880947** |

Structured hallucination proxy: **0 / 316 emitted values = 0.00%**. This only
measures the tiny generated fixture, not unseen real products. High schema/size
scores reflect explicit labels in this fixture, not verified competition recall.

## Tests / CI

`python3 -m pytest -q` covers string IDs, masks, XLSX mapping, schema variants,
extraction, conflicting sources, fabricated/negated facts, plugin grounding,
format/exclusions, hand-calculated pooled metrics, determinism and small CLI /
end-to-end runs. GitHub Actions runs the same suite on Python 3.11 and 3.12.
Real-data performance, image OCR quality, optional model inference and GPU caps
remain unverified without those assets/backends. This is a CLI-only project;
there is no visual UI to verify.

MIT licensed. Do not distribute challenge data; follow competition rules for
public code sharing and any model licence you elect to use.
