# Executable no-hallucination policy

1. **Record isolation** (`data.sources`): only the current record's nonempty,
   unmasked source cells and its mapped image OCR text can provide evidence.
   The catalogue is not an answer index. `conventions` learns vocabulary only;
   it is never passed to the extractor as factual evidence.
2. **Masked means unavailable**: a whole cell containing `[MASKED]` (case
   insensitive) is excluded. Another source of the *same record* can recover it.
3. **Explicit scopes** (`extraction.extract`): copy named columns or labelled
   sections. Do not infer medical benefits, intended ages, ingredients or safety
   claims from a brand/category. Unknown schema fields abstain unless their name
   is explicitly present as a source label or configured alias.
4. **Conflicts abstain** (`extraction.resolve`): differing canonical values for a
   slot blank that slot. Identity size/brand conflicts also suppress the online
   name. Do not select whichever source happens to be first. This intentionally
   over-abstains for complementary lists; it is a precision-first baseline.
5. **Final grounding gate** (`grounding.validate`): every candidate needs exact
   column, character offsets and quote. Every token, number and unit must be
   present; the full phrase must be extractive. Name composition may concatenate
   complete cited identity fragments. No synonym substitution, dosage arithmetic,
   unit conversion, unsupported negation or invented connective prose. Normalise
   whitespace and unit casing only (`250 ML` becomes `250 ml`).
6. **No placeholders**: no `N/A`, `null`, `none` or masked markers. Use the schema's
   empty string/list/object. `do_not_predict_fields` always writes a blank CSV
   cell, decoded by the schema into its empty shape.
7. **Audit every emitted fact**: `<submission stem>.evidence.jsonl` contains one
   record per article, emitted values with citations, and dropped slots/reasons.
   Character offsets refer to original strings, not normalised strings. The ID
   is a verbatim copy, not an inferred fact. Structured object keys are schema,
   not predicted facts.
8. **No external enrichment**: no websites, paid APIs, product search or other
   datasets. Optional OCR and open local models are untrusted proposal sources;
   their text/quotes still pass through the same grounding gate. OCR may misread
   a label: citations demonstrate provenance, not photographic truth.

These rules reduce hallucination; they cannot prove source text itself is true,
detect every semantic contradiction across unlabelled prose, or replace the
official content guardrail PDF. Read that supplied guide before real submissions.
The offline baseline deliberately leaves unsupported descriptions blank rather
than rewriting plausible marketing content. Token evidence alone is not treated
as permission to rearrange facts.
