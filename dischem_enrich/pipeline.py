"""Schema-shaped extraction, audit logging and strict submission validation."""

import csv
import json
from pathlib import Path

from .data import Evidence, read_csv, write_csv, sources, excluded, image_mapping
from .extraction import extract, resolve, name_candidates
from .grounding import Candidate, validate
from .schema import Schema, leaves


def enrich(row, schema, columns=None, aliases=None, ocr=None, images=(), extractor=None):
    source_text = sources(row, columns, ocr, images)
    output = {"article_id": row["record_id"]}
    audit = {"article_id": row["record_id"], "values": [], "dropped": []}
    blocked = excluded(row)

    def value_for(field, path=()):
        slot = str(path[-1]) if path else field
        candidates = extract(slot, source_text, aliases)
        identity_reason = None
        if slot == "online_name":
            candidates, identity_reason = name_candidates(source_text, aliases)
            if identity_reason:
                audit["dropped"].append({"field": field, "path": list(path), "reason": identity_reason})
                return ""
        if extractor:
            candidates += extractor(slot, source_text)
        candidate, reason = resolve(candidates)
        if candidate:
            valid, reason = validate(candidate, source_text)
            if valid:
                audit["values"].append({"field": field, "path": list(path), "value": candidate.value,
                                        "evidence": [e.as_dict() for e in candidate.evidence]})
                return candidate.value
        audit["dropped"].append({"field": field, "path": list(path), "reason": reason})
        return ""

    def populate(field, shape, path=()):
        if isinstance(shape, dict):
            return {k: populate(field, v, (*path, k)) for k, v in shape.items()}
        if isinstance(shape, list):
            # Empty list contracts do not specify item structure: emit strings only.
            value = value_for(field, path)
            if value and field == "ingredients":
                values = [v.strip() for v in value.split(",") if v.strip()]
                original = audit["values"].pop()
                accepted = []
                seen = set()
                evidence = [Evidence(**e) for e in original["evidence"]]
                for item in values:
                    if item.casefold() in seen:
                        continue
                    valid, reason = validate(Candidate(item, evidence), source_text)
                    if valid:
                        seen.add(item.casefold())
                        audit["values"].append({**original, "value": item, "path": [*path, len(accepted)]})
                        accepted.append(item)
                    else:
                        audit["dropped"].append({"field": field, "value": item, "reason": reason})
                return accepted
            return [value] if value else []
        return value_for(field, path)

    for field in schema.columns:
        if field == "article_id":
            continue
        if field in blocked:
            output[field] = schema.empty(field)
            audit["dropped"].append({"field": field, "reason": "do_not_predict_fields"})
        else:
            output[field] = populate(field, schema.empty(field))
    return output, audit


def write_submission(path, rows, schema, blocked=None):
    encoded = []
    for row in rows:
        result = {}
        for field in schema.columns:
            value = row.get(field, schema.empty(field))
            if blocked and field in blocked.get(row["article_id"], set()):
                # The challenge specifically requires a blank CSV cell here.
                result[field] = ""
            else:
                result[field] = schema.encode(field, value)
        encoded.append(result)
    write_csv(path, schema.columns, encoded)


def _shape(value, shape):
    if isinstance(shape, dict):
        return isinstance(value, dict) and value.keys() == shape.keys() and all(_shape(value[k], v) for k, v in shape.items())
    if isinstance(shape, list):
        return isinstance(value, list)
    return isinstance(value, str)


def validate_submission(path, schema, sample=None, inputs=None):
    errors = []
    with Path(path).open(encoding="utf-8-sig", newline="") as stream:
        header = next(csv.reader(stream), [])
    if header != schema.columns:
        errors.append("Submission columns/order differ from schema")
    if sample:
        with Path(sample).open(encoding="utf-8-sig", newline="") as stream:
            if next(csv.reader(stream), []) != schema.columns:
                errors.append("SampleSubmission columns/order differ from schema")
    rows = read_csv(path)
    ids = [row.get("article_id") for row in rows]
    if any(not identifier for identifier in ids) or len(ids) != len(set(ids)):
        errors.append("Empty or duplicate article_id")
    expected = {r["record_id"]: r for r in inputs} if inputs is not None else None
    if expected is not None and (len(rows) != len(inputs) or set(ids) != set(expected)):
        errors.append("Article IDs do not match input record IDs")
    for row in rows:
        for field in schema.columns:
            raw = row.get(field)
            if raw is None:
                errors.append(f"{row.get('article_id')}: missing {field}")
                continue
            if expected and row.get("article_id") in expected and field in excluded(expected[row["article_id"]]) and raw != "":
                errors.append(f"{row['article_id']}: excluded {field} must be a blank cell")
            try:
                value = schema.decode(field, raw)
                if not _shape(value, schema.fields[field]):
                    raise ValueError("wrong shape")
                if any(v.strip().casefold() in ("n/a", "na", "null", "none") or "[masked]" in v.casefold() for _, v in leaves(value)):
                    raise ValueError("forbidden placeholder")
            except (ValueError, TypeError):
                errors.append(f"{row.get('article_id')}: invalid {field}")
    return errors


def run(data_dir, out, input_name="Test_Masked.csv", config=None, ocr=None, extractor=None):
    config = config or {}
    root = Path(data_dir)
    schema = Schema.load(root / "submission_format.json")
    inputs = read_csv(root / input_name)
    identifiers = [r["record_id"] for r in inputs]
    if any(not i for i in identifiers) or len(set(identifiers)) != len(identifiers):
        raise ValueError("Input record IDs must be nonempty and unique")
    mapping = image_mapping(root / "image_mapping.xlsx", root / "product_images") if ocr else {}
    predictions, audits = [], []
    for row in inputs:
        prediction, audit = enrich(row, schema, config.get("source_columns"), config.get("field_aliases"),
                                   ocr, mapping.get(row["record_id"], ()), extractor)
        predictions.append(prediction)
        audits.append(audit)
    out = Path(out)
    out.parent.mkdir(parents=True, exist_ok=True)
    write_submission(out, predictions, schema, {r["record_id"]: excluded(r) for r in inputs})
    evidence_path = out.with_suffix(".evidence.jsonl")
    evidence_path.write_text("".join(json.dumps(a, ensure_ascii=False) + "\n" for a in audits), encoding="utf-8")
    sample = root / "SampleSubmission.csv"
    errors = validate_submission(out, schema, sample if sample.exists() else None, inputs)
    if errors:
        raise ValueError("; ".join(errors))
    return predictions
