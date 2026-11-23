"""Adapter for the real Dis-Chem input columns and submission contract.

Every emitted leaf is copied from a cell of the *same* record and audited with
its source column, character span and exact quote. Keys of nutrient maps and
``other`` attributes are snake-cased source labels. Nothing is looked up in the
catalogue, converted, defaulted or paraphrased.
"""

import html
import json
import re

from .data import MASK, Evidence, missing
from .grounding import tokens
from .schema import leaves

COLUMNS = ("Product Name", "Short Description", "Product Description", "Nutritional Information", "Specifications")
ROW = re.compile(r"<tr>\s*<td>(.*?)</td>\s*<td>(.*?)</td>\s*</tr>", re.S | re.I)
NUMBER = r"\d+(?:[.,]\d+)?"
QUANTITY = re.compile(rf"\s*({NUMBER})\s*([A-Za-z][^\d]*?)?\s*")
SENTENCE = re.compile(r"[^.!?]+(?:[.!?]+|$)")
NAME_SIZE = re.compile(rf"({NUMBER})\s?(ml|l|g|kg|mg)$", re.I)

SPEC_FIELDS = {
    "pack size": "pack_size", "packaging type": "packaging_type", "packaging material": "packaging_material",
    "application area": "application_area", "weight": "weight", "height": "height", "width": "width",
    "depth": "depth", "serving size": "serving_size", "shade": "shade", "colour": "shade", "color": "shade",
    "flavour": "flavour", "flavor": "flavour", "fragrance": "fragrance", "scent": "fragrance",
    "variant": "variant_other", "ingredients": "ingredients", "key ingredients": "ingredients",
    "active ingredients": "ingredients", "certifications": "certification",
}
NUTRIENT_KEYS = {
    "energy (kj)": "energy_kj", "carbohydrate (g)": "carbohydrate_g", "of which total sugar (g)": "total_sugar_g",
    "dietary fibre (g)": "dietary_fibre_g", "total fat (g)": "total_fat_g",
    "of which saturated fat (g)": "saturated_fat_g", "protein (g)": "protein_g", "total sodium (mg)": "sodium_mg",
}
# Labelled values without meaning on their own (e.g. a flavour cell reading just "free").
BARE_QUALIFIERS = {"free", "yes", "no"}


def normalise(value):
    """Whitespace only: spelling, casing and unit spacing are copied as supplied."""
    return re.sub(r"\s+", " ", str(value)).strip()


def is_dischem(row):
    return all(column in row for column in COLUMNS)


def snake(label):
    return re.sub(r"[^0-9a-z]+", "_", label.casefold()).strip("_")


def _contiguous(value, quotes):
    value_tokens = " ".join(tokens(value))
    return bool(value_tokens) and value_tokens in " ".join(tokens(" ".join(quotes)))


class Record:
    """Collects cited leaves; a leaf without exact, contiguous support is dropped."""

    def __init__(self, row):
        self.row = row
        self.audit = {"article_id": row["record_id"], "values": [], "dropped": []}

    def cell(self, column):
        """Raw cell; masked parts are rejected per quote, so unmasked parts stay usable."""
        value = self.row.get(column) or ""
        return "" if missing(MASK.sub("", value)) else value

    def cite(self, field, path, value, spans):
        evidence = [Evidence(c, s, e, self.row[c][s:e]) for c, s, e in spans]
        quotes = [e.quote for e in evidence]
        if (not evidence or missing(value) or any(missing(q) for q in quotes)
                or str(value).strip().casefold() in ("n/a", "na", "none", "null")
                or not _contiguous(str(value), quotes)):
            self.audit["dropped"].append({"field": field, "path": list(path), "value": value,
                                          "reason": "unsupported or masked"})
            return None
        self.audit["values"].append({"field": field, "path": list(path), "value": value,
                                     "evidence": [e.as_dict() for e in evidence]})
        return value

    def drop(self, field, reason):
        self.audit["dropped"].append({"field": field, "reason": reason})


def spec_rows(record):
    """(label, value, start, end) for unmasked Specifications table cells."""
    text = record.cell("Specifications")
    result = []
    for match in ROW.finditer(text):
        label, value = html.unescape(match[1]).strip(), html.unescape(match[2]).strip()
        if label and not missing(value):
            start, end = match.span(2)
            # Trim the span to the stripped value so the quote is the value itself.
            raw = text[start:end]
            start += len(raw) - len(raw.lstrip())
            end -= len(raw) - len(raw.rstrip())
            result.append((label, value, start, end))
    return result


def _quantity(record, field, path, value, span):
    """Split '110 millilitre (ml)' into value/unit; never invent a unit."""
    column, start, end = span
    match = QUANTITY.fullmatch(record.row[column][start:end])
    if not match:
        return {"value": record.cite(field, (*path, "value"), normalise(value), [span]) or "", "unit": ""}
    number_span = (column, start + match.start(1), start + match.end(1))
    result = {"value": record.cite(field, (*path, "value"), match[1], [number_span]) or "", "unit": ""}
    if match[2] and result["value"]:
        unit_span = (column, start + match.start(2), start + match.end(2))
        result["unit"] = record.cite(field, (*path, "unit"), html.unescape(match[2]).strip(), [unit_span]) or ""
    return result


def _split(record, field, path, value, span, separator=","):
    column, start, end = span
    raw = record.row[column][start:end]
    items, offset = [], 0
    for part in _top_level_split(raw, separator):
        stripped = part.strip().rstrip(".").strip()
        if stripped and not missing(stripped):
            local = raw.index(stripped, offset)
            item = record.cite(field, (*path, len(items)), html.unescape(stripped),
                               [(column, start + local, start + local + len(stripped))])
            if item and item.casefold() not in {i.casefold() for i in items}:
                items.append(item)
        offset += len(part) + len(separator)
    return items


def _top_level_split(text, separator):
    parts, depth, current = [], 0, ""
    for char in text:
        depth += char in "([" 
        depth -= char in ")]" and depth > 0
        if char == separator and depth == 0:
            parts.append(current)
            current = ""
        else:
            current += char
    return parts + [current]


def online_name(record):
    name = record.cell("Product Name")
    if not name:
        return ""
    value = normalise(name)
    spans = [("Product Name", 0, len(name))]
    brand = record.cell("T_BRAND").strip()
    lead = name.lstrip()
    offset = len(name) - len(lead)
    # Restore catalogue brand casing for shouted brands ("PURITY" -> "Purity") only;
    # both spellings are cited from this record.
    if (brand and lead[:len(brand)].casefold() == brand.casefold() and lead[:len(brand)].isupper()
            and not brand.isupper() and not re.search(r"[a-z]'[A-Z]", brand)
            and (len(lead) == len(brand) or not lead[len(brand)].isalnum())):
        column = "T_BRAND"
        start = record.row[column].index(brand)
        rest = name[offset + len(brand):]
        value = normalise(brand + rest)
        spans = [(column, start, start + len(brand)), ("Product Name", offset + len(brand), len(name))]
    return record.cite("online_name", (), value, spans) or ""


def paragraphs(text):
    """Yield (start, end) of blank-line separated blocks."""
    for match in re.finditer(r"(?:(?!\n\s*\n).)+", text, re.S):
        if match[0].strip():
            yield match.start() + len(match[0]) - len(match[0].lstrip()), match.end() - (len(match[0]) - len(match[0].rstrip()))


def storage_blocks(text):
    return [(s, e) for s, e in paragraphs(text) if re.match(r"storage\s*:", text[s:e], re.I)]


def short_description(record):
    text = record.cell("Short Description")
    return record.cite("short_description", (), normalise(text), [("Short Description", 0, len(text))]) or "" if text else ""


def long_description(record, skip_storage):
    text = record.cell("Product Description")
    if not text:
        return ""
    pieces, spans = [], []
    storage = set(storage_blocks(text)) if skip_storage else set()
    for start, end in paragraphs(text):
        if (start, end) in storage:
            continue
        for line in re.finditer(r"[^\n]+", text[start:end]):
            raw = line[0]
            body = raw.lstrip(" \t•-*").strip()
            if not body or missing(body):
                continue
            s = start + line.start() + raw.index(body)
            piece = normalise(body)
            if raw.lstrip().startswith("•") and not re.search(r"[.!?:]$", piece):
                piece += "."
            pieces.append(piece)
            spans.append(("Product Description", s, s + len(body)))
    value = " ".join(pieces)
    return record.cite("long_description", (), value, spans) or "" if pieces else ""


def storage_instructions(record):
    text = record.cell("Product Description")
    blocks = storage_blocks(text)
    sentences = []
    for start, end in blocks:
        label = re.match(r"storage\s*:\s*", text[start:end], re.I)
        body_start = start + label.end()
        for match in SENTENCE.finditer(text[body_start:end]):
            sentence = match[0].strip()
            if len(tokens(sentence)) >= 2:
                s = body_start + match.start() + match[0].index(sentence)
                sentences.append((sentence, s, s + len(sentence)))
    if not blocks:
        for start, end in paragraphs(text):
            for match in SENTENCE.finditer(text[start:end]):
                sentence = match[0].strip().lstrip("•").strip()
                if re.match(r"(store|refrigerate|keep refrigerated)\b", sentence, re.I):
                    s = start + match.start() + match[0].index(sentence)
                    sentences.append((sentence, s, s + len(sentence)))
    result = []
    for sentence, s, e in sentences:
        value = record.cite("storage_instructions", (len(result),), normalise(sentence), [("Product Description", s, e)])
        if value and value.casefold() not in {r.casefold() for r in result}:
            result.append(value)
    return result


DECLARATION = re.compile(r"(?:^|(?<=[.!?,:;]\s)|(?<=\n))(?:Ingredients|ingredients)\s*[:,\-]?\s+")
MAY_CONTAIN = re.compile(r"(?:^|(?<=[.!?•]\s)|(?<=\n))((?:[Mm]anufactured in a facility that processes/)?[Mm]ay contain[^.\n]*)")


def ingredient_information(record, rows):
    text = record.cell("Product Description")
    items = []
    for match in DECLARATION.finditer(text):
        tail = text[match.end():]
        stop = re.search(r"\.\s+(?=[A-Z])|\n", tail)
        body = tail[:stop.start() if stop else len(tail)]
        has_colon = ":" in match[0]
        if len(_top_level_split(body, ",")) < (2 if has_colon else 4):
            continue
        items += _split(record, "ingredient_information", (len(items),), body,
                        ("Product Description", match.end(), match.end() + len(body)))
    for label, value, start, end in rows:
        if SPEC_FIELDS.get(label.casefold()) == "ingredients":
            items += _split(record, "ingredient_information", (len(items),), value, ("Specifications", start, end))
    for match in MAY_CONTAIN.finditer(text):
        phrase = match[1].strip()
        value = record.cite("ingredient_information", (len(items),), normalise(phrase),
                            [("Product Description", match.start(1), match.start(1) + len(phrase))])
        if value:
            items.append(value)
    seen, unique = set(), []
    for item in items:
        if item.casefold() not in seen:
            seen.add(item.casefold())
            unique.append(item)
    return unique


INLINE_VARIANT = re.compile(r"(?:^|[,;\n•]\s*)(colour|color|shade|flavour|flavor|fragrance|scent)\s*:\s*([^,;\n.]+)", re.I)


def variant_attributes(record, rows, empty):
    result = {k: list(v) for k, v in empty.items()}
    slots = {"shade": "shade", "flavour": "flavour", "fragrance": "fragrance", "variant_other": "other"}
    found = {}
    for label, value, start, end in rows:
        slot = slots.get(SPEC_FIELDS.get(label.casefold()))
        if slot and value.casefold() not in BARE_QUALIFIERS:
            found.setdefault(slot, []).append((value, ("Specifications", start, end)))
    text = record.cell("Product Description")
    for match in INLINE_VARIANT.finditer(text):
        slot = slots[SPEC_FIELDS[match[1].casefold()]]
        value = match[2].strip()
        if slot not in found or any(value.casefold() == v.casefold() for v, _ in found[slot]):
            if value.casefold() not in BARE_QUALIFIERS and len(tokens(value)) <= 6:
                found.setdefault(slot, []).append((value, ("Product Description", match.start(2), match.start(2) + len(value))))
    for slot, values in found.items():
        if slot not in result:
            continue
        canonical = {normalise(v).casefold() for v, _ in values}
        if len(canonical) > 1 and slot != "other":
            record.drop("variant_attributes", f"conflicting {slot} sources")
            continue
        for value, span in values:
            if normalise(value).casefold() in {r.casefold() for r in result[slot]}:
                continue
            cited = record.cite("variant_attributes", (slot, len(result[slot])), normalise(value), [span])
            if cited:
                result[slot].append(cited)
    return result


def product_specifications(record, rows, empty):
    result = json.loads(json.dumps(empty))
    field = "product_specifications"
    consumed = set()
    for label, value, start, end in rows:
        key = SPEC_FIELDS.get(label.casefold())
        span = ("Specifications", start, end)
        if key == "pack_size" and "pack_size" in result and not result["pack_size"].get("value"):
            result["pack_size"] = _quantity(record, field, ("pack_size",), value, span)
        elif key == "packaging_type" and "packaging_type" in result and not result["packaging_type"]:
            result["packaging_type"] = record.cite(field, ("packaging_type",), normalise(value), [span]) or ""
        elif key in ("application_area", "packaging_material") and key in result:
            result[key] += [v for v in _split(record, field, (key, len(result[key])), value, span)
                            if v.casefold() not in {x.casefold() for x in result[key]}]
        elif key is None or key == "certification":
            if "other" not in result:
                continue
            values = _split(record, field, ("other", len(result["other"]), "value"), value, span) if key else [normalise(value)]
            label_start = record.row["Specifications"].rfind(label, 0, start)
            if label_start < 0:
                label_start = record.row["Specifications"].rfind(html.escape(label, quote=False), 0, start)
            for item in values:
                pair = (key or snake(label), item)
                if pair in consumed or label_start < 0:
                    continue
                if not key and not record.cite(field, ("other", len(result["other"]), "value"), item, [span]):
                    continue
                if not record.cite(field, ("other", len(result["other"]), "attribute"), pair[0],
                                   [("Specifications", label_start, label_start + len(label))]):
                    continue
                consumed.add(pair)
                result["other"].append({"attribute": pair[0], "value": item})
    pack = result.get("pack_size")
    if isinstance(pack, dict) and not pack.get("value"):
        name = record.cell("Product Name").rstrip()
        match = NAME_SIZE.search(name)
        if match:
            result["pack_size"] = {
                "value": record.cite(field, ("pack_size", "value"), match[1], [("Product Name", *match.span(1))]) or "",
                "unit": record.cite(field, ("pack_size", "unit"), match[2], [("Product Name", *match.span(2))]) or "",
            }
    return result


def weight(record, rows, empty):
    for label, value, start, end in rows:
        if SPEC_FIELDS.get(label.casefold()) == "weight":
            return {**empty, **_quantity(record, "weight", (), value, ("Specifications", start, end))}
    return empty


def dimensions(record, rows, empty):
    result = dict(empty)
    parsed = {}
    for label, value, start, end in rows:
        axis = SPEC_FIELDS.get(label.casefold())
        if axis in ("height", "width", "depth") and axis in result and axis not in parsed:
            parsed[axis] = (QUANTITY.fullmatch(value), value, start, end)
    if not parsed:
        return result
    units = {m[2].strip().casefold() if m and m[2] else "" for m, *_ in parsed.values()}
    shared_unit = len(units) == 1 and "" not in units
    for axis, (match, value, start, end) in parsed.items():
        if shared_unit:
            s = start + match.start(1)
            result[axis] = record.cite("dimensions", (axis,), match[1], [("Specifications", s, s + len(match[1]))]) or ""
        else:
            result[axis] = record.cite("dimensions", (axis,), normalise(value), [("Specifications", start, end)]) or ""
    if shared_unit and "unit" in result:
        match, _, start, _ = next(iter(parsed.values()))
        s = start + match.start(2)
        result["unit"] = record.cite("dimensions", ("unit",), match[2].strip(), [("Specifications", s, s + len(match[2].strip()))]) or ""
    return result


def _number(text):
    if not re.fullmatch(r"\d+(?:\.\d+)?", text or ""):
        return None
    return float(text) if "." in text else int(text)


def nutritional_information(record, rows, empty):
    result = json.loads(json.dumps(empty))
    raw = record.cell("Nutritional Information")
    try:
        data = json.loads(raw) if raw else {}
    except json.JSONDecodeError:
        record.drop("nutritional_information", "unparseable nutrition JSON")
        return result
    column = "Nutritional Information"
    nutrients = data.get("nutrients") if isinstance(data, dict) else None
    if not isinstance(nutrients, dict):
        return result
    for name, values in nutrients.items():
        key = NUTRIENT_KEYS.get(name.casefold(), snake(name))
        name_at = raw.find(json.dumps(name, ensure_ascii=False)[1:-1])
        if name_at < 0 or not isinstance(values, dict):
            continue
        for basis in ("per_100g", "per_serving"):
            value = values.get(basis)
            number = _number(value) if isinstance(value, str) else None
            if number is None or basis not in result:
                continue
            found = re.compile(rf'"{basis}"\s*:\s*"({re.escape(value)})"').search(raw, name_at)
            if not found:
                continue
            s = found.start(1)
            if record.cite("nutritional_information", (basis, key), value, [(column, s, s + len(value))]):
                result[basis][key] = number
    if not (result.get("per_100g") or result.get("per_serving")) or "serving_size" not in result:
        return result
    labels = data.get("source_column_labels") or {}
    label = labels.get("per_serving") if isinstance(labels, dict) else None
    match = re.search(rf"Serving size:\s*({NUMBER})\s*([A-Za-z]+(?: \([A-Za-z]+\))?)", label or "")
    if match:
        label_at = raw.find(json.dumps(label, ensure_ascii=False)[1:-1])
        if label_at >= 0:
            result["serving_size"] = {
                "value": record.cite("nutritional_information", ("serving_size", "value"), match[1],
                                     [(column, label_at + match.start(1), label_at + match.end(1))]) or "",
                "unit": record.cite("nutritional_information", ("serving_size", "unit"), match[2],
                                    [(column, label_at + match.start(2), label_at + match.end(2))]) or "",
            }
            return result
    for spec_label, value, start, end in rows:
        if SPEC_FIELDS.get(spec_label.casefold()) == "serving_size":
            result["serving_size"] = _quantity(record, "nutritional_information", ("serving_size",), value,
                                               ("Specifications", start, end))
            break
    return result


def enrich(row, schema, blocked):
    record = Record(row)
    rows = spec_rows(record)
    output = {"article_id": row["record_id"]}
    builders = {
        "online_name": lambda: online_name(record),
        "short_description": lambda: short_description(record),
        "long_description": lambda: long_description(record, "storage_instructions" not in blocked),
        "ingredient_information": lambda: ingredient_information(record, rows),
        "storage_instructions": lambda: storage_instructions(record),
        "variant_attributes": lambda: variant_attributes(record, rows, schema.empty("variant_attributes")),
        "product_specifications": lambda: product_specifications(record, rows, schema.empty("product_specifications")),
        "weight": lambda: weight(record, rows, schema.empty("weight")),
        "dimensions": lambda: dimensions(record, rows, schema.empty("dimensions")),
        "nutritional_information": lambda: nutritional_information(record, rows, schema.empty("nutritional_information")),
    }
    for field in schema.columns:
        if field == "article_id":
            continue
        if field in blocked:
            output[field] = schema.empty(field)
            record.drop(field, "do_not_predict_fields")
        elif field in builders:
            before = len(record.audit["values"])
            output[field] = builders[field]()
            if len(record.audit["values"]) == before:
                record.drop(field, "no evidence")
        else:
            output[field] = schema.empty(field)
            record.drop(field, "no adapter for this field")
    _reconcile(output, record.audit)
    return output, record.audit


def _same(left, right):
    try:
        return float(left) == float(right)
    except (TypeError, ValueError):
        return str(left) == str(right)


def _reconcile(output, audit):
    """Keep one citation per emitted leaf; refuse to emit any uncited leaf."""
    pool = audit["values"]
    kept = []
    for field, value in output.items():
        if field == "article_id":
            continue
        for path, leaf in leaves(value):
            candidates = [e for e in pool if e["field"] == field and _same(e["value"], leaf)]
            match = next((e for e in candidates if tuple(e["path"]) == path), candidates[0] if candidates else None)
            if match is None:
                raise ValueError(f"{audit['article_id']}: uncited {field} value {leaf!r}")
            pool.remove(match)
            kept.append({**match, "path": list(path)})
    audit["values"] = kept
