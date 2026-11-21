"""Lossless CSV identifiers, source collection and optional image text hooks."""

import csv
from dataclasses import dataclass, asdict
import json
from pathlib import Path
import re

MASK = re.compile(r"\[\s*MASKED\s*\]", re.I)


def missing(value):
    return not str(value or "").strip() or bool(MASK.search(str(value)))


def read_csv(path):
    with Path(path).open(encoding="utf-8-sig", newline="") as stream:
        return list(csv.DictReader(stream))


def write_csv(path, columns, rows):
    with Path(path).open("w", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=columns)
        writer.writeheader()
        writer.writerows(rows)


def excluded(row):
    raw = row.get("do_not_predict_fields", "")
    value = json.loads(raw) if raw.strip() else []
    if not isinstance(value, list) or not all(isinstance(v, str) for v in value):
        raise ValueError("do_not_predict_fields must be a JSON list of field names")
    return set(value)


@dataclass(frozen=True)
class Evidence:
    source_column: str
    start: int
    end: int
    quote: str

    def as_dict(self):
        return asdict(self)


def sources(row, columns=None, ocr=None, image_paths=()):
    """Never accepts a catalogue lookup: evidence is local to this record only.

    OCR plugin: callable(path) -> str. Images must be resolved by record_id.
    Entire cells containing a mask are excluded conservatively.
    """
    result = {}
    selected = columns if columns is not None else row.keys()
    for column in selected:
        if column in ("record_id", "article_id", "do_not_predict_fields"):
            continue
        value = row.get(column, "")
        if not missing(value):
            result[column] = value
    if ocr:
        for path in image_paths:
            value = ocr(path)
            if not missing(value):
                result[f"ocr:{Path(path).name}"] = value
    return result


def image_mapping(path, image_dir):
    """Return all images per string record ID; reject paths outside image_dir."""
    from openpyxl import load_workbook

    workbook = load_workbook(path, read_only=True, data_only=True)
    sheet = iter(workbook.active.values)
    header = next(sheet)
    id_index, file_index = header.index("record_id"), header.index("img_filename")
    result = {}
    root = Path(image_dir).resolve()
    for row in sheet:
        identifier, filename = row[id_index], row[file_index]
        if identifier is None or not filename:
            continue
        if not isinstance(identifier, str):
            raise ValueError("XLSX record_id must be text to preserve leading zeros")
        target = (root / str(filename)).resolve()
        if not target.is_relative_to(root):
            raise ValueError("Unsafe image path")
        result.setdefault(identifier, []).append(target)
    workbook.close()
    return result


def conventions(rows):
    """Learn vocabulary counts only, never per-record product facts."""
    from collections import Counter

    counts = Counter()
    expansions = Counter()
    patterns = Counter()
    for row in rows:
        for column, value in row.items():
            if any(tag in column.lower() for tag in ("brand", "category")) and not missing(value):
                counts[value.strip()] += 1
            if not missing(value):
                for full, short in re.findall(r"([A-Za-z]+(?: [A-Za-z]+){0,3})\s*\(([A-Z]{2,6})\)", value):
                    expansions[f"{short} -> {full}"] += 1
                if any(tag in column.lower() for tag in ("name", "prod_nm", "title")):
                    patterns[re.sub(r"\d+(?:\.\d+)?", "#", value)] += 1
    return {"vocabulary": dict(counts), "observed_expansions": dict(expansions), "name_patterns": dict(patterns)}
