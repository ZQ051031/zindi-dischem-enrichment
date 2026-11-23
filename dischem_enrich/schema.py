"""Read submission contracts without assuming the seven structured field names."""

from copy import deepcopy
import json
from pathlib import Path

TEXT_FIELDS = {"online_name", "short_description", "long_description"}


def _empty(spec):
    if not isinstance(spec, dict):
        if isinstance(spec, str) and spec.strip()[:1] in ("[", "{"):
            return json.loads(spec)
        return spec
    for key in ("empty_value", "empty", "default", "empty_shape"):
        if key in spec:
            value = spec[key]
            if isinstance(value, str) and value[:1] in ("[", "{"):
                return json.loads(value)
            return value
    kind = spec.get("type", spec.get("dtype"))
    if kind in ("list", "array"):
        return []
    if kind in ("str", "string", "text"):
        return ""
    if kind in ("dict", "object"):
        properties = spec.get("properties", spec.get("keys"))
        if not isinstance(properties, dict):
            raise ValueError("Object specs must define keys/properties or an empty value")
        return {k: _empty(v) for k, v in properties.items()}
    return spec


class Schema:
    def __init__(self, fields):
        self.fields = {k: deepcopy(v) for k, v in fields.items()}
        if "article_id" not in self.fields:
            self.fields = {"article_id": "", **self.fields}
        if any(not isinstance(v, (str, list, dict)) for v in self.fields.values()):
            raise ValueError("Only text, list and object empty shapes are supported")

    @classmethod
    def load(cls, path):
        raw = json.loads(Path(path).read_text(encoding="utf-8"))
        if isinstance(raw, dict) and "required_columns" in raw:
            # Official Dis-Chem contract: ordered names plus per-field formats.
            formats = raw.get("field_formats", {})
            if not all(isinstance(c, str) for c in raw["required_columns"]):
                raise ValueError("required_columns must be a list of names")
            return cls({c: _empty(formats.get(c, {"type": "string"})) for c in raw["required_columns"]})
        if isinstance(raw, dict):
            for container in ("columns", "fields", "submission_columns", "output_fields"):
                if container in raw:
                    columns = raw[container]
                    if isinstance(columns, list) and all(isinstance(x, str) for x in columns):
                        empties = raw.get("empty_values", raw.get("empty_shapes", {}))
                        return cls({k: _empty(empties.get(k, "")) for k in columns})
                    raw = columns
                    break
        if isinstance(raw, list):
            fields = {}
            for spec in raw:
                if not isinstance(spec, dict):
                    raise ValueError("Column specifications need names and empty shapes")
                name = spec.get("name", spec.get("column", spec.get("field", spec.get("field_name"))))
                if not name:
                    raise ValueError("Missing column name")
                fields[name] = _empty(spec)
            return cls(fields)
        if isinstance(raw, dict):
            return cls({k: _empty(v) for k, v in raw.items()})
        raise ValueError("Unsupported submission schema")

    def empty(self, field):
        return deepcopy(self.fields[field])

    @property
    def columns(self):
        return list(self.fields)

    def decode(self, field, value):
        if not value:
            return self.empty(field)
        return json.loads(value) if isinstance(self.fields[field], (list, dict)) else value

    def encode(self, field, value):
        return json.dumps(value, ensure_ascii=False) if isinstance(self.fields[field], (list, dict)) else value


def leaves(value, path=()):
    """Yield nonempty atomic values with their structural paths."""
    if isinstance(value, dict):
        for key, item in value.items():
            yield from leaves(item, (*path, key))
    elif isinstance(value, list):
        for index, item in enumerate(value):
            yield from leaves(item, (*path, index))
    elif value not in ("", None):
        yield path, str(value)
