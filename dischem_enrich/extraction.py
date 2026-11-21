"""Conservative labelled-section extraction; unknown fields abstain."""

import re

from .data import Evidence
from .grounding import Candidate, normalise, tokens

ALIASES = {
    "ingredients": ("ingredients", "composition", "contains"),
    "directions": ("directions", "usage", "how to use", "instructions"),
    "warnings": ("warnings", "warning", "caution", "precautions"),
    "features": ("features", "claims", "benefits"),
    "specifications": ("specifications", "specs", "attributes"),
    "categories": ("category", "categories"),
    "variants": ("variant", "variants"),
    "online_name": ("online_name", "product_name", "name", "title", "prod_nm", "item_desc"),
    "short_description": ("short_description", "short description", "summary"),
    "long_description": ("long_description", "long description", "description", "details"),
    "size": ("size", "pack size", "volume", "weight"),
    "flavour": ("flavour", "flavor"),
    "shade": ("shade", "colour", "color"),
    "dimensions": ("dimensions",),
    "brand": ("brand", "brand_name", "brnd"),
}
SIZE = re.compile(r"\b\d+(?:\.\d+)?\s*(?:ml|kg|mg|g|l|cm|mm)\b", re.I)


def field_aliases(field, mapping=None):
    if mapping and field in mapping:
        return tuple(mapping[field])
    return ALIASES.get(field, (field.replace("_", " "),))


def extract(field, source_text, mapping=None):
    aliases = field_aliases(field, mapping)
    found = []
    for column, text in source_text.items():
        if column.casefold() in {a.casefold() for a in aliases}:
            found.append(Candidate(normalise(text), [Evidence(column, 0, len(text), text)]))
        # Explicit labels define scope, not guessed associations to nearby products.
        pattern = r"(?:^|[;\n|])\s*(?:" + "|".join(re.escape(a) for a in aliases) + r")\s*:\s*([^;\n|]+)"
        for match in re.finditer(pattern, text, re.I):
            start, end = match.span(1)
            found.append(Candidate(normalise(match[1]), [Evidence(column, start, end, match[1])]))
        if field == "size":
            for match in SIZE.finditer(text):
                found.append(Candidate(normalise(match[0]), [Evidence(column, *match.span(), match[0])]))
    return found


def resolve(candidates):
    """Disagreement in a semantic slot abstains, rather than choosing a source."""
    unique = {}
    for candidate in candidates:
        key = normalise(candidate.value).casefold()
        unique.setdefault(key, candidate)
    if len(unique) > 1:
        return None, "conflicting sources"
    return (next(iter(unique.values())), None) if unique else (None, "no evidence")


def name_candidates(source_text, mapping=None):
    """Join independently evidenced identity parts, never catalogue facts."""
    names = extract("online_name", source_text, mapping)
    brand, brand_reason = resolve(extract("brand", source_text, mapping))
    size, size_reason = resolve(extract("size", source_text, mapping))
    if brand_reason == "conflicting sources" or size_reason == "conflicting sources":
        return [], "conflicting identity sources"
    variant, variant_reason = resolve(extract("variants", source_text, mapping))
    if variant_reason == "conflicting sources":
        return [], "conflicting identity sources"
    result = []
    for name in names:
        parts = [name]
        if brand and not set(tokens(brand.value)).issubset(set(tokens(name.value))):
            parts.insert(0, brand)
        # A variant is not always part of the formal product name: retain only
        # explicitly named variant evidence when constructing an incomplete name.
        if brand and parts[0] is brand and variant and not set(tokens(variant.value)).issubset(set(tokens(name.value))):
            parts.append(variant)
        if size and not set(tokens(size.value)).issubset(set(tokens(name.value))):
            parts.append(size)
        result.append(Candidate(" ".join(p.value for p in parts), [e for p in parts for e in p.evidence]))
    return result, None
