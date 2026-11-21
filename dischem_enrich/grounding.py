"""Final independent gate for all rule and plugin candidates."""

from dataclasses import dataclass
import re

from .data import Evidence, missing


def normalise(value):
    value = re.sub(r"\s+", " ", str(value)).strip()
    value = re.sub(r"(\d)\s*(ML|L|MG|G|KG|CM|MM)\b", lambda m: m[1] + " " + m[2].lower(), value, flags=re.I)
    return value


def tokens(value):
    return re.findall(r"\d+(?:\.\d+)?|[^\W\d_]+", normalise(value).casefold(), re.UNICODE)


@dataclass
class Candidate:
    value: str
    evidence: list[Evidence]


def validate(candidate, source_text):
    """Numbers and units are exact tokens; no synonyms or factual expansions."""
    if missing(candidate.value) or candidate.value.strip().casefold() in ("n/a", "na", "none", "null"):
        return False, "empty, masked or forbidden placeholder"
    if not candidate.evidence:
        return False, "missing evidence"
    quotes = []
    negations = {"no", "not", "never", "without", "free", "avoid"}
    for evidence in candidate.evidence:
        source = source_text.get(evidence.source_column)
        if (source is None or evidence.start < 0 or evidence.end > len(source) or evidence.end <= evidence.start
                or source[evidence.start:evidence.end] != evidence.quote or missing(evidence.quote)):
            return False, "invalid citation span or quote"
        # Include the enclosing clause so a model cannot cite only 'swallow'
        # from 'do not swallow'. Precision is preferred over ambiguous recovery.
        boundaries = list(re.finditer(r"[.;\n|]", source))
        left = max([m.end() for m in boundaries if m.end() <= evidence.start] or [0])
        right = min([m.start() for m in boundaries if m.start() >= evidence.end] or [len(source)])
        scope = set(tokens(source[left:right])) & negations
        if not scope.issubset(set(tokens(candidate.value))):
            return False, "negation or qualification omitted from source clause"
        quotes.append(evidence.quote)
    supported = set(tokens(" ".join(quotes)))
    if not set(tokens(candidate.value)).issubset(supported):
        return False, "unsupported tokens, numbers or units"
    # Token membership alone could turn 'sugar free' into 'sugar' or invert a warning.
    # Require extractive contiguous phrases within each cited source.
    value = " ".join(tokens(candidate.value))
    if not any(value in " ".join(tokens(quote)) for quote in quotes):
        # A name can concatenate complete, individually quoted identity parts.
        if value != " ".join(" ".join(tokens(quote)) for quote in quotes):
            return False, "non-extractive phrase or reordered meaning"
    return True, "grounded"
