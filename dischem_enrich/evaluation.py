"""Deterministic local proxy, NOT the challenge's proprietary LLM judge."""

from collections import Counter
import math

from .data import read_csv
from .grounding import tokens
from .schema import TEXT_FIELDS, leaves


def canonical(value):
    return " ".join(tokens(value))


def cosine(left, right):
    """Pairwise TF-IDF character 3-5 grams, pure standard library."""
    def grams(text):
        text = " " + canonical(text) + " "
        return Counter(text[i:i+n] for n in (3, 4, 5) for i in range(max(0, len(text)-n+1)))

    a, b = grams(left), grams(right)
    if not a or not b:
        return float(bool(canonical(left)) and canonical(left) == canonical(right))
    vocab = a.keys() | b.keys()
    idf = {k: 1 + math.log(3 / (1 + int(k in a) + int(k in b))) for k in vocab}
    dot = sum(a[k] * b[k] * idf[k] ** 2 for k in vocab)
    norm = math.sqrt(sum((a[k] * idf[k]) ** 2 for k in a) * sum((b[k] * idf[k]) ** 2 for k in b))
    return dot / norm if norm else 0.0


def bucket(similarity):
    return min(1.0, max(0.0, math.floor(similarity * 4 + 0.5) / 4))


def compatible_extra(item, references):
    """Only neutral packaging elaborations of a reference are harmless by default."""
    path, value = item
    neutral = {"bottle", "tube", "jar", "pack", "sachet", "container"}
    for ref_path, ref in references:
        extra = set(value.split()) - set(ref.split())
        if path == ref_path and extra and extra <= neutral and f" {ref} " in f" {value} ":
            return True
    return False


def structured_counts(prediction, reference, compatible=compatible_extra):
    """Paths preserve object meaning; default compatibility is conservative.

    A custom local semantic judge may mark an unmatched value compatible. The
    default regards only canonically equal values as matches/duplicates.
    """
    def atoms(value):
        return [(tuple(p for p in path if not isinstance(p, int)), canonical(v)) for path, v in leaves(value)]

    pred, ref = atoms(prediction), atoms(reference)
    remaining = list(ref)
    result = Counter(TP=0, FP=0, FN=0, duplicate=0, acceptable=0)
    seen = set()
    for item in pred:
        if item in remaining:
            remaining.remove(item)
            result["TP"] += 1
        elif item in seen or item in ref:
            result["duplicate"] += 1
        elif compatible and compatible(item, ref):
            result["acceptable"] += 1
        else:
            result["FP"] += 1
        seen.add(item)
    result["FN"] = len(remaining)
    return result


def evaluate(predictions, references, schema, similarity=cosine):
    pred = {r["article_id"]: r for r in predictions}
    fields = {}
    for field in schema.columns:
        if field == "article_id":
            continue
        scores, counts = [], Counter(TP=0, FP=0, FN=0, duplicate=0, acceptable=0)
        for row in references:
            reference = row.get(field, schema.empty(field))
            if not list(leaves(reference)):
                continue
            prediction = pred.get(row["article_id"], {}).get(field, schema.empty(field))
            if field in TEXT_FIELDS:
                scores.append(bucket(similarity(str(prediction), str(reference))))
            else:
                counts.update(structured_counts(prediction, reference))
                scores.append(0)
        if scores:
            denominator = 2 * counts["TP"] + counts["FP"] + counts["FN"]
            score = sum(scores) / len(scores) if field in TEXT_FIELDS else (2 * counts["TP"] / denominator if denominator else 0)
            fields[field] = {"score": score, "products": len(scores)}
            if field not in TEXT_FIELDS:
                fields[field].update(counts)
    return {"fields": fields, "overall": sum(f["score"] for f in fields.values()) / len(fields) if fields else None}


def load_predictions(path, schema):
    return [{field: row[field] if field == "article_id" else schema.decode(field, row.get(field, ""))
             for field in schema.columns} for row in read_csv(path)]


def hallucination_rate(predictions, references, schema):
    """Exact canonical structured-value mismatch share, including empty refs."""
    refs = {r["article_id"]: r for r in references}
    emitted = unsupported = 0
    for row in predictions:
        for field in schema.columns:
            if field == "article_id" or field in TEXT_FIELDS:
                continue
            reference = refs.get(row["article_id"], {}).get(field, schema.empty(field))
            known = {(tuple(p for p in path if not isinstance(p, int)), canonical(value))
                     for path, value in leaves(reference)}
            for path, value in leaves(row.get(field, schema.empty(field))):
                emitted += 1
                atom = (tuple(p for p in path if not isinstance(p, int)), canonical(value))
                unsupported += int(atom not in known)
    return {"unsupported": unsupported, "emitted": emitted, "rate": unsupported / emitted if emitted else 0.0}
