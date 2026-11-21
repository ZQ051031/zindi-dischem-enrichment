import pytest

from dischem_enrich.evaluation import cosine, bucket, structured_counts, evaluate, hallucination_rate
from dischem_enrich.schema import Schema


def test_text_similarity_and_buckets():
    assert cosine("Soap 250 ML", "Soap 250 ml") == pytest.approx(1)
    assert cosine("", "Soap") == 0
    assert bucket(0.6) == 0.5
    assert bucket(0.9) == 1


def test_structured_hand_counts():
    result = structured_counts(["Water", "water", "Salt", "500 ml"], ["Water", "250 ml"])
    assert result == {"TP": 1, "FP": 2, "FN": 1, "duplicate": 1, "acceptable": 0}
    assert structured_counts(["250 ml", "250 ml bottle"], ["250 ml"])["acceptable"] == 1
    assert structured_counts(["250 ml bottle", "500 ml"], ["250 ml"])["FP"] == 1
    assert structured_counts({"weight": "250 g"}, {"volume": "250 g"})["TP"] == 0


def test_each_reference_matched_once():
    result = structured_counts(["x", "x", "x"], ["x", "x"])
    assert result["TP"] == 2 and result["duplicate"] == 1


def test_pooled_f1_and_ignored_empty_refs():
    schema = Schema({"online_name": "", "claims": [], "never": []})
    refs = [{"article_id": "1", "online_name": "Soap", "claims": ["a", "b"]},
            {"article_id": "2", "online_name": "", "claims": ["c"]},
            {"article_id": "3", "online_name": "", "claims": []}]
    pred = [{"article_id": "1", "online_name": "Soap", "claims": ["a", "x"]},
            {"article_id": "2", "claims": ["c"]}, {"article_id": "3", "claims": ["invented"]}]
    result = evaluate(pred, refs, schema)
    assert result["fields"]["claims"]["score"] == pytest.approx(4 / 6)
    assert result["fields"]["online_name"]["score"] == 1
    assert result["overall"] == pytest.approx((1 + 4 / 6) / 2)
    assert "never" not in result["fields"]
    assert hallucination_rate(pred, refs, schema) == {"unsupported": 2, "emitted": 4, "rate": 0.5}


def test_empty_evaluation():
    assert evaluate([], [], Schema({"x": []}))["overall"] is None


def test_hallucination_proxy_counts_compatible_but_absent_values():
    result = hallucination_rate([{"article_id": "1", "x": ["250 ml bottle"]}],
                                [{"article_id": "1", "x": ["250 ml"]}], Schema({"x": []}))
    assert result == {"emitted": 1, "unsupported": 1, "rate": 1.0}
