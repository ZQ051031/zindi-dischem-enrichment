import pytest

from dischem_enrich.data import Evidence
from dischem_enrich.extraction import extract, resolve
from dischem_enrich.grounding import Candidate, validate, normalise
from dischem_enrich.pipeline import enrich
from dischem_enrich.schema import Schema


def test_label_extraction_and_spans():
    text = "Ingredients: Aqua, glycerin; Usage: Apply daily; Warning: Do not swallow"
    for field, expected in [("ingredients", "Aqua, glycerin"), ("directions", "Apply daily"), ("warnings", "Do not swallow")]:
        candidate, reason = resolve(extract(field, {"supplier": text}))
        assert reason is None
        assert candidate.value == expected
        assert validate(candidate, {"supplier": text}) == (True, "grounded")


@pytest.mark.parametrize("invented", ["500 ml", "250 mg", "Organic wash", "Swallow daily", "N/A", "[MASKED]", "Wash 250 ml daily"])
def test_grounding_rejects_invention(invented):
    text = "Wash 250 ML. Do not swallow daily."
    assert not validate(Candidate(invented, [Evidence("source", 0, len(text), text)]), {"source": text})[0]


def test_forged_evidence():
    assert not validate(Candidate("Soap", [Evidence("x", 0, 4, "Soap")]), {"x": "Soda"})[0]
    assert not validate(Candidate("Soap", []), {"x": "Soap"})[0]
    assert not validate(Candidate("Soap", [Evidence("x", 0, 100, "Soap")]), {"x": "Soap"})[0]
    assert not validate(Candidate("swallow", [Evidence("x", 7, 14, "swallow")]), {"x": "Do not swallow"})[0]


def test_units_and_dedupe():
    assert normalise("  250   ML ") == "250 ml"
    candidates = extract("size", {"size": "250 ML", "description": "Pack size: 250 ml"})
    assert resolve(candidates)[0].value == "250 ml"


def test_conflicts_blank():
    schema = Schema({"online_name": "", "ingredients": [], "specifications": {"size": ""}})
    row = {"record_id": "001", "name": "Brand Soap 250 ml", "a": "Ingredients: Water; Size: 250 ml", "b": "Ingredients: Salt; Size: 500 ml"}
    output, audit = enrich(row, schema)
    assert output["ingredients"] == []
    assert output["specifications"] == {"size": ""}
    assert output["online_name"] == ""
    assert any("conflicting" in dropped["reason"] for dropped in audit["dropped"])


def test_custom_columns_and_recovery():
    schema = Schema({"composition_list": [], "custom_usage": []})
    row = {"record_id": "009", "primary": "[MASKED]", "other": "Ingredients: Water; Usage: Rinse"}
    result, audit = enrich(row, schema, aliases={"composition_list": ["Ingredients"], "custom_usage": ["Usage"]})
    assert result["composition_list"] == ["Water"]
    assert result["custom_usage"] == ["Rinse"]
    assert all(v["evidence"][0]["source_column"] == "other" for v in audit["values"])
    assert enrich(row, schema, columns=["primary"])[0]["composition_list"] == []


def test_name_pattern_and_grounding():
    row = {"record_id": "0001", "brand": "Brand", "name": "Wash", "variant": "Lemon", "size": "250 ML"}
    result, audit = enrich(row, Schema({"online_name": ""}))
    assert result["online_name"] == "Brand Wash Lemon 250 ml"
    assert len(audit["values"][0]["evidence"]) == 4


def test_plugin_is_not_trusted():
    def fake(field, source):
        return [Candidate("Cures flu", [Evidence("description", 0, 4, "Soap")])]
    output, audit = enrich({"record_id": "x", "description": "Soap"}, Schema({"claims": []}), extractor=fake)
    assert output["claims"] == []
    assert "unsupported" in audit["dropped"][0]["reason"]


def test_unknown_and_empty_abstain():
    result, _ = enrich({"record_id": "x"}, Schema({"new_field": [], "long_description": ""}))
    assert result == {"article_id": "x", "new_field": [], "long_description": ""}


def test_final_gate_checks_split_list_values():
    output, audit = enrich({"record_id": "x", "ingredients": "Water, water, N/A"}, Schema({"ingredients": []}))
    assert output["ingredients"] == ["Water"]
    assert len(audit["values"]) == 1
    assert any("forbidden" in item["reason"] for item in audit["dropped"])
