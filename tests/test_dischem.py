import json

import pytest

from dischem_enrich import dischem
from dischem_enrich.schema import Schema, leaves

FORMATS = {
    "required_columns": ["article_id", "online_name", "short_description", "long_description",
                         "ingredient_information", "storage_instructions", "variant_attributes",
                         "product_specifications", "weight", "dimensions", "nutritional_information"],
    "field_formats": {
        "online_name": {"type": "string", "empty_value": ""},
        "short_description": {"type": "string", "empty_value": ""},
        "long_description": {"type": "string", "empty_value": ""},
        "ingredient_information": {"type": "array", "empty_value": []},
        "storage_instructions": {"type": "array", "empty_value": []},
        "variant_attributes": {"type": "object", "empty_value": {"shade": [], "flavour": [], "fragrance": [], "other": []}},
        "product_specifications": {"type": "object", "empty_value": {
            "pack_size": {"value": "", "unit": ""}, "application_area": [], "packaging_type": "",
            "packaging_material": [], "other": []}},
        "weight": {"type": "object", "required_keys": ["value", "unit"], "empty_value": {"value": "", "unit": ""}},
        "dimensions": {"type": "object", "empty_value": {"height": "", "width": "", "depth": "", "unit": ""}},
        "nutritional_information": {"type": "object", "empty_value": {
            "serving_size": {"value": "", "unit": ""}, "per_100g": {}, "per_serving": {}}},
    },
}


def table(*rows):
    return "<table>" + "".join(f"<tr><td>{k}</td><td>{v}</td></tr>" for k, v in rows) + "</table>"


@pytest.fixture
def schema(tmp_path):
    path = tmp_path / "submission_format.json"
    path.write_text(json.dumps(FORMATS))
    return Schema.load(path)


def row(**cells):
    base = {"record_id": "000123", "category_group": "", "Product Name": "", "Short Description": "",
            "Product Description": "", "Nutritional Information": "", "Specifications": "",
            "T_WEIGHT_UNIT": "KG", "VOLUME": "999", "do_not_predict_fields": "[]"}
    return {**base, **cells}


def check_grounded(record, output, audit, schema):
    assert output["article_id"] == record["record_id"]
    cited = {(e["field"], tuple(e["path"])): e for e in audit["values"]}
    count = 0
    for field in schema.columns[1:]:
        for path, value in leaves(output[field]):
            count += 1
            entry = cited[(field, path)]
            for ev in entry["evidence"]:
                assert record[ev["source_column"]][ev["start"]:ev["end"]] == ev["quote"]
                assert "MASKED" not in ev["quote"].upper()
    assert count == len(audit["values"])


def test_official_schema_shapes(schema):
    assert schema.columns == FORMATS["required_columns"]
    assert schema.empty("weight") == {"value": "", "unit": ""}
    assert schema.empty("nutritional_information")["per_100g"] == {}
    assert dischem.is_dischem(row())


def test_copies_source_and_keeps_unmasked_spec_rows(schema):
    record = row(**{
        "Product Name": "Brand Cream 25cm x 25cm",
        "Specifications": table(("Weight", "[MASKED]"), ("Height", "108"), ("Width", "69"),
                                ("Packaging Type", "Box"), ("Colour", "Blue And White")),
        "Product Description": "Soft cream.\n\nStorage: Store in a cool, dry place.",
    })
    output, audit = dischem.enrich(record, schema, set())
    assert output["online_name"] == "Brand Cream 25cm x 25cm"
    assert output["weight"] == {"value": "", "unit": ""}
    assert output["dimensions"]["height"] == "108" and output["dimensions"]["unit"] == ""
    assert output["product_specifications"]["packaging_type"] == "Box"
    assert output["variant_attributes"]["shade"] == ["Blue And White"]
    assert output["storage_instructions"] == ["Store in a cool, dry place."]
    assert "Storage" not in output["long_description"]
    check_grounded(record, output, audit, schema)


def test_never_uses_catalogue_context_or_invents(schema):
    record = row(**{"Product Name": "Plain Product"})
    output, audit = dischem.enrich(record, schema, set())
    for field in schema.columns[2:]:
        assert output[field] == schema.empty(field)
    check_grounded(record, output, audit, schema)


def test_blocked_fields_stay_empty(schema):
    record = row(**{"Product Name": "Brand Oil 200ml", "Specifications": table(("Pack Size", "200ml"))})
    output, audit = dischem.enrich(record, schema, {"product_specifications"})
    assert output["product_specifications"] == schema.empty("product_specifications")
    assert any(d["field"] == "product_specifications" for d in audit["dropped"])
    check_grounded(record, output, audit, schema)


def test_nutrition_skips_masked_values(schema):
    nutrition = json.dumps({"nutrients": {"Energy (kJ)": {"per_100g": "288", "per_serving": "[MASKED]"},
                                          "Protein (g)": {"per_100g": "0.8", "per_serving": "0.9"}}})
    record = row(**{"Nutritional Information": nutrition})
    output, audit = dischem.enrich(record, schema, set())
    info = output["nutritional_information"]
    assert info["per_100g"] == {"energy_kj": 288, "protein_g": 0.8}
    assert info["per_serving"] == {"protein_g": 0.9}
    assert all(not isinstance(v, str) for v in info["per_100g"].values())
    check_grounded(record, output, audit, schema)


def test_uncited_leaf_is_refused():
    with pytest.raises(ValueError):
        dischem._reconcile({"article_id": "1", "online_name": "Invented"}, {"article_id": "1", "values": [], "dropped": []})
