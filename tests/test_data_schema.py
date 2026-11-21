import json

import pytest

from dischem_enrich.data import read_csv, write_csv, missing, sources, excluded, image_mapping, conventions
from dischem_enrich.schema import Schema


def test_csv_preserves_ids_and_quotes(tmp_path):
    path = tmp_path / "input.csv"
    rows = [{"record_id": "00001", "source": 'A, B "C"\nnext'}, {"record_id": "01E5", "source": ""}]
    write_csv(path, rows[0].keys(), rows)
    assert read_csv(path) == rows


@pytest.mark.parametrize("value", ["", "  ", None, "[MASKED]", "prefix [masked] suffix"])
def test_missing(value):
    assert missing(value)


def test_sources_and_masks():
    row = {"record_id": "001", "foo": "[MASKED]", "bar": "Water", "do_not_predict_fields": '[]'}
    assert sources(row) == {"bar": "Water"}
    assert sources(row, ["foo"]) == {}
    assert sources(row, [], lambda path: "Label", ["image.png"]) == {"ocr:image.png": "Label"}


def test_exclusion_validation():
    assert excluded({"do_not_predict_fields": '["x"]'}) == {"x"}
    with pytest.raises(ValueError):
        excluded({"do_not_predict_fields": '{}'})


@pytest.mark.parametrize("raw", [
    {"article_id": "", "online_name": "", "custom": [], "object": {"x": ""}},
    {"columns": [{"name": "article_id", "type": "string"}, {"name": "online_name", "empty_value": ""}, {"name": "custom", "type": "array"}, {"name": "object", "type": "object", "properties": {"x": {"type": "string"}}}]},
    {"fields": {"article_id": "", "online_name": "", "custom": {"empty": "[]"}, "object": {"empty_value": {"x": ""}}}},
    {"columns": ["article_id", "online_name", "custom", "object"], "empty_values": {"custom": [], "object": {"x": ""}}},
])
def test_schema_shapes(tmp_path, raw):
    path = tmp_path / "schema.json"
    path.write_text(json.dumps(raw))
    schema = Schema.load(path)
    assert schema.columns == ["article_id", "online_name", "custom", "object"]
    assert schema.empty("object") == {"x": ""}
    assert schema.decode("custom", "") == []
    copied = schema.empty("object")
    copied["x"] = "changed"
    assert schema.empty("object") == {"x": ""}


def test_schema_invalid(tmp_path):
    path = tmp_path / "schema.json"
    path.write_text('{"columns":[{"type":"array"}]}')
    with pytest.raises(ValueError, match="name"):
        Schema.load(path)


def test_xlsx_multiple_images(tmp_path):
    from openpyxl import Workbook

    book = Workbook()
    sheet = book.active
    sheet.append(["record_id", "img_filename"])
    sheet.append(["001", "a.png"])
    sheet.append(["001", "b.png"])
    sheet.append(["002", ""])
    path = tmp_path / "mapping.xlsx"
    book.save(path)
    assert image_mapping(path, tmp_path) == {"001": [tmp_path / "a.png", tmp_path / "b.png"]}
    sheet.append(["003", "../escape.png"])
    book.save(path)
    with pytest.raises(ValueError, match="Unsafe"):
        image_mapping(path, tmp_path)


def test_conventions_are_not_record_index():
    result = conventions([{"record_id": "0001", "brand": "Cape Care", "name": "Vitamin C (VC) 250 ml"}])
    assert result["vocabulary"] == {"Cape Care": 1}
    assert "0001" not in str(result)
    assert result["observed_expansions"]
