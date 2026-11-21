import json
import subprocess
import sys

import pytest

from dischem_enrich.data import read_csv
from dischem_enrich.evaluation import evaluate, load_predictions
from dischem_enrich.pipeline import run, validate_submission, write_submission
from dischem_enrich.schema import Schema, leaves
from dischem_enrich.synthetic import make_synthetic


def test_writer_shapes_exclusions_and_validator(tmp_path):
    schema = Schema({"article_id": "", "online_name": "", "items": [], "specs": {"size": ""}})
    path = tmp_path / "submission.csv"
    write_submission(path, [{"article_id": "0001", "items": ["Water"]}], schema, {"0001": {"items"}})
    rows = read_csv(path)
    assert rows[0] == {"article_id": "0001", "online_name": "", "items": "", "specs": '{"size": ""}'}
    inputs = [{"record_id": "0001", "do_not_predict_fields": '["items"]'}]
    assert validate_submission(path, schema, inputs=inputs) == []
    path.write_text('article_id,online_name,items,specs\n0001,N/A,not-json,{}\n')
    assert len(validate_submission(path, schema, inputs=inputs)) >= 3


def test_synthetic_end_to_end(tmp_path):
    root = make_synthetic(tmp_path / "dataset", count=18)
    output = tmp_path / "submission.csv"
    predictions = run(root, output)
    schema = Schema.load(root / "submission_format.json")
    inputs = read_csv(root / "Test_Masked.csv")
    assert len(predictions) == 18
    assert validate_submission(output, schema, root / "SampleSubmission.csv", inputs) == []
    audits = [json.loads(line) for line in output.with_suffix(".evidence.jsonl").read_text().splitlines()]
    for row, audit, source in zip(predictions, audits, inputs):
        emitted = sum(len(list(leaves(value))) for key, value in row.items() if key != "article_id")
        assert emitted == len(audit["values"])
        for entry in audit["values"]:
            assert entry["evidence"]
            for evidence in entry["evidence"]:
                assert source[evidence["source_column"]][evidence["start"]:evidence["end"]] == evidence["quote"]
    score = evaluate(load_predictions(output, schema), load_predictions(root / "ground_truth.csv", schema), schema)
    assert 0 < score["overall"] < 1
    second = make_synthetic(tmp_path / "second", count=18)
    assert (root / "Test_Masked.csv").read_bytes() == (second / "Test_Masked.csv").read_bytes()


def test_template_join_and_empty_input(tmp_path):
    root = make_synthetic(tmp_path / "data", count=2)
    output = tmp_path / "template_predictions.csv"
    rows = run(root, output, "template_input.csv")
    refs = read_csv(root / "template.csv")
    assert {r["article_id"] for r in rows} == {r["article_id"] for r in refs}
    (root / "Test_Masked.csv").write_text("record_id,do_not_predict_fields\n")
    assert run(root, output) == []


def test_duplicate_input_rejected(tmp_path):
    root = make_synthetic(tmp_path / "data", count=2)
    (root / "Test_Masked.csv").write_text("record_id\n001\n001\n")
    with pytest.raises(ValueError, match="unique"):
        run(root, tmp_path / "out.csv")


def test_cli(tmp_path):
    root = tmp_path / "data"
    out = tmp_path / "submission.csv"
    commands = [["make-synthetic", "--out", str(root), "--count", "4"],
                ["run", "--data-dir", str(root), "--out", str(out)],
                ["evaluate", "--pred", str(out), "--ref", str(root / "ground_truth.csv")],
                ["validate-submission", "--pred", str(out), "--data-dir", str(root)]]
    for args in commands:
        result = subprocess.run([sys.executable, "-m", "dischem_enrich", *args], capture_output=True, text=True)
        assert result.returncode == 0, result.stderr


def test_local_plugin_rejects_remote():
    from dischem_enrich.plugins import LocalLLM

    with pytest.raises(ValueError, match="local"):
        LocalLLM("https://example.org/v1/chat/completions", "model")


def test_local_plugin_retry_cap(monkeypatch):
    from urllib.error import URLError
    from dischem_enrich import plugins

    calls = []

    def fail(*args, **kwargs):
        calls.append(1)
        raise URLError("offline")

    monkeypatch.setattr(plugins, "urlopen", fail)
    assert plugins.LocalLLM("http://localhost/v1/chat/completions", "open-model")("ingredients", {}) == []
    assert len(calls) == 2
