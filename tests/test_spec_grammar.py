"""Pin docs/spec_grammar.md + plugmem/pipelines/spec_schema.json to the
loader's actual behaviour.

These tests are deliberately small: they catch drift (new node type
added to loader but missing from schema/doc; ops added to the Compute
whitelist but missing from the schema enum; etc.) without trying to
re-validate every loader rule in JSON Schema. The loader is still the
authoritative validator for things JSON Schema can't easily express
(cycle detection, ref resolution, body scoping, etc.).
"""
from __future__ import annotations

import json
from pathlib import Path

import pytest
import yaml

jsonschema = pytest.importorskip("jsonschema")

from plugmem.pipelines.sample_specs import PLUGMEM_DEFAULT_RETRIEVE_YAML
from plugmem.pipelines.spec_driven import COMPUTE_OPS, NODE_TYPES, load_yaml_str


REPO_ROOT = Path(__file__).resolve().parent.parent
SCHEMA_PATH = REPO_ROOT / "plugmem" / "pipelines" / "spec_schema.json"
GRAMMAR_DOC_PATH = REPO_ROOT / "docs" / "spec_grammar.md"


def _load_schema() -> dict:
    return json.loads(SCHEMA_PATH.read_text())


# --------------------------------------------------------------------- #
# Schema is well-formed
# --------------------------------------------------------------------- #


def test_schema_is_valid_draft_2020_12():
    schema = _load_schema()
    # `check_schema` raises SchemaError if the schema itself is invalid.
    jsonschema.Draft202012Validator.check_schema(schema)


def test_schema_advertises_every_loader_node_type():
    schema = _load_schema()
    type_consts = {
        d["properties"]["type"]["const"]
        for d in schema["$defs"].values()
        if "properties" in d
        and "type" in d["properties"]
        and "const" in d["properties"]["type"]
    }
    assert type_consts == NODE_TYPES, (
        f"Schema node types {type_consts} drift from loader {NODE_TYPES}"
    )


def test_schema_compute_op_enum_matches_loader():
    schema = _load_schema()
    schema_ops = set(
        schema["$defs"]["ComputeNode"]["properties"]["config"]
        ["properties"]["op"]["enum"]
    )
    assert schema_ops == set(COMPUTE_OPS.keys()), (
        f"Schema Compute.op enum {schema_ops} drifts from loader "
        f"{set(COMPUTE_OPS.keys())}"
    )


# --------------------------------------------------------------------- #
# Schema accepts the shipped sample (and the loader runs it cleanly)
# --------------------------------------------------------------------- #


def test_sample_passes_both_jsonschema_and_loader():
    schema = _load_schema()
    doc = yaml.safe_load(PLUGMEM_DEFAULT_RETRIEVE_YAML)
    # JSON Schema validation.
    jsonschema.validate(instance=doc, schema=schema)
    # Loader validation.
    g = load_yaml_str(PLUGMEM_DEFAULT_RETRIEVE_YAML)
    assert g.phase == "retrieve"


def test_schema_rejects_unknown_node_type():
    schema = _load_schema()
    bad = {
        "phase": "retrieve",
        "nodes": [
            {"id": "in", "type": "Input"},
            {"id": "huh", "type": "UnknownNodeType"},
            {
                "id": "out", "type": "Output",
                "inputs": {
                    "mode": {"const": "semantic_memory"},
                    "reasoning_prompt": {"const": []},
                    "variables": {"const": {}},
                },
            },
        ],
    }
    with pytest.raises(jsonschema.ValidationError):
        jsonschema.validate(instance=bad, schema=schema)


def test_schema_rejects_unknown_compute_op():
    schema = _load_schema()
    bad = {
        "phase": "retrieve",
        "nodes": [
            {"id": "in", "type": "Input"},
            {
                "id": "bad", "type": "Compute",
                "config": {"op": "no_such_op"},
                "inputs": {"a": {"const": 1}, "b": {"const": 2}},
            },
            {
                "id": "out", "type": "Output",
                "inputs": {
                    "mode": {"const": "semantic_memory"},
                    "reasoning_prompt": {"const": []},
                    "variables": {"const": {}},
                },
            },
        ],
    }
    with pytest.raises(jsonschema.ValidationError):
        jsonschema.validate(instance=bad, schema=schema)


# --------------------------------------------------------------------- #
# Doc lists every node type + every Compute op
# --------------------------------------------------------------------- #


def test_grammar_doc_mentions_every_node_type():
    text = GRAMMAR_DOC_PATH.read_text()
    for ntype in NODE_TYPES:
        assert f"`{ntype}`" in text, (
            f"Node type {ntype!r} not mentioned in {GRAMMAR_DOC_PATH.name}"
        )


def test_grammar_doc_mentions_every_compute_op():
    text = GRAMMAR_DOC_PATH.read_text()
    for op in COMPUTE_OPS:
        assert f"`{op}`" in text, (
            f"Compute op {op!r} not mentioned in {GRAMMAR_DOC_PATH.name}"
        )
