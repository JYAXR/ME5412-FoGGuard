"""Frozen training outputs cannot drift during the separate test process."""

import ast
import hashlib
import importlib
from pathlib import Path

import pytest


def test_lock_detects_changed_model_artifacts(tmp_path, monkeypatch):
    scripts = Path(__file__).resolve().parents[1] / "scripts"
    monkeypatch.syspath_prepend(str(scripts))
    module = importlib.import_module("evaluate_a1_fixed_split")
    artifact = tmp_path / "model.joblib"
    artifact.write_bytes(b"frozen model fixture")
    lock = {"files": {"model.joblib": hashlib.sha256(artifact.read_bytes()).hexdigest()}}
    module.check_lock(tmp_path, lock)
    artifact.write_bytes(b"changed model fixture")
    with pytest.raises(ValueError, match="changed"):
        module.check_lock(tmp_path, lock)


def test_test_evaluator_has_no_fitting_calls():
    path = Path(__file__).resolve().parents[1] / "scripts/evaluate_a1_fixed_split.py"
    source = ast.parse(path.read_text("utf-8"))
    forbidden = [node for node in ast.walk(source) if isinstance(node, ast.Call)
                 and isinstance(node.func, ast.Attribute) and node.func.attr in {"fit", "fit_transform", "partial_fit"}]
    assert not forbidden
