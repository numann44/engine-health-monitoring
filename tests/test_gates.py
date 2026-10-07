from pathlib import Path

import numpy as np
import pytest

from engine_health.common import atomic_json, digest, environment, identity, source_identity
from engine_health.evaluation import study as evaluator
from engine_health.training.study import verify


def test_all_subsets_must_freeze_before_any_test_read(tmp_path, monkeypatch):
    out = tmp_path / "outputs/study-v1"
    atomic_json(out / "declaration.json", {"source": {}})
    registry = {"subsets": {"FD001": {}}, "schema": 1}
    registry["digest"] = identity(registry)
    atomic_json(out / "frozen-models.json", registry)
    monkeypatch.setattr(evaluator, "verify", lambda *args: None)
    def forbidden(*args):
        raise AssertionError("Official test must not be read before global freeze")
    monkeypatch.setattr(evaluator, "load_table", forbidden)
    monkeypatch.setattr(np, "loadtxt", forbidden)
    with pytest.raises(ValueError, match="All four"):
        evaluator.evaluate(tmp_path)


def test_source_split_environment_identity(tmp_path):
    (tmp_path / "src").mkdir()
    (tmp_path / "configs").mkdir()
    (tmp_path / "src/module.py").write_text("x = 1\n")
    (tmp_path / "requirements.lock").write_text("example==1\n")
    (tmp_path / "pyproject.toml").write_text("# test\n")
    split = tmp_path / "protocols/study_v1/splits.json"
    atomic_json(split, {"immutable": True})
    record = {"source": source_identity(tmp_path), "environment": environment(), "split_sha256": digest(split)}
    record["digest"] = identity(record)
    verify(tmp_path, record)
    (tmp_path / "src/module.py").write_text("x = 2\n")
    with pytest.raises(ValueError, match="source/environment"):
        verify(tmp_path, record)
    (tmp_path / "src/module.py").write_text("x = 1\n")
    atomic_json(split, {"immutable": False})
    with pytest.raises(ValueError, match="split"):
        verify(tmp_path, record)


def test_declared_split_has_no_engine_overlap():
    import json
    root = Path(__file__).resolve().parents[1]
    record = json.loads((root / "protocols/study_v1/splits.json").read_text())
    assert record["digest"] == identity({k: v for k, v in record.items() if k != "digest"})
    for subset, split in record["subsets"].items():
        assert not set(split["train"]) & set(split["validation"])
        assert set(b["unit"] for b in split["bank"]) == set(split["validation"])
        assert len(split["bank"]) == 4*len(split["validation"])
