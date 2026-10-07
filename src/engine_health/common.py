"""Atomic artifacts and reproducibility identities."""
import contextlib
import fcntl
import hashlib
import importlib.metadata
import json
import os
from pathlib import Path
import platform
import subprocess
import sys
import tempfile
from datetime import datetime, timezone


def now():
    return datetime.now(timezone.utc).isoformat()


def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def identity(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True).encode()).hexdigest()


def atomic_json(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, name = tempfile.mkstemp(dir=path.parent, prefix=".pending-")
    try:
        with os.fdopen(fd, "w") as stream:
            json.dump(value, stream, indent=2, sort_keys=True, allow_nan=False)
            stream.write("\n")
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(name, path)
    finally:
        if os.path.exists(name):
            os.unlink(name)


def read_json(path):
    return json.loads(Path(path).read_text())


def source_identity(root):
    root = Path(root)
    paths = list((root / "src").rglob("*.py"))
    paths += list((root / "configs").rglob("*.json"))
    paths += [root / "requirements.lock", root / "pyproject.toml"]
    return {str(p.relative_to(root)): digest(p) for p in sorted(paths)}


def environment():
    return {
        "python": sys.version,
        "platform": platform.platform(),
        "packages": {p: importlib.metadata.version(p) for p in
                     ["numpy", "pandas", "torch", "scikit-learn"]},
        "threads": {k: os.environ.get(k) for k in ["OMP_NUM_THREADS", "MKL_NUM_THREADS"]},
    }


def commit(root):
    return subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=root, text=True).strip()


@contextlib.contextmanager
def heavy_lock(path):
    """OS lease, held on an inode never unlinked; PID JSON is only diagnostic."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a+") as stream:
        try:
            fcntl.flock(stream, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError as exc:
            raise RuntimeError("Another heavy job owns the study lock") from exc
        stream.seek(0)
        stream.truncate()
        json.dump({"pid": os.getpid(), "started": now(), "command": sys.argv}, stream)
        stream.flush()
        try:
            yield
        finally:
            fcntl.flock(stream, fcntl.LOCK_UN)
