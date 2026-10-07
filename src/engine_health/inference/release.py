"""Fetch only the small, immutable model bundle selected by a public manifest."""
import hashlib
import io
from pathlib import Path
import tempfile
import urllib.request
import zipfile

from engine_health.common import digest, read_json


def release_root(project_root):
    root = Path(project_root)
    local = root / "outputs/release-v0.1.0"
    if (local / "registry.json").exists():
        manifest = read_json(root / "assets/model-registry.json")
        if digest(local / "registry.json") != manifest["registry_sha256"]:
            raise ValueError("Local registry checksum mismatch")
        return local
    manifest_path = root / "assets/model-registry.json"
    if not manifest_path.exists():
        raise FileNotFoundError("The declared study is still running; no models have been published")
    manifest = read_json(manifest_path)
    cache = Path(tempfile.gettempdir()) / "engine-health" / manifest["archive_sha256"]
    if (cache / "registry.json").exists():
        if digest(cache / "registry.json") != manifest["registry_sha256"]:
            raise ValueError("Cached registry checksum mismatch")
        return cache
    with urllib.request.urlopen(manifest["url"], timeout=90) as r:
        data = r.read(200_000_001)
    if len(data) > 200_000_000 or hashlib.sha256(data).hexdigest() != manifest["archive_sha256"]:
        raise ValueError("Release archive identity mismatch")
    with zipfile.ZipFile(io.BytesIO(data)) as z:
        if sum(i.file_size for i in z.infolist()) > 500_000_000:
            raise ValueError("Expanded bundle exceeds limit")
        for name in z.namelist():
            path = Path(name)
            if path.is_absolute() or ".." in path.parts:
                raise ValueError("Invalid bundle path")
        cache.mkdir(parents=True, exist_ok=True)
        # Registry written last so an interrupted download cannot look complete.
        for name in z.namelist():
            if name == "registry.json":
                continue
            path = cache / name
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_bytes(z.read(name))
        registry = z.read("registry.json")
        if hashlib.sha256(registry).hexdigest() != manifest["registry_sha256"]:
            raise ValueError("Release registry identity mismatch")
        (cache / "registry.json").write_bytes(registry)
    return cache
