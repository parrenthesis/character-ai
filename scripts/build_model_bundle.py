#!/usr/bin/env python3
"""
Build an offline model bundle with manifest and checksums.

Outputs models_bundle.tar.gz with manifest.json describing included files.
"""

import hashlib
import json
import tarfile
from pathlib import Path

BUNDLE_NAME = "models_bundle.tar.gz"
INCLUDE_DIRS = [
    Path("models/llm"),
    Path("models/stt"),
    Path("models/tts"),
]


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(8192), b""):
            h.update(chunk)
    return h.hexdigest()


def build_manifest(root: Path) -> dict:
    files = []
    for d in INCLUDE_DIRS:
        full = root / d
        if not full.exists():
            continue
        for p in full.rglob("*"):
            if p.is_file():
                files.append(
                    {
                        "path": str(p.relative_to(root)),
                        "size": p.stat().st_size,
                        "sha256": sha256(p),
                    }
                )
    return {"files": files}


def main() -> None:
    root = Path.cwd()
    manifest = build_manifest(root)
    manifest_path = root / "manifest.json"
    manifest_path.write_text(json.dumps(manifest, indent=2))

    with tarfile.open(BUNDLE_NAME, "w:gz") as tar:
        tar.add(manifest_path, arcname="manifest.json")
        for entry in manifest["files"]:
            tar.add(root / entry["path"], arcname=entry["path"])

    print(f"Created {BUNDLE_NAME} with {len(manifest['files'])} files")


if __name__ == "__main__":
    main()
