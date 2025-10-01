#!/usr/bin/env python3
"""
Install a prebuilt models_bundle.tar.gz onto the device, verifying checksums.
"""

import json
import tarfile
import hashlib
from pathlib import Path

BUNDLE = Path("models_bundle.tar.gz")
TARGET = Path("models")


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(8192), b""):
            h.update(chunk)
    return h.hexdigest()


def main() -> None:
    if not BUNDLE.exists():
        raise SystemExit(f"Bundle not found: {BUNDLE}")

    TARGET.mkdir(parents=True, exist_ok=True)

    with tarfile.open(BUNDLE, "r:gz") as tar:
        tar.extractall(TARGET.parent)

    manifest = json.loads((Path("manifest.json")).read_text())
    for entry in manifest["files"]:
        p = Path(entry["path"])  # already extracted to correct relative path
        if not p.exists():
            raise SystemExit(f"Missing file after extract: {p}")
        if sha256(p) != entry["sha256"]:
            raise SystemExit(f"Checksum mismatch: {p}")

    print("Model bundle installed and verified.")


if __name__ == "__main__":
    main()


