#!/usr/bin/env python3
import json
from pathlib import Path
from character_ai.characters.profile_models import export_json_schema


def main() -> int:
    out_dir = Path("schemas").resolve()
    out_dir.mkdir(parents=True, exist_ok=True)
    schemas = export_json_schema()
    (out_dir / "profile.schema.json").write_text(json.dumps(schemas["profile"], indent=2))
    (out_dir / "consent.schema.json").write_text(json.dumps(schemas["consent"], indent=2))
    (out_dir / "index.schema.json").write_text(json.dumps(schemas["index"], indent=2))
    print(f"Wrote schemas to {out_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())


