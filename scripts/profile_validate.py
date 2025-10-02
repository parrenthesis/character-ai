#!/usr/bin/env python3
import sys
from pathlib import Path

from character_ai.characters.profile_models import export_json_schema, load_profile_dir


def main() -> int:
    if len(sys.argv) < 2:
        print("Usage: profile_validate.py <character_dir> [--print-schema]")
        return 2
    char_dir = Path(sys.argv[1]).resolve()
    if len(sys.argv) > 2 and sys.argv[2] == "--print-schema":
        schemas = export_json_schema()
        import json

        print(json.dumps(schemas, indent=2))
        return 0
    try:
        info = load_profile_dir(char_dir)
        print("OK:", info["id"], "->", char_dir)
        return 0
    except Exception as e:
        print("ERROR:", e)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
