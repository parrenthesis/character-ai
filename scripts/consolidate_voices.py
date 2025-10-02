#!/usr/bin/env python3
"""
Consolidate voice storage from models/voices/ to catalog/voices/.
This script moves voice files and updates all code references.
"""

import re
import shutil
from pathlib import Path


def consolidate_voice_storage():
    """Move voice files from models/voices/ to catalog/voices/."""

    # Paths
    old_voices_dir = Path("models/voices")
    new_voices_dir = Path("catalog/voices")

    # Create new directory structure
    new_voices_dir.mkdir(parents=True, exist_ok=True)

    # Move voice files
    if old_voices_dir.exists():
        for voice_file in old_voices_dir.glob("*"):
            if voice_file.is_file():
                dest_path = new_voices_dir / voice_file.name
                shutil.move(str(voice_file), str(dest_path))
                print(f"  ✓ Moved {voice_file.name} to {dest_path}")

        # Remove old directory if empty
        try:
            old_voices_dir.rmdir()
            print(f"  ✓ Removed empty directory: {old_voices_dir}")
        except OSError:
            print(f"  ⚠️  Directory not empty: {old_voices_dir}")

    print("✅ Voice consolidation complete!")
    print(f"📁 New voice storage: {new_voices_dir}")


def update_code_references():
    """Update all code references to use new voice paths."""

    # Files to update
    files_to_update = [
        "src/character_ai/characters/voice_manager.py",
        "src/character_ai/characters/catalog_voice_manager.py",
        "src/character_ai/characters/catalog_storage.py",
        "configs/production.yaml",
        "configs/runtime.yaml",
        "Dockerfile",
        "Dockerfile.models",
    ]

    # Path mappings
    path_mappings = {
        "models/voices/": "catalog/voices/",
        "models/voices": "catalog/voices",
        "CAI_PATHS__VOICES_DIR": "CAI_PATHS__VOICES_DIR",
        "voices_dir": "voices_dir",
    }

    updated_files = []

    for file_path in files_to_update:
        if Path(file_path).exists():
            print(f"Updating {file_path}...")

            with open(file_path, "r") as f:
                content = f.read()

            original_content = content

            # Update path references
            for old_path, new_path in path_mappings.items():
                if old_path != new_path:  # Only update if different
                    content = content.replace(old_path, new_path)

            # Update specific patterns
            content = re.sub(r"models/voices", "catalog/voices", content)
            content = re.sub(
                r"CAI_PATHS__VOICES_DIR.*models/voices",
                "CAI_PATHS__VOICES_DIR=catalog/voices",
                content,
            )

            if content != original_content:
                with open(file_path, "w") as f:
                    f.write(content)
                updated_files.append(file_path)
                print(f"  ✓ Updated {file_path}")
            else:
                print(f"  - No changes needed in {file_path}")

    print(f"\n✅ Updated {len(updated_files)} files:")
    for file_path in updated_files:
        print(f"  - {file_path}")


def main():
    """Main consolidation function."""
    print("🔄 Consolidating voice storage...")

    # Step 1: Move voice files
    consolidate_voice_storage()

    # Step 2: Update code references
    print("\n🔄 Updating code references...")
    update_code_references()

    print("\n✅ Voice consolidation complete!")
    print("📁 All voice files are now in: catalog/voices/")
    print("🔧 All code references have been updated")


if __name__ == "__main__":
    main()
