#!/usr/bin/env python3
import argparse
import asyncio
from pathlib import Path
from typing import Iterable

from character_ai.characters.voice_manager import VoiceManager


async def _recompute_for_character(
    vm: VoiceManager, character: str, *, force: bool
) -> bool:
    voice_path = await vm.get_character_voice_path(character)
    if not voice_path:
        # Try conventional path under voices dir
        candidate = Path(vm.voice_storage_dir) / f"{character}_voice.wav"
        if not candidate.exists():
            print(f"[skip] no voice for {character}")
            return False
        voice_path = str(candidate)
    ok = await vm.recompute_embedding_from_artifact(character, voice_path, force=force)
    print(f"[{'ok' if ok else 'fail'}] {character}")
    return ok


async def main_async(characters: Iterable[str], force: bool) -> int:
    vm = VoiceManager()
    if characters:
        results = await asyncio.gather(
            *[_recompute_for_character(vm, c, force=force) for c in characters]
        )
        return 0 if all(results) else 1
    # all known voices
    names = vm.list_character_voices()
    results = await asyncio.gather(
        *[_recompute_for_character(vm, c, force=force) for c in names]
    )
    return 0 if all(results) else 1


def main() -> int:
    parser = argparse.ArgumentParser(description="Recompute stored voice embeddings")
    parser.add_argument(
        "characters", nargs="*", help="Character names (default: all known)"
    )
    parser.add_argument(
        "--force", action="store_true", help="Force recompute even if checksum matches"
    )
    args = parser.parse_args()
    return asyncio.run(main_async(args.characters, args.force))


if __name__ == "__main__":
    raise SystemExit(main())
