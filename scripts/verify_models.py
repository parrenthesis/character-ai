#!/usr/bin/env python3
"""Verify that all required models are available locally."""

import logging
from pathlib import Path

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def check_files(model_type: str, base_path: Path, required: list[str]) -> bool:
    """Check if required files exist."""
    logger.info(f"\n{model_type}:")
    logger.info(f"  Path: {base_path}")

    if not base_path.exists():
        logger.error("  ❌ Directory not found")
        return False

    all_exist = True
    for filename in required:
        filepath = base_path / filename
        if filepath.exists():
            size_mb = filepath.stat().st_size / (1024 * 1024)
            logger.info(f"  ✅ {filename} ({size_mb:.1f} MB)")
        else:
            logger.error(f"  ❌ {filename} NOT FOUND")
            all_exist = False

    return all_exist


def main():
    logger.info("=" * 60)
    logger.info("Model Verification Report")
    logger.info("=" * 60)

    results = {}

    # Check STT
    results["stt"] = check_files(
        "STT (Wav2Vec2)",
        Path("models/stt/wav2vec2-base-960h"),
        ["config.json", "preprocessor_config.json", "vocab.json"],
    )

    # Check TTS
    results["tts"] = check_files(
        "TTS (XTTS v2)",
        Path("models/tts/xtts_v2"),
        ["config.json", "model.pth", "vocab.json"],
    )

    # Check LLM
    llm_dir = Path("models/llm")
    logger.info("\nLLM Models:")
    logger.info(f"  Path: {llm_dir}")
    if llm_dir.exists():
        gguf_files = list(llm_dir.glob("*.gguf"))
        if gguf_files:
            for f in gguf_files:
                size_gb = f.stat().st_size / (1024**3)
                logger.info(f"  ✅ {f.name} ({size_gb:.2f} GB)")
            results["llm"] = True
        else:
            logger.error("  ❌ No GGUF files found")
            results["llm"] = False
    else:
        logger.error("  ❌ Directory not found")
        results["llm"] = False

    # Summary
    logger.info("\n" + "=" * 60)
    all_ok = all(results.values())
    for model_type, ok in results.items():
        status = "✅ OK" if ok else "❌ MISSING"
        logger.info(f"{model_type.upper()}: {status}")

    if all_ok:
        logger.info("\n✅ All models available for offline operation")
        return 0
    else:
        logger.info("\n❌ Run 'make download-models' to download missing models")
        return 1


if __name__ == "__main__":
    exit(main())
