#!/usr/bin/env python3
"""Download and cache STT/TTS models locally for offline operation."""

import logging
from pathlib import Path

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def download_stt_models() -> bool:
    """Download Wav2Vec2 models to local directory."""
    try:
        from transformers import Wav2Vec2ForCTC, Wav2Vec2Processor

        models_dir = Path("models/stt")
        models_dir.mkdir(parents=True, exist_ok=True)

        model_name = "facebook/wav2vec2-base-960h"
        target_dir = models_dir / "wav2vec2-base-960h"

        if target_dir.exists() and (target_dir / "config.json").exists():
            logger.info(f"✅ STT model already exists at {target_dir}")
            return True

        logger.info(f"Downloading {model_name}...")
        processor = Wav2Vec2Processor.from_pretrained(model_name)
        model = Wav2Vec2ForCTC.from_pretrained(model_name)

        processor.save_pretrained(target_dir)
        model.save_pretrained(target_dir)

        logger.info(f"✅ STT model saved to {target_dir}")
        return True

    except Exception as e:
        logger.error(f"❌ Failed to download STT models: {e}")
        return False


def download_tts_models() -> bool:
    """Download Coqui TTS models to local directory."""
    try:
        import os

        from TTS.api import TTS

        target_dir = Path("models/tts/xtts_v2")

        if target_dir.exists() and (target_dir / "config.json").exists():
            logger.info(f"✅ TTS model already exists at {target_dir}")
            return True

        logger.info("Downloading XTTS v2 model...")
        target_dir.mkdir(parents=True, exist_ok=True)

        # Set env var for PyTorch 2.8 compatibility
        os.environ["TORCH_FORCE_NO_WEIGHTS_ONLY_LOAD"] = "1"

        # Download model - TTS library will cache to ~/.local/share/tts
        model_name = "tts_models/multilingual/multi-dataset/xtts_v2"
        TTS(model_name=model_name, progress_bar=True)

        # Copy from cache to our local directory
        import shutil

        cache_dir = Path.home() / ".local/share/tts" / model_name.replace("/", "--")

        if cache_dir.exists():
            logger.info(f"Copying model from cache to {target_dir}...")
            for item in cache_dir.iterdir():
                if item.is_file():
                    shutil.copy2(item, target_dir / item.name)
            logger.info(f"✅ TTS model saved to {target_dir}")
            return True
        else:
            logger.error(f"❌ Cache directory not found: {cache_dir}")
            return False

    except Exception as e:
        logger.error(f"❌ Failed to download TTS models: {e}")
        return False


def verify_models() -> bool:
    """Verify that all required models are present."""
    required = {
        "STT": Path("models/stt/wav2vec2-base-960h"),
        "TTS": Path("models/tts/xtts_v2"),
        "LLM": Path("models/llm"),
    }

    all_ok = True
    for model_type, path in required.items():
        if path.exists():
            logger.info(f"✅ {model_type}: {path}")
        else:
            logger.error(f"❌ {model_type} NOT FOUND: {path}")
            all_ok = False

    return all_ok


def main():
    logger.info("=" * 60)
    logger.info("Model Download & Verification")
    logger.info("=" * 60)

    stt_ok = download_stt_models()
    tts_ok = download_tts_models()

    logger.info("\nVerifying models...")
    all_ok = verify_models()

    if stt_ok and tts_ok and all_ok:
        logger.info("\n✅ All models ready for offline operation")
        return 0
    else:
        logger.error("\n❌ Some models are missing")
        return 1


if __name__ == "__main__":
    exit(main())
