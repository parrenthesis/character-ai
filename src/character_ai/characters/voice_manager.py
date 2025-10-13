"""
Voice management system for character voice injection.

Makes it easy for manufacturers to inject character voices at runtime.
"""

import hashlib
import logging
import os
from pathlib import Path
from typing import Any, Dict, List, Optional

import librosa
import numpy as np

from ..core.config import Config
from ..core.protocols import AudioData

logger = logging.getLogger(__name__)

# Versioned algorithm fingerprint for voice embeddings. Bump to force recompute.
EMBEDDING_ALGO_ID = "librosa.mel-mean-v1"
EMBEDDING_PARAMS = {
    "n_fft": 1024,
    "hop_length": 256,
    "n_mels": 64,
    "sr": 22050,
}


class VoiceManager:
    """Manages character voice injection and storage."""

    def __init__(self, voice_storage_dir: Optional[str] = None) -> None:
        cfg = Config()
        self.voice_storage_dir = Path(voice_storage_dir or str(cfg.paths.voices_dir))
        # Don't create directories during instantiation - create when actually needed
        self.character_voices: Dict[
            str, str
        ] = {}  # character_name -> processed_voice_file_path
        self.min_duration_s: float = 1.5
        # Derived artifacts storage (embeddings) - use catalog voices directory
        self.voices_artifacts_root: Path = Path(cfg.paths.voices_dir)
        self._compute_emb_on_ingest: bool = bool(
            getattr(cfg.tts, "compute_voice_embedding_on_ingest", False)
        )
        # Don't create directories during instantiation - create when actually needed

    async def inject_character_voice(
        self,
        character_name: str,
        voice_file_path: str,
        tts_processor: Any,
        *,
        force_recompute_embedding: bool = False,
    ) -> bool:
        """Inject a character voice from audio file."""
        try:
            # Validate file exists
            if not os.path.exists(voice_file_path):
                logger.error(f"Voice file not found: {voice_file_path}")
                return False

            # Create character-specific directories
            character_dir = self.voice_storage_dir / character_name
            voice_samples_dir = character_dir / "voice_samples"
            processed_voices_dir = character_dir / "processed_voices"

            # Ensure directories exist
            voice_samples_dir.mkdir(parents=True, exist_ok=True)
            processed_voices_dir.mkdir(parents=True, exist_ok=True)

            # Copy voice file to character's voice_samples directory
            voice_filename = f"{character_name}_voice.wav"
            voice_samples_path = voice_samples_dir / voice_filename
            processed_voice_path = processed_voices_dir / voice_filename

            # Copy file to voice_samples
            import shutil

            shutil.copy2(voice_file_path, voice_samples_path)

            # Best-effort audio sanity check
            try:
                import io

                import soundfile as sf

                with open(voice_samples_path, "rb") as rf:
                    data = rf.read()
                audio_io = io.BytesIO(data)
                audio_array, sr = sf.read(audio_io)
                if hasattr(audio_array, "shape"):
                    duration = len(audio_array) / float(sr or 22050)
                    if duration < self.min_duration_s:
                        logger.error(
                            f"Voice sample too short (<{self.min_duration_s}s): "
                            f"{voice_samples_path}"
                        )
                        return False
            except Exception:
                logger.warning(
                    f"Unable to pre-validate audio format for {voice_samples_path}"
                )

            # Validate via Coqui TTS by attempting a short synthesis using the reference
            try:
                await tts_processor.inject_character_voice(
                    character_name, str(voice_samples_path), "test", "en"
                )
            except Exception as e:
                logger.error(
                    f"Coqui TTS inject_character_voice failed for {character_name}: {e}"
                )
                return False

            # Store reference to processed voice path
            self.character_voices[character_name] = str(processed_voice_path)

            # Optionally compute and store a voice embedding for faster ingest
            if self._compute_emb_on_ingest:
                try:
                    await self._maybe_compute_embedding(
                        character_name,
                        voice_samples_path,
                        force_recompute=force_recompute_embedding,
                    )
                except Exception as e:
                    logger.warning(f"Voice embedding computation skipped: {e}")

            logger.info(
                f"Voice injected for character '{character_name}' from "
                f"{voice_file_path}"
            )
            return True

        except Exception as e:
            logger.error(f"Failed to inject voice for {character_name}: {e}")
            return False

    async def get_character_voice_path(self, character_name: str) -> Optional[str]:
        """Get the voice file path for a character."""
        character_dir = self.voice_storage_dir / character_name
        processed_voices_dir = character_dir / "processed_voices"

        if processed_voices_dir.exists():
            # Look for voice files in processed_voices directory
            for voice_file in processed_voices_dir.iterdir():
                if voice_file.is_file() and voice_file.suffix.lower() in [
                    ".wav",
                    ".mp3",
                    ".flac",
                ]:
                    return str(voice_file)

        return None

    async def synthesize_with_character_voice(
        self,
        character_name: str,
        text: str,
        tts_processor: Any,
    ) -> AudioData:
        """Synthesize speech using character's injected voice and return AudioData."""
        try:
            voice_path = await self.get_character_voice_path(character_name)
            if not voice_path:
                # Fallback: synthesize without a reference voice
                result = await tts_processor.synthesize(text)
                return result.audio_data if not result.error else result  # type: ignore

            # Use the easy injection method
            result = await tts_processor.inject_character_voice(
                character_name, voice_path, text
            )
            if result.error:
                raise RuntimeError(f"Voice synthesis failed: {result.error}")
            return result.audio_data  # type: ignore

        except Exception as e:
            logger.error(f"Failed to synthesize with character voice: {e}")
            # Return an error-like object consistent with AudioData usage patterns
            return type("AudioDataError", (), {"error": str(e), "audio_data": None})()  # type: ignore

    async def list_available_voices(self) -> List[str]:
        """List all available character voices."""
        return list(self.character_voices.keys())

    async def remove_character_voice(self, character_name: str) -> bool:
        """Remove a character's voice."""
        try:
            if character_name in self.character_voices:
                # Remove processed voice file
                processed_voice_path = self.character_voices[character_name]
                if os.path.exists(processed_voice_path):
                    os.remove(processed_voice_path)

                # Remove character's entire directory (voice_samples + processed_voices)
                character_dir = self.voice_storage_dir / character_name
                if character_dir.exists():
                    import shutil

                    shutil.rmtree(character_dir)

                del self.character_voices[character_name]
                logger.info(f"Removed voice for character '{character_name}'")
                return True
            return False
        except Exception as e:
            logger.error(f"Failed to remove voice for {character_name}: {e}")
            return False

    def has_character_voice(self, character_name: str) -> bool:
        """Check if character has a voice."""
        character_dir = self.voice_storage_dir / character_name
        processed_voices_dir = character_dir / "processed_voices"
        return processed_voices_dir.exists() and any(processed_voices_dir.iterdir())

    def list_character_voices(self) -> List[str]:
        """List all character names with voices."""
        # Scan the directory structure for characters with voice data
        character_voices = []
        if self.voice_storage_dir.exists():
            for character_dir in self.voice_storage_dir.iterdir():
                if character_dir.is_dir():
                    processed_voices_dir = character_dir / "processed_voices"
                    if processed_voices_dir.exists() and any(
                        processed_voices_dir.iterdir()
                    ):
                        character_voices.append(character_dir.name)
        return character_voices

    def clear_character_voices(self) -> None:
        """Clear all character voices."""
        # Remove all character directories
        if self.voice_storage_dir.exists():
            import shutil

            for character_dir in self.voice_storage_dir.iterdir():
                if character_dir.is_dir():
                    shutil.rmtree(character_dir)
        self.character_voices.clear()

    def get_character_count(self) -> int:
        """Get number of characters with voices."""
        return len(self.list_character_voices())

    def is_empty(self) -> bool:
        """Check if no character voices are stored."""
        return len(self.list_character_voices()) == 0

    async def get_voice_info(self, character_name: str) -> Dict:
        """Get information about a character's voice."""
        voice_path = await self.get_character_voice_path(character_name)
        if not voice_path:
            return {"error": f"No voice found for character '{character_name}'"}

        try:
            file_size = os.path.getsize(voice_path)
            return {
                "character_name": character_name,
                "voice_file": voice_path,
                "file_size_bytes": file_size,
                "file_size_mb": round(file_size / (1024 * 1024), 2),
                "exists": True,
            }
        except Exception as e:
            return {"error": f"Failed to get voice info: {e}"}

    async def recompute_embedding_from_artifact(
        self, character_name: str, voice_file_path: str, *, force: bool = False
    ) -> bool:
        """Public entrypoint to recompute and persist embedding from a given artifact.

        Returns True on success, False on failure. Best-effort; logs errors.
        """
        try:
            storage_path = Path(voice_file_path)
            if not storage_path.exists():
                logger.error(
                    f"Voice artifact not found for '{character_name}': "
                    f"{voice_file_path}"
                )
                return False
            await self._maybe_compute_embedding(
                character_name, storage_path, force_recompute=force
            )
            return True
        except Exception as e:
            logger.error(f"Embedding recompute failed for '{character_name}': {e}")
            return False

    async def _maybe_compute_embedding(
        self, character_name: str, storage_path: Path, *, force_recompute: bool = False
    ) -> None:
        """Compute a light-weight voice embedding and persist to artifacts.

        This is optional and best-effort; failures are logged and ignored.
        """
        try:
            # Prepare output paths and compute file checksum to avoid redundant work
            out_dir = self.voices_artifacts_root / character_name
            out_dir.mkdir(parents=True, exist_ok=True)
            out_path = out_dir / "voice_emb.npz"
            sha_path = out_dir / "voice.sha256"

            hasher = hashlib.sha256()
            with open(storage_path, "rb") as rf:
                for chunk in iter(lambda: rf.read(8192), b""):
                    hasher.update(chunk)
            file_sha = hasher.hexdigest()

            if not force_recompute and sha_path.exists() and out_path.exists():
                try:
                    prev_sha = sha_path.read_text().strip()
                    if prev_sha == file_sha:
                        # Also check algorithm fingerprint to avoid stale embeddings
                        # when algo changes
                        try:
                            import numpy as _np

                            data = _np.load(out_path)
                            algo = str(data.get("algorithm", ""))
                            if algo == EMBEDDING_ALGO_ID:
                                logger.info(
                                    f"Voice embedding up-to-date for "
                                    f"'{character_name}', skipping"
                                )
                                return
                        except Exception as e:
                            logger.warning(
                                f"Failed to check voice embedding for "
                                f"'{character_name}': {e}"
                            )
                except Exception as e:
                    logger.warning(
                        f"Failed to load voice embedding metadata for "
                        f"'{character_name}': {e}"
                    )

            # Load audio, resample to 22050 mono
            try:
                y, sr = librosa.load(str(storage_path), sr=22050, mono=True)
                if y.size == 0:
                    return
                # Compute mel-spectrogram and take mean across time
                mel = librosa.feature.melspectrogram(
                    y=y,
                    sr=EMBEDDING_PARAMS["sr"],
                    n_fft=EMBEDDING_PARAMS["n_fft"],
                    hop_length=EMBEDDING_PARAMS["hop_length"],
                    n_mels=EMBEDDING_PARAMS["n_mels"],
                )
                mel_db = librosa.power_to_db(mel + 1e-10)
                emb = mel_db.mean(axis=1).astype(np.float32)  # 64-dim
            except Exception as e:
                logger.warning(
                    f"Failed to compute voice embedding for {character_name}: {e}"
                )
                # Create a dummy embedding to avoid blocking the system
                emb = np.zeros(EMBEDDING_PARAMS["n_mels"], dtype=np.float32)
                sr = 22050  # Set default sample rate

            # Persist under catalog/voices/<character_name>/voice_emb.npz
            np.savez_compressed(
                out_path,
                embedding=emb,
                sample_rate=sr,
                algorithm=EMBEDDING_ALGO_ID,
                algorithm_params=EMBEDDING_PARAMS,  # type: ignore
                source=str(storage_path),
            )
            # Write checksum for future change detection
            sha_path.write_text(file_sha)
            logger.info(f"Stored voice embedding for '{character_name}' at {out_path}")
        except Exception as e:
            logger.debug(f"Embedding computation failed: {e}")
