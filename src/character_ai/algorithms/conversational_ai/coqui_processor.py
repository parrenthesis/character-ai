# CRITICAL: Import torch_init FIRST to set environment variables before any torch imports
# isort: off
from ...core import torch_init  # noqa: F401

# isort: on

import io
import logging
import time
import warnings
from contextlib import redirect_stderr, redirect_stdout
from typing import Any, Dict, List, Optional

import numpy as np
import soundfile as sf
import torch
import torch.serialization
from pydub import AudioSegment
from TTS.api import TTS

from ...core.config import Config
from ...core.exceptions import ModelError
from ...core.protocols import AudioData, AudioResult, BaseAudioProcessor, ModelInfo

logger = logging.getLogger(__name__)


class CoquiProcessor(BaseAudioProcessor):
    """
    Coqui TTS processor for text-to-speech synthesis and voice cloning.

    Implements Coqui TTS for high-quality text-to-speech conversion with
    native voice cloning capabilities.
    """

    def __init__(
        self,
        config: Config,
        model_name: str = "tts_models/en/ljspeech/tacotron2-DDC",
        gpu_device: Optional[str] = None,
        use_half_precision: Optional[bool] = None,
    ):
        self.config = config
        self.model_name = model_name
        self.tts: Any = None

        # Store GPU device preference - actual detection happens in initialize()
        self.gpu_device = gpu_device
        self.device = "cpu"  # Default to CPU, will be updated in initialize()
        self.use_gpu = False
        self._use_half_precision_override = use_half_precision  # Store config override
        self.use_half_precision = False

        logger.info(
            "CoquiProcessor initialized (GPU detection deferred to initialize())"
        )
        self._initialized = False

    async def initialize(self) -> None:
        """Initialize the Coqui TTS model."""
        if self._initialized:
            logger.info("CoquiProcessor already initialized.")
            return

        # GPU device detection and configuration (moved from __init__)
        if self.gpu_device:
            self.device = self.gpu_device
            self.use_gpu = self.gpu_device == "cuda"
        else:
            # If gpu_device is None, it means hardware config disabled GPU
            # Use CPU to avoid CUDA conflicts
            self.device = "cpu"
            self.use_gpu = False

        # Enable half-precision for GPU acceleration
        # Respect explicit override from hardware config, otherwise default to GPU state
        if self._use_half_precision_override is not None:
            self.use_half_precision = self._use_half_precision_override and self.use_gpu
        else:
            self.use_half_precision = self.use_gpu

        logger.info(
            f"Initializing Coqui TTS model: {self.model_name} on {self.device} (GPU: {self.use_gpu}, Half-precision: {self.use_half_precision})"
        )
        try:
            # Suppress GPT2InferenceModel warning
            warnings.filterwarnings("ignore", message=".*GPT2InferenceModel.*")

            # Configure PyTorch to allow XTTS model loading (PyTorch 2.6+ security)
            # Note: TORCH_FORCE_NO_WEIGHTS_ONLY_LOAD=1 environment variable handles
            # PyTorch 2.8 compatibility with XTTS v2 models

            # Suppress verbose TTS library print statements
            import logging as stdlib_logging

            tts_logger = stdlib_logging.getLogger("TTS")
            tts_logger.setLevel(stdlib_logging.ERROR)

            # Initialize TTS with GPU fallback
            tts_kwargs = {"model_name": self.model_name, "progress_bar": False}

            # Initialize TTS model with output suppression
            with redirect_stdout(io.StringIO()), redirect_stderr(io.StringIO()):
                self.tts = TTS(**tts_kwargs)

            # Explicitly move to the configured device to avoid auto-detection conflicts
            if self.device == "cpu":
                self.tts.to("cpu")
                logger.info("TTS model explicitly moved to CPU")
            elif self.device == "cuda":
                try:
                    self.tts.to(self.device)
                    logger.info(f"TTS model explicitly moved to {self.device}")
                except Exception as e:
                    if "CUDA error" in str(e) or "device-side assert" in str(e):
                        import traceback

                        logger.error(f"CRITICAL: CUDA error during .to(cuda): {e}")
                        logger.error(f"Stack trace: {traceback.format_exc()}")
                        logger.warning("Falling back to CPU for TTS model")
                        self.tts.to("cpu")
                        self.device = "cpu"
                        self.use_gpu = False
                        self.use_half_precision = False
                        logger.info("TTS model moved to CPU as fallback")
                    else:
                        raise e

            logger.info(f"TTS model initialized: {self.model_name}")

            self._initialized = True
            logger.info("CoquiProcessor initialized successfully.")
        except Exception as e:
            logger.error(f"Failed to initialize CoquiProcessor: {e}")
            raise ModelError(f"Coqui TTS model initialization failed: {e}")

    async def shutdown(self) -> None:
        """Shutdown the Coqui TTS model and release resources."""
        if not self._initialized:
            logger.info("CoquiProcessor not initialized, nothing to shut down.")
            return

        logger.info("Shutting down CoquiProcessor.")
        self.tts = None
        self._initialized = False

        # Attempt to clear CUDA cache if on GPU
        if self.device == "cuda":
            torch.cuda.empty_cache()
        logger.info("CoquiProcessor shutdown complete.")

    async def process_audio(
        self, audio: AudioData, language: Optional[str] = None
    ) -> AudioResult:
        """Process text to generate speech audio."""
        if not self._initialized:
            raise RuntimeError(
                "CoquiProcessor not initialized. Call initialize() first."
            )

        time.time()
        try:
            # For TTS, we need text input, not audio input
            # This method signature is from the base class but TTS works differently
            # We'll handle this in the voice cloning service instead
            logger.warning(
                "CoquiProcessor.process_audio called with audio input - this is for TTS, not STT"
            )

            return await self._create_error_result(
                "CoquiProcessor is for TTS, not audio processing"
            )

        except Exception as e:
            logger.error(f"Error processing audio with Coqui TTS: {e}")
            return await self._create_error_result(f"Coqui TTS processing failed: {e}")

    async def synthesize_speech(
        self,
        text: str,
        voice_path: Optional[str] = None,
        language: str = "en",
        speed: float = 1.0,
    ) -> AudioResult:
        """Synthesize speech from text using Coqui TTS."""
        if not self._initialized:
            raise RuntimeError(
                "CoquiProcessor not initialized. Call initialize() first."
            )

        start_time = time.time()
        try:
            logger.info(f"Coqui TTS synthesizing: '{text[:50]}...'")

            # Determine multilingual capability safely
            is_multilingual = False
            try:
                # Some models expose `is_multi_lingual` or `is_multilingual`
                if hasattr(self.tts, "is_multi_lingual"):
                    is_multilingual = bool(getattr(self.tts, "is_multi_lingual"))
                elif hasattr(self.tts, "is_multilingual"):
                    is_multilingual = bool(getattr(self.tts, "is_multilingual"))
            except Exception:
                is_multilingual = False

            try:
                bool(getattr(self.tts, "is_multi_speaker", False))
            except Exception:
                pass

            # Build kwargs conditionally to avoid passing unsupported parameters
            tts_kwargs: Dict[str, Any] = {"text": text}

            # XTTS v2 models require speaker_wav parameter for voice cloning
            if voice_path:
                tts_kwargs["speaker_wav"] = voice_path
            else:
                logger.warning("No voice_path provided for XTTS v2 model")

            # XTTS v2 models always require a language parameter
            if is_multilingual:
                tts_kwargs["language"] = (
                    language or "en"
                )  # Default to English if not specified

            # Try to add speed control if supported by the model
            # Coqui XTTS-v2 accepts speed parameters but doesn't actually apply them
            # Force post-processing with pydub/ffmpeg for reliable speed control
            native_speed_control = False
            if speed != 1.0:
                logger.info(
                    f"Speed control requested ({speed}x) - will use pydub post-processing"
                )

            # Synthesize speech with robust GPU fallback handling
            if self.use_gpu:
                try:
                    # Try GPU synthesis first
                    # Add explicit synchronization for stability with XTTS v2
                    if torch.cuda.is_available():
                        torch.cuda.synchronize()

                    # Suppress verbose TTS library output in quiet mode
                    import builtins
                    import os

                    quiet_mode = os.getenv("CAI_QUIET_MODE") == "1"
                    original_print = builtins.print

                    if quiet_mode:
                        builtins.print = lambda *args, **kwargs: None

                    try:
                        if quiet_mode:
                            with redirect_stdout(io.StringIO()), redirect_stderr(
                                io.StringIO()
                            ):
                                if self.use_half_precision:
                                    # Enable half-precision for GPU acceleration (updated API)
                                    with torch.amp.autocast("cuda"):
                                        audio_array = self.tts.tts(**tts_kwargs)
                                    logger.info(
                                        "Used half-precision (FP16) for GPU acceleration"
                                    )
                                else:
                                    audio_array = self.tts.tts(**tts_kwargs)
                                    logger.info(
                                        "Used GPU acceleration (full precision)"
                                    )
                        else:
                            if self.use_half_precision:
                                # Enable half-precision for GPU acceleration (updated API)
                                with torch.amp.autocast("cuda"):
                                    audio_array = self.tts.tts(**tts_kwargs)
                                logger.info(
                                    "Used half-precision (FP16) for GPU acceleration"
                                )
                            else:
                                audio_array = self.tts.tts(**tts_kwargs)
                                logger.info("Used GPU acceleration (full precision)")
                    finally:
                        if quiet_mode:
                            builtins.print = original_print

                    # Synchronize after synthesis for stability
                    if torch.cuda.is_available():
                        torch.cuda.synchronize()
                        # Clear CUDA cache after each synthesis to prevent state corruption
                        torch.cuda.empty_cache()
                except Exception as e:
                    error_msg = str(e)
                    if any(
                        cuda_error in error_msg.lower()
                        for cuda_error in [
                            "cuda error",
                            "device-side assert",
                            "assertion",
                            "inf",
                            "nan",
                            "element < 0",
                        ]
                    ):
                        logger.warning(f"GPU synthesis failed with CUDA error: {e}")
                        logger.info("Falling back to CPU synthesis")

                        # Move model to CPU and update state
                        self.tts.to("cpu")
                        self.device = "cpu"
                        self.use_gpu = False
                        self.use_half_precision = False

                        # Clear CUDA cache to prevent further issues
                        if torch.cuda.is_available():
                            torch.cuda.empty_cache()
                            logger.info("Cleared CUDA cache after fallback")

                        # Retry synthesis on CPU
                        import builtins
                        import os

                        quiet_mode = os.getenv("CAI_QUIET_MODE") == "1"
                        original_print = builtins.print

                        if quiet_mode:
                            builtins.print = lambda *args, **kwargs: None

                        try:
                            if quiet_mode:
                                with redirect_stdout(io.StringIO()), redirect_stderr(
                                    io.StringIO()
                                ):
                                    audio_array = self.tts.tts(**tts_kwargs)
                            else:
                                audio_array = self.tts.tts(**tts_kwargs)
                        finally:
                            if quiet_mode:
                                builtins.print = original_print
                        logger.info("Successfully synthesized with CPU fallback")
                    else:
                        # Re-raise non-CUDA errors
                        raise e
            else:
                # CPU-only synthesis
                if "xtts" in self.model_name.lower():
                    logger.info(
                        "Using CPU synthesis for XTTS v2 (CUDA compatibility workaround)"
                    )
                else:
                    logger.info("Using CPU synthesis")
                # Suppress verbose TTS library output in quiet mode
                import builtins
                import os

                quiet_mode = os.getenv("CAI_QUIET_MODE") == "1"
                original_print = builtins.print

                if quiet_mode:
                    builtins.print = lambda *args, **kwargs: None

                try:
                    if quiet_mode:
                        with redirect_stdout(io.StringIO()), redirect_stderr(
                            io.StringIO()
                        ):
                            audio_array = self.tts.tts(**tts_kwargs)
                    else:
                        audio_array = self.tts.tts(**tts_kwargs)
                finally:
                    if quiet_mode:
                        builtins.print = original_print
            logger.info(
                f"Raw TTS output type: {type(audio_array)}, shape: {getattr(audio_array, 'shape', 'no shape')}"
            )

            # Coqui may return a list of chunks; normalize to a single numpy array
            if isinstance(audio_array, list):
                try:
                    audio_array = np.concatenate(
                        [
                            np.asarray(chunk, dtype=np.float32).reshape(-1)
                            for chunk in audio_array
                        ]
                    )
                    logger.info(
                        f"After list concatenation: shape={audio_array.shape}, dtype={audio_array.dtype}, range=[{np.min(audio_array):.6f}, {np.max(audio_array):.6f}]"
                    )
                except Exception:
                    # Fallback: take first chunk
                    audio_array = np.asarray(audio_array[0], dtype=np.float32).reshape(
                        -1
                    )
                    logger.info(
                        f"After fallback: shape={audio_array.shape}, dtype={audio_array.dtype}, range=[{np.min(audio_array):.6f}, {np.max(audio_array):.6f}]"
                    )
            else:
                logger.info(
                    f"Single array: shape={audio_array.shape}, dtype={audio_array.dtype}, range=[{np.min(audio_array):.6f}, {np.max(audio_array):.6f}]"
                )

            # Apply post-processing speed control only if native control wasn't used
            if speed != 1.0 and not native_speed_control:
                logger.info(f"Applying post-processing speed control: {speed}x")
                # Convert to float32 first
                audio_float = audio_array.astype(np.float32)
                # Use pydub for speed control (simpler, fewer dependencies)
                try:
                    # Convert numpy array to AudioSegment
                    # First, convert to int16 for pydub
                    audio_int16 = (audio_float * 32767).astype(np.int16)

                    # Create AudioSegment from numpy array
                    audio_segment = AudioSegment(
                        audio_int16.tobytes(),
                        frame_rate=22050,
                        sample_width=2,  # 16-bit = 2 bytes
                        channels=1,
                    )

                    # Apply speed change
                    if speed > 1.0:
                        # Speed up
                        new_audio = audio_segment.speedup(playback_speed=speed)
                    else:
                        # Slow down (pydub doesn't have direct slowdown, so we use frame rate manipulation)
                        new_frame_rate = int(22050 * speed)
                        new_audio = audio_segment._spawn(
                            audio_segment.raw_data,
                            overrides={"frame_rate": new_frame_rate},
                        )
                        # Set frame rate back to original for playback
                        new_audio = new_audio.set_frame_rate(22050)

                    # Convert back to numpy array
                    audio_float = (
                        np.frombuffer(new_audio.raw_data, dtype=np.int16).astype(
                            np.float32
                        )
                        / 32767.0
                    )
                    logger.info(
                        f"Successfully applied pydub speed control with rate={speed}"
                    )

                except ImportError:
                    logger.warning("pydub not available, speed control disabled")
                    audio_float = audio_array.astype(np.float32)
                except Exception as e:
                    logger.warning(
                        f"pydub speed control failed: {e}, using original audio"
                    )
                    audio_float = audio_array.astype(np.float32)
            else:
                # Convert to float32 for soundfile
                audio_float = audio_array.astype(np.float32)
                logger.info(
                    f"After speed control (none): shape={audio_float.shape}, dtype={audio_float.dtype}, range=[{np.min(audio_float):.6f}, {np.max(audio_float):.6f}]"
                )

            effective_sample_rate = 22050  # Standard Coqui TTS sample rate

            # Convert to proper WAV format using soundfile

            # Create WAV data in memory
            wav_buffer = io.BytesIO()
            logger.info(
                f"Writing to soundfile: shape={audio_float.shape}, dtype={audio_float.dtype}, range=[{np.min(audio_float):.6f}, {np.max(audio_float):.6f}]"
            )
            sf.write(
                wav_buffer,
                audio_float,
                effective_sample_rate,
                format="WAV",
                subtype="PCM_16",
            )
            wav_bytes = wav_buffer.getvalue()
            wav_buffer.close()
            logger.info(f"WAV bytes created: {len(wav_bytes)} bytes")

            # Test reading back the WAV bytes to see if they're valid
            test_buffer = io.BytesIO(wav_bytes)
            test_audio, test_sr = sf.read(test_buffer)
            test_buffer.close()

            end_time = time.time()
            processing_time = end_time - start_time

            logger.info(f"Coqui TTS synthesis completed in {processing_time:.2f}s")

            return AudioResult(
                text=text,
                audio_data=AudioData(
                    data=wav_bytes,
                    sample_rate=effective_sample_rate,  # Use effective sample rate for speed control
                    channels=1,  # Mono
                ),
                metadata={
                    "model": "Coqui TTS",
                    "language": language,
                    "voice_path": voice_path,
                    "processing_time": processing_time,
                    "sample_rate": 22050,  # Coqui TTS default
                    "speed": speed,
                },
                processing_time=processing_time,
            )

        except Exception as e:
            logger.error(f"Error synthesizing speech with Coqui TTS: {e}")
            return await self._create_error_result(f"Coqui TTS synthesis failed: {e}")

    async def clone_voice(
        self, reference_audio_path: str, text: str, output_path: str
    ) -> Dict[str, Any]:
        """Clone a voice from reference audio and synthesize text."""
        if not self._initialized:
            raise RuntimeError(
                "CoquiProcessor not initialized. Call initialize() first."
            )

        start_time = time.time()
        try:
            logger.info(
                f"Cloning voice from {reference_audio_path} for text: '{text[:50]}...'"
            )

            # Clone voice and synthesize
            audio_array = self.tts.tts(text=text, speaker_wav=reference_audio_path)

            # Save to output path

            sf.write(output_path, audio_array, 22050)

            end_time = time.time()
            processing_time = end_time - start_time

            logger.info(
                f"Voice cloning completed in {processing_time:.2f}s, saved to {output_path}"
            )

            return {
                "success": True,
                "output_path": output_path,
                "processing_time": processing_time,
                "reference_audio": reference_audio_path,
                "synthesized_text": text,
            }

        except Exception as e:
            logger.error(f"Error cloning voice with Coqui TTS: {e}")
            return {
                "success": False,
                "error": str(e),
                "processing_time": time.time() - start_time,
            }

    async def _create_error_result(self, error_message: str) -> AudioResult:
        """Helper to create an error AudioResult."""
        return AudioResult(
            text="",
            error=error_message,
            metadata={"component": "CoquiProcessor", "error": error_message},
            processing_time=0.0,
        )

    async def get_model_info(self) -> ModelInfo:
        """Get information about the loaded model."""
        if not self._initialized:
            raise RuntimeError("CoquiProcessor not initialized.")
        return ModelInfo(
            name="Coqui TTS",
            type="tts",
            size=f"Coqui TTS model: {self.model_name}",
            memory_usage="Unknown",
            precision="fp32",
            quantization="none",
            loaded_at=time.time(),
        )

    async def get_supported_languages(self) -> List[str]:
        """Get list of supported languages."""
        # Coqui TTS supports multiple languages
        return [
            "en",
            "es",
            "fr",
            "de",
            "it",
            "pt",
            "pl",
            "tr",
            "ru",
            "nl",
            "cs",
            "ar",
            "zh",
            "ja",
            "hi",
        ]

    async def get_model_capabilities(self) -> Dict[str, Any]:
        """Get model capabilities."""
        return {
            "supported_languages": await self.get_supported_languages(),
            "voice_cloning": True,
            "multilingual": True,
            "sample_rate": 22050,
            "channels": 1,
            "memory_usage_gb": await self._estimate_memory_usage(),
            "processing_speed": "fast",
            "quality": "high",
        }

    async def get_embeddings(self, audio: AudioData) -> Any:
        """Extract embeddings from audio data for voice cloning."""
        try:
            # Coqui TTS doesn't typically extract embeddings in the same way as STT models
            # This would be more relevant for voice similarity analysis
            return {
                "embeddings": [],
                "metadata": {
                    "component": "CoquiProcessor",
                    "note": "Voice embeddings not implemented",
                },
                "processing_time": 0.0,
            }
        except Exception as e:
            logger.error(f"Error extracting embeddings: {e}")
            return {
                "embeddings": [],
                "metadata": {"component": "CoquiProcessor", "error": str(e)},
                "processing_time": 0.0,
                "error": str(e),
            }

    async def _estimate_memory_usage(self) -> float:
        """Estimate memory usage of the model in GB."""
        if self.tts:
            # Estimate memory usage based on model parameters
            try:
                # Coqui TTS models vary in size, estimate based on typical usage
                return 2.0  # Typical Coqui TTS model size
            except Exception:
                return 2.0
        return 0.0
