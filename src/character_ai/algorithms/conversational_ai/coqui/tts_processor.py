"""
Core TTS processing functionality for Coqui TTS.
"""

import logging
import time
from typing import Any, AsyncGenerator, List, Optional

import numpy as np
import soundfile as sf
from pydub import AudioSegment

from ....core.protocols import AudioData, AudioResult
from .text_processor import CoquiTextProcessor

logger = logging.getLogger(__name__)


class CoquiTTSProcessor:
    """Core TTS processing functionality for Coqui TTS."""

    def __init__(self, tts_model: Any, config: Any):
        self.tts = tts_model
        self.config = config
        self.text_processor = CoquiTextProcessor()

    async def process_audio(
        self, audio: AudioData, language: Optional[str] = None
    ) -> AudioResult:
        """
        Process audio data (placeholder for future audio processing).

        Args:
            audio: Input audio data
            language: Optional language specification

        Returns:
            AudioResult with processed audio
        """
        # This is a placeholder for future audio processing functionality
        # For now, just return the input audio
        return AudioResult(
            text="",
            audio_data=audio,
            metadata={"processed": True, "language": language},
            processing_time=0.0,
        )

    async def synthesize_speech(
        self,
        text: str,
        language: str = "en",
        speaker_id: Optional[str] = None,
        speed: float = 1.0,
        voice_path: Optional[str] = None,
        **kwargs: Any,
    ) -> AudioResult:
        """
        Synthesize speech from text using Coqui TTS.

        Args:
            text: Text to synthesize
            language: Language code (default: "en")
            speaker_id: Optional speaker ID for multi-speaker models
            speed: Speech speed multiplier (default: 1.0)
            **kwargs: Additional synthesis parameters

        Returns:
            AudioResult with synthesized audio
        """
        start_time = time.time()

        try:
            if not self.config.model_loaded or self.tts is None:
                return await self._create_error_result("TTS model not loaded")

            logger.info(f"Synthesizing speech: '{text[:50]}...'")

            # Split text into sentences for better processing
            sentences = self.text_processor.split_into_sentences(text)

            if not sentences:
                return await self._create_error_result("No text to synthesize")

            # Synthesize each sentence
            audio_segments = []
            for sentence in sentences:
                if not sentence.strip():
                    continue

                # Synthesize sentence
                audio_data = await self._synthesize_sentence(
                    sentence, language, speaker_id, speed, voice_path, **kwargs
                )

                if audio_data is not None:
                    audio_segments.append(audio_data)

            if not audio_segments:
                return await self._create_error_result("Failed to synthesize any audio")

            # Combine audio segments
            combined_audio = await self._combine_audio_segments(audio_segments)

            processing_time = time.time() - start_time

            return AudioResult(
                text=text,
                audio_data=combined_audio,
                metadata={
                    "language": language,
                    "speaker_id": speaker_id,
                    "speed": speed,
                    "sentence_count": len(sentences),
                    "processing_time": processing_time,
                },
                processing_time=processing_time,
            )

        except Exception as e:
            logger.error(f"Speech synthesis failed: {e}")
            return await self._create_error_result(f"Speech synthesis failed: {e}")

    async def _synthesize_sentence(
        self,
        sentence: str,
        language: str,
        speaker_id: Optional[str],
        speed: float,
        voice_path: Optional[str] = None,
        **kwargs: Any,
    ) -> Optional[AudioData]:
        """Synthesize a single sentence."""
        try:
            import os
            import tempfile

            # Create temporary file for audio output
            with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as temp_file:
                temp_path = temp_file.name

            try:
                # Prepare synthesis parameters
                synthesis_kwargs = {
                    "text": sentence,
                    "file_path": temp_path,
                    **kwargs,
                }

                # Add speaker parameter if provided
                if speaker_id:
                    synthesis_kwargs["speaker"] = speaker_id

                # Add voice_path parameter if provided (for voice cloning)
                if voice_path:
                    synthesis_kwargs["speaker_wav"] = voice_path

                logger.debug(
                    f"CoquiTTSProcessor._synthesize_sentence: synthesis_kwargs = {synthesis_kwargs}"
                )
                logger.debug(
                    f"CoquiTTSProcessor._synthesize_sentence: voice_path = {voice_path}"
                )

                # Check if model is multilingual
                if hasattr(self.tts, "is_multi_lingual") and self.tts.is_multi_lingual:
                    synthesis_kwargs["language"] = language

                # Synthesize to temporary file
                self.tts.tts_to_file(**synthesis_kwargs)

                # Read audio data from file
                audio_data, sample_rate = sf.read(temp_path)
                logger.info(
                    f"TTS generated audio: sample_rate={sample_rate} Hz, shape={audio_data.shape}"
                )

                # Apply speed adjustment only if needed (skip when speed = 1.0 to avoid quantization artifacts)
                if speed != 1.0:
                    audio_data = await self._adjust_speed(
                        audio_data, sample_rate, speed
                    )

                # Use centralized audio_io utility for WAV encoding
                from ....core.audio_io.audio_utils import audio_data_to_wav_bytes
                from ....core.protocols import AudioData

                # Get bit depth from config
                bit_depth = getattr(self.config.tts, "audio_bit_depth", 32)

                # Create temporary AudioData with numpy array
                temp_audio_data = AudioData(
                    data=audio_data,
                    sample_rate=sample_rate,
                    duration=len(audio_data) / sample_rate,
                    channels=1 if len(audio_data.shape) == 1 else audio_data.shape[1],
                )

                # Convert to WAV bytes using configurable bit depth
                wav_bytes = audio_data_to_wav_bytes(temp_audio_data, bit_depth)

                # Return AudioData with bytes (as per protocol)
                return AudioData(
                    data=wav_bytes,
                    sample_rate=sample_rate,
                    duration=len(audio_data) / sample_rate,
                    channels=1 if len(audio_data.shape) == 1 else audio_data.shape[1],
                )

            finally:
                # Clean up temporary file
                if os.path.exists(temp_path):
                    os.unlink(temp_path)

        except Exception as e:
            import traceback

            logger.error(f"Failed to synthesize sentence: {e}")
            logger.error(f"Traceback: {traceback.format_exc()}")
            return None

    async def _adjust_speed(
        self, audio_data: np.ndarray, sample_rate: int, speed: float
    ) -> np.ndarray:
        """Adjust audio speed using pydub."""
        try:
            # Ensure audio_data is float32 and normalized
            if audio_data.dtype != np.float32:
                audio_data = audio_data.astype(np.float32)

            # Normalize to [-1, 1] range
            if np.any(audio_data > 1.0) or np.any(audio_data < -1.0):
                audio_data = audio_data / np.max(np.abs(audio_data))

            # Convert to 16-bit integer for pydub
            audio_int16 = (audio_data * 32767).astype(np.int16)

            # Convert numpy array to AudioSegment
            audio_segment = AudioSegment(
                audio_int16.tobytes(),
                frame_rate=sample_rate,
                sample_width=2,  # 16-bit = 2 bytes
                channels=1 if len(audio_data.shape) == 1 else audio_data.shape[1],
            )

            # Adjust speed
            adjusted_segment = audio_segment.speedup(playback_speed=speed)

            # Convert back to numpy array
            adjusted_data = np.array(
                adjusted_segment.get_array_of_samples(), dtype=np.float32
            )

            # Normalize back to [-1, 1] range
            adjusted_data = adjusted_data / 32767.0

            # Reshape if needed
            if len(audio_data.shape) > 1:
                adjusted_data = adjusted_data.reshape(-1, audio_data.shape[1])

            return adjusted_data

        except Exception as e:
            logger.error(f"Failed to adjust speed: {e}")
            return audio_data

    async def _combine_audio_segments(
        self, audio_segments: List[AudioData]
    ) -> AudioData:
        """Combine multiple audio segments into one."""
        if len(audio_segments) == 1:
            return audio_segments[0]

        try:
            # Get sample rate from first segment
            sample_rate = audio_segments[0].sample_rate

            # Combine all audio data
            combined_data = np.concatenate([seg.data for seg in audio_segments])

            return AudioData(
                data=combined_data,  # type: ignore
                sample_rate=sample_rate,
                duration=len(combined_data) / sample_rate,
                channels=1 if len(combined_data.shape) == 1 else combined_data.shape[1],
            )

        except Exception as e:
            logger.error(f"Failed to combine audio segments: {e}")
            return audio_segments[0]  # Return first segment as fallback

    async def synthesize_speech_stream(
        self,
        text: str,
        language: str = "en",
        speaker_id: Optional[str] = None,
        speed: float = 1.0,
        **kwargs: Any,
    ) -> AsyncGenerator[AudioData, None]:
        """
        Synthesize speech in streaming mode.

        Args:
            text: Text to synthesize
            language: Language code
            speaker_id: Optional speaker ID
            **kwargs: Additional parameters

        Yields:
            AudioData chunks as they are generated
        """
        try:
            if not self.config.model_loaded or self.tts is None:
                logger.error("TTS model not loaded")
                return

            # Split text into sentences
            sentences = self.text_processor.split_into_sentences(text)

            for sentence in sentences:
                if not sentence.strip():
                    continue

                # Synthesize sentence
                audio_data = await self._synthesize_sentence(
                    sentence, language, speaker_id, speed, **kwargs
                )

                if audio_data is not None:
                    yield audio_data

        except Exception as e:
            logger.error(f"Streaming synthesis failed: {e}")

    async def _create_error_result(self, error_message: str) -> AudioResult:
        """Helper to create an error AudioResult."""
        return AudioResult(
            text="",
            audio_data=None,
            metadata={"error": error_message},
            processing_time=0.0,
        )
