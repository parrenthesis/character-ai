"""Voice Activity Detection session management.

Manages VAD state for real-time sessions, wrapping the existing VoiceActivityDetector
with session-specific state management and configuration.
"""

import asyncio
import logging
import time
from enum import Enum
from typing import Any, Dict, List, Optional

import numpy as np

from .voice_activity_detection import VADConfig, VoiceActivityDetector
from .wake_word_detector import WakeWordDetector

logger = logging.getLogger(__name__)


class VADSessionState(Enum):
    """VAD session states."""

    IDLE = "idle"
    WAITING_FOR_WAKE_WORD = "waiting_for_wake_word"
    SPEECH_DETECTED = "speech_detected"
    SPEECH_ACTIVE = "speech_active"
    SPEECH_ENDING = "speech_ending"
    PROCESSING = "processing"


class VADSessionManager:
    """Manages VAD state for real-time voice interaction sessions."""

    def __init__(
        self,
        vad_config: Optional[VADConfig] = None,
        wake_word_detector: Optional[WakeWordDetector] = None,
        wake_word_config: Optional[Dict[str, Any]] = None,
        character_wake_words: Optional[List[str]] = None,
        hardware_vad_settings: Optional[Dict[str, Any]] = None,
    ) -> None:
        """Initialize VAD session manager.

        Args:
            vad_config: VAD configuration, uses default if None
            wake_word_detector: Optional wake word detector (legacy)
            wake_word_config: Wake word configuration from hardware profile
            character_wake_words: Wake word phrases from character config
            hardware_vad_settings: Hardware profile VAD settings (max_silence_duration_s, etc.)
        """
        self.vad_config = vad_config or VADConfig.for_toy_interaction()
        self.vad_detector = VoiceActivityDetector(self.vad_config)
        self.hardware_vad_settings = hardware_vad_settings or {}

        # Initialize wake word detector from config if provided
        # Wake word API is experimental and has type mismatches
        if (
            wake_word_config
            and wake_word_config.get("enabled")
            and character_wake_words
        ):
            method = wake_word_config.get("method", "energy")

            if method == "energy":
                from .wake_word_detector import EnergyBasedWakeWord

                engine = EnergyBasedWakeWord(character_wake_words, threshold=0.6)  # type: ignore
                self.wake_word_detector = WakeWordDetector(engine, enabled=True)  # type: ignore
                logger.info(
                    f"Wake word detection enabled (energy-based) with phrases: {character_wake_words}"
                )
            elif method == "openwakeword":
                from .wake_word_detector import OpenWakeWordEngine

                engine = OpenWakeWordEngine(models=character_wake_words, threshold=0.6)  # type: ignore
                self.wake_word_detector = WakeWordDetector(engine, enabled=True)  # type: ignore
                logger.info(
                    f"Wake word detection enabled (OpenWakeWord) with phrases: {character_wake_words}"
                )
            else:
                logger.warning(
                    f"Unknown wake word method: {method}, disabling wake word"
                )
                self.wake_word_detector = wake_word_detector  # type: ignore
        else:
            self.wake_word_detector = wake_word_detector  # type: ignore
            if (
                wake_word_config
                and wake_word_config.get("enabled")
                and not character_wake_words
            ):
                logger.warning(
                    "Wake word enabled in hardware config but no character wake words provided"
                )

        self.wake_word_detected = (
            False
            if self.wake_word_detector and self.wake_word_detector.enabled
            else True
        )

        # Session state
        self.state = (
            VADSessionState.WAITING_FOR_WAKE_WORD
            if self.wake_word_detector and self.wake_word_detector.enabled
            else VADSessionState.IDLE
        )
        self.is_speaking = False
        self.speech_buffer: List[Any] = []
        self.silence_duration = 0.0
        self.last_speech_time = 0.0
        self.speech_start_time = 0.0

        # Session parameters - use hardware profile settings if provided, otherwise use defaults
        # Use separate thresholds for speech start (high, avoid noise) and continuation (lower, detect all speech)
        vad_threshold = self.hardware_vad_settings.get("threshold", 0.75)
        self.speech_start_threshold = (
            vad_threshold * 0.02
        )  # Convert to energy level (0.75 -> 0.015)
        self.speech_continue_threshold = (
            vad_threshold * 0.005
        )  # Lower threshold during speech
        self.speech_threshold = self.speech_start_threshold  # Will be set dynamically

        # Use hardware settings for silence threshold, fallback to VADConfig
        self.silence_threshold = self.hardware_vad_settings.get(
            "max_silence_duration_s",
            self.vad_config.min_silence_duration_ms / 1000.0
            if hasattr(self.vad_config, "min_silence_duration_ms")
            else 1.0,
        )

        self.min_speech_duration = self.hardware_vad_settings.get(
            "min_speech_duration_s", 0.3
        )
        self.max_speech_duration = 8.0  # Longer max to allow complete sentences

        logger.info(
            f"VAD session parameters: speech_start={self.speech_start_threshold}, speech_continue={self.speech_continue_threshold}, silence={self.silence_threshold}s, min_speech={self.min_speech_duration}s"
        )

        # Statistics
        self.total_speech_detections = 0
        self.total_silence_detections = 0
        self.session_start_time = time.time()

    def process_audio_chunk(self, audio_chunk: Any) -> VADSessionState:
        """Process an audio chunk and return current VAD state.

        Args:
            audio_chunk: Audio data chunk (numpy array or similar)

        Returns:
            Current VAD session state
        """
        if self.state == VADSessionState.PROCESSING:
            # Skip processing if already processing to prevent overflow
            return self.state

        # Check wake word first if enabled
        if self.wake_word_detector and self.wake_word_detector.enabled:
            if not self.wake_word_detected:
                try:
                    detected, confidence = asyncio.run(
                        self.wake_word_detector.process_audio(audio_chunk)
                    )
                    if detected:
                        self.wake_word_detected = True
                        self.state = VADSessionState.IDLE
                        logger.info(
                            f"Wake word detected (confidence: {confidence:.2f})"
                        )
                    else:
                        return (
                            VADSessionState.WAITING_FOR_WAKE_WORD
                        )  # Don't process until wake word
                except Exception as e:
                    logger.error(f"Wake word detection error: {e}")
                    return VADSessionState.WAITING_FOR_WAKE_WORD

        # Calculate audio level
        audio_level = np.max(np.abs(audio_chunk))
        current_time = time.time()

        # Use dynamic threshold: high for start, lower for continuation
        active_threshold = (
            self.speech_start_threshold
            if not self.is_speaking
            else self.speech_continue_threshold
        )

        if audio_level > active_threshold:
            # Speech detected
            if not self.is_speaking:
                logger.info(
                    f"🎤 Speech started - level: {audio_level:.6f} > threshold: {active_threshold:.6f}"
                )
                self.is_speaking = True
                self.speech_buffer = []
                self.silence_duration = 0.0
                self.speech_start_time = current_time
                self.state = VADSessionState.SPEECH_DETECTED
                self.total_speech_detections += 1
                # Switch to lower threshold for continuation
                self.speech_threshold = self.speech_continue_threshold
                # Return immediately on first detection
                self.speech_buffer.append(audio_chunk)
                self.last_speech_time = current_time
                return self.state

            self.speech_buffer.append(audio_chunk)
            self.last_speech_time = current_time

            # Check for maximum speech duration
            speech_duration = current_time - self.speech_start_time
            if speech_duration > self.max_speech_duration:
                logger.info(f"🛑 Speech too long ({speech_duration:.1f}s), forcing end")
                self.state = VADSessionState.SPEECH_ENDING
                return self.state

            self.state = VADSessionState.SPEECH_ACTIVE
            logger.debug(
                f"Speaking - level: {audio_level:.6f}, duration: {speech_duration:.1f}s"
            )

        elif self.is_speaking:
            # Still in speech mode, check for silence
            self.speech_buffer.append(audio_chunk)  # Include low-level audio

            # Update silence duration based on current time
            if audio_level < self.speech_threshold:
                # This is silence - don't update last_speech_time
                self.silence_duration = current_time - self.last_speech_time
            else:
                # This is still speech - update last_speech_time and reset silence
                self.last_speech_time = current_time
                self.silence_duration = 0.0

            # Check if we should end speech
            should_end = False
            speech_duration = current_time - self.speech_start_time

            # End speech if silence is too long
            if self.silence_duration > self.silence_threshold:
                should_end = True
                logger.info(
                    f"🛑 Speech ended - silence duration: {self.silence_duration:.2f}s > threshold: {self.silence_threshold:.2f}s, total speech: {speech_duration:.2f}s"
                )

            if should_end:
                self.state = VADSessionState.SPEECH_ENDING
                return self.state

            logger.debug(
                f"Silence accumulating: {self.silence_duration:.2f}s / {self.silence_threshold:.2f}s"
            )
        else:
            # No speech detected
            if not self.is_speaking:
                self.total_silence_detections += 1
            # Track silence duration for testing even when not speaking
            if hasattr(self, "last_speech_time") and self.last_speech_time:
                self.silence_duration = current_time - self.last_speech_time
            if self.state != VADSessionState.IDLE:
                self.state = VADSessionState.IDLE

        return self.state

    def is_speech_detected(self) -> bool:
        """Check if speech is currently detected."""
        return self.is_speaking

    def should_end_speech(self) -> bool:
        """Check if speech should be ended based on current state."""
        return self.state == VADSessionState.SPEECH_ENDING

    def get_speech_buffer(self) -> List[Any]:
        """Get the accumulated speech buffer."""
        return self.speech_buffer.copy()

    def get_combined_speech_audio(self) -> Optional[Any]:
        """Get the combined speech audio from the buffer."""
        if not self.speech_buffer:
            return None

        # Combine audio chunks
        if len(self.speech_buffer) > 1:
            combined = np.concatenate(self.speech_buffer)
        else:
            combined = self.speech_buffer[0]

        # Return combined audio without trimming - VAD already handles silence detection
        return combined

    def _trim_silence(self, audio: np.ndarray, threshold: float = 0.003) -> np.ndarray:
        """Trim leading and trailing silence from audio.

        Args:
            audio: Audio array to trim
            threshold: Energy threshold for silence detection (higher = more aggressive trimming)

        Returns:
            Trimmed audio array
        """
        # Calculate energy for each frame
        frame_length = 512
        energy = np.array(
            [
                np.sum(np.abs(audio[i : i + frame_length])) / frame_length
                for i in range(0, len(audio) - frame_length, frame_length)
            ]
        )

        # Find first and last non-silent frames with lower threshold to preserve speech
        non_silent = energy > threshold
        if not np.any(non_silent):
            # If all silent with threshold, fall back to even lower threshold
            non_silent = energy > (threshold / 3)
            if not np.any(non_silent):
                return audio  # Truly all silent, return as-is

        first_frame = np.argmax(non_silent)
        last_frame = len(non_silent) - np.argmax(non_silent[::-1]) - 1

        # Add larger padding to preserve word edges
        padding_frames = 5  # Increased from 2 to preserve beginning/end of words
        first_frame = int(max(0, first_frame - padding_frames))  # type: ignore[call-overload]
        last_frame = int(min(len(non_silent) - 1, last_frame + padding_frames))  # type: ignore[call-overload]

        # Convert frame indices to sample indices
        start_sample = first_frame * frame_length
        end_sample = (last_frame + 1) * frame_length

        return audio[start_sample:end_sample]

    def reset_session(self) -> None:
        """Reset the session for a new interaction."""
        self.state = (
            VADSessionState.WAITING_FOR_WAKE_WORD
            if self.wake_word_detector and self.wake_word_detector.enabled
            else VADSessionState.IDLE
        )
        self.is_speaking = False
        self.speech_buffer = []
        self.silence_duration = 0.0
        self.last_speech_time = 0.0
        self.speech_start_time = 0.0
        # Reset threshold back to high for next speech detection
        self.speech_threshold = self.speech_start_threshold
        if self.wake_word_detector:
            self.wake_word_detected = False
        logger.debug("VAD session reset")

    def set_processing_state(self) -> None:
        """Set the session to processing state."""
        self.state = VADSessionState.PROCESSING

    def get_session_statistics(self) -> dict:
        """Get session statistics."""
        session_duration = time.time() - self.session_start_time
        return {
            "session_duration": session_duration,
            "total_speech_detections": self.total_speech_detections,
            "total_silence_detections": self.total_silence_detections,
            "current_state": self.state.value,
            "is_speaking": self.is_speaking,
            "speech_buffer_size": len(self.speech_buffer),
            "silence_duration": self.silence_duration,
        }

    def update_config(self, vad_config: VADConfig) -> None:
        """Update VAD configuration."""
        self.vad_config = vad_config
        self.vad_detector = VoiceActivityDetector(vad_config)
        logger.info("VAD configuration updated")

    def get_audio_level(self, audio_chunk: Any) -> float:
        """Get audio level from chunk for debugging."""
        return float(np.max(np.abs(audio_chunk)))

    def get_speech_duration(self) -> float:
        """Get current speech duration if speaking."""
        if self.is_speaking:
            return time.time() - self.speech_start_time
        return 0.0

    def get_silence_duration(self) -> float:
        """Get current silence duration."""
        if self.is_speaking:
            return time.time() - self.last_speech_time
        return 0.0
