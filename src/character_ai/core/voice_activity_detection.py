"""
Voice Activity Detection (VAD) for real-time audio processing.

This module provides voice activity detection capabilities for streaming audio,
enabling the system to detect when a user is speaking and optimize processing accordingl
y.
"""

import logging
import time
from dataclasses import dataclass
from enum import Enum
from typing import Any, AsyncGenerator, Callable, Dict, Optional, Tuple

import numpy as np

logger = logging.getLogger(__name__)


class VoiceState(Enum):
    """Voice activity states."""

    SILENCE = "silence"
    SPEECH_START = "speech_start"
    SPEECH_ACTIVE = "speech_active"
    SPEECH_END = "speech_end"


@dataclass
class VADConfig:
    """Configuration for Voice Activity Detection."""

    # Audio processing parameters
    sample_rate: int = 16000
    frame_duration_ms: int = 20  # 20ms frames
    hop_length: int = 320  # 20ms at 16kHz

    # VAD thresholds
    energy_threshold: float = 0.01
    spectral_centroid_threshold: float = 1000.0
    zero_crossing_rate_threshold: float = 0.1

    # Temporal parameters
    min_speech_duration_ms: int = 200  # Minimum speech duration
    min_silence_duration_ms: int = 500  # Minimum silence duration
    speech_pause_threshold_ms: int = 1000  # Pause to consider speech ended

    # Advanced parameters
    use_spectral_features: bool = True
    use_energy_features: bool = True
    use_zero_crossing: bool = True
    smoothing_window: int = 5  # Frames to smooth over

    # Adaptive thresholds
    adaptive_threshold: bool = True
    adaptation_rate: float = 0.1
    noise_floor_samples: int = 100  # Samples to estimate noise floor


@dataclass
class VADResult:
    """Result of voice activity detection."""

    is_voice: bool
    confidence: float
    energy: float
    spectral_centroid: float
    zero_crossing_rate: float
    state: VoiceState
    timestamp: float
    frame_index: int


class VoiceActivityDetector:
    """
    Real-time Voice Activity Detection using multiple audio features.

    Combines energy, spectral centroid, and zero-crossing rate for robust
    voice activity detection in noisy environments.
    """

    def __init__(self, config: VADConfig):
        self.config = config
        self.frame_size = int(
            self.config.sample_rate * self.config.frame_duration_ms / 1000
        )
        self.hop_size = self.config.hop_length

        # State tracking
        self.current_state = VoiceState.SILENCE
        self.speech_start_time: Optional[float] = None
        self.silence_start_time: Optional[float] = None
        self.frame_index = 0

        # Adaptive thresholds
        self.adaptive_energy_threshold = self.config.energy_threshold
        self.adaptive_spectral_threshold = self.config.spectral_centroid_threshold
        self.noise_floor_energy = 0.0
        self.noise_floor_spectral = 0.0

        # Feature history for smoothing
        self.energy_history: list = []
        self.spectral_history: list = []
        self.zcr_history: list = []

        # Callbacks
        self.on_speech_start: Optional[Callable] = None
        self.on_speech_end: Optional[Callable] = None
        self.on_state_change: Optional[Callable] = None

        logger.info(f"VAD initialized with config: {config}")

    def _extract_features(self, audio_frame: np.ndarray) -> Tuple[float, float, float]:
        """Extract audio features from a frame."""

        # Energy (RMS)
        energy = np.sqrt(np.mean(audio_frame**2))

        # Spectral centroid
        if self.config.use_spectral_features and len(audio_frame) > 0:
            fft = np.fft.fft(audio_frame)
            freqs = np.fft.fftfreq(len(audio_frame), 1 / self.config.sample_rate)
            magnitude = np.abs(fft[: len(fft) // 2])
            freqs = freqs[: len(magnitude)]

            if np.sum(magnitude) > 0:
                spectral_centroid = np.sum(freqs * magnitude) / np.sum(magnitude)
            else:
                spectral_centroid = 0.0
        else:
            spectral_centroid = 0.0

        # Zero crossing rate
        if self.config.use_zero_crossing and len(audio_frame) > 1:
            zero_crossings = np.sum(np.diff(np.sign(audio_frame)) != 0)
            zero_crossing_rate = zero_crossings / (len(audio_frame) - 1)
        else:
            zero_crossing_rate = 0.0

        return energy, spectral_centroid, zero_crossing_rate

    def _update_adaptive_thresholds(self, energy: float, spectral_centroid: float) -> None:
        """Update adaptive thresholds based on noise floor estimation."""

        if self.config.adaptive_threshold:
            # Update noise floor estimates
            if self.current_state == VoiceState.SILENCE:
                self.noise_floor_energy = (
                    1 - self.config.adaptation_rate
                ) * self.noise_floor_energy + self.config.adaptation_rate * energy
                self.noise_floor_spectral = (
                    (1 - self.config.adaptation_rate) * self.noise_floor_spectral
                    + self.config.adaptation_rate * spectral_centroid
                )

                # Update adaptive thresholds
                self.adaptive_energy_threshold = max(
                    self.config.energy_threshold, self.noise_floor_energy * 2.0
                )
                self.adaptive_spectral_threshold = max(
                    self.config.spectral_centroid_threshold,
                    self.noise_floor_spectral * 0.8,
                )

    def _smooth_features(
        self, energy: float, spectral_centroid: float, zcr: float
    ) -> Tuple[float, float, float]:
        """Apply smoothing to features using moving average."""

        # Add to history
        self.energy_history.append(energy)
        self.spectral_history.append(spectral_centroid)
        self.zcr_history.append(zcr)

        # Keep only recent history
        if len(self.energy_history) > self.config.smoothing_window:
            self.energy_history.pop(0)
            self.spectral_history.pop(0)
            self.zcr_history.pop(0)

        # Calculate smoothed values
        smoothed_energy = np.mean(self.energy_history)
        smoothed_spectral = np.mean(self.spectral_history)
        smoothed_zcr = np.mean(self.zcr_history)

        return smoothed_energy, smoothed_spectral, smoothed_zcr

    def _classify_voice_activity(
        self, energy: float, spectral_centroid: float, zcr: float
    ) -> Tuple[bool, float]:
        """Classify whether the frame contains voice activity."""

        # Energy-based detection
        energy_voice = energy > self.adaptive_energy_threshold

        # Spectral centroid-based detection
        spectral_voice = spectral_centroid > self.adaptive_spectral_threshold

        # Zero crossing rate-based detection
        zcr_voice = zcr > self.config.zero_crossing_rate_threshold

        # Combine features
        voice_indicators = []
        if self.config.use_energy_features:
            voice_indicators.append(energy_voice)
        if self.config.use_spectral_features:
            voice_indicators.append(spectral_voice)
        if self.config.use_zero_crossing:
            voice_indicators.append(zcr_voice)

        # Majority vote
        is_voice = sum(voice_indicators) >= len(voice_indicators) // 2 + 1

        # Calculate confidence based on feature strength
        confidence = 0.0
        if is_voice:
            confidence = min(
                1.0,
                (
                    (energy / self.adaptive_energy_threshold)
                    + (spectral_centroid / self.adaptive_spectral_threshold)
                    + (zcr / self.config.zero_crossing_rate_threshold)
                )
                / 3.0,
            )
        else:
            confidence = 1.0 - min(
                1.0,
                (
                    (energy / self.adaptive_energy_threshold)
                    + (spectral_centroid / self.adaptive_spectral_threshold)
                    + (zcr / self.config.zero_crossing_rate_threshold)
                )
                / 3.0,
            )

        return is_voice, confidence

    def _update_state(self, is_voice: bool, timestamp: float) -> VoiceState:
        """Update the current voice activity state."""

        new_state = self.current_state

        if is_voice:
            if self.current_state == VoiceState.SILENCE:
                # Speech started
                new_state = VoiceState.SPEECH_START
                self.speech_start_time = timestamp
                self.silence_start_time = None

                if self.on_speech_start:
                    self.on_speech_start(timestamp)

            elif self.current_state == VoiceState.SPEECH_START:
                # Check if minimum speech duration reached
                if (
                    self.speech_start_time is not None and
                    timestamp - self.speech_start_time
                ) * 1000 >= self.config.min_speech_duration_ms:
                    new_state = VoiceState.SPEECH_ACTIVE

            elif self.current_state == VoiceState.SPEECH_ACTIVE:
                # Continue speech
                new_state = VoiceState.SPEECH_ACTIVE

        else:  # Not voice
            if self.current_state in [
                VoiceState.SPEECH_START,
                VoiceState.SPEECH_ACTIVE,
            ]:
                # Check if we should end speech
                if self.silence_start_time is None:
                    self.silence_start_time = timestamp
                elif (
                    timestamp - self.silence_start_time
                ) * 1000 >= self.config.speech_pause_threshold_ms:
                    new_state = VoiceState.SPEECH_END
                    self.speech_start_time = None
                    self.silence_start_time = None

                    if self.on_speech_end:
                        self.on_speech_end(timestamp)

            elif self.current_state == VoiceState.SPEECH_END:
                # Check if minimum silence duration reached
                if self.silence_start_time is None:
                    self.silence_start_time = timestamp
                elif (
                    timestamp - self.silence_start_time
                ) * 1000 >= self.config.min_silence_duration_ms:
                    new_state = VoiceState.SILENCE
                    self.silence_start_time = None

        # State change callback
        if new_state != self.current_state:
            if self.on_state_change:
                self.on_state_change(self.current_state, new_state, timestamp)
            self.current_state = new_state

        return new_state

    def process_frame(self, audio_frame: np.ndarray) -> VADResult:
        """
        Process a single audio frame for voice activity detection.

        Args:
            audio_frame: Audio samples as numpy array

        Returns:
            VADResult with detection results
        """

        timestamp = time.time()

        # Extract features
        energy, spectral_centroid, zcr = self._extract_features(audio_frame)

        # Update adaptive thresholds
        self._update_adaptive_thresholds(energy, spectral_centroid)

        # Smooth features
        smoothed_energy, smoothed_spectral, smoothed_zcr = self._smooth_features(
            energy, spectral_centroid, zcr
        )

        # Classify voice activity
        is_voice, confidence = self._classify_voice_activity(
            smoothed_energy, smoothed_spectral, smoothed_zcr
        )

        # Update state
        state = self._update_state(is_voice, timestamp)

        # Create result
        result = VADResult(
            is_voice=is_voice,
            confidence=confidence,
            energy=smoothed_energy,
            spectral_centroid=smoothed_spectral,
            zero_crossing_rate=smoothed_zcr,
            state=state,
            timestamp=timestamp,
            frame_index=self.frame_index,
        )

        self.frame_index += 1

        return result

    async def process_stream(
        self, audio_stream: AsyncGenerator[np.ndarray, None]
    ) -> AsyncGenerator[VADResult, None]:
        """
        Process a stream of audio frames for voice activity detection.

        Args:
            audio_stream: Async generator yielding audio frames

        Yields:
            VADResult for each processed frame
        """

        async for audio_frame in audio_stream:
            result = self.process_frame(audio_frame)
            yield result

    def set_callbacks(
        self,
        on_speech_start: Optional[Callable[[float], None]] = None,
        on_speech_end: Optional[Callable[[float], None]] = None,
        on_state_change: Optional[
            Callable[[VoiceState, VoiceState, float], None]
        ] = None,
    ) -> None:
        """Set callback functions for voice activity events."""

        self.on_speech_start = on_speech_start
        self.on_speech_end = on_speech_end
        self.on_state_change = on_state_change

    def reset(self) -> None:
        """Reset the VAD state."""

        self.current_state = VoiceState.SILENCE
        self.speech_start_time = None
        self.silence_start_time = None
        self.frame_index = 0

        # Reset adaptive thresholds
        self.adaptive_energy_threshold = self.config.energy_threshold
        self.adaptive_spectral_threshold = self.config.spectral_centroid_threshold
        self.noise_floor_energy = 0.0
        self.noise_floor_spectral = 0.0

        # Clear history
        self.energy_history.clear()
        self.spectral_history.clear()
        self.zcr_history.clear()

        logger.info("VAD state reset")

    def get_state(self) -> Dict[str, Any]:
        """Get current VAD state information."""

        return {
            "current_state": self.current_state.value,
            "frame_index": self.frame_index,
            "adaptive_energy_threshold": self.adaptive_energy_threshold,
            "adaptive_spectral_threshold": self.adaptive_spectral_threshold,
            "noise_floor_energy": self.noise_floor_energy,
            "noise_floor_spectral": self.noise_floor_spectral,
            "speech_start_time": self.speech_start_time,
            "silence_start_time": self.silence_start_time,
        }
