"""Audio processing utilities.

Consolidates audio processing functions used across multiple modules
to eliminate duplication and provide consistent behavior.
"""

# CRITICAL: Import torch_init FIRST to set environment variables before any torch imports
# isort: off
from .. import torch_init  # noqa: F401

# isort: on

import io
import logging
from typing import TYPE_CHECKING, Any, Optional, Tuple

import numpy as np

if TYPE_CHECKING:
    from ..protocols import AudioData

logger = logging.getLogger(__name__)

# Try to import optional dependencies
try:
    import torch
    import torchaudio

    TORCHAUDIO_AVAILABLE = True
except ImportError:
    torch = None  # type: ignore[assignment]
    torchaudio = None
    TORCHAUDIO_AVAILABLE = False

try:
    import librosa

    LIBROSA_AVAILABLE = True
except ImportError:
    librosa = None  # type: ignore[assignment]
    LIBROSA_AVAILABLE = False

try:
    import scipy.signal

    SCIPY_AVAILABLE = True
except ImportError:
    scipy = None
    SCIPY_AVAILABLE = False

try:
    import soundfile as sf

    SOUNDFILE_AVAILABLE = True
except ImportError:
    sf = None
    SOUNDFILE_AVAILABLE = False


def resample_audio(
    audio_array: np.ndarray, orig_sr: int, target_sr: int, method: str = "auto"
) -> np.ndarray:
    """
    Resample audio using best available method.

    Tries methods in order of quality:
    1. torchaudio (highest quality)
    2. librosa (good quality)
    3. scipy (basic quality)

    Args:
        audio_array: Input audio data
        orig_sr: Original sample rate
        target_sr: Target sample rate
        method: Method to use ("auto", "torchaudio", "librosa", "scipy")

    Returns:
        Resampled audio array

    Raises:
        ValueError: If no resampling method is available
    """
    if orig_sr == target_sr:
        return audio_array.copy()

    if method == "auto":
        # Try methods in order of quality
        if TORCHAUDIO_AVAILABLE:
            method = "torchaudio"
        elif LIBROSA_AVAILABLE:
            method = "librosa"
        elif SCIPY_AVAILABLE:
            method = "scipy"
        else:
            raise ValueError(
                "No resampling method available. Install torchaudio, librosa, or scipy."
            )

    try:
        if method == "torchaudio" and TORCHAUDIO_AVAILABLE:
            return _resample_torchaudio(audio_array, orig_sr, target_sr)
        elif method == "librosa" and LIBROSA_AVAILABLE:
            return _resample_librosa(audio_array, orig_sr, target_sr)
        elif method == "scipy" and SCIPY_AVAILABLE:
            return _resample_scipy(audio_array, orig_sr, target_sr)
        else:
            raise ValueError(f"Resampling method '{method}' not available")
    except Exception as e:
        logger.warning(f"Resampling with {method} failed: {e}")
        if method != "scipy" and SCIPY_AVAILABLE:
            logger.info("Falling back to scipy resampling")
            return _resample_scipy(audio_array, orig_sr, target_sr)
        else:
            logger.error("All resampling methods failed, returning original audio")
            return audio_array.copy()


def _resample_torchaudio(
    audio_array: np.ndarray, orig_sr: int, target_sr: int
) -> np.ndarray:
    """Resample using torchaudio (highest quality)."""
    # Convert to tensor
    audio_tensor = torch.from_numpy(audio_array).unsqueeze(0)

    # Resample
    resampler = torchaudio.transforms.Resample(orig_sr, target_sr)
    resampled = resampler(audio_tensor)

    return resampled.squeeze(0).numpy()  # type: ignore


def _resample_librosa(
    audio_array: np.ndarray, orig_sr: int, target_sr: int
) -> np.ndarray:
    """Resample using librosa (good quality)."""
    return librosa.resample(audio_array, orig_sr=orig_sr, target_sr=target_sr)


def _resample_scipy(
    audio_array: np.ndarray, orig_sr: int, target_sr: int
) -> np.ndarray:
    """Resample using scipy (basic quality)."""
    num_samples = int(len(audio_array) * target_sr / orig_sr)
    return scipy.signal.resample(audio_array, num_samples).astype(np.float32)  # type: ignore[no-any-return]


def validate_and_fix_audio(audio_array: np.ndarray) -> np.ndarray:
    """
    Validate audio data and fix common issues.

    - Replace NaN values with zeros
    - Ensure float32 dtype
    - Clip values to [-1.0, 1.0] range

    Args:
        audio_array: Input audio data

    Returns:
        Fixed audio array
    """
    # Make a copy to avoid modifying original
    audio = audio_array.copy()

    # Replace NaN values with zeros
    if np.any(np.isnan(audio)):
        logger.warning("Audio data contains NaN values, replacing with zeros")
        audio = np.nan_to_num(audio, nan=0.0)

    # Ensure float32 dtype
    if audio.dtype != np.float32:
        audio = audio.astype(np.float32)

    # Normalize values to valid range instead of clipping
    if np.any(np.abs(audio) > 1.0):
        max_val = np.max(np.abs(audio))
        logger.info(
            f"Audio data contains values outside [-1.0, 1.0], normalizing by factor {max_val:.3f}"
        )
        audio = audio / max_val

    return audio


class AudioNormalizer:
    """Centralized audio normalization utilities.

    Use these helpers instead of inlined normalization logic to ensure
    consistent behavior across the codebase.
    """

    @staticmethod
    def normalize_peak(audio_array: np.ndarray) -> np.ndarray:
        """Peak-normalize to [-1.0, 1.0] range without clipping."""
        if audio_array.size == 0:
            return audio_array
        audio = audio_array.astype(np.float32, copy=False)
        max_val = np.max(np.abs(audio)) if audio.size else 0.0
        if max_val > 1.0:
            return audio / max_val
        return audio

    @staticmethod
    def normalize_if_needed(audio_array: np.ndarray) -> np.ndarray:
        """Normalize only when values exceed [-1.0, 1.0]."""
        return validate_and_fix_audio(audio_array)


def decode_wav_bytes(wav_bytes: bytes) -> Tuple[np.ndarray, int]:
    """
    Decode WAV bytes to numpy array and sample rate.

    Args:
        wav_bytes: WAV file bytes

    Returns:
        Tuple of (audio_array, sample_rate)

    Raises:
        ValueError: If WAV decoding fails
    """
    if not SOUNDFILE_AVAILABLE:
        raise ValueError("soundfile not available for WAV decoding")

    try:
        wav_buffer = io.BytesIO(wav_bytes)
        audio_array, sample_rate = sf.read(wav_buffer)
        wav_buffer.close()

        # Convert to float32 if needed
        if audio_array.dtype != np.float32:
            audio_array = audio_array.astype(np.float32)

        return audio_array, sample_rate
    except Exception as e:
        raise ValueError(f"Failed to decode WAV bytes: {e}")


def convert_audio_input(audio_data: Any) -> np.ndarray:
    """
    Convert audio input from various formats to float32 numpy array.

    Consolidates common audio conversion patterns used across processors:
    - bytes/bytearray: Convert from int16 PCM to float32 [-1.0, 1.0]
    - numpy array: Convert to float32, handle multi-channel to mono

    Args:
        audio_data: Audio data (bytes, bytearray, or numpy array)

    Returns:
        1-D float32 numpy array in range [-1.0, 1.0]

    Raises:
        ValueError: If audio data type is unsupported
    """
    if isinstance(audio_data, (bytes, bytearray)):
        # Convert int16 PCM bytes to float32
        audio_array = (
            np.frombuffer(audio_data, dtype=np.int16).astype(np.float32) / 32768.0
        )
    elif isinstance(audio_data, np.ndarray):
        # Convert to float32
        audio_array = audio_data.astype(np.float32)
        # If multi-channel, average to mono
        if audio_array.ndim == 2 and audio_array.shape[1] > 1:
            audio_array = audio_array.mean(axis=1)
    else:
        raise ValueError(f"Unsupported audio data type: {type(audio_data)}")

    # Ensure 1-D contiguous array
    if audio_array.ndim > 1:
        audio_array = audio_array.reshape(-1)
    audio_array = np.ascontiguousarray(audio_array)

    return audio_array


def prepare_audio_for_playback(
    audio_data: Any, target_sr: int = 44100
) -> Tuple[np.ndarray, int]:
    """
    Prepare audio data for playback.

    - Decode if WAV bytes
    - Resample to target rate
    - Validate and fix

    Args:
        audio_data: Audio data (numpy array or WAV bytes)
        target_sr: Target sample rate

    Returns:
        Tuple of (audio_array, sample_rate)
    """
    # Handle different input types
    if isinstance(audio_data, bytes):
        # WAV bytes - decode first
        audio_array, orig_sr = decode_wav_bytes(audio_data)
    elif isinstance(audio_data, np.ndarray):
        # Assume it's already decoded
        audio_array = audio_data
        orig_sr = target_sr  # Assume it's already at target rate
    else:
        raise ValueError(f"Unsupported audio data type: {type(audio_data)}")

    # Resample if necessary
    if orig_sr != target_sr:
        logger.info(f"Resampling from {orig_sr}Hz to {target_sr}Hz")
        audio_array = resample_audio(audio_array, orig_sr, target_sr)

    # Validate and fix
    audio_array = validate_and_fix_audio(audio_array)

    return audio_array, target_sr


def concatenate_audio_chunks(
    chunks: list[Any], target_sample_rate: int = 44100
) -> Optional["AudioData"]:
    """
    Concatenate multiple audio chunks into a single AudioData object.

    Handles various audio formats and resamples to target sample rate.

    Args:
        chunks: List of audio chunks (bytes, numpy arrays, or WAV bytes)
        target_sample_rate: Target sample rate for output audio

    Returns:
        AudioData object with concatenated and resampled audio
    """
    if not chunks:
        return None

    from ..protocols import AudioData

    audio_arrays = []
    tts_sample_rate = 22050  # TTS generates at 22050 Hz

    for chunk in chunks:
        try:
            if isinstance(chunk, np.ndarray):
                # Already a numpy array - use directly
                audio_array = chunk.astype(np.float32)
            elif isinstance(chunk, bytes):
                # Try to decode as WAV first
                if chunk.startswith(b"RIFF"):
                    audio_array, chunk_sr = decode_wav_bytes(chunk)
                    if chunk_sr != tts_sample_rate:
                        audio_array = resample_audio(
                            audio_array, chunk_sr, tts_sample_rate
                        )
                else:
                    # Convert raw bytes to audio array
                    audio_array = convert_audio_input(chunk)
            else:
                logger.warning(f"Unsupported chunk type: {type(chunk)}")
                continue

            audio_arrays.append(audio_array)
        except Exception as e:
            logger.warning(f"Failed to process audio chunk: {e}")
            continue

    if not audio_arrays:
        return None

    # Concatenate all audio arrays
    combined_audio = np.concatenate(audio_arrays)

    # Resample from TTS sample rate (22050 Hz) to target sample rate
    if len(combined_audio) > 0 and tts_sample_rate != target_sample_rate:
        resampled_audio = resample_audio(
            combined_audio, tts_sample_rate, target_sample_rate
        )
        logger.info(
            f"Resampled audio from {tts_sample_rate}Hz to {target_sample_rate}Hz: {len(combined_audio)} -> {len(resampled_audio)} samples"
        )
    else:
        resampled_audio = combined_audio

    # Convert numpy array to WAV bytes

    wav_buffer = io.BytesIO()
    sf.write(
        wav_buffer, resampled_audio, target_sample_rate, format="WAV", subtype="PCM_16"
    )
    wav_bytes = wav_buffer.getvalue()
    wav_buffer.close()

    return AudioData(
        data=wav_bytes,
        sample_rate=target_sample_rate,
        duration=len(resampled_audio) / target_sample_rate,
        channels=1,
    )


def write_wav_file(
    file_path: str,
    audio_data: Any,
    sample_rate: int,
    subtype: str = "PCM_16",
    channels: int = 1,
) -> None:
    """
    Write audio data to WAV file with proper format handling.

    Handles both numpy arrays and raw bytes, ensuring compatibility
    with various audio devices and players.

    Args:
        file_path: Path to output WAV file
        audio_data: Audio data (numpy array or bytes)
        sample_rate: Sample rate in Hz
        subtype: WAV subtype (PCM_16, PCM_24, etc.)
    """
    if not SOUNDFILE_AVAILABLE:
        raise ValueError("soundfile not available for WAV writing")

    try:
        if isinstance(audio_data, bytes):
            # Check if it's already WAV format
            if audio_data.startswith(b"RIFF"):
                # Already WAV format - write directly
                with open(file_path, "wb") as f:
                    f.write(audio_data)
                logger.info(
                    f"Wrote WAV file: {file_path} (raw WAV bytes, {len(audio_data)} bytes)"
                )
                return
            else:
                # Use convert_audio_input to handle raw bytes properly
                audio_array = convert_audio_input(audio_data)
                logger.info(
                    f"Converted raw bytes to float32 array: {len(audio_array)} samples"
                )
        elif isinstance(audio_data, np.ndarray):
            # Numpy array - use directly
            audio_array = audio_data
        else:
            raise ValueError(f"Unsupported audio data type: {type(audio_data)}")

        # Normalize audio to prevent clipping
        audio_array = validate_and_fix_audio(audio_array)

        # Convert mono to stereo if needed
        if channels == 2 and len(audio_array.shape) == 1:
            audio_array = np.column_stack((audio_array, audio_array))
            logger.info("Converted mono to stereo for device compatibility")

        # Always use sf.write for numpy arrays with proper subtype
        sf.write(file_path, audio_array, sample_rate, subtype=subtype)
        logger.info(
            f"Wrote WAV file: {file_path} ({len(audio_array)} samples, {sample_rate}Hz, {subtype}, {channels}ch)"
        )

    except Exception as e:
        logger.error(f"Failed to write WAV file {file_path}: {e}")
        raise


def load_audio_file(file_path: str) -> Optional["AudioData"]:
    """
    Load audio file and return AudioData object.

    Args:
        file_path: Path to the audio file

    Returns:
        AudioData object with loaded audio data

    Raises:
        FileNotFoundError: If the file doesn't exist
        ValueError: If the file format is unsupported
    """
    if not SOUNDFILE_AVAILABLE:
        raise ValueError("soundfile not available for audio file loading")

    try:
        from ..protocols import AudioData

        # Load audio file
        audio_data, sample_rate = sf.read(file_path)

        # Convert to AudioData object
        audio_obj = AudioData(
            data=audio_data,
            sample_rate=sample_rate,
            duration=len(audio_data) / sample_rate,
            channels=1 if len(audio_data.shape) == 1 else audio_data.shape[1],
        )

        logger.info(
            f"Loaded audio file: {file_path} ({len(audio_data)} samples, {sample_rate}Hz, {audio_obj.channels}ch)"
        )
        return audio_obj

    except Exception as e:
        logger.error(f"Failed to load audio file {file_path}: {e}")
        raise
