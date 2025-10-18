"""Real audio implementations using sounddevice for hardware interaction.

These implementations provide actual microphone and speaker functionality
when hardware is available.
"""

import asyncio
import logging
from typing import Any, AsyncGenerator, List, Optional

import numpy as np
import sounddevice as sd

from .interfaces import AudioCapture, AudioDevice, AudioDeviceManager, AudioOutput

logger = logging.getLogger(__name__)


class RealAudioDevice:
    """Real audio device wrapper."""

    def __init__(self, device_info: dict, is_input: bool = True) -> None:
        self.name = device_info.get("name", "Unknown Device")
        self.index = device_info.get("index", -1)
        self.channels = device_info.get(
            "max_input_channels" if is_input else "max_output_channels", 1
        )
        self.sample_rate = 16000 if is_input else 22050  # Default rates
        self.is_input = is_input
        self.is_output = not is_input


class RealAudioDeviceManager(AudioDeviceManager):
    """Real device manager using sounddevice."""

    async def list_input_devices(self) -> List[AudioDevice]:
        """List real input devices."""
        try:
            devices = sd.query_devices()
            input_devices: List[AudioDevice] = []

            for i, device in enumerate(devices):
                if device["max_input_channels"] > 0:
                    input_devices.append(RealAudioDevice(device, is_input=True))

            return input_devices
        except Exception as e:
            logger.error(f"Failed to list input devices: {e}")
            return []

    async def list_output_devices(self) -> List[AudioDevice]:
        """List real output devices."""
        try:
            devices = sd.query_devices()
            output_devices: List[AudioDevice] = []

            for i, device in enumerate(devices):
                if device["max_output_channels"] > 0:
                    output_devices.append(RealAudioDevice(device, is_input=False))

            return output_devices
        except Exception as e:
            logger.error(f"Failed to list output devices: {e}")
            return []

    async def get_default_input_device(self) -> Optional[AudioDevice]:
        """Get default input device."""
        try:
            default_idx = sd.default.device[0]  # Input device
            if default_idx is not None:
                device_info = sd.query_devices(default_idx)
                return RealAudioDevice(device_info, is_input=True)
        except Exception as e:
            logger.error(f"Failed to get default input device: {e}")
        return None

    async def get_default_output_device(self) -> Optional[AudioDevice]:
        """Get default output device."""
        try:
            default_idx = sd.default.device[1]  # Output device
            if default_idx is not None:
                device_info = sd.query_devices(default_idx)
                return RealAudioDevice(device_info, is_input=False)
        except Exception as e:
            logger.error(f"Failed to get default output device: {e}")
        return None

    async def get_device_by_name(self, name: str) -> Optional[AudioDevice]:
        """Find device by name."""
        try:
            devices = sd.query_devices()
            for device in devices:
                if name.lower() in device["name"].lower():
                    if device["max_input_channels"] > 0:
                        return RealAudioDevice(device, is_input=True)
                    elif device["max_output_channels"] > 0:
                        return RealAudioDevice(device, is_input=False)
        except Exception as e:
            logger.error(f"Failed to find device by name '{name}': {e}")
        return None

    async def get_device_by_alsa_id(self, alsa_id: str) -> Optional[AudioDevice]:
        """Get device by ALSA hardware ID (e.g., 'hw:3,0')"""
        try:
            device_info = sd.query_devices(alsa_id)
            return RealAudioDevice(device_info, is_input=True)
        except Exception as e:
            logger.debug(f"ALSA device {alsa_id} not found: {e}")
            return None

    async def get_device_by_pattern(
        self, pattern: str, is_input: bool = True
    ) -> Optional[AudioDevice]:
        """Find device by name pattern (case-insensitive substring match)"""
        devices = (
            await self.list_input_devices()
            if is_input
            else await self.list_output_devices()
        )
        for device in devices:
            if pattern.lower() in device.name.lower():
                return device
        return None


class RealAudioCapture(AudioCapture):
    """Real audio capture using sounddevice."""

    def __init__(self) -> None:
        self._is_capturing = False
        self._stream: Optional[sd.InputStream] = None
        self._audio_queue: asyncio.Queue = asyncio.Queue()
        self._sample_rate = 16000
        self._channels = 1
        self._chunk_size = 1024

    def _audio_callback(
        self, indata: np.ndarray, frames: int, time: Any, status: Any
    ) -> None:
        """Callback for audio input stream."""
        if status:
            logger.warning(f"Audio input status: {status}")

        # Convert to mono if needed
        if len(indata.shape) > 1:
            indata = indata[:, 0]

        # Add to queue (non-blocking)
        try:
            self._audio_queue.put_nowait(indata.copy())
        except asyncio.QueueFull:
            logger.debug(
                "Audio queue full, dropping frame"
            )  # Changed to debug to reduce noise

    async def start_capture(
        self,
        device: AudioDevice,
        sample_rate: int = 16000,
        channels: int = 1,
        chunk_size: int = 1024,
    ) -> None:
        """Start real audio capture."""
        if self._is_capturing:
            await self.stop_capture()

        self._sample_rate = sample_rate
        self._channels = channels
        self._chunk_size = chunk_size
        self._audio_queue = asyncio.Queue(
            maxsize=100
        )  # Smaller buffer to prevent overflow

        try:
            # Try callback-based stream first
            try:
                self._stream = sd.InputStream(
                    device=device.index,
                    channels=channels,
                    samplerate=sample_rate,
                    blocksize=chunk_size,
                    callback=self._audio_callback,
                    dtype=np.float32,
                )
                self._stream.start()
                self._is_capturing = True
                logger.info(
                    f"Started real audio capture with callback from device {device.name}"
                )
            except Exception as callback_error:
                logger.warning(f"Callback-based capture failed: {callback_error}")
                logger.info("Falling back to non-callback approach...")

                # Fallback to non-callback approach
                self._stream = sd.InputStream(
                    device=device.index,
                    channels=channels,
                    samplerate=sample_rate,
                    blocksize=chunk_size,
                    dtype=np.float32,
                )
                self._stream.start()
                self._is_capturing = True
                logger.info(
                    f"Started real audio capture without callback from device {device.name}"
                )

            logger.info(
                f"Audio capture started: sample_rate={sample_rate}, channels={channels}"
            )
        except Exception as e:
            logger.error(f"Failed to start audio capture: {e}")
            raise

    async def stop_capture(self) -> None:
        """Stop real audio capture."""
        if self._stream:
            self._stream.stop()
            self._stream.close()
            self._stream = None

        self._is_capturing = False
        logger.info("Stopped real audio capture")

    async def read_audio_chunk(self) -> Optional[np.ndarray]:
        """Read next audio chunk."""
        if not self._is_capturing:
            return None

        try:
            # Wait for audio data with timeout
            chunk = await asyncio.wait_for(self._audio_queue.get(), timeout=1.0)
            return chunk  # type: ignore
        except asyncio.TimeoutError:
            return None
        except Exception as e:
            logger.error(f"Error reading audio chunk: {e}")
            return None

    async def capture_stream(  # type: ignore[override]
        self, sample_rate: int = 22050, channels: int = 1
    ) -> AsyncGenerator[np.ndarray, None]:
        """Stream audio chunks in real-time."""
        while self._is_capturing:
            chunk = await self.read_audio_chunk()
            if chunk is not None:
                yield chunk
            else:
                # If no chunk from queue, try reading directly from stream (non-callback mode)
                if self._stream and not self._stream.callback:
                    try:
                        # Read directly from stream
                        chunk, overflowed = self._stream.read(self._chunk_size)
                        if overflowed:
                            logger.warning("Audio input overflowed")
                        if len(chunk) > 0:
                            # Convert to mono if needed
                            if len(chunk.shape) > 1:
                                chunk = chunk[:, 0]
                            yield chunk
                        else:
                            await asyncio.sleep(0.01)
                    except Exception as e:
                        logger.error(f"Error reading from stream: {e}")
                        await asyncio.sleep(0.01)
                else:
                    await asyncio.sleep(0.01)  # Small delay if no data

    def is_capturing(self) -> bool:
        """Check if capturing."""
        return self._is_capturing


class RealAudioOutput(AudioOutput):
    """Real audio output using sounddevice."""

    def __init__(self) -> None:
        self._is_playing = False
        self._stream: Optional[sd.OutputStream] = None
        self._sample_rate = 22050
        self._channels = 1

    def _audio_callback(
        self, outdata: np.ndarray, frames: int, time: Any, status: Any
    ) -> None:
        """Callback for audio output stream."""
        if status:
            logger.warning(f"Audio output status: {status}")

        # Fill with silence (will be overridden by write_audio_chunk)
        outdata.fill(0)

    async def start_playback(
        self, device: AudioDevice, sample_rate: int = 22050, channels: int = 1
    ) -> None:
        """Start real audio playback."""
        if self._is_playing:
            await self.stop_playback()

        self._sample_rate = sample_rate
        self._channels = channels

        try:
            # Use non-callback stream for direct audio writing
            self._stream = sd.OutputStream(
                device=device.index,
                channels=channels,
                samplerate=sample_rate,
                dtype=np.float32,
            )

            self._stream.start()
            self._is_playing = True

            logger.info(
                f"Started real audio playback to device {device.name}, "
                f"sample_rate={sample_rate}, channels={channels}"
            )
        except Exception as e:
            logger.error(f"Failed to start audio playback: {e}")
            raise

    async def stop_playback(self) -> None:
        """Stop real audio playback."""
        if self._stream:
            self._stream.stop()
            self._stream.close()
            self._stream = None

        self._is_playing = False
        logger.info("Stopped real audio playback")

    async def write_audio_chunk(self, audio_data: np.ndarray) -> None:
        """Write audio chunk for playback."""
        if not self._is_playing or not self._stream:
            return

        try:
            # Ensure correct shape and type
            if len(audio_data.shape) == 1:
                audio_data = audio_data.reshape(-1, 1)

            # Write to stream
            self._stream.write(audio_data.astype(np.float32))
        except Exception as e:
            logger.error(f"Error writing audio chunk: {e}")

    async def play_audio_data(self, audio_data: np.ndarray) -> None:
        """Play complete audio data."""
        if not self._is_playing:
            logger.warning("Cannot play audio data - playback not started")
            return

        try:
            # Ensure correct shape
            if len(audio_data.shape) == 1:
                audio_data = audio_data.reshape(-1, 1)

            # Play the entire audio data
            self._stream.write(audio_data.astype(np.float32))  # type: ignore
        except Exception as e:
            logger.error(f"Error playing audio data: {e}")

    async def play_audio_blocking(
        self,
        audio_data: np.ndarray,
        sample_rate: int = 22050,
        device: Optional[str] = "default",
    ) -> None:
        """Play audio data as a blocking operation (waits for completion)."""
        import sounddevice as sd

        # Ensure correct shape
        if len(audio_data.shape) == 1:
            audio_data = audio_data.reshape(-1, 1)

        # Play and wait
        sd.play(audio_data.astype(np.float32), samplerate=sample_rate, device=device)
        sd.wait()

    async def play_audio_stream(
        self,
        audio_chunks: AsyncGenerator[bytes, None],
        sample_rate: int = 22050,
    ) -> None:
        """
        Stream audio playback chunk-by-chunk.

        Starts playback immediately when first chunk arrives,
        continues as chunks stream in from TTS synthesis.

        Args:
            audio_chunks: Async generator yielding audio bytes
            sample_rate: Audio sample rate
        """
        import io
        import wave

        chunk_count = 0
        total_bytes = 0

        try:
            async for chunk_bytes in audio_chunks:
                if chunk_count == 0:
                    import time

                    time.time()
                    logger.info(
                        "🔊 Starting streaming audio playback (first chunk received)"
                    )

                # Convert bytes to numpy array for playback
                # Assuming chunk_bytes is WAV format
                try:
                    with wave.open(io.BytesIO(chunk_bytes), "rb") as wav_file:
                        audio_data = (
                            np.frombuffer(
                                wav_file.readframes(wav_file.getnframes()),
                                dtype=np.int16,
                            ).astype(np.float32)
                            / 32768.0
                        )

                        # Write to stream immediately
                        await self.write_audio_chunk(audio_data)

                except Exception as e:
                    logger.warning(f"Failed to decode audio chunk {chunk_count}: {e}")
                    continue

                chunk_count += 1
                total_bytes += len(chunk_bytes)

                # Yield control to allow other async tasks
                await asyncio.sleep(0)

            logger.info(
                f"✅ Streaming playback complete: {chunk_count} chunks, "
                f"{total_bytes} bytes total"
            )

        except Exception as e:
            logger.error(f"Streaming playback failed: {e}")
            raise

    def is_playing(self) -> bool:
        """Check if playing."""
        return self._is_playing
