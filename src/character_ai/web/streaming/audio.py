"""
Streaming audio processing for real-time STT/TTS.

This module provides streaming capabilities for speech-to-text and text-to-speech
processing, enabling real-time audio interaction with the character platform.
"""

import asyncio
import base64
import json
import logging
import time
from dataclasses import dataclass
from enum import Enum
from typing import Any, Callable, Dict, List, Optional

import numpy as np
import websockets
from websockets.server import WebSocketServerProtocol

logger = logging.getLogger(__name__)


class AudioFormat(Enum):
    """Supported audio formats."""

    PCM_16KHZ_16BIT = "pcm_16k_16bit"
    PCM_22KHZ_16BIT = "pcm_22k_16bit"
    PCM_44KHZ_16BIT = "pcm_44k_16bit"
    WAV = "wav"
    MP3 = "mp3"


class StreamingState(Enum):
    """Streaming state enumeration."""

    IDLE = "idle"
    LISTENING = "listening"
    PROCESSING = "processing"
    SPEAKING = "speaking"
    ERROR = "error"


@dataclass
class AudioChunk:
    """Audio chunk for streaming processing."""

    data: bytes
    format: AudioFormat
    sample_rate: int
    channels: int
    timestamp: float
    sequence_number: int
    is_final: bool = False


@dataclass
class StreamingConfig:
    """Configuration for streaming audio processing."""

    # Audio parameters
    sample_rate: int = 16000
    channels: int = 1
    bit_depth: int = 16
    chunk_size_ms: int = 100  # 100ms chunks
    buffer_size_ms: int = 1000  # 1 second buffer

    # STT parameters
    stt_model: str = "wav2vec2-base"
    stt_language: str = "en"
    stt_vad_enabled: bool = True
    stt_partial_results: bool = True
    stt_max_alternatives: int = 3

    # TTS parameters
    tts_model: str = "coqui"
    tts_voice: str = "default"
    tts_speed: float = 1.0
    tts_emotion: str = "neutral"

    # Streaming parameters
    max_audio_duration_ms: int = 30000  # 30 seconds max
    silence_timeout_ms: int = 2000  # 2 seconds silence timeout
    connection_timeout_ms: int = 30000  # 30 seconds connection timeout (configurable via streaming.connection_timeout_ms)

    # WebSocket parameters
    websocket_ping_interval: int = 20
    websocket_ping_timeout: int = 10
    max_message_size: int = 1024 * 1024  # 1MB


class StreamingAudioProcessor:
    """
    Real-time streaming audio processor for STT/TTS.

    Handles chunked audio processing, voice activity detection,
    and real-time transcription/synthesis.
    """

    def __init__(self, config: StreamingConfig):
        self.config = config
        self.state = StreamingState.IDLE

        # Audio buffers
        self.audio_buffer: List[AudioChunk] = []
        self.processing_buffer: bytes = b""
        self.output_buffer: bytes = b""

        # State tracking
        self.session_id: Optional[str] = None
        self.start_time: Optional[float] = None
        self.last_audio_time: Optional[float] = None
        self.sequence_number = 0

        # Processing components
        self.stt_processor: Optional[Any] = None
        self.tts_processor: Optional[Any] = None
        self.vad_detector: Optional[Any] = None

        # Callbacks
        self.on_transcript: Optional[Callable] = None
        self.on_audio_output: Optional[Callable] = None
        self.on_state_change: Optional[Callable] = None
        self.on_error: Optional[Callable] = None

        logger.info(f"StreamingAudioProcessor initialized with config: {config}")

    async def initialize(self) -> None:
        """Initialize the streaming processor."""

        try:
            # Initialize STT processor
            try:
                from ..algorithms.conversational_ai.wav2vec2_processor import (
                    Wav2Vec2Processor,
                )

                # Create a mock config for the processor
                from ..config import Config

                mock_config = Config()
                self.stt_processor = Wav2Vec2Processor(mock_config)
                if self.stt_processor is not None:
                    await self.stt_processor.initialize()
            except ImportError:
                logger.warning("Wav2Vec2Processor not available, using placeholder")
                self.stt_processor = None

            # Initialize TTS processor
            try:
                from ..algorithms.conversational_ai.coqui_processor import (
                    CoquiProcessor,
                )
                from ..config import Config

                # Create a mock config for the processor
                mock_config = Config()
                from .config import DEFAULT_COQUI_MODEL

                tts_model = getattr(
                    mock_config.models, "coqui_model", DEFAULT_COQUI_MODEL
                )
                self.tts_processor = CoquiProcessor(mock_config, model_name=tts_model)
                if self.tts_processor is not None:
                    await self.tts_processor.initialize()
            except ImportError:
                logger.warning("CoquiTTSProcessor not available, using placeholder")
                self.tts_processor = None

            # Initialize VAD detector
            from ...core.audio_io.voice_activity_detection import (
                VADConfig,
                VoiceActivityDetector,
            )

            vad_config = VADConfig(
                sample_rate=self.config.sample_rate,
                frame_duration_ms=20,
                min_speech_duration_ms=200,
                min_silence_duration_ms=500,
            )
            self.vad_detector = VoiceActivityDetector(vad_config)

            # Set VAD callbacks
            if self.vad_detector is not None:
                self.vad_detector.set_callbacks(
                    on_speech_start=self._on_speech_start,
                    on_speech_end=self._on_speech_end,
                    on_state_change=self._on_vad_state_change,
                )

            self.state = StreamingState.IDLE
            logger.info("StreamingAudioProcessor initialized successfully")

        except Exception as e:
            logger.error(f"Failed to initialize StreamingAudioProcessor: {e}")
            self.state = StreamingState.ERROR
            raise

    def _on_speech_start(self, timestamp: float) -> None:
        """Handle speech start event."""
        logger.info(f"Speech started at {timestamp}")
        self.state = StreamingState.LISTENING

        if self.on_state_change:
            self.on_state_change(self.state, timestamp)

    def _on_speech_end(self, timestamp: float) -> None:
        """Handle speech end event."""
        logger.info(f"Speech ended at {timestamp}")
        self.state = StreamingState.PROCESSING

        if self.on_state_change:
            self.on_state_change(self.state, timestamp)

        # Process accumulated audio
        asyncio.create_task(self._process_accumulated_audio())

    def _on_vad_state_change(
        self, old_state: Any, new_state: Any, timestamp: float
    ) -> None:
        """Handle VAD state change."""
        logger.debug(
            f"VAD state changed from {old_state} to {new_state} at {timestamp}"
        )

    async def _process_accumulated_audio(self) -> None:
        """Process accumulated audio for transcription."""

        try:
            if not self.audio_buffer:
                return

            # Combine audio chunks
            combined_audio = b"".join(chunk.data for chunk in self.audio_buffer)

            # Convert to numpy array for processing
            audio_array = np.frombuffer(combined_audio, dtype=np.int16)

            # Process with STT
            if self.stt_processor:
                transcript = await self.stt_processor.transcribe_audio(
                    audio_array,
                    sample_rate=self.config.sample_rate,
                    language=self.config.stt_language,
                )

                if transcript and self.on_transcript:
                    self.on_transcript(transcript, is_final=True)
            else:
                # Placeholder transcript when STT is not available
                transcript = "Audio processed (STT not available)"
                if self.on_transcript:
                    self.on_transcript(transcript, is_final=True)

            # Clear processed audio
            self.audio_buffer.clear()
            self.processing_buffer = b""

        except Exception as e:
            logger.error(f"Error processing accumulated audio: {e}")
            if self.on_error:
                self.on_error(e)

    async def process_audio_chunk(self, chunk: AudioChunk) -> Optional[str]:
        """
        Process a single audio chunk.

        Args:
            chunk: Audio chunk to process

        Returns:
            Partial transcript if available
        """

        try:
            # Update state
            self.last_audio_time = time.time()
            self.sequence_number = chunk.sequence_number

            # Add to buffer
            self.audio_buffer.append(chunk)

            # Apply VAD
            if self.vad_detector:
                audio_array = np.frombuffer(chunk.data, dtype=np.int16)
                vad_result = self.vad_detector.process_frame(audio_array)

                # Update state based on VAD
                if vad_result.is_voice and self.state == StreamingState.IDLE:
                    self.state = StreamingState.LISTENING
                elif not vad_result.is_voice and self.state == StreamingState.LISTENING:
                    # Check for silence timeout
                    if (
                        time.time() - self.last_audio_time
                    ) * 1000 > self.config.silence_timeout_ms:
                        self.state = StreamingState.PROCESSING
                        await self._process_accumulated_audio()

            # Process partial audio if configured
            if self.config.stt_partial_results and self.stt_processor:
                try:
                    # Convert chunk to numpy array
                    audio_array = np.frombuffer(chunk.data, dtype=np.int16)

                    # Get partial transcript
                    partial_transcript = await self.stt_processor.transcribe_audio(
                        audio_array,
                        sample_rate=self.config.sample_rate,
                        language=self.config.stt_language,
                        partial=True,
                    )

                    if partial_transcript and self.on_transcript:
                        self.on_transcript(partial_transcript, is_final=False)

                    return str(partial_transcript) if partial_transcript else None

                except Exception as e:
                    logger.warning(f"Error in partial transcription: {e}")
            elif self.config.stt_partial_results and not self.stt_processor:
                # Placeholder partial transcript when STT is not available
                partial_transcript = "Processing audio... (STT not available)"
                if self.on_transcript:
                    self.on_transcript(partial_transcript, is_final=False)
                return partial_transcript

            return None

        except Exception as e:
            logger.error(f"Error processing audio chunk: {e}")
            if self.on_error:
                self.on_error(e)
            return None

    async def synthesize_speech(self, text: str, voice: Optional[str] = None) -> bytes:
        """
        Synthesize speech from text.

        Args:
            text: Text to synthesize
            voice: Voice to use (optional)

        Returns:
            Audio data as bytes
        """

        try:
            self.state = StreamingState.SPEAKING

            if not self.tts_processor:
                # Return placeholder audio when TTS is not available
                logger.warning(
                    "TTS processor not available, returning placeholder audio"
                )
                audio_data = b"placeholder_audio_data"
            else:
                # Use provided voice or default
                voice_to_use = voice or self.config.tts_voice

                # Synthesize audio
                audio_data = await self.tts_processor.synthesize(
                    text=text,
                    voice=voice_to_use,
                    speed=self.config.tts_speed,
                    emotion=self.config.tts_emotion,
                )

            # Update output buffer
            self.output_buffer = audio_data

            if self.on_audio_output:
                self.on_audio_output(audio_data)

            return audio_data

        except Exception as e:
            logger.error(f"Error synthesizing speech: {e}")
            self.state = StreamingState.ERROR
            if self.on_error:
                self.on_error(e)
            raise

    async def start_session(self, session_id: str) -> None:
        """Start a new streaming session."""

        self.session_id = session_id
        self.start_time = time.time()
        self.state = StreamingState.IDLE

        # Clear buffers
        self.audio_buffer.clear()
        self.processing_buffer = b""
        self.output_buffer = b""
        self.sequence_number = 0

        logger.info(f"Started streaming session: {session_id}")

    async def end_session(self) -> None:
        """End the current streaming session."""

        # Process any remaining audio
        if self.audio_buffer:
            await self._process_accumulated_audio()

        # Reset state
        self.session_id = None
        self.start_time = None
        self.state = StreamingState.IDLE

        logger.info("Ended streaming session")

    def get_state(self) -> Dict[str, Any]:
        """Get current processor state."""

        return {
            "state": self.state.value,
            "session_id": self.session_id,
            "start_time": self.start_time,
            "last_audio_time": self.last_audio_time,
            "sequence_number": self.sequence_number,
            "buffer_size": len(self.audio_buffer),
            "processing_buffer_size": len(self.processing_buffer),
            "output_buffer_size": len(self.output_buffer),
        }

    def set_callbacks(
        self,
        on_transcript: Optional[Callable] = None,
        on_audio_output: Optional[Callable] = None,
        on_state_change: Optional[Callable] = None,
        on_error: Optional[Callable] = None,
    ) -> None:
        """Set callback functions for streaming events."""

        self.on_transcript = on_transcript
        self.on_audio_output = on_audio_output
        self.on_state_change = on_state_change
        self.on_error = on_error


class StreamingWebSocketController:
    """
    WebSocket handler for streaming audio processing.

    Manages WebSocket connections for real-time audio streaming
    between clients and the character platform.
    """

    def __init__(self, processor: StreamingAudioProcessor):
        self.processor = processor
        self.active_connections: Dict[str, WebSocketServerProtocol] = {}

        logger.info("StreamingWebSocketController initialized")

    async def handle_connection(
        self, websocket: WebSocketServerProtocol, path: str
    ) -> None:
        """Handle a new WebSocket connection."""

        connection_id = f"{websocket.remote_address[0]}:{websocket.remote_address[1]}"
        self.active_connections[connection_id] = websocket

        logger.info(f"New WebSocket connection: {connection_id}")

        try:
            # Start session
            await self.processor.start_session(connection_id)

            # Set up callbacks
            self.processor.set_callbacks(
                on_transcript=lambda text, is_final: self._send_transcript(
                    websocket, text, is_final
                ),
                on_audio_output=lambda audio: self._send_audio(websocket, audio),
                on_state_change=lambda state, timestamp: self._send_state(
                    websocket, state, timestamp
                ),
                on_error=lambda error: self._send_error(websocket, error),
            )

            # Handle messages
            async for message in websocket:
                if isinstance(message, str):
                    await self._handle_message(websocket, message)
                else:
                    # Handle binary messages (audio data)
                    await self._handle_audio_data(websocket, message)

        except websockets.exceptions.ConnectionClosed:
            logger.info(f"WebSocket connection closed: {connection_id}")
        except Exception as e:
            logger.error(f"Error handling WebSocket connection {connection_id}: {e}")
        finally:
            # Clean up
            await self.processor.end_session()
            if connection_id in self.active_connections:
                del self.active_connections[connection_id]

    async def _handle_audio_data(
        self, websocket: WebSocketServerProtocol, audio_data: bytes
    ) -> None:
        """Handle incoming audio data."""
        try:
            # Process audio data
            audio_chunk = AudioChunk(
                data=audio_data,
                format=AudioFormat.PCM_16KHZ_16BIT,
                sample_rate=16000,
                channels=1,
                timestamp=time.time(),
                sequence_number=0,
                is_final=False,
            )
            await self.processor.process_audio_chunk(audio_chunk)
        except Exception as e:
            logger.error(f"Error processing audio data: {e}")
            await self._send_error(websocket, str(e))

    async def _handle_message(
        self, websocket: WebSocketServerProtocol, message: str
    ) -> None:
        """Handle incoming WebSocket message."""

        try:
            data = json.loads(message)
            message_type = data.get("type")

            if message_type == "audio_chunk":
                await self._handle_audio_chunk(websocket, data)
            elif message_type == "synthesize":
                await self._handle_synthesize_request(websocket, data)
            elif message_type == "ping":
                await self._send_pong(websocket)
            else:
                logger.warning(f"Unknown message type: {message_type}")

        except json.JSONDecodeError:
            logger.error("Invalid JSON message received")
        except Exception as e:
            logger.error(f"Error handling message: {e}")

    async def _handle_audio_chunk(
        self, websocket: WebSocketServerProtocol, data: Dict[str, Any]
    ) -> None:
        """Handle audio chunk message."""

        try:
            # Decode base64 audio data
            audio_data = base64.b64decode(data["audio"])

            # Create audio chunk
            chunk = AudioChunk(
                data=audio_data,
                format=AudioFormat(data.get("format", "pcm_16k_16bit")),
                sample_rate=data.get("sample_rate", 16000),
                channels=data.get("channels", 1),
                timestamp=time.time(),
                sequence_number=data.get("sequence_number", 0),
                is_final=data.get("is_final", False),
            )

            # Process chunk
            partial_transcript = await self.processor.process_audio_chunk(chunk)

            # Send partial transcript if available
            if partial_transcript:
                await self._send_transcript(
                    websocket, partial_transcript, is_final=False
                )

        except Exception as e:
            logger.error(f"Error handling audio chunk: {e}")
            await self._send_error(websocket, str(e))

    async def _handle_synthesize_request(
        self, websocket: WebSocketServerProtocol, data: Dict[str, Any]
    ) -> None:
        """Handle speech synthesis request."""

        try:
            text = data.get("text", "")
            voice = data.get("voice")

            if not text:
                await self._send_error(websocket, "No text provided for synthesis")
                return

            # Synthesize speech
            audio_data = await self.processor.synthesize_speech(text, voice)

            # Send audio response
            await self._send_audio(websocket, audio_data)

        except Exception as e:
            logger.error(f"Error handling synthesize request: {e}")
            await self._send_error(websocket, str(e))

    async def _send_transcript(
        self, websocket: WebSocketServerProtocol, text: str, is_final: bool
    ) -> None:
        """Send transcript to client."""

        message = {
            "type": "transcript",
            "text": text,
            "is_final": is_final,
            "timestamp": time.time(),
        }

        try:
            await websocket.send(json.dumps(message))
        except Exception as e:
            logger.error(f"Error sending transcript: {e}")

    async def _send_audio(
        self, websocket: WebSocketServerProtocol, audio_data: bytes
    ) -> None:
        """Send audio data to client."""

        # Encode audio as base64
        audio_b64 = base64.b64encode(audio_data).decode("utf-8")

        message = {"type": "audio", "audio": audio_b64, "timestamp": time.time()}

        try:
            await websocket.send(json.dumps(message))
        except Exception as e:
            logger.error(f"Error sending audio: {e}")

    async def _send_state(
        self,
        websocket: WebSocketServerProtocol,
        state: StreamingState,
        timestamp: float,
    ) -> None:
        """Send state update to client."""

        message = {"type": "state", "state": state.value, "timestamp": timestamp}

        try:
            await websocket.send(json.dumps(message))
        except Exception as e:
            logger.error(f"Error sending state: {e}")

    async def _send_error(self, websocket: WebSocketServerProtocol, error: str) -> None:
        """Send error message to client."""

        message = {"type": "error", "error": error, "timestamp": time.time()}

        try:
            await websocket.send(json.dumps(message))
        except Exception as e:
            logger.error(f"Error sending error message: {e}")

    async def _send_pong(self, websocket: WebSocketServerProtocol) -> None:
        """Send pong response to ping."""

        message = {"type": "pong", "timestamp": time.time()}

        try:
            await websocket.send(json.dumps(message))
        except Exception as e:
            logger.error(f"Error sending pong: {e}")

    def get_connection_count(self) -> int:
        """Get number of active connections."""
        return len(self.active_connections)

    def get_connections(self) -> List[str]:
        """Get list of active connection IDs."""
        return list(self.active_connections.keys())
