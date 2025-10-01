"""
WebSocket API endpoints for streaming audio and text processing.

This module provides WebSocket endpoints for real-time audio streaming,
voice activity detection, and streaming LLM text generation.
"""

import time
from typing import Any, Dict, Optional

from fastapi import APIRouter, Depends, HTTPException, WebSocket, WebSocketDisconnect
from fastapi.responses import HTMLResponse

from ..core.logging import get_logger
from ..core.security import DeviceIdentity
from ..core.streaming_audio import StreamingAudioProcessor
from ..core.streaming_audio import StreamingConfig as AudioConfig
from ..core.streaming_audio import StreamingWebSocketHandler
from ..core.streaming_llm import StreamingConfig as LLMConfig
from ..core.streaming_llm import (
    StreamingLLM,
    StreamingLLMWebSocketHandler,
    StreamingMode,
)
from .security_deps import require_authentication

logger = get_logger(__name__)

# Create routers
streaming_router = APIRouter(prefix="/api/v1/streaming", tags=["streaming"])

# Global instances
audio_processor: Optional[StreamingAudioProcessor] = None
llm_processor: Optional[StreamingLLM] = None
audio_handler: Optional[StreamingWebSocketHandler] = None
llm_handler: Optional[StreamingLLMWebSocketHandler] = None


async def get_audio_processor() -> StreamingAudioProcessor:
    """Get or create the audio processor instance."""
    global audio_processor

    if audio_processor is None:
        config = AudioConfig(
            sample_rate=16000,
            chunk_size_ms=100,
            stt_model="whisper-base",
            tts_model="xtts",
        )
        audio_processor = StreamingAudioProcessor(config)
        await audio_processor.initialize()

    return audio_processor


async def get_llm_processor() -> StreamingLLM:
    """Get or create the LLM processor instance."""
    global llm_processor

    if llm_processor is None:
        config = LLMConfig(
            model_name="llama-2-7b-chat",
            max_tokens=512,
            temperature=0.7,
            streaming_mode=StreamingMode.TOKEN_BY_TOKEN,
        )
        llm_processor = StreamingLLM(config)
        await llm_processor.initialize()

    return llm_processor


@streaming_router.get("/audio/websocket")
async def audio_websocket_endpoint() -> Dict[str, Any]:
    """Get WebSocket endpoint for audio streaming."""
    return {
        "websocket_url": "ws://localhost:8000/api/v1/streaming/audio/ws",
        "supported_formats": ["pcm_16k_16bit", "pcm_22k_16bit", "pcm_44k_16bit", "wav"],

        "features": [
            "real_time_stt",
            "voice_activity_detection",
            "streaming_tts",
            "partial_transcripts",
        ],
    }


@streaming_router.get("/llm/websocket")
async def llm_websocket_endpoint() -> Dict[str, Any]:
    """Get WebSocket endpoint for LLM streaming."""
    return {
        "websocket_url": "ws://localhost:8000/api/v1/streaming/llm/ws",
        "supported_models": ["llama-2-7b-chat", "llama-2-13b-chat"],
        "streaming_modes": ["token_by_token", "word_by_word", "sentence_by_sentence"],
        "features": [
            "real_time_generation",
            "token_streaming",
            "response_cancellation",
            "multiple_sessions",
        ],
    }


@streaming_router.websocket("/audio/ws")
async def audio_websocket(websocket: WebSocket) -> None:
    """WebSocket endpoint for audio streaming."""

    await websocket.accept()

    try:
        # Get audio processor
        processor = await get_audio_processor()

        # Create handler
        global audio_handler
        if audio_handler is None:
            audio_handler = StreamingWebSocketHandler(processor)

        # Handle connection
        await audio_handler.handle_connection(websocket, "/audio/ws")  # type: ignore

    except WebSocketDisconnect:
        logger.info("Audio WebSocket disconnected")
    except Exception as e:
        logger.error(f"Error in audio WebSocket: {e}")
        try:
            await websocket.close(code=1011, reason="Internal server error")
        except Exception as close_error:
            logger.warning(f"Failed to close audio WebSocket gracefully: {close_error}")



@streaming_router.websocket("/llm/ws")
async def llm_websocket(websocket: WebSocket) -> None:
    """WebSocket endpoint for LLM streaming."""

    await websocket.accept()

    try:
        # Get LLM processor
        processor = await get_llm_processor()

        # Create handler
        global llm_handler
        if llm_handler is None:
            llm_handler = StreamingLLMWebSocketHandler(processor)

        # Handle connection
        await llm_handler.handle_connection(websocket, "/llm/ws")  # type: ignore

    except WebSocketDisconnect:
        logger.info("LLM WebSocket disconnected")
    except Exception as e:
        logger.error(f"Error in LLM WebSocket: {e}")
        try:
            await websocket.close(code=1011, reason="Internal server error")
        except Exception as close_error:
            logger.warning(f"Failed to close LLM WebSocket gracefully: {close_error}")


@streaming_router.get("/audio/status")
async def get_audio_status() -> Dict[str, Any]:
    """Get audio streaming status."""

    processor = await get_audio_processor()
    handler = audio_handler

    status = {
        "processor_state": processor.get_state(),
        "active_connections": handler.get_connection_count() if handler else 0,
        "connections": handler.get_connections() if handler else [],
    }

    return status


@streaming_router.get("/llm/status")
async def get_llm_status() -> Dict[str, Any]:
    """Get LLM streaming status."""

    processor = await get_llm_processor()
    handler = llm_handler

    status = {
        "processor_state": processor.get_state(),
        "active_sessions": handler.get_session_count() if handler else 0,
        "sessions": handler.get_active_sessions() if handler else [],
    }

    return status


@streaming_router.post("/audio/configure")
async def configure_audio_streaming(
    config: Dict[str, Any], device: DeviceIdentity = Depends(require_authentication)
) -> Dict[str, Any]:
    """Configure audio streaming parameters."""

    try:
        processor = await get_audio_processor()

        # Update configuration
        if "sample_rate" in config:
            processor.config.sample_rate = config["sample_rate"]
        if "chunk_size_ms" in config:
            processor.config.chunk_size_ms = config["chunk_size_ms"]
        if "stt_model" in config:
            processor.config.stt_model = config["stt_model"]
        if "tts_model" in config:
            processor.config.tts_model = config["tts_model"]

        logger.info(
            f"Audio streaming configured by device {device.device_id}: {config}"
        )

        return {
            "success": True,
            "config": processor.config.__dict__,
            "message": "Audio streaming configuration updated",
        }

    except Exception as e:
        logger.error(f"Error configuring audio streaming: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@streaming_router.post("/llm/configure")
async def configure_llm_streaming(
    config: Dict[str, Any], device: DeviceIdentity = Depends(require_authentication)
) -> Dict[str, Any]:
    """Configure LLM streaming parameters."""

    try:
        processor = await get_llm_processor()

        # Update configuration
        if "model_name" in config:
            processor.config.model_name = config["model_name"]
        if "max_tokens" in config:
            processor.config.max_tokens = config["max_tokens"]
        if "temperature" in config:
            processor.config.temperature = config["temperature"]
        if "streaming_mode" in config:
            processor.config.streaming_mode = config["streaming_mode"]

        logger.info(f"LLM streaming configured by device {device.device_id}: {config}")

        return {
            "success": True,
            "config": processor.config.__dict__,
            "message": "LLM streaming configuration updated",
        }

    except Exception as e:
        logger.error(f"Error configuring LLM streaming: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@streaming_router.get("/test/client")
async def get_test_client() -> HTMLResponse:
    """Get HTML test client for WebSocket testing."""

    html_content = """
    <!DOCTYPE html>
    <html>
    <head>
        <title>Streaming Test Client</title>
        <style>
            body { font-family: Arial, sans-serif; margin: 20px; }
            .container { max-width: 800px; margin: 0 auto; }
            .section { margin: 20px 0; padding: 20px; border: 1px solid #ccc; }
            button { padding: 10px 20px; margin: 5px; }
            textarea { width: 100%; height: 100px; }
            #messages { height: 300px; overflow-y: scroll; border: 1px solid #ccc; paddi
ng: 10px; }
        </style>
    </head>
    <body>
        <div class="container">
            <h1>Streaming Test Client</h1>

            <div class="section">
                <h2>Audio Streaming Test</h2>
                <button onclick="connectAudio()">Connect Audio</button>
                <button onclick="disconnectAudio()">Disconnect Audio</button>
                <button onclick="startRecording()">Start Recording</button>
                <button onclick="stopRecording()">Stop Recording</button>
                <div id="audioStatus">Disconnected</div>
            </div>

            <div class="section">
                <h2>LLM Streaming Test</h2>
                <button onclick="connectLLM()">Connect LLM</button>
                <button onclick="disconnectLLM()">Disconnect LLM</button>
                <textarea id="promptInput" placeholder="Enter your prompt here..."></tex
tarea>
                <button onclick="sendPrompt()">Send Prompt</button>
                <button onclick="cancelGeneration()">Cancel Generation</button>
                <div id="llmStatus">Disconnected</div>
            </div>

            <div class="section">
                <h2>Messages</h2>
                <div id="messages"></div>
            </div>
        </div>

        <script>
            let audioWs = null;
            let llmWs = null;
            let mediaRecorder = null;
            let audioChunks = [];

            function log(message) {
                const messages = document.getElementById('messages');
                messages.innerHTML += '<div>' + new Date().toLocaleTimeString() + ': ' +
 message + '</div>';
                messages.scrollTop = messages.scrollHeight;
            }

            function connectAudio() {
                audioWs = new WebSocket('ws://localhost:8000/api/v1/streaming/audio/ws')
;

                audioWs.onopen = function() {
                    log('Audio WebSocket connected');
                    document.getElementById('audioStatus').textContent = 'Connected';
                };

                audioWs.onmessage = function(event) {
                    const data = JSON.parse(event.data);
                    log('Audio: ' + JSON.stringify(data));
                };

                audioWs.onclose = function() {
                    log('Audio WebSocket disconnected');
                    document.getElementById('audioStatus').textContent = 'Disconnected';

                };
            }

            function disconnectAudio() {
                if (audioWs) {
                    audioWs.close();
                    audioWs = null;
                }
            }

            function connectLLM() {
                llmWs = new WebSocket('ws://localhost:8000/api/v1/streaming/llm/ws');

                llmWs.onopen = function() {
                    log('LLM WebSocket connected');
                    document.getElementById('llmStatus').textContent = 'Connected';
                };

                llmWs.onmessage = function(event) {
                    const data = JSON.parse(event.data);
                    log('LLM: ' + JSON.stringify(data));
                };

                llmWs.onclose = function() {
                    log('LLM WebSocket disconnected');
                    document.getElementById('llmStatus').textContent = 'Disconnected';
                };
            }

            function disconnectLLM() {
                if (llmWs) {
                    llmWs.close();
                    llmWs = null;
                }
            }

            function sendPrompt() {
                const prompt = document.getElementById('promptInput').value;
                if (llmWs && prompt) {
                    llmWs.send(JSON.stringify({
                        type: 'generate',
                        prompt: prompt,
                        streaming_mode: 'token_by_token'
                    }));
                }
            }

            function cancelGeneration() {
                if (llmWs) {
                    llmWs.send(JSON.stringify({
                        type: 'cancel'
                    }));
                }
            }

            function startRecording() {
                navigator.mediaDevices.getUserMedia({ audio: true })
                    .then(stream => {
                        mediaRecorder = new MediaRecorder(stream);
                        audioChunks = [];

                        mediaRecorder.ondataavailable = function(event) {
                            audioChunks.push(event.data);
                        };

                        mediaRecorder.onstop = function() {
                            const audioBlob = new Blob(audioChunks, { type: 'audio/wav'
});
                            // Convert to base64 and send
                            const reader = new FileReader();
                            reader.onload = function() {
                                const base64 = reader.result.split(',')[1];
                                if (audioWs) {
                                    audioWs.send(JSON.stringify({
                                        type: 'audio_chunk',
                                        audio: base64,
                                        format: 'wav',
                                        sample_rate: 44100,
                                        channels: 1,
                                        sequence_number: 0
                                    }));
                                }
                            };
                            reader.readAsDataURL(audioBlob);
                        };

                        mediaRecorder.start();
                        log('Recording started');
                    })
                    .catch(err => log('Error accessing microphone: ' + err));
            }

            function stopRecording() {
                if (mediaRecorder) {
                    mediaRecorder.stop();
                    log('Recording stopped');
                }
            }
        </script>
    </body>
    </html>
    """

    return HTMLResponse(content=html_content)


@streaming_router.get("/health")
async def streaming_health() -> Dict[str, Any]:
    """Get streaming system health status."""

    try:
        audio_processor = await get_audio_processor()
        llm_processor = await get_llm_processor()

        health = {
            "status": "healthy",
            "timestamp": time.time(),
            "components": {
                "audio_processor": {
                    "state": audio_processor.get_state(),
                    "active_connections": (
                        audio_handler.get_connection_count() if audio_handler else 0
                    ),
                },
                "llm_processor": {
                    "state": llm_processor.get_state(),
                    "active_sessions": (
                        llm_handler.get_session_count() if llm_handler else 0
                    ),
                },
            },
        }

        return health

    except Exception as e:
        logger.error(f"Error checking streaming health: {e}")
        return {"status": "unhealthy", "timestamp": time.time(), "error": str(e)}
