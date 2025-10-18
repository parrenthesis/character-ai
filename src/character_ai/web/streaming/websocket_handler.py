"""
WebSocket handler for streaming LLM functionality.
"""

import asyncio
import json
import logging
from typing import Any, Dict, List

import websockets
from websockets.server import WebSocketServerProtocol

from .core import StreamingLLM
from .types import Token

logger = logging.getLogger(__name__)


class StreamingLLMWebSocketController:
    """WebSocket handler for streaming LLM interactions."""

    def __init__(self, llm: StreamingLLM):
        self.llm = llm
        self.active_sessions: Dict[str, WebSocketServerProtocol] = {}
        self.session_tasks: Dict[str, asyncio.Task] = {}

    async def handle_connection(
        self, websocket: WebSocketServerProtocol, path: str
    ) -> None:
        """Handle WebSocket connection."""
        session_id = f"session_{id(websocket)}"
        self.active_sessions[session_id] = websocket

        logger.info(f"New WebSocket connection: {session_id}")

        try:
            async for message in websocket:
                if isinstance(message, bytes):
                    message = message.decode("utf-8")
                await self._handle_message(websocket, message, session_id)
        except websockets.exceptions.ConnectionClosed:
            logger.info(f"WebSocket connection closed: {session_id}")
        except Exception as e:
            logger.error(f"Error handling WebSocket connection {session_id}: {e}")
        finally:
            await self._cleanup_session(session_id)

    async def _handle_message(
        self, websocket: WebSocketServerProtocol, message: str, session_id: str
    ) -> None:
        """Handle incoming WebSocket message."""
        try:
            data = json.loads(message)
            message_type = data.get("type")

            if message_type == "generate":
                await self._handle_generate_request(websocket, data, session_id)
            elif message_type == "cancel":
                await self._handle_cancel_request(websocket, session_id)
            elif message_type == "ping":
                await self._send_pong(websocket)
            else:
                await self._send_error(
                    websocket, f"Unknown message type: {message_type}"
                )

        except json.JSONDecodeError:
            await self._send_error(websocket, "Invalid JSON message")
        except Exception as e:
            logger.error(f"Error handling message: {e}")
            await self._send_error(websocket, str(e))

    async def _handle_generate_request(
        self, websocket: WebSocketServerProtocol, data: Dict[str, Any], session_id: str
    ) -> None:
        """Handle generate request."""
        try:
            prompt = data.get("prompt", "")
            if not prompt:
                await self._send_error(websocket, "Prompt is required")
                return

            # Cancel any existing generation for this session
            if session_id in self.session_tasks:
                self.session_tasks[session_id].cancel()

            # Start new generation task
            task = asyncio.create_task(
                self._generate_and_stream(websocket, prompt, session_id)
            )
            self.session_tasks[session_id] = task

        except Exception as e:
            logger.error(f"Error handling generate request: {e}")
            await self._send_error(websocket, str(e))

    async def _handle_cancel_request(
        self, websocket: WebSocketServerProtocol, session_id: str
    ) -> None:
        """Handle cancel request."""
        try:
            # Cancel generation
            await self.llm.cancel_generation()

            # Cancel session task
            if session_id in self.session_tasks:
                self.session_tasks[session_id].cancel()
                del self.session_tasks[session_id]

            # Send cancellation confirmation
            await websocket.send(
                json.dumps(
                    {
                        "type": "cancelled",
                        "session_id": session_id,
                        "timestamp": asyncio.get_event_loop().time(),
                    }
                )
            )

        except Exception as e:
            logger.error(f"Error handling cancel request: {e}")
            await self._send_error(websocket, str(e))

    async def _generate_and_stream(
        self, websocket: WebSocketServerProtocol, prompt: str, session_id: str
    ) -> None:
        """Generate and stream tokens to WebSocket."""
        try:
            async for token in self.llm.generate_stream(prompt, session_id):
                await self._send_token(websocket, token)

            # Send completion message
            response = self.llm._create_response()
            await self._send_complete(websocket, response)

        except asyncio.CancelledError:
            logger.info(f"Generation cancelled for session {session_id}")
        except Exception as e:
            logger.error(f"Error during generation: {e}")
            await self._send_error(websocket, str(e))

    async def _send_token(
        self, websocket: WebSocketServerProtocol, token: Token
    ) -> None:
        """Send token to WebSocket client."""
        try:
            message = {
                "type": "token",
                "token": {
                    "text": token.text,
                    "token_id": token.token_id,
                    "logprob": token.logprob,
                    "is_final": token.is_final,
                    "timestamp": token.timestamp,
                    "position": token.position,
                },
                "timestamp": asyncio.get_event_loop().time(),
            }
            await websocket.send(json.dumps(message))
        except Exception as e:
            logger.error(f"Error sending token: {e}")

    async def _send_chunk(
        self, websocket: WebSocketServerProtocol, chunk: List[Token]
    ) -> None:
        """Send chunk of tokens to WebSocket client."""
        try:
            message = {
                "type": "chunk",
                "chunk": [
                    {
                        "text": token.text,
                        "token_id": token.token_id,
                        "logprob": token.logprob,
                        "is_final": token.is_final,
                        "timestamp": token.timestamp,
                        "position": token.position,
                    }
                    for token in chunk
                ],
                "timestamp": asyncio.get_event_loop().time(),
            }
            await websocket.send(json.dumps(message))
        except Exception as e:
            logger.error(f"Error sending chunk: {e}")

    async def _send_complete(
        self, websocket: WebSocketServerProtocol, response: Any
    ) -> None:
        """Send completion message to WebSocket client."""
        try:
            message = {
                "type": "complete",
                "response": {
                    "text": response.text,
                    "token_count": response.token_count,
                    "generation_time": response.generation_time,
                    "is_complete": response.is_complete,
                    "metadata": response.metadata,
                },
                "timestamp": asyncio.get_event_loop().time(),
            }
            await websocket.send(json.dumps(message))
        except Exception as e:
            logger.error(f"Error sending completion: {e}")

    async def _send_error(self, websocket: WebSocketServerProtocol, error: str) -> None:
        """Send error message to client."""
        try:
            message = {
                "type": "error",
                "error": error,
                "timestamp": asyncio.get_event_loop().time(),
            }
            await websocket.send(json.dumps(message))
        except Exception as e:
            logger.error(f"Error sending error message: {e}")

    async def _send_pong(self, websocket: WebSocketServerProtocol) -> None:
        """Send pong response to ping."""
        try:
            message = {"type": "pong", "timestamp": asyncio.get_event_loop().time()}
            await websocket.send(json.dumps(message))
        except Exception as e:
            logger.error(f"Error sending pong: {e}")

    async def _cleanup_session(self, session_id: str) -> None:
        """Clean up session resources."""
        if session_id in self.active_sessions:
            del self.active_sessions[session_id]

        if session_id in self.session_tasks:
            self.session_tasks[session_id].cancel()
            del self.session_tasks[session_id]

        logger.info(f"Cleaned up session: {session_id}")

    def get_active_sessions(self) -> List[str]:
        """Get list of active session IDs."""
        return list(self.active_sessions.keys())

    def get_session_count(self) -> int:
        """Get number of active sessions."""
        return len(self.active_sessions)
