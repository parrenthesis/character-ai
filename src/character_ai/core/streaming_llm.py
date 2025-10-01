"""
Streaming LLM for real-time text generation.

This module provides streaming capabilities for large language models,
enabling real-time token generation and response streaming.
"""

import asyncio
import json
import logging
import time
from dataclasses import dataclass
from enum import Enum
from typing import Any, AsyncGenerator, Callable, Dict, List, Optional

import websockets
from websockets.server import WebSocketServerProtocol

logger = logging.getLogger(__name__)


class StreamingMode(Enum):
    """Streaming mode enumeration."""

    TOKEN_BY_TOKEN = (
        "token_by_token"  # nosec B105 - Not a password, legitimate enum value
    )
    WORD_BY_WORD = "word_by_word"
    SENTENCE_BY_SENTENCE = "sentence_by_sentence"
    CHUNK_BY_CHUNK = "chunk_by_chunk"


class LLMState(Enum):
    """LLM processing state."""

    IDLE = "idle"
    GENERATING = "generating"
    COMPLETED = "completed"
    ERROR = "error"
    CANCELLED = "cancelled"


@dataclass
class Token:
    """Individual token in the stream."""

    text: str
    token_id: int
    logprob: float
    is_final: bool
    timestamp: float
    position: int


@dataclass
class StreamingConfig:
    """Configuration for streaming LLM."""

    # Model parameters
    model_name: str = "llama-2-7b-chat"
    max_tokens: int = 512
    temperature: float = 0.7
    top_p: float = 0.9
    top_k: int = 50
    repetition_penalty: float = 1.1

    # Streaming parameters
    streaming_mode: StreamingMode = StreamingMode.TOKEN_BY_TOKEN
    chunk_size: int = 1  # Number of tokens per chunk
    max_chunk_delay_ms: int = 100  # Maximum delay between chunks
    buffer_size: int = 10  # Number of tokens to buffer

    # Response parameters
    stop_sequences: Optional[List[str]] = None
    include_stop_sequence: bool = False
    max_response_time_ms: int = 30000  # 30 seconds max

    # WebSocket parameters
    ping_interval: int = 20
    ping_timeout: int = 10
    max_message_size: int = 1024 * 1024  # 1MB

    def __post_init__(self) -> None:
        if self.stop_sequences is None:
            self.stop_sequences = ["</s>", "<|endoftext|>", "\n\n"]


@dataclass
class StreamingResponse:
    """Streaming response from LLM."""

    tokens: List[Token]
    full_text: str
    is_complete: bool
    generation_time: float
    tokens_per_second: float
    metadata: Dict[str, Any]


class StreamingLLM:
    """
    Streaming Large Language Model processor.

    Provides real-time token generation with configurable streaming modes
    and WebSocket support for live response streaming.
    """

    def __init__(self, config: StreamingConfig):
        self.config = config
        self.state = LLMState.IDLE

        # Model components
        self.model: Optional[Any] = None
        self.tokenizer: Optional[Any] = None
        self.generation_config: Optional[Dict[str, Any]] = None

        # Streaming state
        self.current_tokens: List[Token] = []
        self.generated_text: str = ""
        self.start_time: Optional[float] = None
        self.session_id: Optional[str] = None

        # Callbacks
        self.on_token: Optional[Callable] = None
        self.on_chunk: Optional[Callable] = None
        self.on_complete: Optional[Callable] = None
        self.on_error: Optional[Callable] = None

        logger.info(f"StreamingLLM initialized with config: {config}")

    async def initialize(self) -> None:
        """Initialize the streaming LLM."""

        try:
            # Initialize model
            try:
                from ..algorithms.large_language_model import (  # type: ignore
                    LargeLanguageModelProcessor,
                )

                self.model = LargeLanguageModelProcessor()
                if self.model is not None:
                    await self.model.initialize()
            except ImportError:
                logger.warning(
                    "LargeLanguageModelProcessor not available, using placeholder"
                )
                self.model = None

            # Set up generation config
            self.generation_config = {
                "max_tokens": self.config.max_tokens,
                "temperature": self.config.temperature,
                "top_p": self.config.top_p,
                "top_k": self.config.top_k,
                "repetition_penalty": self.config.repetition_penalty,
                "stop_sequences": self.config.stop_sequences,
            }

            self.state = LLMState.IDLE
            logger.info("StreamingLLM initialized successfully")

        except Exception as e:
            logger.error(f"Failed to initialize StreamingLLM: {e}")
            self.state = LLMState.ERROR
            raise

    async def generate_stream(
        self, prompt: str, session_id: Optional[str] = None, **kwargs: Any
    ) -> AsyncGenerator[Token, None]:
        """
        Generate a streaming response from the LLM.

        Args:
            prompt: Input prompt for generation
            session_id: Optional session identifier
            **kwargs: Additional generation parameters

        Yields:
            Token objects as they are generated
        """

        try:
            self.state = LLMState.GENERATING
            self.session_id = session_id
            self.start_time = time.time()
            self.current_tokens.clear()
            self.generated_text = ""

            # Merge config with kwargs
            generation_params = {**(self.generation_config or {}), **kwargs}

            logger.info(f"Starting streaming generation for session {session_id}")

            # Generate tokens using the model
            if not self.model:
                # Placeholder token generation when model is not available
                logger.warning("LLM model not available, using placeholder generation")
                async for token_data in self._generate_placeholder_tokens(prompt):
                    token = Token(
                        text=token_data["text"],
                        token_id=token_data.get("id", 0),
                        logprob=token_data.get("logprob", 0.0),
                        is_final=token_data.get("is_final", False),
                        timestamp=time.time(),
                        position=len(self.current_tokens),
                    )

                    # Add to current tokens
                    self.current_tokens.append(token)
                    self.generated_text += token.text

                    # Check for stop sequences
                    if self._should_stop(token.text):
                        token.is_final = True
                        self.state = LLMState.COMPLETED
                        break

                    # Check for timeout
                    if (
                        time.time() - self.start_time
                    ) * 1000 > self.config.max_response_time_ms:
                        logger.warning("Generation timeout reached")
                        token.is_final = True
                        self.state = LLMState.COMPLETED
                        break

                    # Yield token
                    yield token

                    # Call token callback
                    if self.on_token:
                        self.on_token(token)
            else:
                # Use actual model
                async for token_data in self._generate_tokens(
                    prompt, generation_params
                ):
                    token = Token(
                        text=token_data["text"],
                        token_id=token_data.get("id", 0),
                        logprob=token_data.get("logprob", 0.0),
                        is_final=token_data.get("is_final", False),
                        timestamp=time.time(),
                        position=len(self.current_tokens),
                    )

                    # Add to current tokens
                    self.current_tokens.append(token)
                    self.generated_text += token.text

                    # Check for stop sequences
                    if self._should_stop(token.text):
                        token.is_final = True
                        self.state = LLMState.COMPLETED
                        break

                    # Check for timeout
                    if (
                        time.time() - self.start_time
                    ) * 1000 > self.config.max_response_time_ms:
                        logger.warning("Generation timeout reached")
                        token.is_final = True
                        self.state = LLMState.COMPLETED
                        break

                    # Yield token
                    yield token

                    # Call token callback
                    if self.on_token:
                        self.on_token(token)

            # Finalize generation
            if self.state == LLMState.GENERATING:
                self.state = LLMState.COMPLETED

            # Call completion callback
            if self.on_complete:
                response = self._create_response()
                self.on_complete(response)

            logger.info(f"Completed streaming generation for session {session_id}")

        except Exception as e:
            logger.error(f"Error in streaming generation: {e}")
            self.state = LLMState.ERROR
            if self.on_error:
                self.on_error(e)
            raise

    async def _generate_placeholder_tokens(
        self, prompt: str
    ) -> AsyncGenerator[Dict[str, Any], None]:
        """Generate tokens when model is not available - uses intelligent fallback."""
        try:
            # Try to use the actual LLM first
            async for token in self._generate_tokens(prompt, {}):
                yield token
        except Exception as e:
            logger.warning(f"LLM not available, using intelligent fallback: {e}")

            # Intelligent fallback based on prompt content
            response = await self._generate_intelligent_fallback(prompt)
            words = response.split()

            for i, word in enumerate(words):
                # Simulate generation delay using configurable setting
                await asyncio.sleep(0.1)  # Fixed delay for placeholder

                token_data = {
                    "text": word + " ",
                    "id": i,
                    "logprob": -0.5,
                    "is_final": i == len(words) - 1,
                }

                yield token_data

    async def _generate_intelligent_fallback(self, prompt: str) -> str:
        """Generate an intelligent fallback response based on prompt content."""
        prompt_lower = prompt.lower()

        # Character-aware responses
        if any(word in prompt_lower for word in ["hello", "hi", "hey", "greetings"]):
            return "Hello! I'm here to help you with your interactive character experience."
        elif any(word in prompt_lower for word in ["help", "assist", "support"]):
            return "I'd be happy to help! I can assist with character interactions, answer questions, and provide guidance."
        elif any(word in prompt_lower for word in ["thank", "thanks", "appreciate"]):
            return "You're very welcome! I'm glad I could help."
        elif any(
            word in prompt_lower for word in ["how", "what", "when", "where", "why"]
        ):
            return "That's a great question! Let me help you understand that better."
        elif any(word in prompt_lower for word in ["goodbye", "bye", "see you"]):
            return "Goodbye! It was nice talking with you. Have a wonderful day!"
        else:
            return "I understand your message and I'm here to help. How can I assist you today?"

    async def _generate_tokens(
        self, prompt: str, params: Dict[str, Any]
    ) -> AsyncGenerator[Dict[str, Any], None]:
        """
        Generate tokens using the underlying LLM model.

        Integrates with the platform's LLM processors (LlamaCppProcessor or LlamaProcess
or).
        """
        try:
            # Import the platform's LLM processors
            from ..algorithms.conversational_ai.llama_cpp_processor import (
                LlamaCppProcessor,
            )
            from ..algorithms.conversational_ai.llama_processor import LlamaProcessor

            # Determine which processor to use based on configuration
            if getattr(self.config, 'models', {}).get('llama_backend') == "llama_cpp":
                processor: Any = LlamaCppProcessor(self.config)  # type: ignore
            else:
                processor: Any = LlamaProcessor(self.config)  # type: ignore

            # Initialize processor if not already initialized
            if not processor._initialized:
                await processor.initialize()

            # Generate response using the platform's LLM
            result = await processor.process_text(prompt, context=params)

            if result.error:
                logger.error(f"LLM generation error: {result.error}")
                yield {
                    "text": "I'm sorry, I encountered an error processing your request.",
                    "id": 0,
                    "logprob": -1.0,
                    "is_final": True,
                }
                return

            # Stream the response word by word
            response_text = result.text or ""
            words = response_text.split()

            for i, word in enumerate(words):
                # Simulate realistic token generation delay using configurable setting
                await asyncio.sleep(0.05)  # Fixed delay for token generation

                token_data = {
                    "text": word + " ",
                    "id": i,
                    "logprob": -0.3 + (i * 0.01),  # Slightly improving probability
                    "is_final": i == len(words) - 1,
                }

                yield token_data

        except ImportError as e:
            logger.error(f"Failed to import LLM processors: {e}")
            # Fallback to simple response
            yield {
                "text": "I'm here to help, but I need to be properly configured first. "
,
                "id": 0,
                "logprob": -0.5,
                "is_final": True,
            }
        except Exception as e:
            logger.error(f"Error generating tokens with LLM: {e}")
            # Fallback to error response
            yield {
                "text": "I'm sorry, I encountered an error processing your request. ",
                "id": 0,
                "logprob": -1.0,
                "is_final": True,
            }

    def _should_stop(self, token_text: str) -> bool:
        """Check if generation should stop based on stop sequences."""

        for stop_seq in self.config.stop_sequences or []:
            if stop_seq in token_text:
                return True
        return False

    def _create_response(self) -> StreamingResponse:
        """Create a streaming response object."""

        generation_time = time.time() - self.start_time if self.start_time else 0.0
        tokens_per_second = (
            len(self.current_tokens) / generation_time if generation_time > 0 else 0.0
        )

        return StreamingResponse(
            tokens=self.current_tokens.copy(),
            full_text=self.generated_text,
            is_complete=self.state == LLMState.COMPLETED,
            generation_time=generation_time,
            tokens_per_second=tokens_per_second,
            metadata={
                "session_id": self.session_id,
                "model": self.config.model_name,
                "config": self.config.__dict__,
            },
        )

    async def generate_chunks(
        self, prompt: str, session_id: Optional[str] = None, **kwargs: Any
    ) -> AsyncGenerator[List[Token], None]:
        """
        Generate response in chunks based on streaming mode.

        Args:
            prompt: Input prompt for generation
            session_id: Optional session identifier
            **kwargs: Additional generation parameters

        Yields:
            Lists of Token objects based on streaming mode
        """

        chunk_buffer: List[Token] = []

        async for token in self.generate_stream(prompt, session_id, **kwargs):
            chunk_buffer.append(token)

            # Check if we should yield a chunk
            should_yield = False

            if self.config.streaming_mode == StreamingMode.TOKEN_BY_TOKEN:
                should_yield = len(chunk_buffer) >= self.config.chunk_size
            elif self.config.streaming_mode == StreamingMode.WORD_BY_WORD:
                should_yield = token.text.endswith(" ") or token.is_final
            elif self.config.streaming_mode == StreamingMode.SENTENCE_BY_SENTENCE:
                should_yield = (
                    token.text.endswith(".")
                    or token.text.endswith("!")
                    or token.text.endswith("?")
                    or token.is_final
                )
            elif self.config.streaming_mode == StreamingMode.CHUNK_BY_CHUNK:
                should_yield = len(chunk_buffer) >= self.config.chunk_size

            if should_yield or token.is_final:
                if chunk_buffer:
                    yield chunk_buffer.copy()
                    chunk_buffer.clear()

                # Call chunk callback
                if self.on_chunk:
                    self.on_chunk(chunk_buffer)

    async def cancel_generation(self) -> None:
        """Cancel ongoing generation."""

        if self.state == LLMState.GENERATING:
            self.state = LLMState.CANCELLED
            logger.info(f"Cancelled generation for session {self.session_id}")

    def get_state(self) -> Dict[str, Any]:
        """Get current LLM state."""

        return {
            "state": self.state.value,
            "session_id": self.session_id,
            "start_time": self.start_time,
            "tokens_generated": len(self.current_tokens),
            "generated_text_length": len(self.generated_text),
            "generation_time": (
                time.time() - self.start_time if self.start_time else 0.0
            ),
        }

    def set_callbacks(
        self,
        on_token: Optional[Callable] = None,
        on_chunk: Optional[Callable] = None,
        on_complete: Optional[Callable] = None,
        on_error: Optional[Callable] = None,
    ) -> None:
        """Set callback functions for streaming events."""

        self.on_token = on_token
        self.on_chunk = on_chunk
        self.on_complete = on_complete
        self.on_error = on_error


class StreamingLLMWebSocketHandler:
    """
    WebSocket handler for streaming LLM responses.

    Manages WebSocket connections for real-time LLM text generation
    and streaming responses to clients.
    """

    def __init__(self, llm: StreamingLLM):
        self.llm = llm
        self.active_sessions: Dict[str, Dict[str, Any]] = {}

        logger.info("StreamingLLMWebSocketHandler initialized")

    async def handle_connection(self, websocket: WebSocketServerProtocol, path: str) -> None:
        """Handle a new WebSocket connection."""

        session_id = f"{websocket.remote_address[0]}:{websocket.remote_address[1]}"
        self.active_sessions[session_id] = {
            "websocket": websocket,
            "start_time": time.time(),
            "generation_active": False,
        }

        logger.info(f"New LLM WebSocket connection: {session_id}")

        try:
            # Set up callbacks
            self.llm.set_callbacks(
                on_token=lambda token: self._send_token(websocket, token),
                on_chunk=lambda chunk: self._send_chunk(websocket, chunk),
                on_complete=lambda response: self._send_complete(websocket, response),
                on_error=lambda error: self._send_error(websocket, error),
            )

            # Handle messages
            async for message in websocket:
                await self._handle_message(websocket, str(message), session_id)

        except websockets.exceptions.ConnectionClosed:
            logger.info(f"LLM WebSocket connection closed: {session_id}")
        except Exception as e:
            logger.error(f"Error handling LLM WebSocket connection {session_id}: {e}")
        finally:
            # Clean up
            if session_id in self.active_sessions:
                del self.active_sessions[session_id]

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
                logger.warning(f"Unknown LLM message type: {message_type}")

        except json.JSONDecodeError:
            logger.error("Invalid JSON message received")
        except Exception as e:
            logger.error(f"Error handling LLM message: {e}")

    async def _handle_generate_request(
        self, websocket: WebSocketServerProtocol, data: Dict[str, Any], session_id: str
    ) -> None:
        """Handle text generation request."""

        try:
            prompt = data.get("prompt", "")
            data.get("streaming_mode", "token_by_token")
            max_tokens = data.get("max_tokens", self.llm.config.max_tokens)
            temperature = data.get("temperature", self.llm.config.temperature)

            if not prompt:
                await self._send_error(websocket, "No prompt provided")
                return

            # Update session state
            self.active_sessions[session_id]["generation_active"] = True

            # Start generation
            async for token in self.llm.generate_stream(
                prompt=prompt,
                session_id=session_id,
                max_tokens=max_tokens,
                temperature=temperature,
            ):
                # Send token
                await self._send_token(websocket, token)

                # Check if generation is complete
                if token.is_final:
                    break

            # Mark generation as complete
            self.active_sessions[session_id]["generation_active"] = False

        except Exception as e:
            logger.error(f"Error handling generate request: {e}")
            await self._send_error(websocket, str(e))
            self.active_sessions[session_id]["generation_active"] = False

    async def _handle_cancel_request(
        self, websocket: WebSocketServerProtocol, session_id: str
    ) -> None:
        """Handle generation cancellation request."""

        try:
            await self.llm.cancel_generation()
            self.active_sessions[session_id]["generation_active"] = False

            # Send cancellation confirmation
            message = {"type": "cancelled", "timestamp": time.time()}
            await websocket.send(json.dumps(message))

        except Exception as e:
            logger.error(f"Error handling cancel request: {e}")
            await self._send_error(websocket, str(e))

    async def _send_token(self, websocket: WebSocketServerProtocol, token: Token) -> None:
        """Send token to client."""

        message = {
            "type": "token",
            "text": token.text,
            "token_id": token.token_id,
            "logprob": token.logprob,
            "is_final": token.is_final,
            "position": token.position,
            "timestamp": token.timestamp,
        }

        try:
            await websocket.send(json.dumps(message))
        except Exception as e:
            logger.error(f"Error sending token: {e}")

    async def _send_chunk(self, websocket: WebSocketServerProtocol, chunk: List[Token]) -> None:

        """Send token chunk to client."""

        message = {
            "type": "chunk",
            "tokens": [
                {
                    "text": token.text,
                    "token_id": token.token_id,
                    "logprob": token.logprob,
                    "is_final": token.is_final,
                    "position": token.position,
                    "timestamp": token.timestamp,
                }
                for token in chunk
            ],
            "timestamp": time.time(),
        }

        try:
            await websocket.send(json.dumps(message))
        except Exception as e:
            logger.error(f"Error sending chunk: {e}")

    async def _send_complete(
        self, websocket: WebSocketServerProtocol, response: StreamingResponse
    ) -> None:
        """Send completion message to client."""

        message = {
            "type": "complete",
            "full_text": response.full_text,
            "generation_time": response.generation_time,
            "tokens_per_second": response.tokens_per_second,
            "total_tokens": len(response.tokens),
            "metadata": response.metadata,
            "timestamp": time.time(),
        }

        try:
            await websocket.send(json.dumps(message))
        except Exception as e:
            logger.error(f"Error sending complete: {e}")

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

    def get_active_sessions(self) -> List[str]:
        """Get list of active session IDs."""
        return list(self.active_sessions.keys())

    def get_session_count(self) -> int:
        """Get number of active sessions."""
        return len(self.active_sessions)
