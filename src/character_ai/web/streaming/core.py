"""
Core streaming LLM functionality.
"""

import asyncio
import logging
import time
from typing import Any, AsyncGenerator, Callable, Dict, List, Optional

from .types import LLMState, StreamingConfig, StreamingResponse, Token

logger = logging.getLogger(__name__)


class StreamingLLM:
    """Streaming LLM for real-time text generation."""

    def __init__(self, config: StreamingConfig):
        self.config = config
        self.state = LLMState.IDLE
        self.current_tokens: List[Token] = []
        self.generation_start_time: Optional[float] = None
        self.cancelled = False
        self.callbacks: Dict[str, Callable] = {}
        self.session_data: Dict[str, Any] = {}

    async def initialize(self) -> None:
        """Initialize the streaming LLM."""
        try:
            logger.info(
                f"Initializing streaming LLM with model: {self.config.model_name}"
            )

            # Initialize model-specific components
            await self._initialize_model()

            # Set up callbacks
            self._setup_default_callbacks()

            self.state = LLMState.IDLE
            logger.info("Streaming LLM initialized successfully")

        except Exception as e:
            logger.error(f"Failed to initialize streaming LLM: {e}")
            self.state = LLMState.ERROR
            raise

    async def _initialize_model(self) -> None:
        """Initialize the underlying model."""
        # This would contain model-specific initialization
        # For now, just simulate initialization
        await asyncio.sleep(0.1)
        logger.debug("Model initialized")

    def _setup_default_callbacks(self) -> None:
        """Set up default callbacks."""
        self.callbacks = {
            "on_token": self._default_token_callback,
            "on_chunk": self._default_chunk_callback,
            "on_complete": self._default_complete_callback,
            "on_error": self._default_error_callback,
        }

    async def generate_stream(
        self, prompt: str, session_id: Optional[str] = None, **kwargs: Any
    ) -> AsyncGenerator[Token, None]:
        """Generate a stream of tokens from the given prompt."""
        if self.state == LLMState.GENERATING:
            raise RuntimeError("LLM is already generating")

        self.state = LLMState.GENERATING
        self.generation_start_time = time.time()
        self.current_tokens = []
        self.cancelled = False

        try:
            # Store session data
            if session_id:
                self.session_data[session_id] = {
                    "prompt": prompt,
                    "start_time": self.generation_start_time,
                    "tokens": [],
                }

            # Generate placeholder tokens if enabled
            if self.config.enable_placeholder:
                async for token in self._generate_placeholder_tokens(prompt):
                    if self.cancelled:
                        break
                    yield token

            # Generate actual tokens
            async for token in self._generate_tokens(prompt, kwargs):
                if self.cancelled:
                    break

                self.current_tokens.append(token)

                # Store in session data
                if session_id and session_id in self.session_data:
                    self.session_data[session_id]["tokens"].append(token)

                # Call token callback
                if "on_token" in self.callbacks:
                    await self.callbacks["on_token"](token, session_id)

                yield token

            if not self.cancelled:
                self.state = LLMState.COMPLETED
                if "on_complete" in self.callbacks:
                    response = self._create_response()
                    await self.callbacks["on_complete"](response, session_id)

        except Exception as e:
            self.state = LLMState.ERROR
            logger.error(f"Error during generation: {e}")
            if "on_error" in self.callbacks:
                await self.callbacks["on_error"](e, session_id)
            raise

        finally:
            if self.cancelled:
                self.state = LLMState.CANCELLED

    async def _generate_placeholder_tokens(
        self, prompt: str
    ) -> AsyncGenerator[Token, None]:
        """Generate placeholder tokens while model loads."""
        placeholder_texts = ["Thinking", "...", "Processing", "...", "Generating"]

        for i, text in enumerate(placeholder_texts):
            if self.cancelled:
                break

            token = Token(
                text=text,
                token_id=i,
                logprob=0.0,
                is_final=False,
                timestamp=time.time(),
                position=i,
            )

            await asyncio.sleep(self.config.placeholder_delay)
            yield token

    async def _generate_intelligent_fallback(self, prompt: str) -> str:
        """Generate an intelligent fallback response based on prompt content."""
        # Simple keyword-based fallback responses
        prompt_lower = prompt.lower()

        if "hello" in prompt_lower or "hi" in prompt_lower:
            return "Hello! How can I help you today?"
        elif "how are you" in prompt_lower:
            return "I'm doing well, thank you for asking!"
        elif "what" in prompt_lower and "time" in prompt_lower:
            return f"The current time is {time.strftime('%H:%M:%S')}."
        else:
            return "I understand your request. Let me process that for you."

    async def _generate_tokens(
        self, prompt: str, params: Dict[str, Any]
    ) -> AsyncGenerator[Token, None]:
        """Generate tokens using the underlying model."""
        try:
            # Simulate token generation
            fallback_response = await self._generate_intelligent_fallback(prompt)
            words = fallback_response.split()

            for i, word in enumerate(words):
                if self.cancelled:
                    break

                # Add space before word (except first)
                if i > 0:
                    space_token = Token(
                        text=" ",
                        token_id=i * 2 - 1,
                        logprob=0.0,
                        is_final=False,
                        timestamp=time.time(),
                        position=i * 2 - 1,
                    )
                    yield space_token

                # Add word token
                word_token = Token(
                    text=word,
                    token_id=i * 2,
                    logprob=0.0,
                    is_final=(i == len(words) - 1),
                    timestamp=time.time(),
                    position=i * 2,
                )
                yield word_token

                # Simulate processing delay
                await asyncio.sleep(0.05)

        except Exception as e:
            logger.error(f"Error generating tokens: {e}")
            raise

    def _should_stop(self, token_text: str) -> bool:
        """Check if generation should stop based on stop sequences."""
        if not self.config.stop_sequences:
            return False
        return any(seq in token_text for seq in self.config.stop_sequences)

    def _create_response(self) -> StreamingResponse:
        """Create a streaming response object."""
        generation_time = (
            time.time() - self.generation_start_time
            if self.generation_start_time
            else 0.0
        )

        return StreamingResponse(
            text="".join(token.text for token in self.current_tokens),
            tokens=self.current_tokens.copy(),
            metadata={
                "model": self.config.model_name,
                "generation_time": generation_time,
                "token_count": len(self.current_tokens),
                "state": self.state.value,
            },
            generation_time=generation_time,
            token_count=len(self.current_tokens),
            is_complete=self.state == LLMState.COMPLETED,
        )

    async def generate_chunks(
        self, prompt: str, session_id: Optional[str] = None, **kwargs: Any
    ) -> AsyncGenerator[List[Token], None]:
        """Generate chunks of tokens."""
        chunk = []

        async for token in self.generate_stream(prompt, session_id, **kwargs):
            chunk.append(token)

            if len(chunk) >= self.config.chunk_size:
                yield chunk
                chunk = []

                # Call chunk callback
                if "on_chunk" in self.callbacks:
                    await self.callbacks["on_chunk"](chunk, session_id)

        # Yield remaining tokens
        if chunk:
            yield chunk

    async def cancel_generation(self) -> None:
        """Cancel ongoing generation."""
        if self.state == LLMState.GENERATING:
            self.cancelled = True
            logger.info("Generation cancelled")

    def get_state(self) -> Dict[str, Any]:
        """Get current LLM state."""
        return {
            "state": self.state.value,
            "token_count": len(self.current_tokens),
            "generation_time": (
                time.time() - self.generation_start_time
                if self.generation_start_time
                else 0.0
            ),
            "active_sessions": len(self.session_data),
        }

    def set_callbacks(
        self,
        on_token: Optional[Callable] = None,
        on_chunk: Optional[Callable] = None,
        on_complete: Optional[Callable] = None,
        on_error: Optional[Callable] = None,
    ) -> None:
        """Set custom callbacks."""
        if on_token:
            self.callbacks["on_token"] = on_token
        if on_chunk:
            self.callbacks["on_chunk"] = on_chunk
        if on_complete:
            self.callbacks["on_complete"] = on_complete
        if on_error:
            self.callbacks["on_error"] = on_error

    # Default callback implementations
    async def _default_token_callback(
        self, token: Token, session_id: Optional[str]
    ) -> None:
        """Default token callback."""
        logger.debug(f"Token generated: {token.text}")

    async def _default_chunk_callback(
        self, chunk: List[Token], session_id: Optional[str]
    ) -> None:
        """Default chunk callback."""
        logger.debug(f"Chunk generated: {len(chunk)} tokens")

    async def _default_complete_callback(
        self, response: StreamingResponse, session_id: Optional[str]
    ) -> None:
        """Default complete callback."""
        logger.info(
            f"Generation complete: {response.token_count} tokens in {response.generation_time:.2f}s"
        )

    async def _default_error_callback(
        self, error: Exception, session_id: Optional[str]
    ) -> None:
        """Default error callback."""
        logger.error(f"Generation error: {error}")
