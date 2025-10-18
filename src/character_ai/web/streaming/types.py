"""
Types and data structures for streaming LLM functionality.
"""

from dataclasses import dataclass
from enum import Enum
from typing import Any, Dict, List, Optional


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

    model_name: str
    max_tokens: int = 1000
    temperature: float = 0.7
    top_p: float = 0.9
    frequency_penalty: float = 0.0
    presence_penalty: float = 0.0
    stop_sequences: Optional[List[str]] = None
    streaming_mode: StreamingMode = StreamingMode.TOKEN_BY_TOKEN
    chunk_size: int = 5
    buffer_size: int = 100
    timeout: float = 30.0
    retry_attempts: int = 3
    enable_placeholder: bool = True
    placeholder_delay: float = 0.1
    enable_fallback: bool = True
    fallback_threshold: float = 0.5

    def __post_init__(self) -> None:
        if self.stop_sequences is None:
            self.stop_sequences = ["\n\n", "Human:", "Assistant:"]


@dataclass
class StreamingResponse:
    """Response from streaming LLM."""

    text: str
    tokens: List[Token]
    metadata: Dict[str, Any]
    generation_time: float
    token_count: int
    is_complete: bool
