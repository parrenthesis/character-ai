"""
Session memory management for conversational AI.

Provides rolling conversation windows with strict token limits and character-specific
retention policies to maintain context while respecting memory constraints.
"""

import logging
import time
from collections import deque
from dataclasses import dataclass, field
from typing import Any, Callable, Dict, List, Optional, Tuple

logger = logging.getLogger(__name__)


@dataclass
class ConversationTurn:
    """A single turn in the conversation."""

    timestamp: float
    user_input: str
    character_response: str
    character_name: str
    turn_id: str = field(default_factory=lambda: f"turn_{int(time.time() * 1000)}")

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for serialization."""
        return {
            "timestamp": self.timestamp,
            "user_input": self.user_input,
            "character_response": self.character_response,
            "character_name": self.character_name,
            "turn_id": self.turn_id,
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "ConversationTurn":
        """Create from dictionary."""
        return cls(
            timestamp=data["timestamp"],
            user_input=data["user_input"],
            character_response=data["character_response"],
            character_name=data["character_name"],
            turn_id=data["turn_id"],
        )


@dataclass
class MemoryConfig:
    """Configuration for session memory management."""

    max_turns: int = 10  # Maximum number of conversation turns to keep
    max_tokens: int = 1000  # Maximum total tokens across all turns
    max_age_seconds: int = 3600  # Maximum age of any turn (1 hour)
    character_specific_limits: Dict[str, Dict[str, int]] = field(default_factory=dict)

    def get_limits_for_character(self, character_name: str) -> Tuple[int, int, int]:
        """Get memory limits for a specific character."""
        char_limits = self.character_specific_limits.get(character_name.lower(), {})
        return (
            char_limits.get("max_turns", self.max_turns),
            char_limits.get("max_tokens", self.max_tokens),
            char_limits.get("max_age_seconds", self.max_age_seconds),
        )


class SessionMemory:
    """Manages rolling conversation memory with strict limits."""

    def __init__(self, config: Optional[MemoryConfig] = None):
        self.config = config or MemoryConfig()
        self.conversations: Dict[str, deque] = {}  # character_name -> deque of turns
        self._token_estimator = self._create_token_estimator()

    def _create_token_estimator(self) -> Callable[[str], int]:
        """Create a simple token estimator (rough approximation)."""

        def estimate_tokens(text: str) -> int:
            # Rough approximation: 1 token ≈ 4 characters for English
            # This is conservative and works well for most LLMs
            return max(1, len(text) // 4)

        return estimate_tokens

    def add_turn(
        self, character_name: str, user_input: str, character_response: str
    ) -> None:
        """Add a new conversation turn for a character."""
        character_name = character_name.lower()

        # Create new turn
        turn = ConversationTurn(
            timestamp=time.time(),
            user_input=user_input,
            character_response=character_response,
            character_name=character_name,
        )

        # Initialize conversation if needed
        if character_name not in self.conversations:
            self.conversations[character_name] = deque()

        # Add turn
        self.conversations[character_name].append(turn)

        # Apply memory limits
        self._apply_memory_limits(character_name)

        logger.debug(
            f"Added turn for {character_name}, total turns: "
            f"{len(self.conversations[character_name])}"
        )

    def _apply_memory_limits(self, character_name: str) -> None:
        """Apply memory limits to a character's conversation."""
        if character_name not in self.conversations:
            return

        conversation = self.conversations[character_name]
        max_turns, max_tokens, max_age = self.config.get_limits_for_character(
            character_name
        )
        current_time = time.time()

        # Remove old turns based on age
        while conversation and (current_time - conversation[0].timestamp) > max_age:
            removed = conversation.popleft()
            logger.debug(f"Removed old turn for {character_name}: {removed.turn_id}")

        # Remove turns based on count limit
        while len(conversation) > max_turns:
            removed = conversation.popleft()
            logger.debug(
                f"Removed turn for {character_name} (count limit): {removed.turn_id}"
            )

        # Remove turns based on token limit
        while conversation:
            total_tokens = sum(
                self._token_estimator(turn.user_input)
                + self._token_estimator(turn.character_response)
                for turn in conversation
            )
            if total_tokens <= max_tokens:
                break

            removed = conversation.popleft()
            logger.debug(
                f"Removed turn for {character_name} (token limit): {removed.turn_id}"
            )

    def get_conversation_context(
        self, character_name: str, max_turns: Optional[int] = None
    ) -> List[ConversationTurn]:
        """Get conversation context for a character."""
        character_name = character_name.lower()

        if character_name not in self.conversations:
            return []

        conversation = self.conversations[character_name]

        # Apply max_turns limit if specified
        if max_turns is not None and len(conversation) > max_turns:
            return list(conversation)[-max_turns:]

        return list(conversation)

    def get_conversation_depth(self, character_name: str) -> int:
        """Get number of turns in current conversation (for template selection)."""
        character_name = character_name.lower()
        if character_name not in self.conversations:
            return 0
        return len(self.conversations[character_name])

    def get_conversation_summary(self, character_name: str) -> Dict[str, Any]:
        """Get a summary of conversation state for a character."""
        character_name = character_name.lower()

        if character_name not in self.conversations:
            return {
                "character_name": character_name,
                "total_turns": 0,
                "total_tokens": 0,
                "oldest_turn_age": 0,
                "newest_turn_age": 0,
            }

        conversation = self.conversations[character_name]
        if not conversation:
            return {
                "character_name": character_name,
                "total_turns": 0,
                "total_tokens": 0,
                "oldest_turn_age": 0,
                "newest_turn_age": 0,
            }

        current_time = time.time()
        total_tokens = sum(
            self._token_estimator(turn.user_input)
            + self._token_estimator(turn.character_response)
            for turn in conversation
        )

        return {
            "character_name": character_name,
            "total_turns": len(conversation),
            "total_tokens": total_tokens,
            "oldest_turn_age": current_time - conversation[0].timestamp,
            "newest_turn_age": current_time - conversation[-1].timestamp,
            "memory_limits": {
                "max_turns": self.config.get_limits_for_character(character_name)[0],
                "max_tokens": self.config.get_limits_for_character(character_name)[1],
                "max_age_seconds": self.config.get_limits_for_character(character_name)[
                    2
                ],
            },
        }

    def clear_conversation(self, character_name: str) -> None:
        """Clear conversation history for a character."""
        character_name = character_name.lower()
        if character_name in self.conversations:
            del self.conversations[character_name]
            logger.info(f"Cleared conversation history for {character_name}")

    def clear_all_conversations(self) -> None:
        """Clear all conversation history."""
        self.conversations.clear()
        logger.info("Cleared all conversation history")

    def get_all_conversations(self) -> Dict[str, List[Dict[str, Any]]]:
        """Get all conversations as dictionaries for serialization."""
        return {
            character_name: [turn.to_dict() for turn in conversation]
            for character_name, conversation in self.conversations.items()
        }

    def load_conversations(
        self, conversations_data: Dict[str, List[Dict[str, Any]]]
    ) -> None:
        """Load conversations from serialized data."""
        self.conversations.clear()

        for character_name, turns_data in conversations_data.items():
            conversation: deque[ConversationTurn] = deque()
            for turn_data in turns_data:
                turn = ConversationTurn.from_dict(turn_data)
                conversation.append(turn)
            self.conversations[character_name.lower()] = conversation

        logger.info(f"Loaded conversations for {len(self.conversations)} characters")

    def format_context_for_llm(
        self,
        character_name: str,
        current_user_input: str,
        max_turns: Optional[int] = None,
    ) -> str:
        """Format conversation context for LLM consumption."""
        context_turns = self.get_conversation_context(character_name, max_turns)

        if not context_turns:
            return ""

        context_lines = ["Previous conversation:"]
        for turn in context_turns:
            context_lines.append(f"User: {turn.user_input}")
            context_lines.append(f"{character_name}: {turn.character_response}")

        context_lines.append(f"\nCurrent user input: {current_user_input}")

        return "\n".join(context_lines)
