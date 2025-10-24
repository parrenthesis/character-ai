"""
LLM-based conversation summarization for infinite context.

Compresses conversation history using LLM summarization while preserving
key information and maintaining conversation flow.
"""

import logging
import time
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Protocol

from .session_memory import ConversationTurn

logger = logging.getLogger(__name__)


class LLMProvider(Protocol):
    """Protocol for LLM providers used in summarization."""

    def generate_response(self, prompt: str, max_tokens: int = 150) -> str:
        """Generate a response from the LLM."""
        ...


@dataclass
class ConversationSummary:
    """A summarized conversation segment."""

    summary_id: str
    user_id: str
    character_name: str
    start_turn_id: str
    end_turn_id: str
    summary_text: str
    turn_count: int
    created_at: float
    token_count: int = 0

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for serialization."""
        return {
            "summary_id": self.summary_id,
            "user_id": self.user_id,
            "character_name": self.character_name,
            "start_turn_id": self.start_turn_id,
            "end_turn_id": self.end_turn_id,
            "summary_text": self.summary_text,
            "turn_count": self.turn_count,
            "created_at": self.created_at,
            "token_count": self.token_count,
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "ConversationSummary":
        """Create from dictionary."""
        return cls(
            summary_id=data["summary_id"],
            user_id=data["user_id"],
            character_name=data["character_name"],
            start_turn_id=data["start_turn_id"],
            end_turn_id=data["end_turn_id"],
            summary_text=data["summary_text"],
            turn_count=data["turn_count"],
            created_at=data["created_at"],
            token_count=data.get("token_count", 0),
        )


class ConversationSummarizer:
    """LLM-based conversation summarization."""

    def __init__(
        self,
        llm_provider: Optional[LLMProvider],
        max_summary_tokens: int = 150,
        summarize_every_n_turns: int = 10,
        keep_recent_turns: int = 5,
    ):
        """Initialize with LLM provider and summarization settings."""
        self.llm_provider = llm_provider
        self.max_summary_tokens = max_summary_tokens
        self.summarize_every_n_turns = summarize_every_n_turns
        self.keep_recent_turns = keep_recent_turns
        self.summaries: Dict[
            str, List[ConversationSummary]
        ] = {}  # user_character -> summaries

    def should_summarize(self, turn_count: int) -> bool:
        """Check if conversation should be summarized."""
        return turn_count > 0 and turn_count % self.summarize_every_n_turns == 0

    def create_summary(
        self, user_id: str, character_name: str, turns: List[ConversationTurn]
    ) -> Optional[ConversationSummary]:
        """Create a summary from conversation turns."""
        if not turns:
            return None

        # Check if LLM provider is available
        if not self.llm_provider:
            logger.debug("No LLM provider available for summarization")
            return None

        try:
            # Build conversation text for summarization
            conversation_text = self._format_conversation_for_summarization(turns)

            # Create summarization prompt
            prompt = self._create_summarization_prompt(conversation_text)

            # Generate summary using LLM
            summary_text = self.llm_provider.generate_response(
                prompt, max_tokens=self.max_summary_tokens
            )

            # Create summary object
            summary = ConversationSummary(
                summary_id=f"summary_{int(time.time() * 1000)}",
                user_id=user_id,
                character_name=character_name,
                start_turn_id=turns[0].turn_id,
                end_turn_id=turns[-1].turn_id,
                summary_text=summary_text.strip(),
                turn_count=len(turns),
                created_at=time.time(),
                token_count=self._estimate_tokens(summary_text),
            )

            # Store summary
            key = f"{user_id}_{character_name}"
            if key not in self.summaries:
                self.summaries[key] = []

            self.summaries[key].append(summary)

            logger.debug(f"Created summary for {user_id} with {character_name}")
            return summary

        except Exception as e:
            logger.error(f"Failed to create summary: {e}")
            return None

    def get_summaries(
        self, user_id: str, character_name: str
    ) -> List[ConversationSummary]:
        """Get all summaries for a user-character pair."""
        key = f"{user_id}_{character_name}"
        return self.summaries.get(key, [])

    def get_recent_summaries(
        self, user_id: str, character_name: str, limit: int = 3
    ) -> List[ConversationSummary]:
        """Get recent summaries for a user-character pair."""
        summaries = self.get_summaries(user_id, character_name)
        return summaries[-limit:] if summaries else []

    def format_summaries_for_context(self, user_id: str, character_name: str) -> str:
        """Format summaries for inclusion in LLM context."""
        summaries = self.get_recent_summaries(user_id, character_name, limit=3)

        if not summaries:
            return ""

        context_lines = ["Previous conversation summaries:"]
        for summary in summaries:
            context_lines.append(f"- {summary.summary_text}")

        return "\n".join(context_lines)

    def _format_conversation_for_summarization(
        self, turns: List[ConversationTurn]
    ) -> str:
        """Format conversation turns for summarization."""
        lines = []
        for turn in turns:
            lines.append(f"User: {turn.user_input}")
            lines.append(f"{turn.character_name}: {turn.character_response}")

        return "\n".join(lines)

    def _create_summarization_prompt(self, conversation_text: str) -> str:
        """Create prompt for LLM summarization."""
        return f"""Please summarize the following conversation, focusing on:
1. Key topics discussed
2. Important decisions or preferences mentioned
3. Relationship developments
4. Any facts about the user

Conversation:
{conversation_text}

Summary:"""

    def _estimate_tokens(self, text: str) -> int:
        """Estimate token count for text (rough approximation)."""
        # Conservative estimate: 1 token ≈ 4 characters
        return max(1, len(text) // 4)

    def clear_summaries(self, user_id: str, character_name: str) -> None:
        """Clear summaries for a user-character pair."""
        key = f"{user_id}_{character_name}"
        if key in self.summaries:
            del self.summaries[key]
            logger.debug(f"Cleared summaries for {user_id} with {character_name}")

    def get_summary_stats(self) -> Dict[str, Any]:
        """Get statistics about stored summaries."""
        total_summaries = sum(len(summaries) for summaries in self.summaries.values())
        total_users = len(self.summaries)

        return {
            "total_summaries": total_summaries,
            "total_user_character_pairs": total_users,
            "max_summary_tokens": self.max_summary_tokens,
            "summarize_every_n_turns": self.summarize_every_n_turns,
        }


class FallbackSummarizer:
    """Fallback summarization when LLM is unavailable."""

    def __init__(self, max_length: int = 200):
        """Initialize with maximum summary length."""
        self.max_length = max_length

    def create_simple_summary(self, turns: List[ConversationTurn]) -> str:
        """Create a simple summary without LLM."""
        if not turns:
            return ""

        # Extract key phrases from user inputs
        key_phrases = []
        for turn in turns:
            # Simple extraction of important words
            words = turn.user_input.lower().split()
            important_words = [w for w in words if len(w) > 3 and w.isalpha()]
            key_phrases.extend(important_words[:3])  # Take first 3 important words

        # Create simple summary
        if key_phrases:
            summary = f"Discussed: {', '.join(set(key_phrases[:5]))}"
        else:
            summary = f"Had {len(turns)} exchanges"

        # Truncate if too long
        if len(summary) > self.max_length:
            summary = summary[: self.max_length - 3] + "..."

        return summary
