"""
Hybrid memory system combining preferences, storage, and summarization.

Orchestrates preference extraction, persistent storage, and conversation
summarization to provide comprehensive memory capabilities.
"""

import logging
import time
from dataclasses import dataclass
from typing import Any, Dict, List, Optional

from ...observability.memory_metrics import get_memory_metrics
from .conversation_summarizer import ConversationSummarizer, LLMProvider
from .optimized_conversation_storage import OptimizedConversationStorage
from .preference_extractor import (
    PreferenceExtractor,
    PreferenceStorage,
    UserPreferences,
)
from .session_memory import ConversationTurn, SessionMemory

logger = logging.getLogger(__name__)


@dataclass
class MemoryConfig:
    """Configuration for hybrid memory system."""

    enabled: bool = True
    data_directory: str = "data"

    # Preference settings
    preferences_enabled: bool = True
    preferences_storage_path: str = "data/user_preferences.json"

    # Storage settings
    storage_enabled: bool = True
    storage_db_path: str = "data/conversations.db"
    max_age_days: int = 30

    # Summarization settings
    summarization_enabled: bool = True
    summarize_every_n_turns: int = 10
    keep_recent_turns: int = 5
    max_summary_tokens: int = 150

    # Context settings
    include_recent_turns: int = 3
    include_summaries: bool = True
    include_preferences: bool = True


class HybridMemorySystem:
    """Unified hybrid memory system."""

    def __init__(
        self, llm_provider: Optional[LLMProvider], config: MemoryConfig, device_id: str
    ):
        """Initialize hybrid memory system."""
        self.llm_provider = llm_provider
        self.config = config
        self.device_id = device_id

        # Initialize components
        self.preference_extractor = PreferenceExtractor()
        self.preference_storage = PreferenceStorage(
            self.config.preferences_storage_path
        )
        self.conversation_storage = OptimizedConversationStorage(
            self.config.storage_db_path, self.config.max_age_days
        )
        self.conversation_summarizer = ConversationSummarizer(
            llm_provider=self.llm_provider,
            max_summary_tokens=self.config.max_summary_tokens,
            summarize_every_n_turns=self.config.summarize_every_n_turns,
            keep_recent_turns=self.config.keep_recent_turns,
        )

        # Fallback to SessionMemory
        self.session_memory = SessionMemory()

        # Current session tracking
        self.current_session_id: Optional[str] = None

        # Metrics
        self.metrics = get_memory_metrics()

        logger.info(f"Initialized hybrid memory system for device {device_id}")

    def start_session(self, character_name: str) -> str:
        """Start a new conversation session."""
        if not self.config.enabled:
            return "fallback_session"

        session_id = f"session_{int(time.time() * 1000)}"
        self.current_session_id = session_id

        # Start session in storage
        if self.config.storage_enabled:
            self.conversation_storage.start_session(
                session_id, self.device_id, character_name
            )
            # Record metrics
            self.metrics.record_session_created(character_name)

        logger.debug(f"Started session {session_id} for {character_name}")
        return session_id

    def end_session(self) -> None:
        """End the current conversation session."""
        if not self.config.enabled or not self.current_session_id:
            return

        # End session in storage
        if self.config.storage_enabled:
            self.conversation_storage.end_session(self.current_session_id)

        self.current_session_id = None
        logger.debug("Ended current session")

    def process_turn(
        self, character_name: str, user_input: str, character_response: str
    ) -> None:
        """Process a conversation turn through all memory components."""
        if not self.config.enabled:
            # Fallback to SessionMemory
            self.session_memory.add_turn(character_name, user_input, character_response)
            return

        # Create conversation turn
        turn = ConversationTurn(
            timestamp=time.time(),
            user_input=user_input,
            character_response=character_response,
            character_name=character_name,
        )

        # Store in persistent storage
        if self.config.storage_enabled and self.current_session_id:
            try:
                start_time = time.time()
                self.conversation_storage.store_turn(
                    turn, self.current_session_id, self.device_id
                )
                duration = time.time() - start_time

                # Record metrics
                self.metrics.record_turn_stored(character_name)
                self.metrics.record_db_operation("store_turn", "success", duration)

                logger.debug(
                    f"Stored turn {turn.turn_id} in session {self.current_session_id}"
                )
            except Exception as e:
                self.metrics.record_db_operation("store_turn", "error", 0.0)
                logger.error(f"Failed to store turn in database: {e}", exc_info=True)

        # Extract and update preferences
        if self.config.preferences_enabled:
            try:
                self._update_preferences(user_input, character_response)
            except Exception as e:
                self.metrics.record_preference_extraction("error", "failed")
                logger.error(f"Failed to update preferences: {e}", exc_info=True)

        # Check if we should summarize
        if self.config.summarization_enabled:
            recent_turns = self.conversation_storage.get_recent_turns(
                self.device_id, character_name, self.config.summarize_every_n_turns
            )

            if len(recent_turns) >= self.config.summarize_every_n_turns:
                try:
                    start_time = time.time()
                    self.conversation_summarizer.create_summary(
                        self.device_id, character_name, recent_turns
                    )
                    duration = time.time() - start_time
                    self.metrics.record_summarization(character_name, duration)
                except Exception as e:
                    self.metrics.record_summarization(character_name, 0.0)
                    logger.error(f"Failed to create summary: {e}", exc_info=True)

        # Also add to SessionMemory for immediate context
        self.session_memory.add_turn(character_name, user_input, character_response)

        logger.debug(f"Processed turn for {character_name}")

    def build_context_for_llm(
        self, character_name: str, current_user_input: str
    ) -> str:
        """Build comprehensive context for LLM including preferences, summaries, and recent turns."""
        start_time = time.time()

        if not self.config.enabled:
            return self.session_memory.format_context_for_llm(
                character_name, current_user_input
            )

        context_parts = []

        # Add preferences
        if self.config.include_preferences and self.config.preferences_enabled:
            preferences = self._get_user_preferences()
            if preferences:
                context_parts.append(self._format_preferences_for_context(preferences))

        # Add conversation summaries
        if self.config.include_summaries and self.config.summarization_enabled:
            summaries_context = (
                self.conversation_summarizer.format_summaries_for_context(
                    self.device_id, character_name
                )
            )
            if summaries_context:
                context_parts.append(summaries_context)

        # Add recent conversation turns
        if self.config.storage_enabled:
            recent_turns = self.conversation_storage.get_recent_turns(
                self.device_id, character_name, self.config.include_recent_turns
            )
            if recent_turns:
                context_parts.append(self._format_turns_for_context(recent_turns))
        else:
            # Fallback to SessionMemory
            session_context = self.session_memory.format_context_for_llm(
                character_name, current_user_input, self.config.include_recent_turns
            )
            if session_context:
                context_parts.append(session_context)

        # Add current user input
        context_parts.append(f"Current user input: {current_user_input}")

        # Build final context
        final_context = "\n\n".join(context_parts)

        # Record metrics
        duration = time.time() - start_time
        len(final_context.split())  # Rough token estimation
        self.metrics.record_context_build(character_name, duration)

        return final_context

    def _update_preferences(self, user_input: str, character_response: str) -> None:
        """Update user preferences from conversation."""
        try:
            # Load current preferences
            current_preferences = self.preference_storage.load_preferences(
                self.device_id
            )
            if not current_preferences:
                current_preferences = UserPreferences(user_id=self.device_id)

            # Extract preferences from user input
            user_extractions = self.preference_extractor.extract_preferences(user_input)

            # Record metrics for extracted preferences
            for pref_type, values in user_extractions.items():
                if values:  # Only record if values were extracted
                    for value in values:
                        self.metrics.record_preference_extraction(pref_type, "pattern")

            updated_preferences = self.preference_extractor.update_preferences(
                current_preferences, user_extractions
            )

            # Save updated preferences
            self.preference_storage.save_preferences(updated_preferences)

        except Exception as e:
            logger.error(f"Failed to update preferences: {e}")

    def _get_user_preferences(self) -> Optional[UserPreferences]:
        """Get user preferences."""
        try:
            return self.preference_storage.load_preferences(self.device_id)
        except Exception as e:
            logger.error(f"Failed to load preferences: {e}")
            return None

    def get_user_preferences(self) -> Optional[UserPreferences]:
        """Get user preferences (public method)."""
        return self._get_user_preferences()

    def _format_preferences_for_context(self, preferences: UserPreferences) -> str:
        """Format user preferences for LLM context."""
        context_lines = ["User information:"]

        if preferences.name:
            context_lines.append(f"- Name: {preferences.name}")

        if preferences.interests:
            context_lines.append(f"- Interests: {', '.join(preferences.interests)}")

        if preferences.favorite_color:
            context_lines.append(f"- Favorite color: {preferences.favorite_color}")

        if preferences.dislikes:
            context_lines.append(f"- Dislikes: {', '.join(preferences.dislikes)}")

        if preferences.occupation:
            context_lines.append(f"- Occupation: {preferences.occupation}")

        if preferences.age:
            context_lines.append(f"- Age: {preferences.age}")

        if preferences.location:
            context_lines.append(f"- Location: {preferences.location}")

        if preferences.custom_facts:
            for key, value in preferences.custom_facts.items():
                context_lines.append(f"- {key}: {value}")

        return "\n".join(context_lines)

    def _format_turns_for_context(self, turns: List[ConversationTurn]) -> str:
        """Format conversation turns for LLM context."""
        if not turns:
            return ""

        context_lines = ["Recent conversation:"]
        for turn in turns:
            context_lines.append(f"User: {turn.user_input}")
            context_lines.append(f"{turn.character_name}: {turn.character_response}")

        return "\n".join(context_lines)

    def search_conversation_history(
        self, character_name: str, query: str, limit: int = 10
    ) -> List[ConversationTurn]:
        """Search conversation history."""
        if not self.config.enabled or not self.config.storage_enabled:
            return []

        return self.conversation_storage.search_conversations(
            self.device_id, character_name, query, limit
        )

    def get_conversation_stats(self, character_name: str) -> Dict[str, Any]:
        """Get conversation statistics."""
        if not self.config.enabled or not self.config.storage_enabled:
            return {}

        return self.conversation_storage.get_session_stats(
            self.device_id, character_name
        )

    def export_user_data(self) -> Dict[str, Any]:
        """Export all user data (GDPR compliance)."""
        if not self.config.enabled:
            return {}

        export_data = {
            "user_id": self.device_id,
            "export_timestamp": time.time(),
        }

        # Export preferences
        if self.config.preferences_enabled:
            preferences = self._get_user_preferences()
            if preferences:
                export_data["preferences"] = preferences.to_dict()

        # Export conversation data
        if self.config.storage_enabled:
            export_data["conversations"] = self.conversation_storage.export_user_data(
                self.device_id
            )

        return export_data

    def delete_user_data(self) -> bool:
        """Delete all user data (GDPR compliance)."""
        if not self.config.enabled:
            return True

        success = True

        # Delete preferences
        if self.config.preferences_enabled:
            success &= self.preference_storage.delete_user_data(self.device_id)

        # Delete conversation data
        if self.config.storage_enabled:
            success &= self.conversation_storage.delete_user_data(self.device_id)

        # Clear summaries
        if self.config.summarization_enabled:
            self.conversation_summarizer.clear_summaries(
                self.device_id, "all"  # Clear for all characters
            )

        # Clear session memory
        self.session_memory.clear_all_conversations()

        return success

    def cleanup_old_data(self) -> Dict[str, int]:
        """Clean up old data and return cleanup statistics."""
        if not self.config.enabled:
            return {}

        cleanup_stats = {}

        # Cleanup conversation storage
        if self.config.storage_enabled:
            cleanup_stats[
                "conversations_deleted"
            ] = self.conversation_storage.cleanup_old_data()

        return cleanup_stats

    def get_memory_stats(self) -> Dict[str, Any]:
        """Get comprehensive memory system statistics."""
        if not self.config.enabled:
            return {"enabled": False}

        stats = {
            "enabled": True,
            "device_id": self.device_id,
            "current_session_id": self.current_session_id,
        }

        # Storage stats
        if self.config.storage_enabled:
            stats["storage"] = self.conversation_storage.get_database_stats()

        # Summarization stats
        if self.config.summarization_enabled:
            stats["summarization"] = self.conversation_summarizer.get_summary_stats()

        # Session memory stats
        session_stats = self.session_memory.get_conversation_summary("all")
        stats["session_memory"] = session_stats

        return stats
