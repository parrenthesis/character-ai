"""LLM response generation service."""

import logging
from typing import TYPE_CHECKING, Any, Dict

from ..characters import Character, CharacterResponseFilter
from ..core.exceptions import handle_model_error
from .base_service import BaseService
from .error_messages import ServiceErrorMessages

if TYPE_CHECKING:
    from ..algorithms.conversational_ai.session_memory import SessionMemory
    from ..algorithms.conversational_ai.text_normalizer import TextNormalizer
    from ..core.llm.template_prompt_builder import TemplatePromptBuilder
    from ..core.resource_manager import ResourceManager

logger = logging.getLogger(__name__)


class LLMService(BaseService):
    """LLM response generation service.

    Handles character-specific response generation with filtering and normalization.
    """

    def __init__(
        self,
        resource_manager: "ResourceManager",
        text_normalizer: "TextNormalizer",
        prompt_builder: "TemplatePromptBuilder",
        session_memory: "SessionMemory",
    ):
        super().__init__(resource_manager)
        self.text_normalizer = text_normalizer
        self.prompt_builder = prompt_builder
        self.session_memory = session_memory
        self.character_filters: Dict[str, CharacterResponseFilter] = {}

    def _get_processor(self, model_type: str) -> Any:
        """Get LLM processor from resource manager."""
        return self.resource_manager.get_llm_processor()

    @handle_model_error
    async def generate_response(self, text: str, character: Character) -> str:
        """Generate character-specific response.

        Args:
            text: User input text
            character: Character to generate response for

        Returns:
            Generated response text
        """
        # Get or create character-specific filter
        character_name = (
            character.name if hasattr(character, "name") else str(character)
        )

        if character_name not in self.character_filters:
            franchise = (
                getattr(character, "franchise", None)
                or (character.metadata.get("franchise") if character.metadata else None)
                or character_name.lower()
            )
            self.character_filters[character_name] = CharacterResponseFilter(
                character, franchise
            )
            logger.debug(f"Initialized CharacterResponseFilter for {character_name}")

        # Use common processor initialization pattern
        llm_processor = await self.get_or_create_processor("llm")

        # Get conversation depth for template selection
        conversation_depth = self.session_memory.get_conversation_depth(character_name)

        # Build prompt using Jinja2 template (conversation-aware)
        character_prompt = self.prompt_builder.build_prompt(
            user_input=text,
            character=character,
            conversation_context=self.session_memory.format_context_for_llm(
                character_name=character_name, current_user_input=text, max_turns=5
            ),
            conversation_depth=conversation_depth,
        )

        # Use processor
        result = await llm_processor.process_text(character_prompt)
        response_text = (
            result.text
            if result.text
            else ServiceErrorMessages.get_llm_fallback(
                character_name, is_generation_error=False
            )
        )

        # Debug logging to see raw LLM output before cleaning
        logger.debug(f"Raw LLM output before filtering: '{response_text}'")

        # Apply character-specific filtering BEFORE generic text normalization
        conversation_history = self.session_memory.get_conversation_context(
            character_name
        )
        response_text = self.character_filters[character_name].filter_response(
            response_text, conversation_history
        )
        logger.debug(f"After character filter: '{response_text}'")

        # Then apply generic text normalization
        response_text = self.text_normalizer.clean_llm_response(response_text)
        logger.debug(f"After text normalization: '{response_text}'")

        # Mark model as used
        self.resource_manager.mark_model_used("llm")
        return response_text
