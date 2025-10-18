"""Jinja2-based prompt builder for flexible, data-driven prompt templates."""

import logging
import os
from typing import Any, Dict, Optional

from jinja2 import Environment, FileSystemLoader, Template, TemplateNotFound

from ...characters import Character
from ..config import Config

logger = logging.getLogger(__name__)


class TemplatePromptBuilder:
    """
    Builds LLM prompts using Jinja2 templates.

    Replaces hardcoded f-strings with data-driven templates for:
    - Easier iteration on prompt content
    - Conversation-aware prompts
    - Character-specific template inheritance
    - Non-dev-friendly prompt editing
    """

    def __init__(
        self, config: Optional[Config] = None, template_dir: Optional[str] = None
    ):
        self.config = config or Config()
        self.template_dir = template_dir or "configs"

        # Setup Jinja2 environment
        self.env = Environment(
            loader=FileSystemLoader(self.template_dir),
            trim_blocks=True,
            lstrip_blocks=True,
            keep_trailing_newline=False,
        )

        # Add custom filters
        self.env.filters["replace_underscores"] = lambda s: s.replace("_", " ")
        self.env.filters["enum_value"] = (
            lambda e: e.value if hasattr(e, "value") else str(e)
        )

    def build_prompt(
        self,
        user_input: str,
        character: Character,
        conversation_context: Optional[str] = None,
        conversation_depth: int = 0,
    ) -> str:
        """Build prompt from Jinja2 template with conversation awareness."""

        # Determine which template to use
        template = self._get_template(character, conversation_depth)

        # Load character-specific prompt config from prompts.yaml
        prompts_config = self._load_prompts_yaml(character)

        # Build template context
        context = self._build_template_context(
            character=character,
            user_input=user_input,
            conversation_context=conversation_context,
            conversation_depth=conversation_depth,
            prompts_config=prompts_config,
        )

        # Render template
        try:
            prompt = template.render(**context)
            logger.debug(
                f"Rendered prompt template for {character.name} (depth={conversation_depth})"
            )
            return prompt
        except Exception as e:
            logger.error(f"Failed to render template: {e}")
            # Fallback to basic prompt
            return self._fallback_prompt(user_input, character, conversation_context)

    def _get_template(self, character: Character, conversation_depth: int) -> Template:
        """Select appropriate template based on character and conversation state."""
        franchise = getattr(character, "franchise", character.name.lower())

        # Try conversation-aware template first if we have conversation history
        if conversation_depth > 0:
            template_paths = [
                f"characters/{franchise}/{character.name.lower()}/templates/conversational_prompt.jinja2",
                f"characters/{franchise}/data/templates/conversational_prompt.jinja2",
            ]
            for path in template_paths:
                try:
                    return self.env.get_template(path)
                except TemplateNotFound:
                    continue

        # Try base character template
        template_paths = [
            f"characters/{franchise}/{character.name.lower()}/templates/base_prompt.jinja2",
            f"characters/{franchise}/data/templates/base_prompt.jinja2",
            "templates/prompts/base_character.jinja2",  # Generic fallback
        ]

        for path in template_paths:
            try:
                return self.env.get_template(path)
            except TemplateNotFound:
                continue

        # If no template found, create inline fallback
        logger.warning(f"No template found for {character.name}, using inline fallback")
        return Template(self._get_inline_fallback_template())

    def _load_prompts_yaml(self, character: Character) -> Dict[str, Any]:
        """Load prompts.yaml configuration for template variables."""
        franchise = getattr(character, "franchise", character.name.lower())
        paths = [
            os.path.join(
                "configs",
                "characters",
                franchise,
                character.name.lower(),
                "prompts.yaml",
            ),
            os.path.join("configs", "characters", franchise, "data", "prompts.yaml"),
        ]

        for path in paths:
            if os.path.exists(path):
                try:
                    from pathlib import Path

                    from ..config.yaml_loader import YAMLConfigLoader

                    return YAMLConfigLoader.load_yaml(Path(path))
                except Exception as e:
                    logger.warning(f"Failed to load prompts.yaml from {path}: {e}")

        return {}

    def _build_template_context(
        self,
        character: Character,
        user_input: str,
        conversation_context: Optional[str],
        conversation_depth: int,
        prompts_config: Dict[str, Any],
    ) -> Dict[str, Any]:
        """Build context dictionary for template rendering."""

        # Load character-specific filters config for max_words
        filters_config = self._load_filters_config(character)
        max_words = filters_config.get("max_words", 25)

        # Extract examples from prompts.yaml if available, otherwise use generic defaults
        good_examples = prompts_config.get("examples", {}).get(
            "good",
            [
                "Hello! Nice to meet you.",
                "That's interesting. Tell me more.",
                "I understand what you mean.",
            ],
        )

        bad_examples = prompts_config.get("examples", {}).get(
            "bad",
            [
                "I am ready to assist you with any questions you may have at this time.",
                "As an AI system, I process information according to my programming parameters.",
            ],
        )

        return {
            "character": character,
            "user_input": user_input,
            "conversation_context": conversation_context,
            "conversation_depth": conversation_depth,
            "response_config": {
                "max_words": max_words,
                "style": "conversational" if conversation_depth > 3 else "measured",
            },
            "good_examples": good_examples,
            "bad_examples": bad_examples,
        }

    def _load_filters_config(self, character: Character) -> Dict[str, Any]:
        """Load character-specific filters configuration."""
        franchise = getattr(character, "franchise", character.name.lower())
        paths = [
            os.path.join(
                "configs",
                "characters",
                franchise,
                character.name.lower(),
                "filters.yaml",
            ),
            os.path.join("configs", "characters", franchise, "data", "filters.yaml"),
        ]

        for path in paths:
            if os.path.exists(path):
                try:
                    from pathlib import Path

                    from ..config.yaml_loader import YAMLConfigLoader

                    config = YAMLConfigLoader.load_yaml(Path(path))
                    logger.debug(f"Loaded filters config from {path}")
                    return config
                except Exception as e:
                    logger.warning(f"Failed to load filters from {path}: {e}")

        # Return defaults if no file found
        return {"max_words": 25}

    def _fallback_prompt(
        self, user_input: str, character: Character, conversation_context: Optional[str]
    ) -> str:
        """Simple fallback if template rendering fails."""
        character_name = character.name
        context_section = (
            f"\n\n<conversation_history>\n{conversation_context}\n</conversation_history>"
            if conversation_context
            else ""
        )

        return f"""You are {character_name}, a character.

CRITICAL: Respond with ONE brief statement (under 30 words).

{context_section}

User: {user_input}

{character_name}:"""

    def _get_inline_fallback_template(self) -> str:
        """Inline fallback template string."""
        return """You are {{ character.name }}, a {{ character.dimensions.species|enum_value }}.

CRITICAL: Respond with ONE brief statement.

{% if conversation_context %}
<conversation_history>
{{ conversation_context }}
</conversation_history>
{% endif %}

User: {{ user_input }}

{{ character.name }}:"""
