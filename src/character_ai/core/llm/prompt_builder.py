"""LLM prompt building utilities."""

import os
from typing import Any, Dict, Optional

import yaml

from ...characters.types import Character
from ..config import Config


class LLMPromptBuilder:
    """Builds LLM prompts from character profiles and conversation context."""

    def __init__(self, config: Optional[Config] = None):
        self.config = config or Config()

    def build_prompt(
        self,
        user_input: str,
        character: Character,
        conversation_context: Optional[str] = None,
    ) -> str:
        """Build complete prompt from character prompts.yaml + context"""
        # Try to load character-specific prompts first
        franchise = getattr(character, "franchise", None)
        if not franchise:
            # Try to infer franchise from character metadata or use a default
            franchise = (
                character.metadata.get("franchise") if character.metadata else None
            )
        if not franchise:
            # Last resort: use character name as franchise (for backward compatibility)
            franchise = character.name.lower()

        prompts = self.load_character_prompts(character.name, franchise)

        # Use system prompt from prompts.yaml if available, otherwise build from character info
        if prompts.get("system_prompt"):
            system_prompt = prompts["system_prompt"]
        else:
            # Build system prompt from character information
            character_name = character.name
            character_type = (
                character.dimensions.species.value
                if hasattr(character, "dimensions")
                else "character"
            )
            voice_style = (
                character.voice_style
                if hasattr(character, "voice_style")
                else "friendly"
            )
            topics = (
                ", ".join([topic.value for topic in character.dimensions.topics[:5]])
                if hasattr(character, "dimensions") and character.dimensions.topics
                else "general topics"
            )

            system_prompt = f"""You are {character_name}, a {character_type} with a {voice_style} voice.
You love talking about: {topics}.

Key characteristics:
- {character_type}
- {voice_style} personality
- Interested in {topics}

CRITICAL: Respond with EXACTLY ONE brief statement (under 50 words).

Output rules:
- Reply as a single spoken line only.
- Do not include any dialogue labels such as "user:" or "{character_name.lower()}:".
- Do not include thoughts, stage directions, or explanations.
- Do not ask questions unless the user's input requires clarification."""

        # Add conversation context if provided
        if conversation_context:
            system_prompt += f"\n\n<conversation_history>\n{conversation_context}\n</conversation_history>"

        # Build the complete prompt
        prompt = f"{system_prompt}\n\nUser: {user_input}\n\n{character.name}:"

        return prompt

    def load_character_prompts(
        self, character_name: str, franchise: str
    ) -> Dict[str, Any]:
        """Load prompts.yaml for character"""
        try:
            # Construct path to prompts.yaml - try multiple possible locations
            possible_paths = [
                os.path.join(
                    "configs",
                    "characters",
                    franchise,
                    character_name.lower(),
                    "prompts.yaml",
                ),
                os.path.join("configs", "characters", franchise, "prompts.yaml"),
                os.path.join(
                    "configs", "characters", character_name.lower(), "prompts.yaml"
                ),
                os.path.join(
                    "configs", "characters", franchise, "data", "prompts.yaml"
                ),  # Legacy path for backward compatibility
            ]

            prompts_path = None
            for path in possible_paths:
                if os.path.exists(path):
                    prompts_path = path
                    break

            if prompts_path:
                with open(prompts_path, "r", encoding="utf-8") as f:
                    prompts = yaml.safe_load(f)
                return prompts or {}
            else:
                # Return empty dict if no prompts file found - let build_prompt handle the fallback
                return {}
        except Exception:
            # Return empty dict on error - let build_prompt handle the fallback
            return {}
