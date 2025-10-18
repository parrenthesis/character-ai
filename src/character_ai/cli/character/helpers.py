"""
Shared helper functions for character CLI commands.
"""

import json
import logging
from pathlib import Path
from typing import List

import yaml

from ...characters import Character, CharacterService
from ...characters.management import CharacterProfile

logger = logging.getLogger(__name__)


def _save_character(character_profile: CharacterProfile, file_path: str) -> None:
    """Save character profile to YAML file."""
    character_data = {
        "id": character_profile.id,
        "display_name": character_profile.display_name,
        "character_type": character_profile.character_type,
        "voice_style": character_profile.voice_style,
        "language": character_profile.language,
        "traits": character_profile.traits,
    }

    with open(file_path, "w") as f:
        yaml.dump(character_data, f, default_flow_style=False)


def _save_enhanced_character(character: Character, file_path: str) -> None:
    """Save enhanced character to YAML file."""
    character_data = {
        "name": character.name,
        "voice_style": character.voice_style,
        "language": character.language,
        "metadata": character.metadata,
        "dimensions": {
            "species": character.dimensions.species.value,
            "archetype": character.dimensions.archetype.value,
            "personality_traits": [
                trait.value for trait in character.dimensions.personality_traits
            ],
            "abilities": [ability.value for ability in character.dimensions.abilities],
            "topics": [topic.value for topic in character.dimensions.topics],
            "backstory": character.dimensions.backstory,
            "goals": character.dimensions.goals,
            "fears": character.dimensions.fears,
            "likes": character.dimensions.likes,
            "dislikes": character.dimensions.dislikes,
        },
    }

    with open(file_path, "w") as f:
        yaml.dump(character_data, f, default_flow_style=False)


def _save_character_profile(
    character_profile: CharacterProfile, file_path: str
) -> None:
    """Save character profile to YAML file with enhanced structure."""
    profile_data = {
        "id": character_profile.id,
        "display_name": character_profile.display_name,
        "character_type": character_profile.character_type,
        "voice_style": character_profile.voice_style,
        "language": character_profile.language,
        "traits": character_profile.traits,
        "metadata": {
            "created_at": "2024-01-01T00:00:00Z",
            "version": "1.0",
            "source": "cli_creation",
        },
    }

    with open(file_path, "w") as f:
        yaml.dump(profile_data, f, default_flow_style=False, allow_unicode=True)


def _save_character_prompts(
    character_profile: CharacterProfile, file_path: str
) -> None:
    """Save character prompts to YAML file."""
    prompts_data = {
        "system_prompt": f"You are {character_profile.display_name}, a {character_profile.character_type} with a {character_profile.voice_style} voice style.",
        "greeting": f"Hello! I'm {character_profile.display_name}. How can I help you today?",
        "personality": character_profile.traits.get("personality", ""),
        "voice_style": character_profile.voice_style,
        "language": character_profile.language,
    }

    with open(file_path, "w") as f:
        yaml.dump(prompts_data, f, default_flow_style=False, allow_unicode=True)


def _save_character_filters(
    character_profile: CharacterProfile, file_path: str
) -> None:
    """Save character response filters to YAML file."""
    filters_data = {
        "response_filters": {
            "max_length": 500,
            "min_length": 10,
            "personality_consistency": True,
            "voice_style_consistency": True,
            "safety_filters": True,
        },
        "content_guidelines": {
            "personality": character_profile.traits.get("personality", ""),
            "voice_style": character_profile.voice_style,
            "character_type": character_profile.character_type,
        },
    }

    with open(file_path, "w") as f:
        yaml.dump(filters_data, f, default_flow_style=False, allow_unicode=True)


def _save_character_templates(
    character_profile: CharacterProfile, templates_dir: Path
) -> None:
    """Save character Jinja2 templates to templates directory."""
    # System prompt template
    system_template = f"""You are {character_profile.display_name}, a {character_profile.character_type}.

Personality: {character_profile.traits.get('personality', '')}
Voice Style: {character_profile.voice_style}
Language: {character_profile.language}

Always stay in character and respond according to your personality and voice style."""

    # Greeting template
    greeting_template = (
        f"""Hello! I'm {character_profile.display_name}. How can I help you today?"""
    )

    # Response template
    response_template = """{{ character_name }}: {{ response }}"""

    # Save templates
    (templates_dir / "system_prompt.j2").write_text(system_template)
    (templates_dir / "greeting.j2").write_text(greeting_template)
    (templates_dir / "response.j2").write_text(response_template)


def _load_character_manager() -> CharacterService:
    """Load and initialize character service."""
    import asyncio

    service = CharacterService()
    asyncio.run(service.initialize())
    return service


def _format_character_list(characters: List[Character], format: str = "table") -> str:
    """Format character list for display."""
    if not characters:
        return "No characters found."

    if format == "json":
        char_data = []
        for char in characters:
            char_data.append(
                {
                    "name": char.name,
                    "species": char.dimensions.species.value,
                    "archetype": char.dimensions.archetype.value,
                    "personality_traits": [
                        t.value for t in char.dimensions.personality_traits
                    ],
                    "abilities": [a.value for a in char.dimensions.abilities],
                    "topics": [t.value for t in char.dimensions.topics],
                    "voice_style": char.voice_style,
                    "backstory": char.dimensions.backstory,
                }
            )
        return json.dumps(char_data, indent=2)
    elif format == "yaml":
        char_data = []
        for char in characters:
            char_data.append(
                {
                    "name": char.name,
                    "species": char.dimensions.species.value,
                    "archetype": char.dimensions.archetype.value,
                    "personality_traits": [
                        t.value for t in char.dimensions.personality_traits
                    ],
                    "abilities": [a.value for a in char.dimensions.abilities],
                    "topics": [t.value for t in char.dimensions.topics],
                    "voice_style": char.voice_style,
                    "backstory": char.dimensions.backstory,
                }
            )
        return yaml.dump(char_data, default_flow_style=False, allow_unicode=True)
    else:  # table format
        from tabulate import tabulate

        table_data = []
        for char in characters:
            table_data.append(
                [
                    char.name,
                    char.dimensions.species.value,
                    char.dimensions.archetype.value,
                    ", ".join(
                        [t.value for t in char.dimensions.personality_traits[:2]]
                    ),
                    char.voice_style,
                ]
            )

        headers = ["Name", "Species", "Archetype", "Personality", "Voice Style"]
        return str(tabulate(table_data, headers=headers, tablefmt="grid"))
