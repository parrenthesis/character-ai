"""
Character creation commands for the Character AI CLI.
"""

import json
import logging
import re
from pathlib import Path
from typing import Any

import click

from ...characters import CharacterService
from ...characters.management import CharacterProfile
from ...core.llm.config import LLMConfigService
from ...core.llm.factory import LLMFactory
from ...core.llm.manager import OpenModelService
from .helpers import (
    _save_character,
    _save_character_filters,
    _save_character_profile,
    _save_character_prompts,
    _save_character_templates,
    _save_enhanced_character,
)

logger = logging.getLogger(__name__)


@click.command()
@click.option("--name", help="Name of the character")
@click.option("--species", help="Character species (pony, robot, dragon, etc.)")
@click.option("--personality", help="Character personality traits")
@click.option("--voice-style", help="Voice style (happy, friendly, mystical, etc.)")
@click.option(
    "--interactive", "-i", is_flag=True, help="Interactive character creation"
)
@click.option("--template", help="Use a character template")
@click.option("--ai-generate", help="Generate character from AI description")
@click.option("--output", "-o", help="Save character to file")
def create(
    name: str,
    species: str,
    personality: str,
    voice_style: str,
    interactive: bool,
    template: str,
    ai_generate: str,
    output: str,
) -> None:
    """Create a new character."""
    try:
        if interactive:
            import asyncio

            asyncio.run(_interactive_character_creation())
            return

        if template:
            _create_from_template(template, name)
            return

        if ai_generate:
            _create_from_ai_description(ai_generate, name)
            return

        # Prompt for missing required fields
        if not name:
            name = click.prompt("Character name")
        if not species:
            species = click.prompt("Species")
        if not personality:
            personality = click.prompt("Personality")
        if not voice_style:
            voice_style = click.prompt("Voice style")

        # Create character profile
        character_profile = CharacterProfile(
            id=name.lower().replace(" ", "_"),
            display_name=name,
            character_type=species,
            voice_style=voice_style,
            language="en",
            traits={"personality": personality, "voice_style": voice_style},
        )

        # Save character
        if output:
            _save_character(character_profile, output)
        else:
            # Save to new schema format in configs/characters/
            character_id = name.lower().replace(" ", "_")
            character_dir = Path.cwd() / "configs" / "characters" / character_id
            character_dir.mkdir(parents=True, exist_ok=True)

            # Create voice_samples directory
            voice_samples_dir = character_dir / "voice_samples"
            voice_samples_dir.mkdir(exist_ok=True)

            # Create profile.yaml
            profile_file = character_dir / "profile.yaml"
            _save_character_profile(character_profile, str(profile_file))

            # Create prompts.yaml
            prompts_file = character_dir / "prompts.yaml"
            _save_character_prompts(character_profile, str(prompts_file))

            # Create filters.yaml (Jinja2 templates + CharacterResponseFilter)
            filters_file = character_dir / "filters.yaml"
            _save_character_filters(character_profile, str(filters_file))

            # Create templates directory with Jinja2 templates
            templates_dir = character_dir / "templates"
            templates_dir.mkdir(exist_ok=True)
            _save_character_templates(character_profile, templates_dir)

            click.echo(f"✓ Character directory created: {character_dir}")
            click.echo(f"✓ Voice samples directory created: {voice_samples_dir}")
            click.echo(f"✓ Templates directory created: {templates_dir}")
            click.echo(f"✓ Response filters configured: {filters_file}")

        click.echo(f"✓ Character '{name}' created successfully!")

    except Exception as e:
        click.echo(f"Error creating character: {e}", err=True)
        raise click.Abort()


async def _interactive_character_creation() -> None:
    """Interactive character creation using AI."""
    try:
        click.echo("🤖 AI-Powered Character Creation")
        click.echo("=" * 35)

        # Get user input
        description = click.prompt(
            "Describe your character (e.g., 'a friendly robot who loves music')"
        )

        # Use LLM to generate character
        config_manager = LLMConfigService()
        model_manager = OpenModelService()
        factory = LLMFactory(config_manager, model_manager)

        # Get character creation LLM
        llm = factory.get_character_creation_llm()

        # Generate character using LLM
        prompt = f"""Create a character profile based on this description: "{description}"

Please provide a JSON response with:
- name: Character name
- species: Character species
- personality: Key personality traits
- voice_style: Voice characteristics
- traits: Additional character traits
- backstory: Brief character backstory

Format as valid JSON only."""

        click.echo("Generating character with AI...")
        response = await llm.generate(prompt, max_tokens=500, temperature=0.8)

        # Parse JSON response
        try:
            json_match = re.search(r"\{.*\}", response, re.DOTALL)
            if json_match:
                character_data = json.loads(json_match.group())

                # Create character profile
                character_profile = CharacterProfile(
                    id=character_data.get("name", "Unknown").lower().replace(" ", "_"),
                    display_name=character_data.get("name", "Unknown"),
                    character_type=character_data.get("species", "Unknown"),
                    voice_style=character_data.get("voice_style", "neutral"),
                    language="en",
                    traits=character_data.get("traits", {}),
                )

                # Display generated character
                click.echo("\n🎭 Generated Character:")
                click.echo(f"Name: {character_profile.display_name}")
                click.echo(f"Species: {character_profile.character_type}")
                click.echo(
                    f"Personality: {character_profile.traits.get('personality', '')}"
                )
                click.echo(f"Voice Style: {character_profile.voice_style}")
                if character_profile.traits.get("backstory"):
                    click.echo(
                        f"Backstory: {character_profile.traits.get('backstory')}"
                    )

                # Ask if user wants to save
                if click.confirm("\nSave this character?"):
                    # Save to new schema format in configs/characters/
                    character_id = character_profile.id
                    character_dir = Path.cwd() / "configs" / "characters" / character_id
                    character_dir.mkdir(parents=True, exist_ok=True)

                    # Create voice_samples directory
                    voice_samples_dir = character_dir / "voice_samples"
                    voice_samples_dir.mkdir(exist_ok=True)

                    # Create profile.yaml
                    profile_file = character_dir / "profile.yaml"
                    _save_character_profile(character_profile, str(profile_file))

                    # Create prompts.yaml
                    prompts_file = character_dir / "prompts.yaml"
                    _save_character_prompts(character_profile, str(prompts_file))

                    # Create filters.yaml (Jinja2 templates + CharacterResponseFilter)
                    filters_file = character_dir / "filters.yaml"
                    _save_character_filters(character_profile, str(filters_file))

                    # Create templates directory with Jinja2 templates
                    templates_dir = character_dir / "templates"
                    templates_dir.mkdir(exist_ok=True)
                    _save_character_templates(character_profile, templates_dir)

                    click.echo(f"✓ Character directory created: {character_dir}")
                    click.echo(
                        f"✓ Voice samples directory created: {voice_samples_dir}"
                    )
                    click.echo(f"✓ Templates directory created: {templates_dir}")
                    click.echo(f"✓ Response filters configured: {filters_file}")
                else:
                    click.echo("Character not saved.")
            else:
                click.echo("Could not parse AI response as JSON")
                click.echo(f"Raw response: {response}")
        except json.JSONDecodeError as e:
            click.echo(f"Error parsing AI response: {e}")
            click.echo(f"Raw response: {response}")

    except Exception as e:
        click.echo(f"Error in interactive creation: {e}", err=True)
        raise click.Abort()


def _create_from_template(template_name: str, name: str) -> None:
    """Create character from template."""
    try:
        # Use character manager
        manager = CharacterService()

        # Get available templates
        templates = manager.get_available_templates()

        # Find the template by name or key
        template = None
        for t in templates:
            if (
                t.name.lower().replace(" ", "_") == template_name
                or t.name.lower() == template_name
                or template_name in t.name.lower()
            ):
                template = t
                break

        if not template:
            click.echo(f"Unknown template: {template_name}")
            template_names = [t.name.lower().replace(" ", "_") for t in templates]
            click.echo(f"Available templates: {', '.join(template_names)}")
            return

        # Create character from template
        character = template.create_character(name)

        # Save character to new schema format
        character_id = name.lower().replace(" ", "_")
        character_dir = Path.cwd() / "configs" / "characters" / character_id
        character_dir.mkdir(parents=True, exist_ok=True)

        # Create voice_samples directory (raw input)
        voice_samples_dir = character_dir / "voice_samples"
        voice_samples_dir.mkdir(exist_ok=True)

        # Create profile.yaml
        profile_file = character_dir / "profile.yaml"
        _save_enhanced_character(character, str(profile_file))

        # Create prompts.yaml
        prompts_file = character_dir / "prompts.yaml"
        _save_enhanced_character_prompts(character, str(prompts_file))

        # Create filters.yaml (Jinja2 templates + CharacterResponseFilter)
        filters_file = character_dir / "filters.yaml"
        # Convert Character to CharacterProfile format for _save_character_filters
        from ...characters.management import CharacterProfile

        temp_profile = CharacterProfile(
            id=character.name.lower().replace(" ", "_"),
            display_name=character.name,
            character_type=character.dimensions.species.value,
            voice_style=character.voice_style,
            language=character.language,
            traits={
                "personality": ", ".join(
                    [t.value for t in character.dimensions.personality_traits]
                )
            },
        )
        _save_character_filters(temp_profile, str(filters_file))

        # Create templates directory with Jinja2 templates
        templates_dir = character_dir / "templates"
        templates_dir.mkdir(exist_ok=True)
        _save_character_templates(temp_profile, templates_dir)

        click.echo(f"✓ Character directory created: {character_dir}")
        click.echo(f"✓ Voice samples directory created: {voice_samples_dir}")
        click.echo(f"✓ Templates directory created: {templates_dir}")
        click.echo(f"✓ Response filters configured: {filters_file}")

        click.echo(f"✓ Character '{name}' created from template '{template_name}'!")
        click.echo(f"Species: {character.dimensions.species.value}")
        click.echo(f"Archetype: {character.dimensions.archetype.value}")
        personality_traits = ", ".join(
            [t.value for t in character.dimensions.personality_traits]
        )
        click.echo(f"Personality: {personality_traits}")

    except Exception as e:
        click.echo(f"Error creating character from template: {e}", err=True)
        raise click.Abort()


def _create_from_ai_description(description: str, name: str) -> None:
    """Create character from AI description."""
    try:
        click.echo(f"🤖 Generating character from description: '{description}'")

        # Use character manager
        import asyncio

        manager = CharacterService()
        asyncio.run(manager.initialize())

        # Generate character
        character = asyncio.run(
            manager.generate_character_from_description(description, name)
        )

        if character:
            click.echo(f"✓ Character '{character.name}' generated successfully!")
            click.echo(f"Species: {character.dimensions.species.value}")
            click.echo(f"Archetype: {character.dimensions.archetype.value}")
            click.echo(
                f"Personality: {', '.join([t.value for t in character.dimensions.personality_traits])}"
            )
            if character.dimensions.backstory:
                click.echo(f"Backstory: {character.dimensions.backstory}")
        else:
            click.echo("✗ Failed to generate character")

    except Exception as e:
        click.echo(f"Error generating character: {e}", err=True)
        raise click.Abort()


def _save_enhanced_character_prompts(character: Any, file_path: str) -> None:
    """Save enhanced character prompts to YAML file."""
    import yaml

    prompts_data = {
        "system_prompt": f"You are {character.name}, a {character.dimensions.species.value} with a {character.voice_style} voice style.",
        "greeting": f"Hello! I'm {character.name}. How can I help you today?",
        "personality": ", ".join(
            [t.value for t in character.dimensions.personality_traits]
        ),
        "voice_style": character.voice_style,
        "language": character.language,
        "backstory": character.dimensions.backstory,
        "goals": character.dimensions.goals,
        "fears": character.dimensions.fears,
        "likes": character.dimensions.likes,
        "dislikes": character.dimensions.dislikes,
    }

    with open(file_path, "w") as f:
        yaml.dump(prompts_data, f, default_flow_style=False, allow_unicode=True)
