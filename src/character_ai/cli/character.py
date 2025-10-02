"""
Character management commands for the Character AI CLI.

Provides Click-based commands for creating, managing, and interacting with characters.
"""

import json
import logging
from pathlib import Path
from typing import Any, Dict, List, Optional

import click

from ..characters.catalog import CharacterCatalog
from ..characters.catalog_storage import CatalogStorage
from ..characters.manager import CharacterManager
from ..characters.profile_models import CharacterProfile
from ..characters.types import Character
from ..core.llm.config import LLMConfigManager
from ..core.llm.factory import LLMFactory
from ..core.llm.manager import OpenModelManager

logger = logging.getLogger(__name__)


@click.group()
def character_commands() -> None:
    """Character creation and management commands."""
    pass


@character_commands.command()
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

            click.echo(f"✓ Character directory created: {character_dir}")
            click.echo(f"✓ Voice samples directory created: {voice_samples_dir}")

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
        config_manager = LLMConfigManager()
        model_manager = OpenModelManager()
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
            import re

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

                    click.echo(f"✓ Character directory created: {character_dir}")
                    click.echo(
                        f"✓ Voice samples directory created: {voice_samples_dir}"
                    )
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
        manager = CharacterManager()

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
        _save_enhanced_character_profile(character, str(profile_file))

        # Create prompts.yaml
        prompts_file = character_dir / "prompts.yaml"
        _save_enhanced_character_prompts(character, str(prompts_file))

        click.echo(f"✓ Character directory created: {character_dir}")
        click.echo(f"✓ Voice samples directory created: {voice_samples_dir}")

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

        manager = CharacterManager()
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


def _save_character(character_profile: CharacterProfile, file_path: str) -> None:
    """Save character profile to YAML file."""
    import yaml

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
    import yaml

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


@character_commands.command()
@click.option(
    "--format",
    type=click.Choice(["table", "json", "yaml"]),
    default="table",
    help="Output format",
)
def list(format: str) -> None:
    """List all available characters."""
    try:
        import asyncio

        manager = CharacterManager()
        asyncio.run(manager.initialize())
        characters = manager.list_characters()

        if not characters:
            click.echo("No characters found.")
            return

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
            click.echo(json.dumps(char_data, indent=2))
        elif format == "yaml":
            import yaml

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
            click.echo(yaml.dump(char_data, default_flow_style=False))
        else:
            # Table format
            click.echo("Available Characters:")
            click.echo("=" * 50)
            for char in characters:
                click.echo(f"Name: {char.name}")
                click.echo(f"Species: {char.dimensions.species.value}")
                click.echo(f"Archetype: {char.dimensions.archetype.value}")
                click.echo(
                    f"Personality: {', '.join([t.value for t in char.dimensions.personality_traits])}"
                )
                click.echo(
                    f"Abilities: {', '.join([a.value for a in char.dimensions.abilities])}"
                )
                click.echo(
                    f"Topics: {', '.join([t.value for t in char.dimensions.topics])}"
                )
                click.echo(f"Voice Style: {char.voice_style}")
                if char.dimensions.backstory:
                    click.echo(f"Backstory: {char.dimensions.backstory}")
                click.echo("-" * 30)

    except Exception as e:
        click.echo(f"Error listing characters: {e}", err=True)
        raise click.Abort()


@character_commands.command()
@click.argument("character_name")
@click.option(
    "--format",
    type=click.Choice(["json", "yaml"]),
    default="yaml",
    help="Output format",
)
def show(character_name: str, format: str) -> None:
    """Show detailed information about a character."""
    try:
        character_manager = CharacterManager()
        character = character_manager.get_character(character_name)

        if not character:
            click.echo(f"Character '{character_name}' not found.", err=True)
            raise click.Abort()

        # Create character data dict
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
                "abilities": [
                    ability.value for ability in character.dimensions.abilities
                ],
                "topics": [topic.value for topic in character.dimensions.topics],
                "backstory": character.dimensions.backstory,
                "goals": character.dimensions.goals,
                "fears": character.dimensions.fears,
                "likes": character.dimensions.likes,
                "dislikes": character.dimensions.dislikes,
            },
        }

        if format == "json":
            click.echo(json.dumps(character_data, indent=2))
        else:
            # YAML format
            import yaml

            click.echo(yaml.dump(character_data, default_flow_style=False))

    except Exception as e:
        click.echo(f"Error showing character: {e}", err=True)
        raise click.Abort()


@character_commands.command()
@click.argument("character_name")
def activate(character_name: str) -> None:
    """Activate a character for interaction."""
    try:
        character_manager = CharacterManager()
        # Check if character exists
        character = character_manager.get_character(character_name)
        success = character is not None

        if success:
            click.echo(f"✓ Character '{character_name}' activated!")
        else:
            click.echo(f"✗ Failed to activate character '{character_name}'", err=True)
            raise click.Abort()

    except Exception as e:
        click.echo(f"Error activating character: {e}", err=True)
        raise click.Abort()


@character_commands.command()
@click.argument("character_name")
def remove(character_name: str) -> None:
    """Remove a character."""
    try:
        if not click.confirm(
            f"Are you sure you want to remove character '{character_name}'?"
        ):
            click.echo("Character removal cancelled.")
            return

        character_manager = CharacterManager()
        success = character_manager.remove_character(character_name)

        if success:
            click.echo(f"✓ Character '{character_name}' removed!")
        else:
            click.echo(f"✗ Failed to remove character '{character_name}'", err=True)
            raise click.Abort()

    except Exception as e:
        click.echo(f"Error removing character: {e}", err=True)
        raise click.Abort()


@character_commands.command()
def templates() -> None:
    """List available character templates."""
    try:
        import asyncio

        manager = CharacterManager()
        asyncio.run(manager.initialize())
        templates = manager.get_available_templates()

        click.echo("Available Character Templates:")
        click.echo("=" * 50)
        for template in templates:
            click.echo(f"{template.name}: {template.description}")
            click.echo(f"  Species: {template.species.value}")
            click.echo(f"  Archetype: {template.archetype.value}")
            click.echo(
                f"  Personality: {', '.join([t.value for t in template.personality_traits])}"
            )
            click.echo(f"  Tags: {', '.join(template.tags)}")
            click.echo()

        click.echo(
            "Usage: cai character create --template <template_name> --name <character_name>"
        )

    except Exception as e:
        click.echo(f"Error listing templates: {e}", err=True)
        raise click.Abort()


@character_commands.command()
@click.option("--species", help="Filter by species")
@click.option("--archetype", help="Filter by archetype")
@click.option("--personality", help="Filter by personality trait")
@click.option("--ability", help="Filter by ability")
@click.option("--topic", help="Filter by topic")
def search(
    species: Optional[str],
    archetype: Optional[str],
    personality: Optional[str],
    ability: Optional[str],
    topic: Optional[str],
) -> None:
    """Search characters by criteria."""
    try:
        import asyncio

        manager = CharacterManager()
        asyncio.run(manager.initialize())

        # Build search criteria
        criteria = {}
        if species:
            criteria["species"] = species
        if archetype:
            criteria["archetype"] = archetype
        if personality:
            criteria["personality_traits"] = personality
        if ability:
            criteria["abilities"] = ability
        if topic:
            criteria["topics"] = topic

        # Search characters
        results = manager.search_characters(criteria)

        if not results:
            click.echo("No characters found matching criteria")
            return

        click.echo(f"Found {len(results)} characters:")
        click.echo("=" * 50)
        for character in results:
            click.echo(f"Name: {character.name}")
            click.echo(f"Species: {character.dimensions.species.value}")
            click.echo(f"Archetype: {character.dimensions.archetype.value}")
            click.echo(
                f"Personality: {', '.join([t.value for t in character.dimensions.personality_traits])}"
            )
            click.echo(
                f"Abilities: {', '.join([a.value for a in character.dimensions.abilities])}"
            )
            click.echo(
                f"Topics: {', '.join([t.value for t in character.dimensions.topics])}"
            )
            if character.dimensions.backstory:
                click.echo(f"Backstory: {character.dimensions.backstory}")
            click.echo("-" * 30)

    except Exception as e:
        click.echo(f"Error searching characters: {e}", err=True)
        raise click.Abort()


@character_commands.command()
def stats() -> None:
    """Show character statistics."""
    try:
        import asyncio

        manager = CharacterManager()
        asyncio.run(manager.initialize())
        stats = manager.get_character_statistics()

        click.echo("Character Statistics:")
        click.echo("=" * 30)
        click.echo(f"Total Characters: {stats['total_characters']}")

        click.echo("\nSpecies Distribution:")
        for species, count in stats["species_distribution"].items():
            click.echo(f"  {species}: {count}")

        click.echo("\nArchetype Distribution:")
        for archetype, count in stats["archetype_distribution"].items():
            click.echo(f"  {archetype}: {count}")

        click.echo("\nTop Personality Traits:")
        sorted_traits = sorted(
            stats["personality_traits_distribution"].items(),
            key=lambda x: x[1],
            reverse=True,
        )
        for trait, count in sorted_traits[:10]:
            click.echo(f"  {trait}: {count}")

        click.echo("\nTop Abilities:")
        sorted_abilities = sorted(
            stats["abilities_distribution"].items(), key=lambda x: x[1], reverse=True
        )
        for ability, count in sorted_abilities[:10]:
            click.echo(f"  {ability}: {count}")

        click.echo("\nTop Topics:")
        sorted_topics = sorted(
            stats["topics_distribution"].items(), key=lambda x: x[1], reverse=True
        )
        for topic, count in sorted_topics[:10]:
            click.echo(f"  {topic}: {count}")

    except Exception as e:
        click.echo(f"Error getting statistics: {e}", err=True)
        raise click.Abort()


@character_commands.command()
@click.argument("character_name")
@click.option("--message", "-m", help="Message to send to character")
@click.option("--interactive", "-i", is_flag=True, help="Interactive conversation mode")
def chat(character_name: str, message: Optional[str], interactive: bool) -> None:
    """Chat with a character."""
    try:
        character_manager = CharacterManager()
        character = character_manager.get_character(character_name)

        if not character:
            click.echo(f"Character '{character_name}' not found.", err=True)
            raise click.Abort()

        if interactive:
            click.echo(f"Starting conversation with {character.name}...")
            click.echo("Type 'exit' to end the conversation.")
            click.echo("=" * 50)

            while True:
                user_input = click.prompt("You", default="", show_default=False)
                if user_input.lower() in ["exit", "quit", "bye"]:
                    click.echo("Goodbye!")
                    break

                # This would integrate with the actual conversation system
                click.echo(f"{character.name}: [Response would be generated here]")
        else:
            if not message:
                click.echo(
                    "Please provide a message with --message or use --interactive mode"
                )
                return

            # This would integrate with the actual conversation system
            click.echo(f"Sending message to {character.name}: {message}")
            click.echo(f"{character.name}: [Response would be generated here]")

    except Exception as e:
        click.echo(f"Error chatting with character: {e}", err=True)
        raise click.Abort()


# Catalog Management Commands


@character_commands.group()
def catalog() -> None:
    """Character catalog management commands."""
    pass


@catalog.command()
@click.argument("catalog_file", type=click.Path(exists=True))
@click.option(
    "--franchise", default="imported", help="Franchise name for imported characters"
)
@click.option(
    "--voice-dir",
    type=click.Path(exists=True),
    help="Directory containing voice files for characters",
)
def import_catalog(
    catalog_file: str, franchise: Optional[str], voice_dir: Optional[str]
) -> None:
    """Import character catalog from YAML file with optional voice file support."""
    try:
        import asyncio

        catalog_storage = CatalogStorage()
        voice_path = Path(voice_dir) if voice_dir else None
        result = asyncio.run(
            catalog_storage.import_catalog(Path(catalog_file), voice_path)
        )

        click.echo(f"✓ Imported {result['imported_count']} characters")
        click.echo(f"Total characters in file: {result['total_characters']}")
        if result.get("voice_processed_count", 0) > 0:
            click.echo(f"✓ Processed {result['voice_processed_count']} voice files")

        if result["errors"]:
            click.echo(f"Errors encountered: {len(result['errors'])}")
            for error in result["errors"]:
                click.echo(f"  - {error}")

    except Exception as e:
        click.echo(f"Error importing catalog: {e}", err=True)
        raise click.Abort()


@catalog.command()
@click.option("--franchise", help="Export specific franchise only")
@click.option("--output", "-o", help="Output file path")
def export_catalog(franchise: Optional[str], output: Optional[str]) -> None:
    """Export character catalog to YAML file."""
    try:
        import asyncio

        catalog_storage = CatalogStorage()
        output_path = asyncio.run(
            catalog_storage.export_catalog(franchise, Path(output) if output else None)
        )

        click.echo(f"✓ Catalog exported to {output_path}")

    except Exception as e:
        click.echo(f"Error exporting catalog: {e}", err=True)
        raise click.Abort()


@catalog.command()
@click.option(
    "--text", help="Text search across character names, backstories, goals, and traits"
)
@click.option("--franchise", help="Filter by franchise")
@click.option("--species", help="Filter by species")
@click.option("--archetype", help="Filter by archetype")
@click.option("--personality", help="Filter by personality trait")
@click.option("--ability", help="Filter by ability")
@click.option("--topic", help="Filter by topic")
@click.option(
    "--voice-available", is_flag=True, help="Only characters with voice available"
)
def search_catalog(
    text: Optional[str],
    franchise: Optional[str],
    species: Optional[str],
    archetype: Optional[str],
    personality: Optional[str],
    ability: Optional[str],
    topic: Optional[str],
    voice_available: bool,
) -> None:
    """Search catalog characters by criteria."""
    try:
        import asyncio

        catalog_storage = CatalogStorage()

        # Build search criteria
        criteria = {}
        if text:
            criteria["text"] = text
        if franchise:
            criteria["franchise"] = franchise
        if species:
            criteria["species"] = species
        if archetype:
            criteria["archetype"] = archetype
        if personality:
            criteria["personality_traits"] = personality
        if ability:
            criteria["abilities"] = ability
        if topic:
            criteria["topics"] = topic
        if voice_available:
            criteria["voice_available"] = "true"

        # Search characters
        results = asyncio.run(catalog_storage.search_characters(criteria))

        if not results:
            click.echo("No characters found matching criteria")
            return

        click.echo(f"Found {len(results)} characters:")
        click.echo("=" * 50)
        for character in results:
            click.echo(f"Name: {character.name}")
            click.echo(f"Species: {character.dimensions.species.value}")
            click.echo(f"Archetype: {character.dimensions.archetype.value}")
            click.echo(
                f"Personality: {', '.join([t.value for t in character.dimensions.personality_traits])}"
            )
            click.echo(
                f"Abilities: {', '.join([a.value for a in character.dimensions.abilities])}"
            )
            click.echo(
                f"Topics: {', '.join([t.value for t in character.dimensions.topics])}"
            )
            if character.dimensions.backstory:
                click.echo(f"Backstory: {character.dimensions.backstory}")
            click.echo("-" * 30)

    except Exception as e:
        click.echo(f"Error searching catalog: {e}", err=True)
        raise click.Abort()


@catalog.command()
def catalog_stats() -> None:
    """Show catalog statistics."""
    try:
        import asyncio

        catalog_storage = CatalogStorage()
        stats = asyncio.run(catalog_storage.get_catalog_statistics())

        click.echo("Catalog Statistics:")
        click.echo("=" * 30)
        click.echo(f"Total Characters: {stats.get('total_characters', 0)}")

        click.echo("\nFranchises:")
        for franchise, count in stats.get("franchises", {}).items():
            click.echo(f"  {franchise}: {count}")

        click.echo("\nSpecies Distribution:")
        for species, count in stats.get("species_distribution", {}).items():
            click.echo(f"  {species}: {count}")

        click.echo("\nArchetype Distribution:")
        for archetype, count in stats.get("archetype_distribution", {}).items():
            click.echo(f"  {archetype}: {count}")

        click.echo("\nTop Personality Traits:")
        sorted_traits = sorted(
            stats.get("personality_traits_distribution", {}).items(),
            key=lambda x: x[1],
            reverse=True,
        )
        for trait, count in sorted_traits[:10]:
            click.echo(f"  {trait}: {count}")

        click.echo("\nTop Abilities:")
        sorted_abilities = sorted(
            stats.get("abilities_distribution", {}).items(),
            key=lambda x: x[1],
            reverse=True,
        )
        for ability, count in sorted_abilities[:10]:
            click.echo(f"  {ability}: {count}")

        click.echo("\nTop Topics:")
        sorted_topics = sorted(
            stats.get("topics_distribution", {}).items(),
            key=lambda x: x[1],
            reverse=True,
        )
        for topic, count in sorted_topics[:10]:
            click.echo(f"  {topic}: {count}")

        # Voice statistics
        voice_stats = stats.get("voice_stats", {})
        click.echo("\nVoice Statistics:")
        click.echo(f"  Characters with voice: {voice_stats.get('total_with_voice', 0)}")

        click.echo(
            f"  Voice availability: {voice_stats.get('voice_availability', 0):.1%}"
        )

        # Usage statistics
        usage_stats = stats.get("usage_stats", {})
        click.echo("\nUsage Statistics:")
        click.echo(f"  Total usage: {usage_stats.get('total_usage', 0)}")

        most_used = usage_stats.get("most_used", [])
        if most_used:
            click.echo("  Most used characters:")
            for char_info in most_used[:5]:
                click.echo(f"    {char_info['name']}: {char_info['usage_count']} uses")

    except Exception as e:
        click.echo(f"Error getting catalog statistics: {e}", err=True)
        raise click.Abort()


# Collection Management Commands


@character_commands.group()
def collection() -> None:
    """Character collection management commands."""
    pass


@collection.command()
@click.argument("collection_name")
@click.option("--description", help="Collection description")
@click.option("--author", help="Collection author")
@click.option("--license", help="Collection license")
def create_collection(
    collection_name: str,
    description: Optional[str],
    author: Optional[str],
    license: Optional[str],
) -> None:
    """Create new character collection."""
    try:
        import asyncio

        catalog = CharacterCatalog()
        collection = asyncio.run(
            catalog.create_collection(
                name=collection_name,
                description=description or "",
                author=author or "",
                license=license or "",
            )
        )

        click.echo(f"✓ Created collection '{collection_name}'")
        click.echo(f"Description: {collection.metadata.description}")
        click.echo(f"Author: {collection.metadata.author}")
        click.echo(f"License: {collection.metadata.license}")

    except Exception as e:
        click.echo(f"Error creating collection: {e}", err=True)
        raise click.Abort()


@collection.command()
@click.argument("collection_name")
@click.argument("catalog_file", type=click.Path(exists=True))
def import_collection(collection_name: str, catalog_file: str) -> None:
    """Import character collection from YAML file."""
    try:
        import asyncio

        catalog = CharacterCatalog()
        collection = asyncio.run(catalog.import_collection(Path(catalog_file)))

        click.echo(f"✓ Imported collection '{collection.metadata.name}'")
        click.echo(f"Characters: {len(collection.characters)}")
        click.echo(f"Description: {collection.metadata.description}")
        click.echo(f"Author: {collection.metadata.author}")

    except Exception as e:
        click.echo(f"Error importing collection: {e}", err=True)
        raise click.Abort()


@collection.command()
@click.argument("collection_name")
@click.option("--output", "-o", help="Output file path")
def export_collection(collection_name: str, output: Optional[str]) -> None:
    """Export character collection to YAML file."""
    try:
        import asyncio

        catalog = CharacterCatalog()
        output_path = asyncio.run(
            catalog.export_collection(collection_name, Path(output) if output else None)
        )

        click.echo(f"✓ Collection exported to {output_path}")

    except Exception as e:
        click.echo(f"Error exporting collection: {e}", err=True)
        raise click.Abort()


@collection.command()
def list_collections() -> None:
    """List all character collections."""
    try:
        import asyncio

        catalog = CharacterCatalog()
        collections = asyncio.run(catalog.get_all_collections())

        if not collections:
            click.echo("No collections found.")
            return

        click.echo("Character Collections:")
        click.echo("=" * 30)
        for collection_name in collections:
            click.echo(f"  {collection_name}")

    except Exception as e:
        click.echo(f"Error listing collections: {e}", err=True)
        raise click.Abort()


@collection.command()
@click.argument("collection_name")
def collection_stats(collection_name: str) -> None:
    """Show statistics for a specific collection."""
    try:
        import asyncio

        catalog = CharacterCatalog()
        stats = asyncio.run(catalog.get_collection_statistics(collection_name))

        if not stats:
            click.echo(f"Collection '{collection_name}' not found.")
            return

        click.echo(f"Collection Statistics: {collection_name}")
        click.echo("=" * 40)
        click.echo(f"Total Characters: {stats.get('total_characters', 0)}")

        metadata = stats.get("metadata", {})
        click.echo(f"Description: {metadata.get('description', 'N/A')}")
        click.echo(f"Author: {metadata.get('author', 'N/A')}")
        click.echo(f"License: {metadata.get('license', 'N/A')}")
        click.echo(f"Created: {metadata.get('created_at', 'N/A')}")
        click.echo(f"Updated: {metadata.get('updated_at', 'N/A')}")

        click.echo("\nSpecies Distribution:")
        for species, count in stats.get("species_distribution", {}).items():
            click.echo(f"  {species}: {count}")

        click.echo("\nArchetype Distribution:")
        for archetype, count in stats.get("archetype_distribution", {}).items():
            click.echo(f"  {archetype}: {count}")

        click.echo("\nTop Personality Traits:")
        sorted_traits = sorted(
            stats.get("personality_traits_distribution", {}).items(),
            key=lambda x: x[1],
            reverse=True,
        )
        for trait, count in sorted_traits[:10]:
            click.echo(f"  {trait}: {count}")

        click.echo("\nTop Abilities:")
        sorted_abilities = sorted(
            stats.get("abilities_distribution", {}).items(),
            key=lambda x: x[1],
            reverse=True,
        )
        for ability, count in sorted_abilities[:10]:
            click.echo(f"  {ability}: {count}")

        click.echo("\nTop Topics:")
        sorted_topics = sorted(
            stats.get("topics_distribution", {}).items(),
            key=lambda x: x[1],
            reverse=True,
        )
        for topic, count in sorted_topics[:10]:
            click.echo(f"  {topic}: {count}")

    except Exception as e:
        click.echo(f"Error getting collection statistics: {e}", err=True)
        raise click.Abort()


# Voice Management Commands


@character_commands.group()
def voice() -> None:
    """Character voice management commands."""
    pass


@voice.command()
@click.argument("character_name")
@click.argument("voice_file", type=click.Path(exists=True))
@click.option("--quality-score", type=float, help="Voice quality score (0.0-1.0)")
def clone_voice(
    character_name: str, voice_file: str, quality_score: Optional[float]
) -> None:
    """Clone voice for a character using new schema format."""
    try:
        import asyncio

        from ..characters.schema_voice_manager import SchemaVoiceManager

        schema_voice_manager = SchemaVoiceManager()
        success = asyncio.run(
            schema_voice_manager.clone_character_voice(
                character_name=character_name,
                voice_file_path=voice_file,
                quality_score=quality_score,
            )
        )

        if success:
            click.echo(f"✅ Voice cloned for character '{character_name}'")
            click.echo(f"🎤 Voice file: {voice_file}")
            if quality_score:
                click.echo(f"⭐ Quality score: {quality_score}")
        else:
            click.echo(
                f"❌ Failed to clone voice for character '{character_name}'", err=True
            )
            raise click.Abort()

    except Exception as e:
        click.echo(f"Error cloning voice: {e}", err=True)
        raise click.Abort()


@voice.command()
@click.argument("character_name")
def voice_info(character_name: str) -> None:
    """Get voice information for a character using new schema format."""
    try:
        import asyncio

        from ..characters.schema_voice_manager import SchemaVoiceManager

        schema_voice_manager = SchemaVoiceManager()
        voice_info = asyncio.run(
            schema_voice_manager.get_character_voice_info(character_name)
        )

        if not voice_info:
            click.echo(f"No voice found for character '{character_name}'")
            return

        click.echo(f"Voice Information for '{character_name}':")
        click.echo("=" * 50)
        click.echo(f"Character: {voice_info.get('character_name', 'N/A')}")
        click.echo(f"Voice File: {voice_info.get('voice_file_path', 'N/A')}")
        click.echo(f"Voice Samples Dir: {voice_info.get('voice_samples_dir', 'N/A')}")
        click.echo(f"File Size: {voice_info.get('file_size_mb', 0)} MB")
        click.echo(f"Quality Score: {voice_info.get('quality_score', 0):.2f}")
        click.echo(f"Quality: {voice_info.get('quality', 'N/A')}")
        click.echo(f"Language: {voice_info.get('language', 'N/A')}")
        click.echo(f"Cloned At: {voice_info.get('cloned_at', 'N/A')}")
        click.echo(f"Available: {voice_info.get('available', False)}")

    except Exception as e:
        click.echo(f"Error getting voice info: {e}", err=True)
        raise click.Abort()


@voice.command()
@click.option("--min-quality", type=float, default=0.0, help="Minimum quality score")
@click.option("--max-quality", type=float, default=1.0, help="Maximum quality score")
def list_voices(min_quality: float, max_quality: float) -> None:
    """List characters with voice information using new schema format."""
    try:
        import asyncio

        from ..characters.schema_voice_manager import SchemaVoiceManager

        schema_voice_manager = SchemaVoiceManager()
        characters = asyncio.run(schema_voice_manager.list_characters_with_voice())

        if not characters:
            click.echo("No characters with voice found.")
            return

        # Filter by quality score
        characters = [
            char
            for char in characters
            if min_quality <= char.get("quality_score", 0) <= max_quality
        ]

        click.echo(f"Characters with Voice ({len(characters)} found):")
        click.echo("=" * 50)
        for char_info in characters:
            click.echo(f"Character: {char_info.get('character_name', 'N/A')}")
            click.echo(f"Quality Score: {char_info.get('quality_score', 0):.2f}")
            click.echo(f"Quality: {char_info.get('quality', 'N/A')}")
            click.echo(f"Language: {char_info.get('language', 'N/A')}")
            click.echo(f"File Size: {char_info.get('file_size_mb', 0)} MB")
            click.echo(f"Available: {char_info.get('available', False)}")
            click.echo("-" * 30)

    except Exception as e:
        click.echo(f"Error listing voices: {e}", err=True)
        raise click.Abort()


@voice.command()
@click.argument("character_name")
def remove_voice(character_name: str) -> None:
    """Remove voice for a character using new schema format."""
    try:
        if not click.confirm(
            f"Are you sure you want to remove voice for character '{character_name}'?"
        ):
            click.echo("Voice removal cancelled.")
            return

        import asyncio

        from ..characters.schema_voice_manager import SchemaVoiceManager

        schema_voice_manager = SchemaVoiceManager()
        success = asyncio.run(
            schema_voice_manager.remove_character_voice(character_name)
        )

        if success:
            click.echo(f"✅ Voice removed for character '{character_name}'")
        else:
            click.echo(
                f"❌ Failed to remove voice for character '{character_name}'", err=True
            )
            raise click.Abort()

    except Exception as e:
        click.echo(f"Error removing voice: {e}", err=True)
        raise click.Abort()


@voice.command()
def voice_stats() -> None:
    """Show voice statistics using new schema format."""
    try:
        import asyncio

        from ..characters.schema_voice_manager import SchemaVoiceManager

        schema_voice_manager = SchemaVoiceManager()
        analytics = asyncio.run(schema_voice_manager.get_voice_analytics())

        click.echo("Voice Statistics:")
        click.echo("=" * 30)
        click.echo(f"Total Characters: {analytics.get('total_characters', 0)}")
        click.echo(f"Available Voices: {analytics.get('available_voices', 0)}")
        click.echo(f"Average Quality: {analytics.get('average_quality', 0):.2f}")
        click.echo(f"Last Updated: {analytics.get('last_updated', 'N/A')}")

    except Exception as e:
        click.echo(f"Error getting voice statistics: {e}", err=True)
        raise click.Abort()


@voice.command()
@click.option("--output", "-o", help="Output file path")
def export_voice_catalog(output: Optional[str]) -> None:
    """Export voice catalog to JSON file using new schema format."""
    try:
        import asyncio
        from pathlib import Path

        from ..characters.schema_voice_manager import SchemaVoiceManager

        schema_voice_manager = SchemaVoiceManager()
        output_path = asyncio.run(
            schema_voice_manager.export_voice_catalog(
                output_file=Path(output) if output else None
            )
        )

        click.echo(f"✅ Voice catalog exported to {output_path}")

    except Exception as e:
        click.echo(f"Error exporting voice catalog: {e}", err=True)
        raise click.Abort()


@voice.command()
@click.argument("voice_catalog_file", type=click.Path(exists=True))
def import_voice_catalog(voice_catalog_file: str) -> None:
    """Import voice catalog from JSON file using new schema format."""
    try:
        import asyncio
        from pathlib import Path

        from ..characters.schema_voice_manager import SchemaVoiceManager

        schema_voice_manager = SchemaVoiceManager()
        result = asyncio.run(
            schema_voice_manager.import_voice_catalog(Path(voice_catalog_file))
        )

        click.echo(f"✅ Imported {result['imported_count']} voice records")
        click.echo(f"Total voice records in file: {result['total_voice_info']}")

        if result["errors"]:
            click.echo(f"Errors encountered: {len(result['errors'])}")
            for error in result["errors"]:
                click.echo(f"  - {error}")

    except Exception as e:
        click.echo(f"Error importing voice catalog: {e}", err=True)
        raise click.Abort()


@voice.command()
@click.argument("character")
@click.argument("voice_samples", type=click.Path(exists=True))
@click.option(
    "--quality",
    default="high",
    type=click.Choice(["low", "medium", "high", "ultra-high"]),
    help="Voice quality setting",
)
@click.option("--language", default="en", help="Voice language")
def clone_from_samples(
    character: str, voice_samples: str, quality: str, language: str
) -> None:
    """Clone character voice from multiple samples using new schema format."""
    try:
        import asyncio
        from pathlib import Path

        from ..characters.schema_voice_manager import SchemaVoiceManager

        schema_voice_manager = SchemaVoiceManager()

        # Process voice samples directory
        samples_dir = Path(voice_samples)
        if not samples_dir.is_dir():
            click.echo(f"Error: {voice_samples} is not a directory", err=True)
            raise click.Abort()

        # Get all audio files from samples directory
        audio_files: List[Path] = []
        for ext in ["*.wav", "*.mp3", "*.flac", "*.m4a"]:
            audio_files.extend(samples_dir.glob(ext))

        if not audio_files:
            click.echo(f"No audio files found in {voice_samples}", err=True)
            raise click.Abort()

        click.echo(f"Found {len(audio_files)} voice samples")
        click.echo(f"Quality setting: {quality}")
        click.echo(f"Language: {language}")

        # Clone voice from samples using new schema format
        success = asyncio.run(
            schema_voice_manager.clone_character_voice_from_samples(
                character_name=character,
                voice_samples_dir=voice_samples,
                quality=quality,
                language=language,
            )
        )

        if success:
            click.echo(
                f"✅ Voice cloned and processed for character '{character}' from {len(audio_files)} samples"
            )
            click.echo(f"🎤 Quality: {quality}")
            click.echo(f"🌍 Language: {language}")
            click.echo("🧠 Coqui TTS voice embeddings created")
            click.echo(f"📁 Raw samples: configs/characters/{character}/voice_samples/")
            click.echo(
                f"📁 Processed samples: configs/characters/{character}/processed_samples/"
            )
            click.echo("🔗 TTS integration ready - character can speak in cloned voice")
        else:
            click.echo(
                f"❌ Failed to clone voice for character '{character}'", err=True
            )
            raise click.Abort()

    except Exception as e:
        click.echo(f"Error cloning voice from samples: {e}", err=True)
        raise click.Abort()


@character_commands.group()
def franchise() -> None:
    """Franchise management commands."""
    pass


@franchise.command()
@click.argument("franchise_name")
@click.option("--description", help="Franchise description")
@click.option("--owner", help="Franchise owner")
@click.option(
    "--permissions",
    help='Permissions JSON (e.g., \'{"read": true, "write": true, "admin": false}\')',
)
def create_franchise(
    franchise_name: str,
    description: str,
    owner: str,
    permissions: Optional[Dict[str, Any]],
) -> None:
    """Create a new franchise."""
    try:
        import asyncio
        import json

        catalog_storage = CatalogStorage()

        # Parse permissions if provided
        permissions_dict = None
        if permissions:
            try:
                permissions_dict = (
                    permissions
                    if isinstance(permissions, dict)
                    else json.loads(permissions)
                )
            except json.JSONDecodeError:
                click.echo("Error: Invalid permissions JSON format", err=True)
                raise click.Abort()

        result = asyncio.run(
            catalog_storage.create_franchise(
                franchise_name, description or "", owner or "", permissions_dict or {}
            )
        )

        if result:
            click.echo(f"✓ Created franchise '{franchise_name}'")
            if description:
                click.echo(f"  Description: {description}")
            if owner:
                click.echo(f"  Owner: {owner}")

    except Exception as e:
        click.echo(f"Error creating franchise: {e}", err=True)
        raise click.Abort()


@franchise.command()
def list_franchises() -> None:
    """List all franchises."""
    try:
        import asyncio

        catalog_storage = CatalogStorage()
        franchises = asyncio.run(catalog_storage.list_franchises())

        if not franchises:
            click.echo("No franchises found.")
            return

        click.echo("Franchises:")
        click.echo("=" * 50)
        for franchise in franchises:
            click.echo(f"  {franchise['name']}")
            if franchise.get("description"):
                click.echo(f"    Description: {franchise['description']}")
            if franchise.get("owner"):
                click.echo(f"    Owner: {franchise['owner']}")
            click.echo(f"    Characters: {franchise.get('character_count', 0)}")
            click.echo(
                f"    Voice Availability: {franchise.get('voice_availability', 0.0):.1%}"
            )
            click.echo(f"    Created: {franchise.get('created_at', 'Unknown')}")
            click.echo()

    except Exception as e:
        click.echo(f"Error listing franchises: {e}", err=True)
        raise click.Abort()


@franchise.command()
@click.argument("franchise_name")
def franchise_info(franchise_name: str) -> None:
    """Get detailed information about a franchise."""
    try:
        import asyncio

        catalog_storage = CatalogStorage()
        franchise_info = asyncio.run(catalog_storage.get_franchise_info(franchise_name))

        if not franchise_info:
            click.echo(f"Franchise '{franchise_name}' not found.", err=True)
            return

        click.echo(f"Franchise: {franchise_info['name']}")
        click.echo("=" * 30)
        click.echo(f"Description: {franchise_info.get('description', 'None')}")
        click.echo(f"Owner: {franchise_info.get('owner', 'None')}")
        click.echo(f"Characters: {franchise_info.get('character_count', 0)}")
        click.echo(
            f"Voice Availability: {franchise_info.get('voice_availability', 0.0):.1%}"
        )
        click.echo(f"Created: {franchise_info.get('created_at', 'Unknown')}")
        click.echo(f"Last Updated: {franchise_info.get('last_updated', 'Unknown')}")

        permissions = franchise_info.get("permissions", {})
        click.echo("Permissions:")
        click.echo(f"  Read: {permissions.get('read', False)}")
        click.echo(f"  Write: {permissions.get('write', False)}")
        click.echo(f"  Admin: {permissions.get('admin', False)}")

    except Exception as e:
        click.echo(f"Error getting franchise info: {e}", err=True)
        raise click.Abort()


@franchise.command()
@click.argument("franchise_name")
@click.option(
    "--force", is_flag=True, help="Force deletion even if franchise has characters"
)
def delete_franchise(franchise_name: str, force: bool) -> None:
    """Delete a franchise and all its characters."""
    try:
        import asyncio

        catalog_storage = CatalogStorage()

        if not force:
            # Check if franchise has characters
            franchise_info = asyncio.run(
                catalog_storage.get_franchise_info(franchise_name)
            )
            if franchise_info and franchise_info.get("character_count", 0) > 0:
                click.echo(
                    f"Franchise '{franchise_name}' has {franchise_info['character_count']} characters"
                )
                click.echo("Use --force to delete anyway.")
                return

        result = asyncio.run(catalog_storage.delete_franchise(franchise_name, force))

        if result:
            click.echo(f"✓ Deleted franchise '{franchise_name}'")

    except Exception as e:
        click.echo(f"Error deleting franchise: {e}", err=True)
        raise click.Abort()


@character_commands.group()
def cost() -> None:
    """Cost monitoring commands for cloud LLM usage."""
    pass


@character_commands.group()
def relationship() -> None:
    """Character relationship management commands."""
    pass


@relationship.command()
@click.argument("character_name")
@click.argument("related_character")
@click.argument("relationship_type")
@click.option("--strength", default=1.0, help="Relationship strength (0.0-1.0)")
@click.option("--description", help="Relationship description")
@click.option("--franchise", default="original", help="Franchise name")
def add_relationship(
    character_name: str,
    related_character: str,
    relationship_type: str,
    strength: float,
    description: Optional[str],
    franchise: str,
) -> None:
    """Add a relationship between characters."""

    async def _add_relationship() -> None:
        from ..characters.catalog_storage import CatalogStorage

        catalog_storage = CatalogStorage()
        character = await catalog_storage.load_character(character_name, franchise)

        if not character:
            click.echo(
                f"Character '{character_name}' not found in franchise '{franchise}'",
                err=True,
            )
            raise click.Abort()

        character.add_relationship(
            related_character, relationship_type, strength, description
        )
        await catalog_storage.store_character(character, franchise)

        click.echo(
            f"✓ Added relationship: {character_name} -> {related_character} ({relationship_type})"
        )

    try:
        import asyncio

        asyncio.run(_add_relationship())
    except Exception as e:
        click.echo(f"Error adding relationship: {e}", err=True)
        raise click.Abort()


@relationship.command()
@click.argument("character_name")
@click.option("--franchise", default="original", help="Franchise name")
def list_relationships(character_name: str, franchise: str) -> None:
    """List all relationships for a character."""

    async def _list_relationships() -> None:
        from ..characters.catalog_storage import CatalogStorage

        catalog_storage = CatalogStorage()
        character = await catalog_storage.load_character(character_name, franchise)

        if not character:
            click.echo(
                f"Character '{character_name}' not found in franchise '{franchise}'",
                err=True,
            )
            raise click.Abort()

        if not character.relationships:
            click.echo(f"No relationships found for {character_name}")
            return

        click.echo(f"Relationships for {character_name}:")
        for rel in character.relationships:
            click.echo(
                f"  - {rel.character} ({rel.relationship}, strength: {rel.strength})"
            )
            if rel.description:
                click.echo(f"    Description: {rel.description}")

    try:
        import asyncio

        asyncio.run(_list_relationships())
    except Exception as e:
        click.echo(f"Error listing relationships: {e}", err=True)
        raise click.Abort()


@character_commands.group()
def localization() -> None:
    """Character localization management commands."""
    pass


@localization.command()
@click.argument("character_name")
@click.argument("language")
@click.argument("localized_name")
@click.option("--backstory", help="Localized backstory")
@click.option("--description", help="Localized description")
@click.option("--franchise", default="original", help="Franchise name")
def add_localization(
    character_name: str,
    language: str,
    localized_name: str,
    backstory: Optional[str],
    description: Optional[str],
    franchise: str,
) -> None:
    """Add a localization for a character."""

    async def _add_localization() -> None:
        from ..characters.catalog_storage import CatalogStorage

        catalog_storage = CatalogStorage()
        character = await catalog_storage.load_character(character_name, franchise)

        if not character:
            click.echo(
                f"Character '{character_name}' not found in franchise '{franchise}'",
                err=True,
            )
            raise click.Abort()

        character.add_localization(language, localized_name, backstory, description)
        await catalog_storage.store_character(character, franchise)

        click.echo(
            f"✓ Added localization: {character_name} -> {localized_name} ({language})"
        )

    try:
        import asyncio

        asyncio.run(_add_localization())
    except Exception as e:
        click.echo(f"Error adding localization: {e}", err=True)
        raise click.Abort()


@localization.command()
@click.argument("character_name")
@click.option("--franchise", default="original", help="Franchise name")
def list_localizations(character_name: str, franchise: str) -> None:
    """List all localizations for a character."""

    async def _list_localizations() -> None:
        from ..characters.catalog_storage import CatalogStorage

        catalog_storage = CatalogStorage()
        character = await catalog_storage.load_character(character_name, franchise)

        if not character:
            click.echo(
                f"Character '{character_name}' not found in franchise '{franchise}'",
                err=True,
            )
            raise click.Abort()

        if not character.localizations:
            click.echo(f"No localizations found for {character_name}")
            return

        click.echo(f"Localizations for {character_name}:")
        for loc in character.localizations:
            click.echo(f"  - {loc.language}: {loc.name}")
            if loc.description:
                click.echo(f"    Description: {loc.description}")

    try:
        import asyncio

        asyncio.run(_list_localizations())
    except Exception as e:
        click.echo(f"Error listing localizations: {e}", err=True)
        raise click.Abort()


@character_commands.group()
def licensing() -> None:
    """Character licensing management commands."""
    pass


@licensing.command()
@click.argument("character_name")
@click.argument("owner")
@click.argument("rights", nargs=-1)
@click.option("--restrictions", multiple=True, help="Usage restrictions")
@click.option("--expiration", help="License expiration date (YYYY-MM-DD)")
@click.option("--territories", multiple=True, help="Allowed territories")
@click.option("--license-type", default="proprietary", help="License type")
@click.option("--franchise", default="original", help="Franchise name")
def set_licensing(
    character_name: str,
    owner: str,
    rights: str,
    restrictions: Optional[str],
    expiration: Optional[str],
    territories: Optional[str],
    license_type: str,
    franchise: str,
) -> None:
    """Set licensing information for a character."""

    async def _set_licensing() -> None:
        from ..characters.catalog_storage import CatalogStorage

        catalog_storage = CatalogStorage()
        character = await catalog_storage.load_character(character_name, franchise)

        if not character:
            click.echo(
                f"Character '{character_name}' not found in franchise '{franchise}'",
                err=True,
            )
            raise click.Abort()

        character.set_licensing(
            owner=owner,
            rights=list(rights),
            restrictions=list(restrictions),
            expiration=expiration,
            territories=list(territories),
            license_type=license_type,
        )
        await catalog_storage.store_character(character, franchise)

        click.echo(f"✓ Set licensing for {character_name}: {owner} ({license_type})")

    try:
        import asyncio

        asyncio.run(_set_licensing())
    except Exception as e:
        click.echo(f"Error setting licensing: {e}", err=True)
        raise click.Abort()


@licensing.command()
@click.argument("character_name")
@click.option("--franchise", default="original", help="Franchise name")
def show_licensing(character_name: str, franchise: str) -> None:
    """Show licensing information for a character."""

    async def _show_licensing() -> None:
        from ..characters.catalog_storage import CatalogStorage

        catalog_storage = CatalogStorage()
        character = await catalog_storage.load_character(character_name, franchise)

        if not character:
            click.echo(
                f"Character '{character_name}' not found in franchise '{franchise}'",
                err=True,
            )
            raise click.Abort()

        if not character.licensing:
            click.echo(f"No licensing information found for {character_name}")
            return

        lic = character.licensing
        click.echo(f"Licensing for {character_name}:")
        click.echo(f"  Owner: {lic.owner}")
        click.echo(f"  Rights: {', '.join(lic.rights)}")
        click.echo(f"  Restrictions: {', '.join(lic.restrictions)}")
        click.echo(f"  Territories: {', '.join(lic.territories)}")
        click.echo(f"  License Type: {lic.license_type}")
        if lic.expiration:
            click.echo(f"  Expiration: {lic.expiration}")

    try:
        import asyncio

        asyncio.run(_show_licensing())
    except Exception as e:
        click.echo(f"Error showing licensing: {e}", err=True)
        raise click.Abort()


@cost.command()
@click.option(
    "--month", help="Specific month (YYYY-MM) or current month if not specified"
)
def summary(month: Optional[str]) -> None:
    """Show cost summary for LLM usage."""
    try:
        from ..characters.cost_monitor import LLMCostMonitor

        cost_monitor = LLMCostMonitor()
        summary_data = cost_monitor.get_cost_summary(month)

        if not summary_data:
            click.echo("No cost data found.")
            return

        click.echo("Cost Summary:")
        click.echo("=" * 30)
        click.echo(f"Total Cost: ${summary_data.get('total_cost', 0):.4f}")
        click.echo(f"Total Tokens: {summary_data.get('total_tokens', 0):,}")
        click.echo(f"Total Entries: {summary_data.get('total_entries', 0)}")

        if month:
            click.echo(f"Month: {month}")
        else:
            click.echo("All Time")

        # Show monthly breakdown if available
        monthly_breakdown = summary_data.get("monthly_breakdown", {})
        if monthly_breakdown:
            click.echo("\nMonthly Breakdown:")
            for month_key, month_data in sorted(monthly_breakdown.items()):
                click.echo(
                    f"  {month_key}: ${month_data.get('total_cost', 0):.4f} ({month_data.get('token_count', 0)} tokens)"
                )

    except Exception as e:
        click.echo(f"Error getting cost summary: {e}", err=True)
        raise click.Abort()


@cost.command()
@click.argument("franchise")
@click.option(
    "--month", help="Specific month (YYYY-MM) or current month if not specified"
)
def franchise_costs(franchise: str, month: Optional[str]) -> None:
    """Show cost breakdown for a specific franchise."""
    try:
        from ..characters.cost_monitor import LLMCostMonitor

        cost_monitor = LLMCostMonitor()
        franchise_data = cost_monitor.get_franchise_costs(franchise, month)

        if not franchise_data:
            click.echo(f"No cost data found for franchise '{franchise}'.")
            return

        click.echo(f"Cost Summary for Franchise: {franchise}")
        click.echo("=" * 40)
        click.echo(f"Total Cost: ${franchise_data.get('total_cost', 0):.4f}")
        click.echo(f"Total Tokens: {franchise_data.get('total_tokens', 0):,}")
        click.echo(f"Total Entries: {franchise_data.get('total_entries', 0)}")

        if month:
            click.echo(f"Month: {month}")

        # Show operations breakdown
        operations = franchise_data.get("operations", {})
        if operations:
            click.echo("\nOperations Breakdown:")
            for operation, data in operations.items():
                click.echo(
                    f"  {operation}: ${data.get('cost', 0):.4f} ({data.get('tokens', 0)} tokens)"
                )

    except Exception as e:
        click.echo(f"Error getting franchise costs: {e}", err=True)
        raise click.Abort()


@cost.command()
@click.argument("provider")
@click.option(
    "--month", help="Specific month (YYYY-MM) or current month if not specified"
)
def provider_costs(provider: str, month: Optional[str]) -> None:
    """Show cost breakdown for a specific provider."""
    try:
        from ..characters.cost_monitor import LLMCostMonitor

        cost_monitor = LLMCostMonitor()
        provider_data = cost_monitor.get_provider_costs(provider, month)

        if not provider_data:
            click.echo(f"No cost data found for provider '{provider}'.")
            return

        click.echo(f"Cost Summary for Provider: {provider}")
        click.echo("=" * 40)
        click.echo(f"Total Cost: ${provider_data.get('total_cost', 0):.4f}")
        click.echo(f"Total Tokens: {provider_data.get('total_tokens', 0):,}")
        click.echo(f"Total Entries: {provider_data.get('total_entries', 0)}")

        if month:
            click.echo(f"Month: {month}")

        # Show models breakdown
        models = provider_data.get("models", {})
        if models:
            click.echo("\nModels Breakdown:")
            for model, data in models.items():
                click.echo(
                    f"  {model}: ${data.get('cost', 0):.4f} ({data.get('tokens', 0):,} tokens)"
                )

    except Exception as e:
        click.echo(f"Error getting provider costs: {e}", err=True)
        raise click.Abort()


@cost.command()
@click.option("--output", "-o", help="Output file path")
def export_costs(output: str) -> None:
    """Export cost data to JSON file."""
    try:
        from ..characters.cost_monitor import LLMCostMonitor

        cost_monitor = LLMCostMonitor()
        output_path = cost_monitor.export_cost_data(Path(output) if output else None)

        click.echo(f"✓ Cost data exported to {output_path}")

    except Exception as e:
        click.echo(f"Error exporting cost data: {e}", err=True)
        raise click.Abort()


# Character Bundling Commands


@character_commands.command()
@click.argument("character")
@click.option("--franchise", required=True, help="Franchise identifier")
@click.option("--include-voice", is_flag=True, help="Include voice models in bundle")
@click.option("--include-models", is_flag=True, help="Include LLM models in bundle")
@click.option("--output", "-o", help="Output path for bundle")
def bundle(
    character: str,
    franchise: str,
    include_voice: bool,
    include_models: bool,
    output: str,
) -> None:
    """Bundle character for production deployment."""
    try:
        from ..characters.bundler import CharacterBundler
        from ..core.config import Config

        config = Config()
        bundler = CharacterBundler(config)

        bundle_path = bundler.bundle_character(
            character_id=character,
            franchise=franchise,
            include_voice=include_voice,
            include_models=include_models,
            output_path=output,
        )

        click.echo(f"✅ Character bundle created: {bundle_path}")
        click.echo(f"📦 Character: {character}")
        click.echo(f"🏢 Franchise: {franchise}")
        click.echo(f"🎤 Voice included: {include_voice}")
        click.echo(f"🤖 Models included: {include_models}")

    except Exception as e:
        click.echo(f"Error bundling character: {e}", err=True)
        raise click.Abort()


@character_commands.command()
@click.argument("bundle_path", type=click.Path(exists=True))
@click.option("--extract-to", help="Directory to extract to")
def extract_bundle(bundle_path: str, extract_to: str) -> None:
    """Extract a character bundle."""
    try:
        from ..characters.bundler import CharacterBundler
        from ..core.config import Config

        config = Config()
        bundler = CharacterBundler(config)

        extract_dir = bundler.extract_bundle(bundle_path, extract_to)

        click.echo(f"✅ Bundle extracted to: {extract_dir}")

    except Exception as e:
        click.echo(f"Error extracting bundle: {e}", err=True)
        raise click.Abort()


def _save_character_profile(
    character_profile: CharacterProfile, file_path: str
) -> None:
    """Save character profile to new schema format."""
    import yaml

    # Create profile data in new schema format
    profile_data = {
        "id": character_profile.id,
        "display_name": character_profile.display_name,
        "character_type": character_profile.character_type,
        "voice_style": character_profile.voice_style,
        "language": character_profile.language,
        "traits": character_profile.traits,
        "safety": {
            "content_filter": True,
            "age_appropriate": True,
            "moderation": "strict",
        },
        "llm": {
            "model": "phi-3-mini-4k-instruct",
            "provider": "local",
            "prompt_template": "You are {character_name}, a {character_type}. {personality_traits}",
        },
        "stt": {"model": "wav2vec2-base", "language": character_profile.language},
        "tts": {
            "model": "coqui",
            "voice_style": character_profile.voice_style,
            "voice_artifact": None,
        },
    }

    with open(file_path, "w") as f:
        yaml.dump(profile_data, f, default_flow_style=False)


def _save_character_prompts(
    character_profile: CharacterProfile, file_path: str
) -> None:
    """Save character prompts to new schema format."""
    import yaml

    # Create prompts data
    prompts_data = {
        "system_prompt": f"You are {character_profile.display_name}, a {character_profile.character_type}. {character_profile.traits.get('personality', '')}",
        "greeting": f"Hello! I'm {character_profile.display_name}. How can I help you today?",
        "topics": character_profile.traits.get("topics", []),
        "personality_traits": character_profile.traits.get("personality", ""),
        "voice_style": character_profile.voice_style,
    }

    with open(file_path, "w") as f:
        yaml.dump(prompts_data, f, default_flow_style=False)


def _save_enhanced_character_profile(character: Character, file_path: str) -> None:
    """Save enhanced character profile to new schema format."""
    import yaml

    # Create profile data in new schema format
    profile_data = {
        "id": character.name.lower().replace(" ", "_"),
        "display_name": character.name,
        "character_type": character.dimensions.species.value,
        "voice_style": character.voice_style,
        "language": character.language,
        "traits": {
            "personality": ", ".join(
                [t.value for t in character.dimensions.personality_traits]
            ),
            "abilities": [a.value for a in character.dimensions.abilities],
            "topics": [t.value for t in character.dimensions.topics],
            "backstory": character.dimensions.backstory,
            "goals": character.dimensions.goals,
            "fears": character.dimensions.fears,
            "likes": character.dimensions.likes,
            "dislikes": character.dimensions.dislikes,
        },
        "safety": {
            "content_filter": True,
            "age_appropriate": True,
            "moderation": "strict",
        },
        "llm": {
            "model": "phi-3-mini-4k-instruct",
            "provider": "local",
            "prompt_template": f"You are {character.name}, a {character.dimensions.species.value}. {', '.join([t.value for t in character.dimensions.personality_traits])}",
        },
        "stt": {"model": "wav2vec2-base", "language": character.language},
        "tts": {
            "model": "coqui",
            "voice_style": character.voice_style,
            "voice_artifact": None,
        },
    }

    with open(file_path, "w") as f:
        yaml.dump(profile_data, f, default_flow_style=False)


def _save_enhanced_character_prompts(character: Character, file_path: str) -> None:
    """Save enhanced character prompts to new schema format."""
    import yaml

    # Create prompts data
    personality_traits = ", ".join(
        [t.value for t in character.dimensions.personality_traits]
    )
    topics = [t.value for t in character.dimensions.topics]

    prompts_data = {
        "system_prompt": f"You are {character.name}, a {character.dimensions.species.value}. {personality_traits}",
        "greeting": f"Hello! I'm {character.name}. How can I help you today?",
        "topics": topics,
        "personality_traits": personality_traits,
        "voice_style": character.voice_style,
        "backstory": character.dimensions.backstory,
        "goals": character.dimensions.goals,
        "fears": character.dimensions.fears,
        "likes": character.dimensions.likes,
        "dislikes": character.dimensions.dislikes,
    }

    with open(file_path, "w") as f:
        yaml.dump(prompts_data, f, default_flow_style=False)


@character_commands.command()
@click.argument("character_name")
@click.option("--text", "-t", help="Send text message to character")
@click.option(
    "--interactive", "-i", is_flag=True, help="Start interactive chat session"
)
@click.option(
    "--voice", "-v", is_flag=True, help="Enable voice interaction (requires microphone)"
)
def test(character_name: str, text: str, interactive: bool, voice: bool) -> None:
    """Test character interaction during development."""
    try:
        import asyncio

        from ..characters.schema_voice_manager import SchemaVoiceManager
        from ..core.llm.config import LLMConfigManager
        from ..core.llm.factory import LLMFactory

        # Load character profile
        character_dir = Path("configs/characters") / character_name
        if not character_dir.exists():
            click.echo(
                f"Error: Character '{character_name}' not found in configs/characters/",
                err=True,
            )
            raise click.Abort()

        profile_file = character_dir / "profile.yaml"
        if not profile_file.exists():
            click.echo(f"Error: Character profile not found: {profile_file}", err=True)
            raise click.Abort()

        # Load character configuration
        import yaml

        with open(profile_file, "r") as f:
            character_config = yaml.safe_load(f)

        click.echo(
            f"🤖 Testing character: {character_config.get('display_name', character_name)}"
        )
        click.echo(f"📝 Species: {character_config.get('character_type', 'unknown')}")
        click.echo(
            f"🎭 Personality: {character_config.get('traits', {}).get('personality', 'unknown')}"
        )

        # Initialize LLM
        llm_config = character_config.get("llm", {})
        model_name = llm_config.get("model", "phi-3-mini-4k-instruct")

        click.echo(f"🧠 Loading LLM model: {model_name}")

        # For development, use local model if available
        if model_name.startswith("phi-3") or model_name.startswith("llama"):
            # Use local model for development
            config_manager = LLMConfigManager()
            from ..core.llm.manager import OpenModelManager

            model_manager = OpenModelManager()
            llm_factory = LLMFactory(config_manager, model_manager)
            llm = llm_factory.get_runtime_llm()
            click.echo("✅ Local LLM loaded for development testing")
        else:
            click.echo(
                "⚠️  Using cloud LLM - consider using local model for development"
            )
            # Fallback to cloud model
            config_manager = LLMConfigManager()
            from ..core.llm.manager import OpenModelManager

            model_manager = OpenModelManager()
            llm_factory = LLMFactory(config_manager, model_manager)
            llm = llm_factory.get_runtime_llm()

        # Test text interaction
        if text:
            click.echo(f"💬 Sending: {text}")

            # Load character prompts
            prompts_file = character_dir / "prompts.yaml"
            if prompts_file.exists():
                with open(prompts_file, "r") as f:
                    prompts_config = yaml.safe_load(f)
                system_prompt = prompts_config.get("system_prompt", "")
                # Use character's system prompt
                full_prompt = f"{system_prompt}\n\nUser: {text}\nData:"
            else:
                full_prompt = text

            response = asyncio.run(llm.generate(full_prompt))
            click.echo(f"🤖 {character_name}: {response}")

        # Test voice interaction
        if voice:
            voice_manager = SchemaVoiceManager()
            voice_info = asyncio.run(
                voice_manager.get_character_voice_info(character_name)
            )
            if voice_info and voice_info.get("available"):
                click.echo(
                    "🎤 Voice interaction enabled - character can speak with cloned voice"
                )
                click.echo("📁 Voice files:")
                click.echo(f"  - Embedding: {voice_info.get('embedding_file', 'N/A')}")
                click.echo(f"  - Quality: {voice_info.get('quality_score', 'N/A')}")
            else:
                click.echo("⚠️  No voice available - run voice cloning first")

        # Interactive mode
        if interactive:
            click.echo("\n🔄 Interactive mode - type 'quit' to exit")

            # Load character prompts for interactive mode
            prompts_file = character_dir / "prompts.yaml"
            system_prompt = ""
            if prompts_file.exists():
                with open(prompts_file, "r") as f:
                    prompts_config = yaml.safe_load(f)
                system_prompt = prompts_config.get("system_prompt", "")

            while True:
                user_input = click.prompt("You", default="", show_default=False)
                if user_input.lower() in ["quit", "exit", "q"]:
                    break

                # Use character's system prompt
                if system_prompt:
                    full_prompt = f"{system_prompt}\n\nUser: {user_input}\nData:"
                else:
                    full_prompt = user_input

                response = asyncio.run(llm.generate(full_prompt))
                click.echo(f"{character_name}: {response}")

        click.echo("✅ Character testing complete!")

    except Exception as e:
        click.echo(f"Error testing character: {e}", err=True)
        raise click.Abort()


@character_commands.command()
@click.argument("bundle_path", type=click.Path(exists=True))
def validate_bundle(bundle_path: str) -> None:
    """Validate a character bundle."""
    try:
        from ..characters.bundler import CharacterBundler
        from ..core.config import Config

        config = Config()
        bundler = CharacterBundler(config)

        is_valid = bundler.validate_bundle(bundle_path)

        if is_valid:
            click.echo(f"✅ Bundle is valid: {bundle_path}")
        else:
            click.echo(f"❌ Bundle is invalid: {bundle_path}", err=True)
            raise click.Abort()

    except Exception as e:
        click.echo(f"Error validating bundle: {e}", err=True)
        raise click.Abort()
