"""
Character management commands for the Character AI CLI.
"""

import json
import logging
from typing import Optional

import click
import yaml

from ...characters import CharacterService
from .helpers import _format_character_list, _load_character_manager

logger = logging.getLogger(__name__)


@click.command()
@click.option(
    "--format",
    type=click.Choice(["table", "json", "yaml"]),
    default="table",
    help="Output format",
)
def list(format: str) -> None:
    """List all available characters."""
    try:
        manager = _load_character_manager()
        characters = manager.list_characters()

        if not characters:
            click.echo("No characters found.")
            return

        formatted_output = _format_character_list(characters, format)
        click.echo(formatted_output)

    except Exception as e:
        click.echo(f"Error listing characters: {e}", err=True)
        raise click.Abort()


@click.command()
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
        character_manager = CharacterService()
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
            click.echo(yaml.dump(character_data, default_flow_style=False))

    except Exception as e:
        click.echo(f"Error showing character: {e}", err=True)
        raise click.Abort()


@click.command()
@click.argument("character_name")
def activate(character_name: str) -> None:
    """Activate a character for interaction."""
    try:
        character_manager = CharacterService()
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


@click.command()
@click.argument("character_name")
def remove(character_name: str) -> None:
    """Remove a character."""
    try:
        if not click.confirm(
            f"Are you sure you want to remove character '{character_name}'?"
        ):
            click.echo("Character removal cancelled.")
            return

        character_manager = CharacterService()
        success = character_manager.remove_character(character_name)

        if success:
            click.echo(f"✓ Character '{character_name}' removed!")
        else:
            click.echo(f"✗ Failed to remove character '{character_name}'", err=True)
            raise click.Abort()

    except Exception as e:
        click.echo(f"Error removing character: {e}", err=True)
        raise click.Abort()


@click.command()
def templates() -> None:
    """List available character templates."""
    try:
        manager = _load_character_manager()
        templates = manager.get_available_templates()

        click.echo("Available Character Templates:")
        click.echo("=" * 50)
        for template in templates:
            click.echo(f"{template.name}: {template.description}")

    except Exception as e:
        click.echo(f"Error listing templates: {e}", err=True)
        raise click.Abort()


@click.command()
@click.option("--species", help="Filter by species")
@click.option("--archetype", help="Filter by archetype")
@click.option("--personality", help="Filter by personality trait")
@click.option("--ability", help="Filter by ability")
@click.option("--topic", help="Filter by topic")
@click.option(
    "--format",
    type=click.Choice(["table", "json", "yaml"]),
    default="table",
    help="Output format",
)
def search(
    species: Optional[str],
    archetype: Optional[str],
    personality: Optional[str],
    ability: Optional[str],
    topic: Optional[str],
    format: str,
) -> None:
    """Search characters by criteria."""
    try:
        manager = _load_character_manager()

        # Build search criteria
        criteria = {}
        if species:
            criteria["species"] = species
        if archetype:
            criteria["archetype"] = archetype
        if personality:
            criteria["personality"] = personality
        if ability:
            criteria["ability"] = ability
        if topic:
            criteria["topic"] = topic

        characters = manager.search_characters(criteria)

        if not characters:
            click.echo("No characters found matching criteria.")
            return

        formatted_output = _format_character_list(characters, format)
        click.echo(formatted_output)

    except Exception as e:
        click.echo(f"Error searching characters: {e}", err=True)
        raise click.Abort()


@click.command()
def stats() -> None:
    """Show character statistics."""
    try:
        manager = _load_character_manager()
        characters = manager.list_characters()

        if not characters:
            click.echo("No characters found.")
            return

        # Calculate statistics
        total_characters = len(characters)
        species_count: dict[str, int] = {}
        archetype_count: dict[str, int] = {}
        personality_count: dict[str, int] = {}

        for char in characters:
            # Count species
            species = char.dimensions.species.value
            species_count[species] = species_count.get(species, 0) + 1

            # Count archetypes
            archetype = char.dimensions.archetype.value
            archetype_count[archetype] = archetype_count.get(archetype, 0) + 1

            # Count personality traits
            for trait in char.dimensions.personality_traits:
                trait_name = str(trait.value)
                personality_count[trait_name] = personality_count.get(trait_name, 0) + 1

        # Display statistics
        click.echo("Character Statistics:")
        click.echo("=" * 30)
        click.echo(f"Total Characters: {total_characters}")
        click.echo()

        click.echo("Species Distribution:")
        for species, count in sorted(species_count.items()):
            click.echo(f"  {species}: {count}")

        click.echo()
        click.echo("Archetype Distribution:")
        for archetype, count in sorted(archetype_count.items()):
            click.echo(f"  {archetype}: {count}")

        click.echo()
        click.echo("Top Personality Traits:")
        sorted_traits = sorted(
            personality_count.items(), key=lambda x: x[1], reverse=True
        )
        for trait, count in sorted_traits[:10]:  # type: ignore[assignment] # Top 10
            click.echo(f"  {trait}: {count}")

    except Exception as e:
        click.echo(f"Error getting statistics: {e}", err=True)
        raise click.Abort()


@click.command()
@click.argument("character_name")
@click.option("--message", "-m", help="Message to send to character")
@click.option("--interactive", "-i", is_flag=True, help="Interactive conversation mode")
def chat(character_name: str, message: Optional[str], interactive: bool) -> None:
    """Chat with a character."""
    try:
        character_manager = CharacterService()
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
