"""
Character management commands for the Character AI CLI.

Provides Click-based commands for creating, managing, and interacting with characters.
"""

import click

from .creation import create
from .management import activate, chat, list, remove, search, show, stats, templates

# Import other modules when they're created
# from .catalog_commands import catalog
# from .voice_commands import voice
# from .cost_commands import cost


@click.group()
def character_commands() -> None:
    """Character creation and management commands."""
    pass


# Add creation commands
character_commands.add_command(create)

# Add management commands
character_commands.add_command(list)
character_commands.add_command(show)
character_commands.add_command(activate)
character_commands.add_command(remove)
character_commands.add_command(templates)
character_commands.add_command(search)
character_commands.add_command(stats)
character_commands.add_command(chat)

# Add other command groups when modules are created
# character_commands.add_command(catalog)
# character_commands.add_command(voice)
# character_commands.add_command(cost)
