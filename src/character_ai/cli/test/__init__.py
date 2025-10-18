"""
Testing commands for the Character AI CLI.

Provides Click-based commands for testing platform functionality.
"""

import click

from .voice_pipeline import voice_pipeline

# Import other modules when they're created
# from .benchmarks import benchmark
# from .connectivity import connectivity, performance
# from .audio import audio


@click.group()
def test_commands() -> None:
    """Testing and validation commands."""
    pass


# Add voice pipeline command
test_commands.add_command(voice_pipeline)

# Add other commands when modules are created
# test_commands.add_command(benchmark)
# test_commands.add_command(connectivity)
# test_commands.add_command(performance)
# test_commands.add_command(audio)
