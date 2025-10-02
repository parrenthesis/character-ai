"""
Character AI CLI

Unified command-line interface for the Character AI.
"""

# Import the CLI function directly to avoid module path issues
from .main import cli

__all__ = ["cli"]


# Export the CLI function directly to avoid import warnings
def main() -> None:
    """Main CLI entry point."""
    cli()


if __name__ == "__main__":
    main()
