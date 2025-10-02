"""
LLM management commands for the Character AI CLI.

Provides Click-based commands for managing LLMs, models, and configurations.
"""

import asyncio
import json
import logging
from typing import List, Optional

import click

from ..core.llm.config import LLMConfigManager
from ..core.llm.config import LLMProvider as ConfigLLMProvider
from ..core.llm.config import LLMType
from ..core.llm.factory import LLMFactory
from ..core.llm.manager import ModelInfo, OpenModelManager

logger = logging.getLogger(__name__)


@click.group()
def llm_commands() -> None:
    """LLM (Large Language Model) management commands."""
    pass


@llm_commands.command()
@click.option("--installed", is_flag=True, help="Show only installed models")
@click.option(
    "--use-case",
    type=click.Choice(["character_creation", "runtime"]),
    help="Filter by use case",
)
def list_models(installed: bool, use_case: Optional[str]) -> None:
    """List available or installed LLM models."""
    try:
        LLMConfigManager()
        model_manager = OpenModelManager()

        if installed:
            installed_models = model_manager.list_installed_models()
            click.echo("Installed models:")
            for model in installed_models:
                click.echo(f"  - {model}")
        else:
            available_models: List[ModelInfo] = model_manager.list_available_models()
            if use_case:
                available_models = [
                    m
                    for m in available_models
                    if m.recommended_for and use_case in m.recommended_for
                ]

            click.echo("Available models:")
            for model in available_models:  # type: ignore
                installed_marker = (
                    "✓" if model_manager.is_model_installed(model.name) else " "  # type: ignore
                )
                click.echo(
                    f"  {installed_marker} {model.name} ({model.size}) - {model.description}"  # type: ignore
                )
                click.echo(
                    f"    Recommended for: {', '.join(model.recommended_for or [])}"
                )  # type: ignore
    except Exception as e:
        click.echo(f"Error listing models: {e}", err=True)
        raise click.Abort()


@llm_commands.command()
@click.argument("model_name")
@click.option("--progress", is_flag=True, help="Show download progress")
def install(model_name: str, progress: bool) -> None:
    """Install an LLM model."""
    try:
        LLMConfigManager()
        model_manager = OpenModelManager()

        if model_manager.is_model_installed(model_name):
            click.echo(f"Model {model_name} is already installed")
            return

        model_info = model_manager.get_model_info(model_name)
        if not model_info:
            click.echo(f"Unknown model: {model_name}", err=True)
            raise click.Abort()

        click.echo(f"Installing {model_name} ({model_info.size})...")

        def progress_callback(progress_pct: float) -> None:
            if progress:
                click.echo(f"\rProgress: {progress_pct:.1f}%", nl=False)

        # Run async download
        success = asyncio.run(
            model_manager.download_model(model_name, progress_callback)
        )

        if success:
            click.echo(f"\n✓ Successfully installed {model_name}")
        else:
            click.echo(f"\n✗ Failed to install {model_name}", err=True)
            raise click.Abort()
    except Exception as e:
        click.echo(f"Error installing model: {e}", err=True)
        raise click.Abort()


@llm_commands.command()
@click.argument("model_name")
def remove(model_name: str) -> None:
    """Remove an installed LLM model."""
    try:
        model_manager = OpenModelManager()

        if not model_manager.is_model_installed(model_name):
            click.echo(f"Model {model_name} is not installed")
            return

        success = model_manager.remove_model(model_name)
        if success:
            click.echo(f"✓ Removed {model_name}")
        else:
            click.echo(f"✗ Failed to remove {model_name}", err=True)
            raise click.Abort()
    except Exception as e:
        click.echo(f"Error removing model: {e}", err=True)
        raise click.Abort()


@llm_commands.command()
@click.argument("model_name")
def model_info(model_name: str) -> None:
    """Show information about a specific model."""
    try:
        model_manager = OpenModelManager()
        model_info = model_manager.get_model_info(model_name)

        if not model_info:
            click.echo(f"Unknown model: {model_name}", err=True)
            raise click.Abort()

        click.echo(f"Model: {model_info.name}")
        click.echo(f"Size: {model_info.size}")
        click.echo(f"Description: {model_info.description}")
        click.echo(f"Format: {model_info.format}")
        click.echo(f"Quantization: {model_info.quantization}")
        click.echo(f"Recommended for: {', '.join(model_info.recommended_for or [])}")

        if model_manager.is_model_installed(model_name):
            model_path = model_manager.get_model_path(model_name)
            click.echo(f"Installed at: {model_path}")
        else:
            click.echo("Status: Not installed")
    except Exception as e:
        click.echo(f"Error getting model info: {e}", err=True)
        raise click.Abort()


@llm_commands.command()
@click.option(
    "--character-creation-provider",
    type=click.Choice(["local", "ollama", "openai", "anthropic"]),
    help="Set character creation provider",
)
@click.option("--character-creation-model", help="Set character creation model")
@click.option(
    "--runtime-provider",
    type=click.Choice(["local", "ollama", "openai", "anthropic"]),
    help="Set runtime provider",
)
@click.option("--runtime-model", help="Set runtime model")
@click.option("--local-model-path", help="Set local model path")
@click.option("--ollama-base-url", help="Set Ollama base URL")
@click.option("--save", help="Save configuration to file")
@click.option("--load", help="Load configuration from file")
@click.option("--show", is_flag=True, help="Show current configuration")
def config(
    character_creation_provider: Optional[str],
    character_creation_model: Optional[str],
    runtime_provider: Optional[str],
    runtime_model: Optional[str],
    local_model_path: Optional[str],
    ollama_base_url: Optional[str],
    save: bool,
    load: Optional[str],
    show: bool,
) -> None:
    """Configure LLM settings."""
    try:
        config_manager = LLMConfigManager()

        if show:
            _show_config(config_manager)
            return

        if load:
            config_manager.load_config(load)
            click.echo(f"Configuration loaded from {load}")
            return

        # Update configuration
        if character_creation_provider:
            config_manager.character_creation.provider = ConfigLLMProvider(
                character_creation_provider
            )
            click.echo(
                f"Character creation provider set to {character_creation_provider}"
            )

        if character_creation_model:
            config_manager.character_creation.model = character_creation_model
            click.echo(f"Character creation model set to {character_creation_model}")

        if runtime_provider:
            config_manager.runtime.provider = ConfigLLMProvider(runtime_provider)
            click.echo(f"Runtime provider set to {runtime_provider}")

        if runtime_model:
            config_manager.runtime.model = runtime_model
            click.echo(f"Runtime model set to {runtime_model}")

        if local_model_path:
            config_manager.providers.local_model_path = local_model_path
            click.echo(f"Local model path set to {local_model_path}")

        if ollama_base_url:
            config_manager.providers.ollama_base_url = ollama_base_url
            click.echo(f"Ollama base URL set to {ollama_base_url}")

        # Save configuration
        if save:
            config_manager.save_config("llm_config.yaml")
            click.echo("Configuration saved to llm_config.yaml")
    except Exception as e:
        click.echo(f"Error configuring LLM: {e}", err=True)
        raise click.Abort()


def _show_config(config_manager: LLMConfigManager) -> None:
    """Show current LLM configuration."""
    click.echo("Current LLM Configuration:")
    click.echo(
        f"Character Creation: {config_manager.character_creation.provider.value} - {config_manager.character_creation.model}"
    )
    click.echo(
        f"Runtime: {config_manager.runtime.provider.value} - {config_manager.runtime.model}"
    )
    click.echo(f"Local Model Path: {config_manager.providers.local_model_path}")
    click.echo(f"Ollama Base URL: {config_manager.providers.ollama_base_url}")
    click.echo(
        f"Cost Tracking: {'Enabled' if config_manager.cost_tracking_enabled else 'Disabled'}"
    )
    click.echo(
        f"Fallback: {'Enabled' if config_manager.fallback_enabled else 'Disabled'}"
    )


@llm_commands.command()
@click.option("--json", "output_json", is_flag=True, help="Output as JSON")
def status(output_json: bool) -> None:
    """Show LLM status and health."""
    try:
        config_manager = LLMConfigManager()
        model_manager = OpenModelManager()
        factory = LLMFactory(config_manager, model_manager)

        status_info = factory.get_llm_status()

        if output_json:
            click.echo(json.dumps(status_info, indent=2))
        else:
            click.echo("LLM Status:")
            for llm_type, info in status_info.items():
                click.echo(f"\n{llm_type.replace('_', ' ').title()}:")
                if "error" in info:
                    click.echo(f"  Error: {info['error']}")
                else:
                    click.echo(f"  Provider: {info['provider']}")
                    click.echo(f"  Connection: {'✓' if info['connection_ok'] else '✗'}")

                    click.echo(f"  Cost per token: ${info['cost_per_token']}")
                    click.echo(
                        f"  Requires internet: {'Yes' if info['requires_internet'] else 'No'}"
                    )
    except Exception as e:
        click.echo(f"Error getting LLM status: {e}", err=True)
        raise click.Abort()


@llm_commands.command()
@click.option(
    "--llm-type",
    type=click.Choice(["character_creation", "runtime", "both"]),
    default="both",
    help="Which LLM to test",
)
def test(llm_type: str) -> None:
    """Test LLM connections."""
    try:
        config_manager = LLMConfigManager()
        model_manager = OpenModelManager()
        factory = LLMFactory(config_manager, model_manager)

        if llm_type in ["character_creation", "both"]:
            click.echo("Testing character creation LLM...")
            success = factory.test_llm_connection(LLMType.CHARACTER_CREATION)
            click.echo(f"Character creation: {'✓' if success else '✗'}")

        if llm_type in ["runtime", "both"]:
            click.echo("Testing runtime LLM...")
            success = factory.test_llm_connection(LLMType.RUNTIME)
            click.echo(f"Runtime: {'✓' if success else '✗'}")
    except Exception as e:
        click.echo(f"Error testing LLM: {e}", err=True)
        raise click.Abort()


@llm_commands.command()
@click.option("--cleanup", is_flag=True, help="Clean up orphaned files")
def storage(cleanup: bool) -> None:
    """Show storage usage for LLM models."""
    try:
        model_manager = OpenModelManager()
        usage = model_manager.get_storage_usage()

        click.echo("Storage Usage:")
        click.echo(f"  Total size: {usage['total_size_gb']:.2f} GB")
        click.echo(f"  Model count: {usage['model_count']}")
        click.echo(f"  Models directory: {usage['models_dir']}")

        if cleanup:
            click.echo("\nCleaning up orphaned files...")
            cleaned = model_manager.cleanup_orphaned_files()
            if cleaned:
                click.echo(f"Cleaned up: {', '.join(cleaned)}")
            else:
                click.echo("No orphaned files found")
    except Exception as e:
        click.echo(f"Error checking storage: {e}", err=True)
        raise click.Abort()
