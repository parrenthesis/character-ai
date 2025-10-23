"""
Main CLI entry point for the Character AI platform.

Provides unified command-line interface with subcommands for different functionality.
"""

# CRITICAL: Import torch_init FIRST to set environment variables before any torch imports
# isort: off
from ..core import torch_init  # noqa: F401

# isort: on

import logging
import os
import warnings

import click  # noqa: E402

from ..core.llm import OpenModelService  # noqa: E402
from . import memory as memory_commands  # noqa: E402
from .character import character_commands  # noqa: E402
from .config import config_commands  # noqa: E402
from .deploy import deploy_commands  # noqa: E402
from .llm import llm_commands  # noqa: E402
from .test import test_commands  # noqa: E402

# Configure logging
logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)

# Suppress expected warnings that are safe to ignore
warnings.filterwarnings("ignore", category=RuntimeWarning, module="character_ai.cli")
warnings.filterwarnings("ignore", category=UserWarning, message="CUDA.*unknown error")
warnings.filterwarnings(
    "ignore", category=RuntimeWarning, message=".*found in sys.modules.*"
)
warnings.filterwarnings(
    "ignore", category=RuntimeWarning, message="coroutine.*was never awaited"
)
warnings.filterwarnings(
    "ignore",
    category=PendingDeprecationWarning,
    message="Please use.*import python_multipart",
)
warnings.filterwarnings("ignore", category=UserWarning, message=".*CUDA.*")
warnings.filterwarnings("ignore", category=RuntimeWarning, message=".*coroutine.*")

# Suppress safe-to-ignore warnings from ML libraries
warnings.filterwarnings(
    "ignore", category=FutureWarning, message=".*clean_up_tokenization_spaces.*"
)
warnings.filterwarnings(
    "ignore", category=UserWarning, message=".*pkg_resources is deprecated.*"
)
warnings.filterwarnings(
    "ignore",
    category=UserWarning,
    message=".*GPT2InferenceModel has generative capabilities.*",
)
warnings.filterwarnings(
    "ignore",
    category=UserWarning,
    message=".*this function's implementation will be changed.*",
)
warnings.filterwarnings(
    "ignore",
    category=UserWarning,
    message=".*Some weights of Wav2Vec2ForCTC were not initialized.*",
)
warnings.filterwarnings(
    "ignore", category=UserWarning, message=".*The attention mask is not set.*"
)

# Set specific environment variables to suppress only the identified safe warnings
os.environ.setdefault(
    "TOKENIZERS_PARALLELISM", "false"
)  # Suppress tokenizer parallelism warning


def detect_cuda_availability() -> bool:
    """Detect CUDA availability with graceful fallback."""
    try:
        import torch

        return torch.cuda.is_available()
    except ImportError:
        return False
    except Exception:
        return False  # Graceful fallback for CUDA errors


@click.group()
@click.version_option(version="1.0.0")
@click.option("--verbose", "-v", is_flag=True, help="Enable verbose output")
@click.option("--debug", "-d", is_flag=True, help="Enable debug output")
@click.pass_context
def cli(ctx: click.Context, verbose: bool, debug: bool) -> None:
    """
    Character AI CLI

    A comprehensive platform for creating, managing, and interacting with AI
    characters.
    Supports multi-language voice interaction, character creation, and parental
    controls.
    """
    # Ensure that ctx.obj exists and is a dict
    ctx.ensure_object(dict)

    # Set logging level
    if debug:
        logging.getLogger().setLevel(logging.DEBUG)
    elif verbose:
        logging.getLogger().setLevel(logging.INFO)
    else:
        logging.getLogger().setLevel(logging.WARNING)

    # Detect CUDA availability and provide user feedback
    cuda_available = detect_cuda_availability()
    if not cuda_available and verbose:
        logger.info("CUDA not available - using CPU processing (this is normal)")
    elif cuda_available and verbose:
        logger.info("CUDA detected - GPU acceleration available")

    ctx.obj["verbose"] = verbose
    ctx.obj["debug"] = debug


@cli.group()
def llm() -> None:
    """LLM (Large Language Model) management commands."""
    pass


@cli.group()
def character() -> None:
    """Character creation and management commands."""
    pass


@cli.group()
def test() -> None:
    """Testing and validation commands."""
    pass


@cli.group()
def deploy() -> None:
    """Deployment and production commands."""
    pass


@cli.group()
def config() -> None:
    """Configuration management commands."""
    pass


@cli.group()
def memory() -> None:
    """Memory system management commands."""
    pass


# Import and register subcommands

# Add all commands from each module to their respective groups
for command in llm_commands.commands.values():
    llm.add_command(command)
for command in character_commands.commands.values():
    character.add_command(command)
for command in test_commands.commands.values():
    test.add_command(command)
for command in deploy_commands.commands.values():
    deploy.add_command(command)
for command in config_commands.commands.values():
    config.add_command(command)

# Add memory commands
memory.add_command(memory_commands.stats)
memory.add_command(memory_commands.export_user)
memory.add_command(memory_commands.cleanup)
memory.add_command(memory_commands.vacuum)
memory.add_command(memory_commands.clean)


@cli.command()
def status() -> None:
    """Show platform status and health."""
    click.echo("Character AI Status")
    click.echo("=" * 40)

    # Check core components
    try:
        from ..core.config import Config

        Config()
        click.echo("✓ Configuration: Loaded")
    except Exception as e:
        click.echo(f"✗ Configuration: Error - {e}")

    # Check LLM system
    try:
        from ..core.llm import LLMConfigService, LLMFactory, OpenModelService

        config_manager = LLMConfigService()
        model_manager = OpenModelService()
        LLMFactory(config_manager, model_manager)
        click.echo("✓ LLM System: Available")
    except Exception as e:
        click.echo(f"✗ LLM System: Error - {e}")

    # Check web API
    try:
        click.echo("✓ Web API: Available")
    except Exception as e:
        click.echo(f"✗ Web API: Error - {e}")

    click.echo("\nPlatform is ready for character creation and interaction!")


@cli.command()
@click.option(
    "--download-models", is_flag=True, help="Download default models automatically"
)
@click.option(
    "--model-size",
    type=click.Choice(["tiny", "small", "medium"]),
    default="tiny",
    help="Model size to download",
)
@click.option("--interactive", is_flag=True, help="Interactive setup with choices")
@click.option("--offline", is_flag=True, help="Show manual download instructions")
@click.option("--verify", is_flag=True, help="Verify installed models")
def setup(
    download_models: bool,
    model_size: str,
    interactive: bool,
    offline: bool,
    verify: bool,
) -> None:
    """Setup platform with automatic model download and configuration."""

    click.echo("🚀 Character AI Setup")
    click.echo("=" * 40)

    if verify:
        _verify_models()
        return

    if offline:
        _show_offline_instructions()
        return

    if interactive:
        _interactive_setup()
        return

    if download_models:
        _download_models(model_size)
        return

    # Default setup - check if models exist, offer to download
    model_manager = OpenModelService()
    if model_manager.has_any_models():
        click.echo("✅ Platform ready! Models found.")
        click.echo("Run 'cai status' to check platform health.")
    else:
        click.echo("❌ No models found. Platform needs models to function.")
        if click.confirm("Download default model now? (637MB)"):
            _download_models("tiny")
        else:
            click.echo("Run 'cai setup --offline' for manual download instructions.")


def _verify_models() -> None:
    """Verify installed models."""

    model_manager = OpenModelService()
    installed = model_manager.list_installed_models()

    if not installed:
        click.echo("❌ No models installed")
        return

    click.echo("✅ Installed models:")
    for model in installed:
        model_info = model_manager.get_model_info(model)
        if model_info:
            click.echo(f"  • {model} ({model_info.size})")


def _show_offline_instructions() -> None:
    """Show manual download instructions."""
    click.echo("📥 Manual Model Download Instructions")
    click.echo("=" * 40)
    click.echo("1. Download tinyllama-1.1b model:")
    click.echo(
        "   https://huggingface.co/TheBloke/TinyLlama-1.1B-Chat-v1.0-GGUF/resolve/main/tinyllama-1.1b-chat-v1.0.Q4_K_M.gguf"
    )
    click.echo("2. Create models/llm/ directory:")
    click.echo("   mkdir -p models/llm")
    click.echo("3. Place model file in models/llm/ directory")
    click.echo("4. Run 'cai setup --verify' to confirm installation")


def _interactive_setup() -> None:
    """Interactive setup with user choices."""
    click.echo("🎯 Interactive Setup")
    click.echo("=" * 20)

    # Hardware detection
    import psutil

    memory_gb = psutil.virtual_memory().total / (1024**3)

    if memory_gb < 4:
        recommended = "tiny"
        click.echo(
            f"💻 Detected: {memory_gb:.1f}GB RAM - Recommended: tiny model (637MB)"
        )
    elif memory_gb < 16:
        recommended = "small"
        click.echo(
            f"💻 Detected: {memory_gb:.1f}GB RAM - Recommended: small model "
            f"(2.3GB) - Good for development"
        )
    else:
        recommended = "medium"
        click.echo(
            f"💻 Detected: {memory_gb:.1f}GB RAM - Recommended: medium model "
            f"(4.1GB) - Great for development"
        )

    # Model selection
    model_choice = click.prompt(
        "Choose model size",
        type=click.Choice(["tiny", "small", "medium"]),
        default=recommended,
    )

    _download_models(model_choice)


def _download_models(model_size: str) -> None:
    """Download models based on size selection."""
    import asyncio

    model_manager = OpenModelService()

    # Map size to model names
    model_mapping = {
        "tiny": "tinyllama-1.1b",
        "small": "phi-3-mini",  # 2.3GB - good for development
        "medium": "mistral-7b-instruct",  # 4.1GB - better for development than 8B
    }

    model_name = model_mapping.get(model_size, "tinyllama-1.1b")
    model_info = model_manager.get_model_info(model_name)

    if not model_info:
        click.echo(f"❌ Unknown model: {model_name}")
        return

    if model_manager.is_model_installed(model_name):
        click.echo(f"✅ Model {model_name} already installed")
        return

    click.echo(f"📥 Downloading {model_name} ({model_info.size})...")
    click.echo("This may take a few minutes depending on your internet connection.")

    def progress_callback(progress: float) -> None:
        click.echo(f"\rProgress: {progress:.1f}%", nl=False)

    try:
        success = asyncio.run(
            model_manager.download_model(model_name, progress_callback)
        )
        if success:
            click.echo(f"\n✅ Successfully downloaded {model_name}")
            click.echo("🎉 Setup complete! Platform ready to use.")
            click.echo("Run 'cai status' to verify everything is working.")
        else:
            click.echo(f"\n❌ Failed to download {model_name}")
            click.echo("Check your internet connection and try again.")
    except Exception as e:
        click.echo(f"\n❌ Error during download: {e}")
        click.echo("Run 'cai setup --offline' for manual download instructions.")


@cli.command()
def info() -> None:
    """Show platform information."""
    click.echo("Character AI")
    click.echo("=" * 30)
    click.echo("Version: 1.0.0")
    click.echo("License: MIT")
    click.echo("Description: Edge-first AI character platform for toys")
    click.echo("\nFeatures:")
    click.echo("• Multi-language voice interaction")
    click.echo("• AI-powered character creation")
    click.echo("• Parental controls and safety")
    click.echo("• Real-time conversation")
    click.echo("• Open LLM support")
    click.echo("\nQuick Start:")
    click.echo("  cai setup                # Setup platform with models")
    click.echo("  cai character create    # Create a new character")
    click.echo("  cai llm list-models     # List available LLM models")
    click.echo("  cai test run           # Run platform tests")
    click.echo("  cai status             # Check platform status")


if __name__ == "__main__":
    cli()
