"""
Configuration management commands for the Character AI CLI.

Provides Click-based commands for managing platform configuration.
"""

import json
import logging
from pathlib import Path
from typing import Optional

import click

from ..core.config import Config

logger = logging.getLogger(__name__)


@click.group()
def config_commands() -> None:
    """Configuration management commands."""
    pass


@config_commands.command()
@click.option(
    "--format",
    type=click.Choice(["json", "yaml", "table"]),
    default="table",
    help="Output format",
)
@click.option("--section", help="Show specific configuration section")
def show(format: str, section: Optional[str]) -> None:
    """Show current configuration."""
    try:
        config = Config()

        if section:
            # Show specific section
            if hasattr(config, section):
                section_config = getattr(config, section)
                if format == "json":
                    click.echo(
                        json.dumps(section_config.__dict__, indent=2, default=str)
                    )
                elif format == "yaml":
                    import yaml

                    click.echo(
                        yaml.dump(section_config.__dict__, default_flow_style=False)
                    )
                else:
                    click.echo(f"{section.title()} Configuration:")
                    click.echo("=" * 40)
                    for key, value in section_config.__dict__.items():
                        click.echo(f"{key}: {value}")
            else:
                click.echo(f"Unknown section: {section}", err=True)
                raise click.Abort()
        else:
            # Show all configuration
            if format == "json":
                click.echo(json.dumps(config.__dict__, indent=2, default=str))
            elif format == "yaml":
                import yaml

                click.echo(yaml.dump(config.__dict__, default_flow_style=False))
            else:
                click.echo("Character AI Configuration")
                click.echo("=" * 50)
                for section_name in [
                    "environment",
                    "models",
                    "api",
                    "security",
                    "language_support",
                    "multilingual_audio",
                    "personalization",
                    "parental_controls",
                ]:
                    if hasattr(config, section_name):
                        section_config = getattr(config, section_name)
                        click.echo(f"\n{section_name.title()}:")
                        for key, value in section_config.__dict__.items():
                            click.echo(f"  {key}: {value}")

    except Exception as e:
        click.echo(f"Error showing configuration: {e}", err=True)
        raise click.Abort()


@config_commands.command()
@click.argument("key")
@click.argument("value")
def set(key: str, value: str) -> None:
    """Set a configuration value."""
    try:
        click.echo(f"Setting {key} = {value}")
        click.echo("Configuration setting not yet implemented")
        click.echo("Use environment variables or configuration files for now")

    except Exception as e:
        click.echo(f"Error setting configuration: {e}", err=True)
        raise click.Abort()


@config_commands.command()
@click.argument("file_path")
def load(file_path: str) -> None:
    """Load configuration from file."""
    try:
        config_path = Path(file_path)

        if not config_path.exists():
            click.echo(f"Configuration file not found: {file_path}", err=True)
            raise click.Abort()

        click.echo(f"Loading configuration from {file_path}...")
        click.echo("Configuration loading not yet implemented")
        click.echo("Use environment variables for configuration")

    except Exception as e:
        click.echo(f"Error loading configuration: {e}", err=True)
        raise click.Abort()


@config_commands.command()
@click.argument("file_path")
def save(file_path: str) -> None:
    """Save current configuration to file."""
    try:
        config = Config()
        output_path = Path(file_path)

        click.echo(f"Saving configuration to {file_path}...")

        # Create directory if needed
        output_path.parent.mkdir(parents=True, exist_ok=True)

        # Save as JSON
        with open(output_path, "w") as f:
            json.dump(config.__dict__, f, indent=2, default=str)

        click.echo(f"✓ Configuration saved to {file_path}")

    except Exception as e:
        click.echo(f"Error saving configuration: {e}", err=True)
        raise click.Abort()


@config_commands.command()
def validate() -> None:
    """Validate current configuration."""
    try:
        click.echo("Validating configuration...")

        config = Config()
        validation_results = {"valid": True, "errors": [], "warnings": []}

        # Check environment
        if config.environment.value not in ["development", "production", "demo"]:
            validation_results["errors"].append(  # type: ignore
                f"Invalid environment: {config.environment}"
            )
            validation_results["valid"] = False

        # Check required directories
        required_dirs = ["data", "models", "logs"]
        for dir_name in required_dirs:
            dir_path = getattr(config, f"{dir_name}_dir", None)
            if dir_path and not dir_path.exists():
                validation_results["warnings"].append(  # type: ignore  # type: ignore
                    f"Directory does not exist: {dir_path}"
                )

        # Check API configuration
        if config.api.host == "0.0.0.0" and config.environment.value == "production":  # nosec B104
            validation_results["warnings"].append(  # type: ignore
                "Binding to 0.0.0.0 in production may be insecure. Consider using a reverse proxy."
            )

        # Display results
        if validation_results["valid"]:
            click.echo("✓ Configuration is valid")
        else:
            click.echo("✗ Configuration has errors")

        if validation_results["errors"]:
            click.echo("\nErrors:")
            for error in validation_results["errors"]:  # type: ignore
                click.echo(f"  - {error}")

        if validation_results["warnings"]:
            click.echo("\nWarnings:")
            for warning in validation_results["warnings"]:  # type: ignore
                click.echo(f"  - {warning}")

        if not validation_results["valid"]:
            raise click.Abort()

    except Exception as e:
        click.echo(f"Error validating configuration: {e}", err=True)
        raise click.Abort()


@config_commands.command()
@click.option(
    "--template",
    type=click.Choice(["development", "production", "demo"]),
    help="Configuration template to generate",
)
@click.option("--output", "-o", help="Output file for generated configuration")
def generate(template: Optional[str], output: Optional[str]) -> None:
    """Generate configuration template."""
    try:
        if not template:
            click.echo("Available templates:")
            click.echo("  development - Development environment")
            click.echo("  production  - Production environment")
            click.echo("  demo        - Demo environment")
            return

        if not output:
            output = f"config-{template}.yaml"

        click.echo(f"Generating {template} configuration template...")

        # Generate template configuration
        template_config = {
            "environment": template,
            "debug": template == "development",
            "api": {
                "host": "127.0.0.1",  # Use localhost by default for security
                "port": 8000,
                "workers": 4 if template == "production" else 1,
            },
            "models": {
                "llm_backend": "llama_cpp",
                "wav2vec2_model": "facebook/wav2vec2-base",
                "coqui_model": "tts_models/en/ljspeech/tacotron2-DDC",
            },
            "security": {
                "jwt_secret": "${CAI_JWT_SECRET}",
                "require_https": template == "production",
            },
        }

        # Save template
        output_path = Path(output)
        output_path.parent.mkdir(parents=True, exist_ok=True)

        if output_path.suffix == ".json":
            with open(output_path, "w") as f:
                json.dump(template_config, f, indent=2)
        else:
            import yaml

            with open(output_path, "w") as f:
                yaml.dump(template_config, f, default_flow_style=False)

        click.echo(f"✓ Configuration template saved to {output_path}")

    except Exception as e:
        click.echo(f"Error generating configuration: {e}", err=True)
        raise click.Abort()


@config_commands.command()
def env() -> None:
    """Show environment variables used by the platform."""
    try:
        click.echo("Character AI Environment Variables")
        click.echo("=" * 55)

        env_vars = {
            "CAI_ENVIRONMENT": "Platform environment (development/production/demo)",
            "CAI_DEBUG": "Enable debug mode (true/false)",
            "CAI_JWT_SECRET": "JWT secret for authentication",
            "CAI_PRIVATE_KEY_FILE": "Path to private key file",
            "CAI_MODELS_DIR": "Directory for model files",
            "CAI_WAV2VEC2_MODEL": "Wav2Vec2 model name",
            "CAI_COQUI_MODEL": "Coqui TTS model name",
            "CAI_ENABLE_HTTPS": "Enable HTTPS (true/false)",
            "CAI_CERT_FILE": "SSL certificate file",
            "CAI_KEY_FILE": "SSL private key file",
            "CAI_LOG_LEVEL": "Logging level (DEBUG/INFO/WARNING/ERROR)",
            "CAI_LOG_FILE": "Log file path",
            "CAI_LANGUAGE_SUPPORT__DEFAULT_LANGUAGE": "Default language",
            "CAI_LANGUAGE_SUPPORT__SUPPORTED_LANGUAGES": "Comma-separated list of supported languages",
            "CAI_MULTILINGUAL_AUDIO__TTS_LANGUAGES": "TTS supported languages",
            "CAI_MULTILINGUAL_AUDIO__STT_LANGUAGES": "STT supported languages",
            "CAI_PERSONALIZATION__ENABLED": "Enable personalization (true/false)",
            "CAI_PARENTAL_CONTROLS__ENABLED": "Enable parental controls (true/false)",
        }

        for var, description in env_vars.items():
            click.echo(f"{var}: {description}")

        click.echo("\nExample .env file:")
        click.echo("CAI_ENVIRONMENT=development")
        click.echo("CAI_DEBUG=true")
        click.echo("CAI_JWT_SECRET=your-secret-key")
        click.echo("CAI_MODELS_DIR=models")
        click.echo("CAI_LOG_LEVEL=INFO")

    except Exception as e:
        click.echo(f"Error showing environment variables: {e}", err=True)
        raise click.Abort()
