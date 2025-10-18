"""Deployment commands for the Character AI CLI.

Provides Click-based commands for creating deployment bundles.
"""
# mypy: ignore-errors

import click

from ..characters.catalog import CharacterBundler


@click.group()
def deploy_commands() -> None:
    """Deployment and bundling commands."""
    pass


@deploy_commands.command()
@click.option("--character", "-c", required=True)
@click.option("--franchise", "-f", required=True)
@click.option(
    "--hardware",
    "-h",
    required=True,
    help="Hardware profile (raspberry_pi, desktop, etc)",
)
@click.option("--format", type=click.Choice(["tar.gz", "docker"]), default="tar.gz")
@click.option(
    "--output", "-o", default="bundles", help="Output directory (default: bundles/)"
)
@click.option("--include-models/--no-models", default=True, help="Bundle models")
def bundle(character, franchise, hardware, format, output, include_models):
    """Create deployment bundle for character on specific hardware."""

    bundler = CharacterBundler()
    output_path = bundler.create_bundle(
        character=character,
        franchise=franchise,
        hardware_profile=hardware,
        output_format=format,
        include_models=include_models,
        output_dir=output,
    )

    click.echo(f"✅ Bundle created: {output_path}")
    click.echo(f"   Format: {format}")
    click.echo(f"   Hardware: {hardware}")
    click.echo(f"   Models included: {include_models}")


@deploy_commands.command()
@click.option("--list-profiles", is_flag=True, help="List available hardware profiles")
def profiles(list_profiles):
    """Manage hardware profiles."""
    if list_profiles:
        from ..services.hardware_profile_service import HardwareProfileManager

        profile_manager = HardwareProfileManager()
        profiles = profile_manager.list_available_profiles()

        click.echo("Available hardware profiles:")
        for profile in profiles:
            click.echo(f"  - {profile}")
    else:
        click.echo("Use --list-profiles to see available hardware profiles")


@deploy_commands.command()
@click.option("--character", "-c", required=True)
@click.option("--franchise", "-f", required=True)
@click.option("--hardware", "-h", required=True)
@click.option("--output", "-o", help="Output directory")
def validate(character, franchise, hardware, output):
    """Validate deployment configuration."""
    from ..services.hardware_profile_service import HardwareProfileManager

    click.echo(f"Validating deployment for {character} ({franchise}) on {hardware}...")

    # Validate hardware profile
    try:
        profile_manager = HardwareProfileManager()
        hardware_config = profile_manager.load_profile(hardware)
        click.echo(f"✅ Hardware profile '{hardware}' is valid")
    except Exception as e:
        click.echo(f"❌ Hardware profile error: {e}")
        return

    # Validate character
    try:
        from .character.helpers import _load_character_manager

        character_service = _load_character_manager()
        character_obj = character_service.get_character(character)
        if character_obj:
            click.echo(f"✅ Character '{character}' is valid")
        else:
            click.echo(f"❌ Character '{character}' not found")
            return
    except Exception as e:
        click.echo(f"❌ Character validation error: {e}")
        return

    # Validate models
    try:
        profile_manager.get_hardware_constraints(hardware)
        models = hardware_config.get("models", {})

        click.echo("Model requirements:")
        for model_type, model_name in models.items():
            click.echo(f"  {model_type}: {model_name}")

        click.echo("✅ All validations passed")

    except Exception as e:
        click.echo(f"❌ Model validation error: {e}")


if __name__ == "__main__":
    deploy_commands()
