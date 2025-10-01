"""
Deployment commands for the Character AI CLI.

Provides Click-based commands for deployment, production setup, and submission.
"""

import json
import logging
import shutil
import subprocess  # nosec B404 - Required for deployment operations
import time
from pathlib import Path
from typing import Any, Optional

import click

logger = logging.getLogger(__name__)

# Allowed commands for security
ALLOWED_COMMANDS = {
    "docker", "cp", "uvicorn", "python"
}

def safe_subprocess_run(cmd: list[str], **kwargs: Any) -> subprocess.CompletedProcess:
    """Safely run subprocess with command validation."""
    if not cmd or cmd[0] not in ALLOWED_COMMANDS:
        raise ValueError(f"Command not allowed: {cmd[0] if cmd else 'empty'}")

    # Use absolute paths for security
    if cmd[0] == "docker":
        cmd[0] = shutil.which("docker") or "docker"
    elif cmd[0] == "cp":
        cmd[0] = shutil.which("cp") or "cp"
    elif cmd[0] == "uvicorn":
        cmd[0] = shutil.which("uvicorn") or "uvicorn"
    elif cmd[0] == "python":
        cmd[0] = shutil.which("python") or "python"

    return subprocess.run(cmd, **kwargs)  # nosec B603 - Validated commands only


@click.group()
def deploy_commands() -> None:
    """Deployment and production commands."""
    pass


@deploy_commands.command()
@click.option(
    "--environment",
    type=click.Choice(["development", "production", "demo"]),
    default="production",
    help="Deployment environment",
)
@click.option("--docker", is_flag=True, help="Use Docker for deployment")
@click.option("--models", is_flag=True, help="Include model files in deployment")
@click.option("--output", "-o", help="Output directory for deployment")
def build(environment: str, docker: bool, models: bool, output: Optional[str]) -> None:
    """Build deployment package."""
    try:
        click.echo(f"Building deployment package for {environment} environment...")

        if docker:
            _build_docker_deployment(environment, models)
        else:
            _build_standard_deployment(environment, models, output)

        click.echo("✓ Deployment package built successfully!")

    except Exception as e:
        click.echo(f"Error building deployment: {e}", err=True)
        raise click.Abort()


def _build_docker_deployment(environment: str, models: bool) -> None:
    """Build Docker-based deployment."""
    click.echo("Building Docker images...")

    # Build main application image
    click.echo("Building main application image...")
    result = safe_subprocess_run(
        ["docker", "build", "-t", f"icp-{environment}", "."],
        capture_output=True,
        text=True,
    )

    if result.returncode != 0:
        click.echo(f"Docker build failed: {result.stderr}", err=True)
        raise click.Abort()

    if models:
        click.echo("Building models image...")
        result = safe_subprocess_run(
            [
                "docker",
                "build",
                "-f",
                "Dockerfile.models",
                "-t",
                f"icp-{environment}-models",
                ".",
            ],
            capture_output=True,
            text=True,
        )

        if result.returncode != 0:
            click.echo(f"Models Docker build failed: {result.stderr}", err=True)
            raise click.Abort()

    click.echo("✓ Docker images built successfully!")


def _build_standard_deployment(environment: str, models: bool, output: Optional[str]) -> None:
    """Build standard deployment package."""
    if not output:
        output = f"deployment-{environment}"

    output_path = Path(output)
    output_path.mkdir(exist_ok=True)

    # Copy source code
    click.echo("Copying source code...")
    safe_subprocess_run(["cp", "-r", "src/", str(output_path / "src")])
    safe_subprocess_run(["cp", "-r", "configs/", str(output_path / "configs")])
    safe_subprocess_run(["cp", "pyproject.toml", str(output_path)])
    safe_subprocess_run(["cp", "README.md", str(output_path)])
    safe_subprocess_run(["cp", "LICENSE", str(output_path)])

    if models:
        click.echo("Copying model files...")
        models_path = output_path / "models"
        models_path.mkdir(exist_ok=True)
        safe_subprocess_run(["cp", "-r", "models/", str(models_path)])

    # Create deployment script
    deployment_script = output_path / "deploy.sh"
    with open(deployment_script, "w") as f:
        f.write(
            f"""#!/bin/bash
# Character AI Deployment Script
# Environment: {environment}

echo "Deploying Character AI..."

# Install dependencies
pip install -e .

# Set environment
export CAI_ENVIRONMENT={environment}

# Start the application
python -m character_ai.web.toy_api
"""
        )

    deployment_script.chmod(0o755)

    click.echo(f"✓ Deployment package created in {output_path}")


@deploy_commands.command()
@click.option(
    "--environment",
    type=click.Choice(["development", "production", "demo"]),
    default="production",
    help="Deployment environment",
)
@click.option("--port", type=int, default=8000, help="Port to run on")
@click.option("--host", default="127.0.0.1", help="Host to bind to (use 0.0.0.0 only with proper security)")
@click.option("--workers", type=int, default=1, help="Number of worker processes")
def start(environment: str, port: int, host: str, workers: int) -> None:
    """Start the platform in production mode."""
    try:
        click.echo("Starting Character AI...")
        click.echo(f"Environment: {environment}")
        click.echo(f"Host: {host}:{port}")
        click.echo(f"Workers: {workers}")

        # Set environment variables
        import os

        os.environ["CAI_ENVIRONMENT"] = environment

        # Start the application
        if environment == "production":
            # Use production server
            safe_subprocess_run(
                [
                    "uvicorn",
                    "character_ai.web.toy_api:app",
                    "--host",
                    host,
                    "--port",
                    str(port),
                    "--workers",
                    str(workers),
                ]
            )
        else:
            # Use development server
            safe_subprocess_run(
                ["python", "-m", "character_ai.web.toy_api"]
            )

    except Exception as e:
        click.echo(f"Error starting platform: {e}", err=True)
        raise click.Abort()


@deploy_commands.command()
@click.option(
    "--environment",
    type=click.Choice(["development", "production", "demo"]),
    default="production",
    help="Deployment environment",
)
@click.option("--output", "-o", help="Output file for health report")
def health(environment: str, output: Optional[str]) -> None:
    """Check deployment health."""
    try:
        click.echo(f"Checking platform health for {environment} environment...")

        health_report = {
            "environment": environment,
            "timestamp": "2025-01-01T00:00:00Z",
            "checks": {
                "configuration": "not_checked",
                "llm_system": "not_checked",
                "web_api": "not_checked",
                "database": "not_checked",
                "models": "not_checked",
            },
            "overall_health": "unknown",
        }

        # This would perform actual health checks
        click.echo("Health check not yet implemented")
        click.echo("Use the web API health endpoint or existing monitoring tools")

        if output:
            with open(output, "w") as f:
                json.dump(health_report, f, indent=2)
            click.echo(f"Health report saved to {output}")
        else:
            click.echo(json.dumps(health_report, indent=2))

    except Exception as e:
        click.echo(f"Error checking health: {e}", err=True)
        raise click.Abort()


@deploy_commands.command()
@click.option(
    "--environment",
    type=click.Choice(["development", "production", "demo"]),
    default="production",
    help="Deployment environment",
)
@click.option("--backup-dir", help="Directory to store backups")
def backup(environment: str, backup_dir: Optional[str]) -> None:
    """Create deployment backup."""
    try:
        if not backup_dir:
            backup_dir = f"backup-{environment}-{int(time.time())}"

        backup_path = Path(backup_dir)
        backup_path.mkdir(exist_ok=True)

        click.echo(f"Creating backup for {environment} environment...")
        click.echo(f"Backup directory: {backup_path}")

        # Backup configuration
        safe_subprocess_run(["cp", "-r", "configs/", str(backup_path / "configs")])

        # Backup characters
        if Path("characters").exists():
            safe_subprocess_run(["cp", "-r", "characters/", str(backup_path / "characters")])


        # Backup models (if not too large)
        if Path("models").exists():
            click.echo("Backing up models (this may take a while)...")
            safe_subprocess_run(["cp", "-r", "models/", str(backup_path / "models")])

        # Create backup metadata
        backup_info = {
            "environment": environment,
            "timestamp": "2025-01-01T00:00:00Z",
            "version": "0.1.0",
            "backup_dir": str(backup_path),
        }

        with open(backup_path / "backup_info.json", "w") as f:
            json.dump(backup_info, f, indent=2)

        click.echo(f"✓ Backup created in {backup_path}")

    except Exception as e:
        click.echo(f"Error creating backup: {e}", err=True)
        raise click.Abort()


@deploy_commands.command()
@click.argument("backup_dir")
def restore(backup_dir: str) -> None:
    """Restore from backup."""
    try:
        backup_path = Path(backup_dir)

        if not backup_path.exists():
            click.echo(f"Backup directory not found: {backup_dir}", err=True)
            raise click.Abort()

        if not click.confirm(
            f"Are you sure you want to restore from {backup_dir}? This will overwrite current data."
        ):
            click.echo("Restore cancelled.")
            return

        click.echo(f"Restoring from backup: {backup_path}")

        # Restore configuration
        if (backup_path / "configs").exists():
            safe_subprocess_run(["cp", "-r", str(backup_path / "configs"), "."])
            click.echo("✓ Configuration restored")

        # Restore characters
        if (backup_path / "characters").exists():
            safe_subprocess_run(["cp", "-r", str(backup_path / "characters"), "."])
            click.echo("✓ Characters restored")

        # Restore models
        if (backup_path / "models").exists():
            safe_subprocess_run(["cp", "-r", str(backup_path / "models"), "."])
            click.echo("✓ Models restored")

        click.echo("✓ Restore completed successfully!")

    except Exception as e:
        click.echo(f"Error restoring backup: {e}", err=True)
        raise click.Abort()


@deploy_commands.command()
@click.option(
    "--environment",
    type=click.Choice(["development", "production", "demo"]),
    default="production",
    help="Deployment environment",
)
@click.option("--api-key", help="API key for submission")
@click.option(
    "--dry-run", is_flag=True, help="Perform dry run without actual submission"
)
def submit(environment: str, api_key: Optional[str], dry_run: bool) -> None:
    """Submit platform to Hasbro API challenge."""
    try:
        if dry_run:
            click.echo("Dry run mode - no actual submission will be made")

        click.echo(f"Preparing submission for {environment} environment...")

        # This would integrate with the existing hasbro_api_challenge.py script
        # For now, just show the path to the script
        script_path = Path("scripts/hasbro_api_challenge.py")

        if script_path.exists():
            click.echo(f"Use the submission script: {script_path}")
            click.echo("Run: python scripts/hasbro_api_challenge.py")
        else:
            click.echo(
                "Submission script not found. Please ensure scripts/hasbro_api_challenge.py exists."
            )

        click.echo("\nSubmission preparation not yet integrated into CLI")
        click.echo("Use the dedicated submission script for now")

    except Exception as e:
        click.echo(f"Error preparing submission: {e}", err=True)
        raise click.Abort()


@deploy_commands.command()
def status() -> None:
    """Show deployment status."""
    try:
        click.echo("Character AI Deployment Status")
        click.echo("=" * 50)

        # Check if running
        try:
            import requests

            response = requests.get("http://localhost:8000/health", timeout=5)
            if response.status_code == 200:
                click.echo("✓ Platform is running")
            else:
                click.echo("✗ Platform is not responding correctly")
        except Exception:
            click.echo("✗ Platform is not running")

        # Check configuration
        config_path = Path("configs")
        if config_path.exists():
            click.echo("✓ Configuration files found")
        else:
            click.echo("✗ Configuration files missing")

        # Check models
        models_path = Path("models")
        if models_path.exists():
            model_count = len(list(models_path.glob("**/*.gguf")))
            click.echo(f"✓ Models directory found ({model_count} models)")
        else:
            click.echo("✗ Models directory missing")

        # Check characters
        characters_path = Path("characters")
        if characters_path.exists():
            character_count = len(list(characters_path.glob("*.yaml")))
            click.echo(f"✓ Characters directory found ({character_count} characters)")
        else:
            click.echo("✗ Characters directory missing")

    except Exception as e:
        click.echo(f"Error checking status: {e}", err=True)
        raise click.Abort()


@deploy_commands.command()
@click.argument("character")
@click.argument("franchise")
@click.option("--environment", default="production", help="Deployment environment")
@click.option("--scale", default="1", help="Number of replicas to deploy")
@click.option("--bundle-path", help="Path to character bundle (optional)")
def deploy_character(character: str, franchise: str, environment: str, scale: str, bundle_path: Optional[str]) -> None:
    """Deploy specific character to production."""
    try:
        click.echo(f"🚀 Deploying character '{character}' from franchise '{franchise}'")
        click.echo(f"Environment: {environment}")
        click.echo(f"Scale: {scale} replicas")

        if bundle_path:
            click.echo(f"Bundle: {bundle_path}")

        # Validate character exists
        character_dir = Path(f"configs/characters/{character}")
        if not character_dir.exists():
            click.echo(f"❌ Character '{character}' not found in configs/characters/", err=True)
            raise click.Abort()

        # Check if bundle exists
        if bundle_path and not Path(bundle_path).exists():
            click.echo(f"❌ Bundle not found: {bundle_path}", err=True)
            raise click.Abort()

        # Deploy character
        click.echo("📦 Preparing character deployment...")

        # Create deployment configuration
        deployment_config = {
            "character": character,
            "franchise": franchise,
            "environment": environment,
            "scale": int(scale),
            "deployment_time": "2025-01-01T00:00:00Z"
        }

        # Save deployment config
        deploy_dir = Path.cwd() / "deployments"
        if not deploy_dir.exists():
            deploy_dir.mkdir(parents=True, exist_ok=True)

        config_file = deploy_dir / f"{character}_{franchise}_{environment}.json"
        with open(config_file, "w") as f:
            json.dump(deployment_config, f, indent=2)

        click.echo(f"✅ Character deployment configuration created: {config_file}")
        click.echo(f"🎭 Character: {character}")
        click.echo(f"🏢 Franchise: {franchise}")
        click.echo(f"🌍 Environment: {environment}")
        click.echo(f"📈 Scale: {scale} replicas")

        if bundle_path:
            click.echo(f"📦 Bundle: {bundle_path}")

        click.echo("\n🚀 Character deployment ready!")
        click.echo("💡 Use your container orchestration system (Docker, Kubernetes, etc.) to deploy the character")

    except Exception as e:
        click.echo(f"Error deploying character: {e}", err=True)
        raise click.Abort()
