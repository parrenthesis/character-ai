"""Character deployment bundler for multi-platform deployment."""

import logging
import shutil
import tempfile
from pathlib import Path
from typing import Optional

try:
    import yaml
except ImportError:
    yaml = None  # type: ignore[assignment]

logger = logging.getLogger(__name__)


class CharacterBundler:
    """Create deployment bundles for characters."""

    def create_bundle(
        self,
        character: str,
        franchise: str,
        hardware_profile: str,
        output_format: str = "tar.gz",
        include_models: bool = True,
        output_dir: Optional[str] = None,
    ) -> str:
        """Create deployment bundle."""

        # Create temp directory
        bundle_dir = Path(tempfile.mkdtemp())

        try:
            # Copy character configs
            self._copy_character_configs(bundle_dir, character, franchise)

            # Copy hardware config
            self._copy_hardware_config(bundle_dir, hardware_profile)

            # Create merged runtime config
            self._create_runtime_config(
                bundle_dir, character, franchise, hardware_profile
            )

            # Copy models if requested
            if include_models:
                self._copy_models(bundle_dir, hardware_profile)

            # Create install script
            self._create_install_script(bundle_dir, include_models)

            # Create README
            self._create_readme(bundle_dir, character, franchise, hardware_profile)

            # Package based on format
            if output_format == "tar.gz":
                output_path = self._create_tarball(
                    bundle_dir, character, hardware_profile, output_dir
                )
            elif output_format == "docker":
                output_path = self._create_docker_image(
                    bundle_dir,
                    character,
                    franchise,
                    hardware_profile,
                    output_dir or ".",
                )
            else:
                raise ValueError(f"Unsupported output format: {output_format}")

            return output_path

        finally:
            # Cleanup temp directory
            shutil.rmtree(bundle_dir)

    def _copy_character_configs(
        self, bundle_dir: Path, character: str, franchise: str
    ) -> None:
        """Copy character configuration files."""
        char_config_dir = bundle_dir / "configs" / "characters" / franchise / "data"
        char_config_dir.mkdir(parents=True, exist_ok=True)

        source_dir = Path(f"configs/characters/{franchise}/data")
        if source_dir.exists():
            # Copy all character config files
            for config_file in source_dir.glob("*"):
                if config_file.is_file():
                    shutil.copy2(config_file, char_config_dir / config_file.name)
            logger.info(f"Copied character configs for {character}")
        else:
            logger.warning(f"Character config directory not found: {source_dir}")

    def _copy_hardware_config(self, bundle_dir: Path, hardware_profile: str) -> None:
        """Copy hardware profile configuration."""
        hw_config_dir = bundle_dir / "configs" / "hardware"
        hw_config_dir.mkdir(parents=True, exist_ok=True)

        source_file = Path(f"configs/hardware/{hardware_profile}.yaml")
        if source_file.exists():
            shutil.copy2(source_file, hw_config_dir / f"{hardware_profile}.yaml")
            logger.info(f"Copied hardware profile: {hardware_profile}")
        else:
            logger.warning(f"Hardware profile not found: {source_file}")

    def _create_runtime_config(
        self, bundle_dir: Path, character: str, franchise: str, hardware_profile: str
    ) -> None:
        """Create merged runtime configuration."""
        from ..config import Config
        from ..hardware_profile import HardwareProfileManager

        # Load base runtime config
        base_config = Config()

        # Load hardware profile
        profile_manager = HardwareProfileManager()
        hardware_config = profile_manager.load_profile(hardware_profile)

        # Merge configurations
        merged_config = profile_manager.merge_with_config(hardware_config, base_config)

        # Create runtime config directory
        runtime_dir = bundle_dir / "configs"
        runtime_dir.mkdir(exist_ok=True)

        # Write merged config
        runtime_config = {
            "runtime": {
                "target_latency_s": getattr(
                    merged_config.runtime, "target_latency_s", 0.5
                ),
                "streaming_enabled": getattr(
                    merged_config.runtime, "streaming_enabled", True
                ),
                "predictive_loading": getattr(
                    merged_config.runtime, "predictive_loading", True
                ),
                "idle_timeout_s": getattr(merged_config.runtime, "idle_timeout_s", 300),
            },
            "interaction": {
                "stt_language": getattr(
                    merged_config.interaction, "stt_language", "en"
                ),
                "min_audio_s": getattr(merged_config.interaction, "min_audio_s", 0.2),
                "max_audio_s": getattr(merged_config.interaction, "max_audio_s", 30.0),
                "max_new_tokens": getattr(
                    merged_config.interaction, "max_new_tokens", 64
                ),
                "temperature": getattr(merged_config.interaction, "temperature", 0.6),
                "sample_rate": getattr(merged_config.interaction, "sample_rate", 16000),
                "channels": getattr(merged_config.interaction, "channels", 1),
            },
            "models": hardware_config.get("models", {}),
            "model_registry": getattr(base_config.runtime, "model_registry", {}),
            "wake_words": getattr(base_config.runtime, "wake_words", {}),
        }

        if yaml:
            with open(runtime_dir / "runtime.yaml", "w") as f:
                yaml.dump(runtime_config, f, default_flow_style=False)
        else:
            # Fallback to JSON if yaml not available
            import json

            with open(runtime_dir / "runtime.json", "w") as f:
                json.dump(runtime_config, f, indent=2)

        logger.info("Created merged runtime configuration")

    def _copy_models(self, bundle_dir: Path, hardware_profile: str) -> None:
        """Copy required models for the hardware profile."""
        from ..hardware_profile import HardwareProfileManager

        profile_manager = HardwareProfileManager()
        hardware_config = profile_manager.load_profile(hardware_profile)
        models = hardware_config.get("models", {})

        models_dir = bundle_dir / "models"
        models_dir.mkdir(exist_ok=True)

        # Copy model files based on hardware profile requirements
        for model_type, model_name in models.items():
            model_dir = models_dir / model_type
            model_dir.mkdir(exist_ok=True)

            # This would copy actual model files in a real implementation
            # For now, create a placeholder
            placeholder_file = model_dir / f"{model_name}.placeholder"
            placeholder_file.write_text(
                f"Model: {model_name}\nType: {model_type}\nHardware: {hardware_profile}"
            )

        logger.info(f"Created model placeholders for {len(models)} models")

    def _create_install_script(self, bundle_dir: Path, include_models: bool) -> None:
        """Create installation script."""
        install_script = bundle_dir / "install.sh"

        script_content = f"""#!/bin/bash
# Character AI Deployment Installer
# Generated by CharacterBundler

set -e

echo "Installing Character AI deployment..."

# Check Python version
python3 --version || {{ echo "Python 3 is required"; exit 1; }}

# Install dependencies
echo "Installing Python dependencies..."
pip3 install -r requirements.txt

# Set up configuration
echo "Setting up configuration..."
mkdir -p ~/.character_ai/configs
cp -r configs/* ~/.character_ai/configs/

# Set up models
{"if [ -d models ]; then" if include_models else "# Models not included in this bundle"}
{"    echo 'Setting up models...'" if include_models else ""}
{"    mkdir -p ~/.character_ai/models" if include_models else ""}
{"    cp -r models/* ~/.character_ai/models/" if include_models else ""}
{"fi" if include_models else ""}

# Set up audio permissions
echo "Setting up audio permissions..."
sudo usermod -a -G audio $USER || true

echo "Installation complete!"
echo "Run: cai test voice-pipeline --character <character> --franchise <franchise> --realtime"
"""

        install_script.write_text(script_content)
        install_script.chmod(0o755)

        logger.info("Created installation script")

    def _create_readme(
        self, bundle_dir: Path, character: str, franchise: str, hardware_profile: str
    ) -> None:
        """Create README for the bundle."""
        readme_file = bundle_dir / "README.md"

        readme_content = f"""# Character AI Deployment Bundle

## Character: {character} ({franchise})
## Hardware Profile: {hardware_profile}

This bundle contains everything needed to deploy the {character} character on {hardware_profile} hardware.

### Contents

- `configs/` - Character and hardware configuration files
- `models/` - AI model files (if included)
- `install.sh` - Installation script
- `requirements.txt` - Python dependencies

### Installation

1. Extract this bundle to your target system
2. Run the installation script:
   ```bash
   chmod +x install.sh
   ./install.sh
   ```

### Usage

After installation, you can test the character:

```bash
# Test real-time interaction
cai test voice-pipeline --character {character} --franchise {franchise} --realtime

# Test with specific hardware profile
cai test voice-pipeline --character {character} --franchise {franchise} --hardware-profile {hardware_profile} --realtime
```

### Hardware Requirements

This bundle is optimized for {hardware_profile} hardware. See `configs/hardware/{hardware_profile}.yaml` for specific requirements.

### Support

For issues or questions, please refer to the Character AI documentation.
"""

        readme_file.write_text(readme_content)

        logger.info("Created README")

    def _create_tarball(
        self,
        bundle_dir: Path,
        character: str,
        hardware_profile: str,
        output_dir: Optional[str],
    ) -> str:
        """Create tar.gz bundle."""
        import tarfile

        output_dir_path = Path(output_dir) if output_dir else Path(".")
        output_dir_path.mkdir(exist_ok=True)

        bundle_name = f"{character}_{hardware_profile}_bundle.tar.gz"
        output_path = output_dir_path / bundle_name

        with tarfile.open(output_path, "w:gz") as tar:
            tar.add(bundle_dir, arcname=".")

        logger.info(f"Created tarball: {output_path}")
        return str(output_path)

    def _create_docker_image(
        self,
        bundle_dir: Path,
        character: str,
        franchise: str,
        hardware_profile: str,
        output_dir: Optional[str],
    ) -> str:
        """Create Docker image bundle."""
        # Create Dockerfile
        dockerfile = bundle_dir / "Dockerfile"

        # Dockerfile template - not SQL, bandit false positive
        dockerfile_content = f"""FROM python:3.11-slim

# Install system dependencies
RUN apt-get update && apt-get install -y \\
    portaudio19-dev \\
    libsndfile1 \\
    && rm -rf /var/lib/apt/lists/*

# Set working directory
WORKDIR /app

# Copy requirements and install Python dependencies
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy application files
COPY . .

# Set up configuration
RUN mkdir -p /app/configs
RUN cp -r configs/* /app/configs/

# Set up models if included
{"RUN mkdir -p /app/models && cp -r models/* /app/models/" if True else "# Models not included"}

# Expose port for web interface (if needed)
EXPOSE 8000

# Set environment variables
ENV CHARACTER_AI_CONFIG_DIR=/app/configs
ENV CHARACTER_AI_MODEL_DIR=/app/models

# Default command
CMD ["python", "-m", "character_ai.cli.main"]
"""  # nosec B608

        dockerfile.write_text(dockerfile_content)

        # Create .dockerignore
        dockerignore = bundle_dir / ".dockerignore"
        dockerignore.write_text(
            """__pycache__/
*.pyc
*.pyo
*.pyd
.Python
env/
venv/
.venv/
pip-log.txt
pip-delete-this-directory.txt
.tox/
.coverage
.coverage.*
.cache
nosetests.xml
coverage.xml
*.cover
*.log
.git/
.mypy_cache/
.pytest_cache/
.hypothesis/
"""
        )

        # Create docker-compose.yml for easy deployment
        compose_file = bundle_dir / "docker-compose.yml"
        compose_content = f"""version: '3.8'

services:
  character-ai:
    build: .
    container_name: character-ai-{character}
    environment:
      - CHARACTER_AI_CONFIG_DIR=/app/configs
      - CHARACTER_AI_MODEL_DIR=/app/models
    volumes:
      - ./configs:/app/configs:ro
      - ./models:/app/models:ro
    devices:
      - /dev/snd:/dev/snd  # Audio device access
    privileged: true  # Required for audio device access
    restart: unless-stopped
"""

        compose_file.write_text(compose_content)

        output_dir_path = Path(output_dir) if output_dir else Path(".")
        output_dir_path.mkdir(exist_ok=True)

        # Create tar.gz of Docker bundle
        bundle_name = f"{character}_{hardware_profile}_docker_bundle.tar.gz"
        output_path = output_dir_path / bundle_name

        import tarfile

        with tarfile.open(output_path, "w:gz") as tar:
            tar.add(bundle_dir, arcname=".")

        logger.info(f"Created Docker bundle: {output_path}")
        return str(output_path)
