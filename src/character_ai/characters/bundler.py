"""
Character bundling functionality for production deployment.
Handles creating character bundles with all necessary assets.
"""

import json
import tarfile
from datetime import datetime
from pathlib import Path
from typing import Optional

import yaml

from character_ai.core.config import Config


class CharacterBundler:
    """Handles character bundling for production deployment."""

    def __init__(self, config: Config, base_path: str | None = None):
        self.config = config
        self.base_path = Path(base_path) if base_path else Path(".")
        # Don't initialize managers - they create directories in project root
        # We only need them for specific operations, not for bundling

    def bundle_character(
        self,
        character_id: str,
        franchise: str,
        include_voice: bool = True,
        include_models: bool = True,
        output_path: str | None = None,
    ) -> str:
        """
        Bundle a character for production deployment.

        Args:
            character_id: Character identifier
            franchise: Franchise identifier
            include_voice: Include voice models in bundle
            include_models: Include LLM models in bundle
            output_path: Output path for bundle (optional)

        Returns:
            Path to created bundle file
        """

        # Generate output path if not provided
        if not output_path:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            bundles_dir = Path("bundles")
            bundles_dir.mkdir(exist_ok=True)
            output_path = str(
                bundles_dir / f"{character_id}_{franchise}_{timestamp}.tar.gz"
            )

        # Create bundle directory
        bundle_dir = Path(f"temp_bundle_{character_id}")
        bundle_dir.mkdir(exist_ok=True)

        try:
            # Copy character profile and prompts
            self._copy_character_assets(character_id, bundle_dir)

            # Copy voice assets if requested
            if include_voice:
                self._copy_voice_assets(character_id, bundle_dir)

            # Copy model assets if requested
            if include_models:
                self._copy_model_assets(character_id, bundle_dir)

            # Create deployment configuration
            self._create_deployment_config(character_id, franchise, bundle_dir)

            # Create bundle manifest
            self._create_bundle_manifest(character_id, franchise, bundle_dir)

            # Create tar.gz bundle
            with tarfile.open(output_path, "w:gz") as tar:
                tar.add(bundle_dir, arcname=character_id)

            print(f"✅ Character bundle created: {output_path}")
            return output_path

        finally:
            # Clean up temporary directory
            if bundle_dir.exists():
                import shutil

                shutil.rmtree(bundle_dir)

    def _copy_character_assets(self, character_id: str, bundle_dir: Path) -> None:
        """Copy character profile and prompts to bundle."""

        character_dir = self.base_path / "configs" / "characters" / character_id
        if not character_dir.exists():
            raise ValueError(f"Character not found: {character_id}")

        # Copy profile and prompts
        for file_name in ["profile.yaml", "prompts.yaml"]:
            src_file = character_dir / file_name
            if src_file.exists():
                dst_file = bundle_dir / file_name
                dst_file.write_text(src_file.read_text())
                print(f"  ✓ Copied {file_name}")

    def _copy_voice_assets(self, character_id: str, bundle_dir: Path) -> None:
        """Copy voice assets to bundle."""

        voice_dir = bundle_dir / "voices"
        voice_dir.mkdir(exist_ok=True)

        # Copy voice files from catalog
        catalog_voice_dir = Path("catalog/voices")
        if catalog_voice_dir.exists():
            for voice_file in catalog_voice_dir.glob(f"{character_id}*"):
                if voice_file.is_file():
                    dst_file = voice_dir / voice_file.name
                    dst_file.write_bytes(voice_file.read_bytes())
                    print(f"  ✓ Copied voice: {voice_file.name}")

        # Copy voice samples if they exist
        voice_samples_dir = (
            self.base_path / "configs" / "characters" / character_id / "voice_samples"
        )
        if voice_samples_dir.exists():
            samples_dir = bundle_dir / "voice_samples"
            samples_dir.mkdir(exist_ok=True)

            for sample_file in voice_samples_dir.glob("*"):
                if sample_file.is_file():
                    dst_file = samples_dir / sample_file.name
                    dst_file.write_bytes(sample_file.read_bytes())
                    print(f"  ✓ Copied voice sample: {sample_file.name}")

    def _copy_model_assets(self, character_id: str, bundle_dir: Path) -> None:
        """Copy model assets to bundle based on character configuration."""

        models_dir = bundle_dir / "models"
        models_dir.mkdir(exist_ok=True)

        # Read character profile to get model names
        profile_file = (
            self.base_path / "configs" / "characters" / character_id / "profile.yaml"
        )
        if not profile_file.exists():
            print(f"  ⚠️  Character profile not found: {profile_file}")
            return

        with open(profile_file, "r") as f:
            profile = yaml.safe_load(f)

        # Get model names from profile
        llm_model = profile.get("llm", {}).get("model", "phi-3-mini-4k-instruct")
        stt_model = profile.get("stt", {}).get("model", "wav2vec2-base")
        tts_model = profile.get("tts", {}).get("model", "coqui")

        print("  📋 Character model requirements:")
        print(f"    LLM: {llm_model}")
        print(f"    STT: {stt_model}")
        print(f"    TTS: {tts_model}")

        # Copy LLM model - simple file lookup
        llm_path = self._find_llm_model(llm_model)
        if llm_path:
            dst_file = models_dir / llm_path.name
            dst_file.write_bytes(llm_path.read_bytes())
            print(f"  ✓ Copied LLM model: {llm_path.name}")
        else:
            print(f"  ⚠️  LLM model not found: {llm_model}")

        # Copy STT model - simple file lookup
        stt_path = self._find_stt_model(stt_model)
        if stt_path:
            wav2vec2_bundle_dir = models_dir / "wav2vec2"
            wav2vec2_bundle_dir.mkdir(exist_ok=True)

            dst_file = wav2vec2_bundle_dir / stt_path.name
            dst_file.write_bytes(stt_path.read_bytes())
            print(f"  ✓ Copied STT model: {stt_path.name}")
        else:
            print(f"  ⚠️  STT model not found: {stt_model}")

        # TTS models are typically downloaded at runtime
        print(f"  ℹ️  TTS model {tts_model} will be downloaded at runtime")

    def _find_llm_model(self, model_name: str) -> Optional[Path]:
        """Find LLM model file by name."""
        llm_dir = self.base_path / "models" / "llm"
        if not llm_dir.exists():
            return None

        # Try common extensions and quantization suffixes
        for ext in [".gguf", ".safetensors", ".bin"]:
            # Try exact match first
            model_path = llm_dir / f"{model_name}{ext}"
            if model_path.exists():
                return model_path

            # Try with common quantization suffixes
            for suffix in ["-q4_k_m", "-q4_0", "-q8_0", "-fp16"]:
                model_path = llm_dir / f"{model_name}{suffix}{ext}"
                if model_path.exists():
                    return model_path

        return None

    def _find_stt_model(self, model_name: str) -> Optional[Path]:
        """Find STT model file by name."""
        wav2vec2_dir = self.base_path / "models" / "wav2vec2"
        if not wav2vec2_dir.exists():
            return None

        # Map model names to filenames
        stt_mapping = {
            "wav2vec2-base": "wav2vec2-base.pt",
            "wav2vec2-large": "wav2vec2-large.pt",
            "facebook/wav2vec2-base": "wav2vec2-base.pt",
            "facebook/wav2vec2-large": "wav2vec2-large.pt",
        }

        filename = stt_mapping.get(model_name, f"{model_name}.pt")
        model_path = wav2vec2_dir / filename

        if model_path.exists():
            return model_path

        return None

    def _create_deployment_config(
        self, character_id: str, franchise: str, bundle_dir: Path
    ) -> None:
        """Create deployment configuration for the character."""

        config = {
            "character": {
                "id": character_id,
                "franchise": franchise,
                "deployment": {
                    "environment": "production",
                    "scaling": {"min_replicas": 1, "max_replicas": 10},
                    "resources": {"cpu": "1000m", "memory": "2Gi"},
                },
            },
            "monitoring": {
                "enabled": True,
                "metrics": ["response_time", "error_rate", "throughput"],
                "alerts": {
                    "error_rate_threshold": 0.05,
                    "response_time_threshold": 2000,
                },
            },
            "security": {
                "content_filter": True,
                "age_appropriate": True,
                "rate_limiting": {"requests_per_minute": 60},
            },
        }

        config_file = bundle_dir / "deployment.yaml"
        config_file.write_text(json.dumps(config, indent=2))
        print("  ✓ Created deployment configuration")

    def _create_bundle_manifest(
        self, character_id: str, franchise: str, bundle_dir: Path
    ) -> None:
        """Create bundle manifest with metadata."""

        # Read character profile to get model names
        profile_file = (
            self.base_path / "configs" / "characters" / character_id / "profile.yaml"
        )
        model_names = {}

        if profile_file.exists():
            with open(profile_file, "r") as f:
                profile = yaml.safe_load(f)

            model_names = {
                "llm": profile.get("llm", {}).get("model", "phi-3-mini-4k-instruct"),
                "stt": profile.get("stt", {}).get("model", "wav2vec2-base"),
                "tts": profile.get("tts", {}).get("model", "coqui"),
            }

        manifest = {
            "bundle_version": "1.0.0",
            "character_id": character_id,
            "franchise": franchise,
            "created_at": datetime.now().isoformat(),
            "character_models": model_names,
            "assets": {
                "profile": "profile.yaml",
                "prompts": "prompts.yaml",
                "voices": "voices/",
                "voice_samples": "voice_samples/",
                "models": "models/",
                "deployment": "deployment.yaml",
            },
            "requirements": {
                "python_version": ">=3.10",
                "dependencies": [
                    "character-ai>=1.0.0",
                    "torch>=2.3.0",
                    "transformers>=4.52.0",
                ],
            },
        }

        manifest_file = bundle_dir / "manifest.json"
        manifest_file.write_text(json.dumps(manifest, indent=2))
        print("  ✓ Created bundle manifest")

    def extract_bundle(self, bundle_path: str, extract_to: str | None = None) -> str:
        """
        Extract a character bundle.

        Args:
            bundle_path: Path to bundle file
            extract_to: Directory to extract to (optional)

        Returns:
            Path to extracted directory
        """

        if not extract_to:
            extract_to = f"extracted_{Path(bundle_path).stem}"

        extract_dir = Path(extract_to)
        extract_dir.mkdir(exist_ok=True)

        with tarfile.open(bundle_path, "r:gz") as tar:
            safe_members = []
            for member in tar.getmembers():
                # Skip members with absolute paths or parent directory traversal
                if member.name.startswith("/") or ".." in member.name:
                    continue
                # Keep the relative directory structure but ensure it's safe
                # Remove any leading slashes and normalize the path
                safe_name = member.name.lstrip("/")
                if safe_name and not safe_name.startswith(".."):
                    member.name = safe_name
                    safe_members.append(member)

            tar.extractall(extract_dir, members=safe_members)  # nosec B202 - members are filtered for security

        print(f"Bundle extracted to: {extract_dir}")
        return str(extract_dir)

    def validate_bundle(self, bundle_path: str) -> bool:
        """
        Validate a character bundle.

        Args:
            bundle_path: Path to bundle file

        Returns:
            True if bundle is valid
        """

        try:
            with tarfile.open(bundle_path, "r:gz") as tar:
                # Check for required files
                required_files = ["profile.yaml", "prompts.yaml", "manifest.json"]

                file_names = tar.getnames()

                # Check if files are at root level
                root_files_present = all(f in file_names for f in required_files)
                if root_files_present:
                    print(f"Bundle validation passed: {bundle_path}")
                    return True

                # If not at root, look for character directory
                # Look for any directory that contains the required files
                character_dirs = [
                    name
                    for name in file_names
                    if name.endswith("/") and name.count("/") == 1
                ]

                # If no single-level directories, look for any subdirectory
                if not character_dirs:
                    # Find any directory that might contain the files
                    potential_dirs = set()
                    for file_name in file_names:
                        if "/" in file_name:
                            dir_name = file_name.split("/")[0] + "/"
                            potential_dirs.add(dir_name)
                    character_dirs = list(potential_dirs)

                if not character_dirs:
                    print("No character directory found in bundle")
                    return False

                # Check each potential directory
                for character_dir in character_dirs:
                    character_dir = character_dir.rstrip("/")
                    all_files_present = True

                    for required_file in required_files:
                        # Check in character directory
                        if f"{character_dir}/{required_file}" not in file_names:
                            all_files_present = False
                            break

                    if all_files_present:
                        print(f"✅ Bundle validation passed: {bundle_path}")
                        return True

                print("Missing required files in any character directory")
                return False

        except Exception as e:
            print(f"Bundle validation failed: {e}")
            return False
