"""
Open model management for the Character AI.

Handles downloading, managing, and recommending open-source LLMs.
"""

import asyncio
import json
import logging
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional

import requests

logger = logging.getLogger(__name__)


@dataclass
class ModelInfo:
    """Information about an open model."""

    name: str
    size: str
    description: str
    url: str
    checksum: Optional[str] = None
    format: str = "gguf"  # gguf, safetensors, etc.
    quantization: str = "q4_k_m"
    recommended_for: Optional[List[str]] = None  # ["character_creation", "runtime"]

    def __post_init__(self) -> None:
        if self.recommended_for is None:
            self.recommended_for = ["runtime"]


@dataclass
class ModelDownload:
    """Model download progress and status."""

    model_name: str
    status: str  # "downloading", "completed", "failed"
    progress: float = 0.0
    downloaded_bytes: int = 0
    total_bytes: int = 0
    error: Optional[str] = None


class OpenModelManager:
    """Manages open-source LLM models."""

    def __init__(self, models_dir: str = "models/llm"):
        self.models_dir = Path(models_dir)

        # Known open models
        self.available_models = {
            "llama-3.2-1b-instruct": ModelInfo(
                name="llama-3.2-1b-instruct",
                size="1.1GB",
                description="Fast, efficient model for runtime conversations",
                url="https://huggingface.co/meta-llama/Llama-3.2-1B-Instruct-GGUF/resolve/main/llama-3.2-1b-instruct-q4_k_m.gguf",
                format="gguf",
                quantization="q4_k_m",
                recommended_for=["runtime"],
            ),
            "llama-3.2-3b-instruct": ModelInfo(
                name="llama-3.2-3b-instruct",
                size="2.1GB",
                description="Balanced model for character creation and runtime",
                url="https://huggingface.co/meta-llama/Llama-3.2-3B-Instruct-GGUF/resolve/main/llama-3.2-3b-instruct-q4_k_m.gguf",
                format="gguf",
                quantization="q4_k_m",
                recommended_for=["character_creation", "runtime"],
            ),
            "tinyllama-1.1b": ModelInfo(
                name="tinyllama-1.1b",
                size="637MB",
                description="Ultra-lightweight model for edge devices",
                url="https://huggingface.co/TheBloke/TinyLlama-1.1B-Chat-v1.0-GGUF/resolve/main/tinyllama-1.1b-chat-v1.0.Q4_K_M.gguf",
                format="gguf",
                quantization="q4_k_m",
                recommended_for=["runtime"],
            ),
            "phi-3-mini": ModelInfo(
                name="phi-3-mini",
                size="2.3GB",
                description="Microsoft's efficient model for conversations",
                url="https://huggingface.co/microsoft/Phi-3-mini-4k-instruct-gguf/resolve/main/Phi-3-mini-4k-instruct-q4.gguf",
                format="gguf",
                quantization="q4",
                recommended_for=["character_creation", "runtime"],
            ),
            "llama-3.2-8b-instruct": ModelInfo(
                name="llama-3.2-8b-instruct",
                size="4.7GB",
                description="Powerful model for character creation and complex tasks",
                url="https://huggingface.co/meta-llama/Llama-3.2-8B-Instruct-GGUF/resolve/main/llama-3.2-8b-instruct-q4_k_m.gguf",
                format="gguf",
                quantization="q4_k_m",
                recommended_for=["character_creation"],
            ),
            "mistral-7b-instruct": ModelInfo(
                name="mistral-7b-instruct",
                size="4.1GB",
                description="High-performance 7B model for development and character creation",
                url="https://huggingface.co/TheBloke/Mistral-7B-Instruct-v0.1-GGUF/resolve/main/mistral-7b-instruct-v0.1.Q4_K_M.gguf",
                format="gguf",
                quantization="q4_k_m",
                recommended_for=["character_creation", "development"],
            ),
        }

        self.downloads: Dict[str, ModelDownload] = {}
        self.executor = ThreadPoolExecutor(max_workers=2)

    def list_available_models(self) -> List[ModelInfo]:
        """List all available open models."""
        return list(self.available_models.values())

    def list_installed_models(self) -> List[str]:
        """List installed models."""
        # Create directory if it doesn't exist
        if not self.models_dir.exists():
            self.models_dir.mkdir(parents=True, exist_ok=True)

        installed = []
        for model_file in self.models_dir.glob("*.gguf"):
            # Extract base model name from filename
            # e.g., "tinyllama-1.1b-q4_k_m.gguf" -> "tinyllama-1.1b"
            base_name = model_file.stem
            # Remove quantization suffixes
            for suffix in ["-q4_k_m", "-q4_0", "-q8_0", "-fp16"]:
                if base_name.endswith(suffix):
                    base_name = base_name[: -len(suffix)]
                    break
            installed.append(base_name)
        return installed

    def get_model_path(self, model_name: str) -> Optional[Path]:
        """Get path to installed model."""
        # Create directory if it doesn't exist
        if not self.models_dir.exists():
            self.models_dir.mkdir(parents=True, exist_ok=True)

        # First try exact match
        for ext in [".gguf", ".safetensors", ".bin"]:
            model_path = self.models_dir / f"{model_name}{ext}"
            if model_path.exists():
                return model_path

        # Then try with common quantization suffixes
        for ext in [".gguf", ".safetensors", ".bin"]:
            for suffix in ["-q4_k_m", "-q4_0", "-q8_0", "-fp16"]:
                model_path = self.models_dir / f"{model_name}{suffix}{ext}"
                if model_path.exists():
                    return model_path

        return None

    def is_model_installed(self, model_name: str) -> bool:
        """Check if model is installed."""
        return self.get_model_path(model_name) is not None

    def has_any_models(self) -> bool:
        """Check if any models are installed."""
        return len(self.list_installed_models()) > 0

    def get_model_info(self, model_name: str) -> Optional[ModelInfo]:
        """Get information about a model."""
        return self.available_models.get(model_name)

    def get_recommended_models(self, use_case: str) -> List[ModelInfo]:
        """Get models recommended for specific use case."""
        recommended = []
        for model in self.available_models.values():
            if model.recommended_for and use_case in model.recommended_for:
                recommended.append(model)
        return recommended

    async def download_model(
        self,
        model_name: str,
        progress_callback: Optional[Callable[[float], None]] = None,
    ) -> bool:
        """Download a model asynchronously."""
        if model_name not in self.available_models:
            logger.error(f"Unknown model: {model_name}")
            return False

        if self.is_model_installed(model_name):
            logger.info(f"Model {model_name} already installed")
            return True

        # Create directory if it doesn't exist
        if not self.models_dir.exists():
            self.models_dir.mkdir(parents=True, exist_ok=True)

        model_info = self.available_models[model_name]
        download = ModelDownload(
            model_name=model_name, status="downloading", progress=0.0
        )
        self.downloads[model_name] = download

        try:
            # Start download in thread pool
            loop = asyncio.get_event_loop()
            success = await loop.run_in_executor(
                self.executor,
                self._download_file,
                model_info.url,
                self.models_dir / f"{model_name}.gguf",
                progress_callback,
            )

            if success:
                download.status = "completed"
                download.progress = 100.0
                logger.info(f"Successfully downloaded {model_name}")
                return True
            else:
                download.status = "failed"
                download.error = "Download failed"
                logger.error(f"Failed to download {model_name}")
                return False

        except Exception as e:
            download.status = "failed"
            download.error = str(e)
            logger.error(f"Error downloading {model_name}: {e}")
            return False

    def _download_file(
        self,
        url: str,
        file_path: Path,
        progress_callback: Optional[Callable[[float], None]] = None,
    ) -> bool:
        """Download file with progress tracking."""
        try:
            response = requests.get(url, stream=True, timeout=30)
            response.raise_for_status()

            total_size = int(response.headers.get("content-length", 0))
            downloaded = 0

            with open(file_path, "wb") as f:
                for chunk in response.iter_content(chunk_size=8192):
                    if chunk:
                        f.write(chunk)
                        downloaded += len(chunk)

                        if progress_callback:
                            progress = (
                                (downloaded / total_size * 100) if total_size > 0 else 0
                            )
                            progress_callback(progress)

            return True

        except Exception as e:
            logger.error(f"Download error: {e}")
            # Clean up partial file
            if file_path.exists():
                file_path.unlink()
            return False

    def get_download_status(self, model_name: str) -> Optional[ModelDownload]:
        """Get download status for a model."""
        return self.downloads.get(model_name)

    def cancel_download(self, model_name: str) -> bool:
        """Cancel a model download."""
        if model_name in self.downloads:
            download = self.downloads[model_name]
            if download.status == "downloading":
                download.status = "cancelled"
                return True
        return False

    def remove_model(self, model_name: str) -> bool:
        """Remove an installed model."""
        model_path = self.get_model_path(model_name)
        if model_path and model_path.exists():
            try:
                model_path.unlink()
                logger.info(f"Removed model {model_name}")
                return True
            except Exception as e:
                logger.error(f"Error removing model {model_name}: {e}")
                return False
        return False

    def get_model_size(self, model_name: str) -> Optional[int]:
        """Get size of installed model."""
        model_path = self.get_model_path(model_name)
        if model_path and model_path.exists():
            return model_path.stat().st_size
        return None

    def verify_model(self, model_name: str) -> bool:
        """Verify model integrity."""
        model_path = self.get_model_path(model_name)
        if not model_path or not model_path.exists():
            return False

        # Basic file size check
        if model_path.stat().st_size < 1024 * 1024:  # Less than 1MB is suspicious
            logger.warning(f"Model {model_name} seems too small")
            return False

        return True

    def get_storage_usage(self) -> Dict[str, Any]:
        """Get storage usage information."""
        total_size = 0
        model_count = 0

        for model_file in self.models_dir.glob("*.gguf"):
            total_size += model_file.stat().st_size
            model_count += 1

        return {
            "total_size_bytes": total_size,
            "total_size_mb": total_size / (1024 * 1024),
            "total_size_gb": total_size / (1024 * 1024 * 1024),
            "model_count": model_count,
            "models_dir": str(self.models_dir),
        }

    def cleanup_orphaned_files(self) -> List[str]:
        """Clean up orphaned model files."""
        cleaned = []

        for model_file in self.models_dir.glob("*.gguf"):
            model_name = model_file.stem
            if model_name not in self.available_models:
                try:
                    model_file.unlink()
                    cleaned.append(model_name)
                    logger.info(f"Cleaned up orphaned model: {model_name}")
                except Exception as e:
                    logger.error(f"Error cleaning up {model_name}: {e}")

        return cleaned

    def get_model_recommendations(
        self, use_case: str, device_type: str = "cpu"
    ) -> List[ModelInfo]:
        """Get model recommendations based on use case and device."""
        recommendations = []

        # Filter by use case
        for model in self.available_models.values():
            if model.recommended_for and use_case in model.recommended_for:
                recommendations.append(model)

        # Sort by size (smaller first for edge devices)
        if device_type == "edge":
            recommendations.sort(key=lambda x: x.size)
        else:
            # Sort by capability (larger models first for server)
            recommendations.sort(key=lambda x: x.size, reverse=True)

        return recommendations

    def export_model_list(self, file_path: str) -> None:
        """Export list of available models to JSON."""
        models_data = {}
        for name, info in self.available_models.items():
            models_data[name] = {
                "name": info.name,
                "size": info.size,
                "description": info.description,
                "url": info.url,
                "format": info.format,
                "quantization": info.quantization,
                "recommended_for": info.recommended_for,
            }

        with open(file_path, "w") as f:
            json.dump(models_data, f, indent=2)

        logger.info(f"Model list exported to {file_path}")

    def import_model_list(self, file_path: str) -> None:
        """Import model list from JSON."""
        if not Path(file_path).exists():
            logger.warning(f"Model list file not found: {file_path}")
            return

        with open(file_path, "r") as f:
            models_data = json.load(f)

        # Update available models
        for name, data in models_data.items():
            self.available_models[name] = ModelInfo(
                name=data["name"],
                size=data["size"],
                description=data["description"],
                url=data["url"],
                format=data.get("format", "gguf"),
                quantization=data.get("quantization", "q4_k_m"),
                recommended_for=data.get("recommended_for", ["runtime"]),
            )

        logger.info(f"Model list imported from {file_path}")

    def __del__(self) -> None:
        """Cleanup thread pool."""
        if hasattr(self, "executor"):
            self.executor.shutdown(wait=False)
