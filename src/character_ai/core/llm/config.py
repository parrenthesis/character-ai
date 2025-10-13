"""
LLM configuration system for the Character AI.

Supports both open (local) and token-based (cloud) LLM configurations.
"""

import logging
import os
from dataclasses import dataclass, field
from enum import Enum
from pathlib import Path
from typing import Any, Dict, List, Optional

import yaml

from .providers import LLMProvider

logger = logging.getLogger(__name__)


# LLMProvider is now imported from .providers


class LLMType(Enum):
    """LLM usage types."""

    CHARACTER_CREATION = "character_creation"
    RUNTIME = "runtime"


@dataclass
class LLMConfig:
    """Base LLM configuration."""

    provider: LLMProvider
    model: str
    temperature: float = 0.7
    max_tokens: int = 150
    timeout: int = 30
    retries: int = 3
    enabled: bool = True

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for serialization."""
        return {
            "provider": self.provider.value,
            "model": self.model,
            "temperature": self.temperature,
            "max_tokens": self.max_tokens,
            "timeout": self.timeout,
            "retries": self.retries,
            "enabled": self.enabled,
        }


@dataclass
class CharacterCreationConfig(LLMConfig):
    """Configuration for character creation LLM (powerful, creative)."""

    provider: LLMProvider = LLMProvider.LOCAL
    model: str = "llama-3.2-3b-instruct"  # Default open model
    temperature: float = 0.8  # More creative
    max_tokens: int = 2000  # Longer responses for detailed creation
    timeout: int = 60  # Longer timeout for complex generation

    def __post_init__(self) -> None:
        """Set default values for character creation."""
        if self.provider == LLMProvider.LOCAL:
            self.model = "llama-3.2-3b-instruct"
        elif self.provider == LLMProvider.OLLAMA:
            self.model = "llama3.2:3b"
        elif self.provider == LLMProvider.OPENAI:
            self.model = "gpt-4o-mini"
        elif self.provider == LLMProvider.ANTHROPIC:
            self.model = "claude-3-haiku-20240307"


@dataclass
class RuntimeConfig(LLMConfig):
    """Configuration for runtime LLM (fast, efficient)."""

    provider: LLMProvider = LLMProvider.LOCAL
    model: str = "llama-3.2-1b-instruct"  # Smaller, faster model
    temperature: float = 0.7  # Balanced creativity
    max_tokens: int = 150  # Shorter responses for real-time
    timeout: int = 30  # Faster timeout for real-time

    def __post_init__(self) -> None:
        """Set default values for runtime."""
        if self.provider == LLMProvider.LOCAL:
            # Prefer a small local edge model by default
            self.model = "TinyLlama/TinyLlama-1.1B-Chat-v1.0"
        elif self.provider == LLMProvider.OLLAMA:
            self.model = "llama3.2:1b"
        elif self.provider == LLMProvider.OPENAI:
            self.model = "gpt-3.5-turbo"
        elif self.provider == LLMProvider.ANTHROPIC:
            self.model = "claude-3-haiku-20240307"


@dataclass
class LLMProviderConfig:
    """Configuration for specific LLM providers."""

    # Local LLM config
    local_model_path: str = "models/llm"
    local_device: str = "cpu"
    local_quantization: str = "q4_k_m"

    # Ollama config
    ollama_base_url: str = "http://localhost:11434"
    ollama_models: List[str] = field(
        default_factory=lambda: ["llama3.2:1b", "llama3.2:3b"]
    )

    # OpenAI config
    openai_api_key: Optional[str] = None
    openai_models: List[str] = field(
        default_factory=lambda: ["gpt-3.5-turbo", "gpt-4o-mini"]
    )

    # Anthropic config
    anthropic_api_key: Optional[str] = None
    anthropic_models: List[str] = field(
        default_factory=lambda: ["claude-3-haiku-20240307"]
    )

    def __post_init__(self) -> None:
        """Load API keys from environment variables."""
        if not self.openai_api_key:
            self.openai_api_key = os.environ.get("OPENAI_API_KEY")
        if not self.anthropic_api_key:
            self.anthropic_api_key = os.environ.get("ANTHROPIC_API_KEY")


@dataclass
class LLMConfigManager:
    """Manages LLM configurations for the platform."""

    character_creation: CharacterCreationConfig = field(
        default_factory=CharacterCreationConfig
    )
    runtime: RuntimeConfig = field(default_factory=RuntimeConfig)
    providers: LLMProviderConfig = field(default_factory=LLMProviderConfig)

    # Cost tracking
    cost_tracking_enabled: bool = True
    max_monthly_cost: float = 50.0  # USD

    # Fallback configuration
    fallback_enabled: bool = True
    fallback_provider: LLMProvider = LLMProvider.LOCAL

    def __post_init__(self) -> None:
        """Initialize configurations."""
        # Load from canonical runtime config first (defaults)
        self._load_from_runtime_yaml()

        # Then load from environment variables (overrides)
        self._load_from_env()

        # Validate configurations
        self._validate_configs()

    def _load_from_env(self) -> None:
        """Load configuration from environment variables."""
        # Character creation config
        if os.environ.get("CAI_CHARACTER_CREATION_PROVIDER"):
            self.character_creation.provider = LLMProvider(
                os.environ.get("CAI_CHARACTER_CREATION_PROVIDER", "local")
            )
        if os.environ.get("CAI_CHARACTER_CREATION_MODEL"):
            model = os.environ.get("CAI_CHARACTER_CREATION_MODEL")
            if model:
                self.character_creation.model = model

        # Runtime config
        if os.environ.get("CAI_RUNTIME_PROVIDER"):
            self.runtime.provider = LLMProvider(
                os.environ.get("CAI_RUNTIME_PROVIDER", "local")
            )
        if os.environ.get("CAI_RUNTIME_MODEL"):
            model = os.environ.get("CAI_RUNTIME_MODEL")
            if model:
                self.runtime.model = model

        # Provider configs
        if os.environ.get("CAI_LOCAL_MODEL_PATH"):
            path = os.environ.get("CAI_LOCAL_MODEL_PATH")
            if path:
                self.providers.local_model_path = path
        if os.environ.get("CAI_OLLAMA_BASE_URL"):
            url = os.environ.get("CAI_OLLAMA_BASE_URL")
            if url:
                self.providers.ollama_base_url = url

    def _load_from_runtime_yaml(self) -> None:
        """Load LLM provider/model from configs/runtime.yaml if present (defaults).

        Expected schema (minimal):
        llm:
          runtime:
            provider: local|ollama|openai|anthropic
            model: <model-id>
          character_creation:
            provider: local|ollama|openai|anthropic
            model: <model-id>
          providers:
            ollama_base_url: http://localhost:11434
            local_model_path: models/llm
        """
        try:
            runtime_path = Path("configs/runtime.yaml")
            if not runtime_path.exists():
                return
            with open(runtime_path, "r") as f:
                data = yaml.safe_load(f) or {}

            llm_section = data.get("llm", {}) or {}

            # Runtime defaults
            rt = llm_section.get("runtime", {}) or {}
            prov = rt.get("provider")
            model = rt.get("model")
            if prov:
                try:
                    self.runtime.provider = LLMProvider(str(prov))
                except Exception:
                    logger.warning(f"Unknown runtime provider in runtime.yaml: {prov}")
            if model:
                self.runtime.model = str(model)

            # Character creation defaults
            cc = llm_section.get("character_creation", {}) or {}
            cprov = cc.get("provider")
            cmodel = cc.get("model")
            if cprov:
                try:
                    self.character_creation.provider = LLMProvider(str(cprov))
                except Exception:
                    logger.warning(f"Unknown character_creation provider: {cprov}")
            if cmodel:
                self.character_creation.model = str(cmodel)

            # Provider-level settings
            provs = llm_section.get("providers", {}) or {}
            if provs.get("ollama_base_url"):
                self.providers.ollama_base_url = str(provs["ollama_base_url"])
            if provs.get("local_model_path"):
                self.providers.local_model_path = str(provs["local_model_path"])

            logger.info("Loaded LLM defaults from configs/runtime.yaml")
        except Exception as e:
            logger.warning(f"Failed to load configs/runtime.yaml for LLM: {e}")

    def _validate_configs(self) -> None:
        """Validate LLM configurations."""
        # Check if required API keys are present for token-based providers
        if self.character_creation.provider in [
            LLMProvider.OPENAI,
            LLMProvider.ANTHROPIC,
        ]:
            if not self._get_api_key(self.character_creation.provider):
                logger.warning(
                    f"API key not found for {self.character_creation.provider.value}"
                )
                # Fallback to local
                self.character_creation.provider = LLMProvider.LOCAL

        if self.runtime.provider in [LLMProvider.OPENAI, LLMProvider.ANTHROPIC]:
            if not self._get_api_key(self.runtime.provider):
                logger.warning(f"API key not found for {self.runtime.provider.value}")
                # Fallback to local
                self.runtime.provider = LLMProvider.LOCAL

    def _get_api_key(self, provider: LLMProvider) -> Optional[str]:
        """Get API key for provider."""
        if provider == LLMProvider.OPENAI:
            return self.providers.openai_api_key
        elif provider == LLMProvider.ANTHROPIC:
            return self.providers.anthropic_api_key
        return None

    def get_config(self, llm_type: LLMType) -> LLMConfig:
        """Get configuration for specific LLM type."""
        if llm_type == LLMType.CHARACTER_CREATION:
            return self.character_creation
        elif llm_type == LLMType.RUNTIME:
            return self.runtime
        else:
            raise ValueError(f"Unknown LLM type: {llm_type}")

    def get_provider_config(self, provider: LLMProvider) -> Dict[str, Any]:
        """Get provider-specific configuration."""
        if provider == LLMProvider.LOCAL:
            return {
                "model_path": self.providers.local_model_path,
                "device": self.providers.local_device,
                "quantization": self.providers.local_quantization,
            }
        elif provider == LLMProvider.OLLAMA:
            return {
                "base_url": self.providers.ollama_base_url,
                "model": (
                    self.character_creation.model
                    if provider == LLMProvider.OLLAMA
                    else self.runtime.model
                ),
            }
        elif provider == LLMProvider.OPENAI:
            return {
                "api_key": self.providers.openai_api_key,
                "model": (
                    self.character_creation.model
                    if provider == LLMProvider.OPENAI
                    else self.runtime.model
                ),
            }
        elif provider == LLMProvider.ANTHROPIC:
            return {
                "api_key": self.providers.anthropic_api_key,
                "model": (
                    self.character_creation.model
                    if provider == LLMProvider.ANTHROPIC
                    else self.runtime.model
                ),
            }
        else:
            raise ValueError(f"Unknown provider: {provider}")

    def save_config(self, file_path: str) -> None:
        """Save configuration to YAML file."""
        config_data = {
            "character_creation": self.character_creation.to_dict(),
            "runtime": self.runtime.to_dict(),
            "providers": {
                "local_model_path": self.providers.local_model_path,
                "local_device": self.providers.local_device,
                "local_quantization": self.providers.local_quantization,
                "ollama_base_url": self.providers.ollama_base_url,
                "ollama_models": self.providers.ollama_models,
                "openai_models": self.providers.openai_models,
                "anthropic_models": self.providers.anthropic_models,
            },
            "cost_tracking": {
                "enabled": self.cost_tracking_enabled,
                "max_monthly_cost": self.max_monthly_cost,
            },
            "fallback": {
                "enabled": self.fallback_enabled,
                "provider": self.fallback_provider.value,
            },
        }

        with open(file_path, "w") as f:
            yaml.dump(config_data, f, default_flow_style=False)

        logger.info(f"LLM configuration saved to {file_path}")

    def load_config(self, file_path: str) -> None:
        """Load configuration from YAML file."""
        if not Path(file_path).exists():
            logger.warning(f"Configuration file not found: {file_path}")
            return

        with open(file_path, "r") as f:
            config_data = yaml.safe_load(f)

        # Load character creation config
        if "character_creation" in config_data:
            cc_data = config_data["character_creation"]
            self.character_creation = CharacterCreationConfig(
                provider=LLMProvider(cc_data.get("provider", "local")),
                model=cc_data.get("model", "llama-3.2-3b-instruct"),
                temperature=cc_data.get("temperature", 0.8),
                max_tokens=cc_data.get("max_tokens", 2000),
                timeout=cc_data.get("timeout", 60),
            )

        # Load runtime config
        if "runtime" in config_data:
            rt_data = config_data["runtime"]
            self.runtime = RuntimeConfig(
                provider=LLMProvider(rt_data.get("provider", "local")),
                model=rt_data.get("model", "llama-3.2-1b-instruct"),
                temperature=rt_data.get("temperature", 0.7),
                max_tokens=rt_data.get("max_tokens", 150),
                timeout=rt_data.get("timeout", 30),
            )

        logger.info(f"LLM configuration loaded from {file_path}")

    def get_cost_estimate(self, provider: LLMProvider, tokens: int) -> float:
        """Get cost estimate for token usage."""
        if provider == LLMProvider.LOCAL or provider == LLMProvider.OLLAMA:
            return 0.0  # Free

        # Approximate costs per 1K tokens
        costs = {
            LLMProvider.OPENAI: 0.15,  # GPT-3.5-turbo
            LLMProvider.ANTHROPIC: 0.10,  # Claude-3-haiku
        }

        return (tokens / 1000) * costs.get(provider, 0.0)

    def should_use_fallback(self, provider: LLMProvider) -> bool:
        """Check if fallback should be used."""
        if not self.fallback_enabled:
            return False

        # Check if API key is missing for token-based providers
        if provider in [LLMProvider.OPENAI, LLMProvider.ANTHROPIC]:
            return not self._get_api_key(provider)

        return False
