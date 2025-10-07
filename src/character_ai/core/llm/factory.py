"""
LLM Factory for creating and managing LLM instances.

Implements the dual LLM system with separate instances for character creation and runtim
e.
"""

import logging
from typing import Any, Dict, List, Optional, Type

from .config import LLMConfigManager, LLMType
from .manager import OpenModelManager
from .providers import (
    AnthropicProvider,
    LLMInterface,
    LLMProvider,
    LocalLLMProvider,
    OllamaProvider,
    OpenAIProvider,
)

logger = logging.getLogger(__name__)


class LLMFactory:
    """Factory for creating LLM instances."""

    def __init__(
        self, config_manager: LLMConfigManager, model_manager: OpenModelManager
    ):
        self.config_manager = config_manager
        self.model_manager = model_manager
        self._instances: Dict[str, LLMInterface] = {}
        self._provider_classes: Dict[LLMProvider, Type[LLMInterface]] = {
            LLMProvider.LOCAL: LocalLLMProvider,
            LLMProvider.OLLAMA: OllamaProvider,
            LLMProvider.OPENAI: OpenAIProvider,
            LLMProvider.ANTHROPIC: AnthropicProvider,
        }

    def get_llm(self, llm_type: LLMType) -> LLMInterface:
        """Get LLM instance for specific type (creation or runtime)."""
        cache_key = f"{llm_type.value}"

        if cache_key not in self._instances:
            self._instances[cache_key] = self._create_llm(llm_type)

        return self._instances[cache_key]

    def _create_llm(self, llm_type: LLMType) -> LLMInterface:
        """Create LLM instance based on configuration."""
        config = self.config_manager.get_config(llm_type)
        provider = self._map_config_provider(config.provider)

        # Check if we should use fallback
        if self.config_manager.should_use_fallback(provider):
            logger.warning(f"Using fallback provider for {llm_type.value}")
            provider = self.config_manager.fallback_provider

        # Get provider configuration
        provider_config = self.config_manager.get_provider_config(provider)

        # For local provider, get the actual model path from model manager
        if provider == LLMProvider.LOCAL:
            model_path = self._get_model_path(config.model)
            if model_path:
                provider_config["model_path"] = str(model_path)
                logger.info(f"Using model: {model_path}")
            else:
                logger.error(f"No model found for {config.model}")
                raise ValueError(f"No model found for {config.model}")

        # Add model-specific configuration
        provider_config.update(
            {
                "model": config.model,
                "temperature": config.temperature,
                "max_tokens": config.max_tokens,
                "timeout": config.timeout,
                "retries": config.retries,
            }
        )

        # Create provider instance
        provider_class = self._provider_classes[provider]
        instance = provider_class(provider_config)  # type: ignore

        logger.info(f"Created {provider.value} LLM for {llm_type.value}")
        return instance

    def _get_model_path(self, model_name: str) -> Optional[str]:
        """Get the actual model file path for local provider.

        Resolution order (path-first):
        1) Explicit GGUF path from runtime config (configs/runtime.yaml → Config.models.llama_gguf_path)
        2) Name-based lookup via OpenModelManager
        3) Fallback to any installed model
        """
        try:
            # 1) Prefer explicit GGUF path from runtime config if present
            try:
                from character_ai.core.config import Config as CoreConfig  # lazy import

                core_cfg = CoreConfig()
                gguf_path = getattr(core_cfg.models, "llama_gguf_path", None)
                if gguf_path:
                    from pathlib import Path as _P

                    p = _P(str(gguf_path))
                    if p.exists():
                        return str(p)
            except Exception:
                # Non-fatal: fall through to model manager lookup
                pass

            # 2) Name-based lookup via model manager
            model_path = self.model_manager.get_model_path(model_name)
            if model_path and model_path.exists():
                return str(model_path)
            else:
                logger.warning(
                    f"Model {model_name} not found, trying to find any available model"
                )
                # 3) Try to find any available model
                installed_models = self.model_manager.list_installed_models()
                if installed_models:
                    # Use the first available model
                    fallback_model = installed_models[0]
                    fallback_path = self.model_manager.get_model_path(fallback_model)
                    if fallback_path and fallback_path.exists():
                        logger.info(f"Using fallback model: {fallback_model}")
                        return str(fallback_path)
                return None
        except Exception as e:
            logger.error(f"Error getting model path for {model_name}: {e}")
            return None

    def _map_config_provider(self, config_provider: LLMProvider) -> LLMProvider:
        """Map configuration provider to factory provider."""
        # Since we're now using the same enum, just return it directly
        return config_provider

    def get_character_creation_llm(self) -> LLMInterface:
        """Get LLM for character creation (powerful, creative)."""
        return self.get_llm(LLMType.CHARACTER_CREATION)

    def get_runtime_llm(self) -> LLMInterface:
        """Get LLM for runtime conversations (fast, efficient)."""
        return self.get_llm(LLMType.RUNTIME)

    def refresh_llm(self, llm_type: LLMType) -> None:
        """Refresh LLM instance (useful after configuration changes)."""
        cache_key = f"{llm_type.value}"
        if cache_key in self._instances:
            del self._instances[cache_key]
        logger.info(f"Refreshed {llm_type.value} LLM")

    def refresh_all_llms(self) -> None:
        """Refresh all LLM instances."""
        self._instances.clear()
        logger.info("Refreshed all LLM instances")

    def get_llm_capabilities(self, llm_type: LLMType) -> Dict[str, Any]:
        """Get capabilities of LLM for specific type."""
        llm = self.get_llm(llm_type)
        return llm.get_capabilities()

    async def test_llm_connection(self, llm_type: LLMType) -> bool:
        """Test LLM connection and basic functionality."""
        try:
            llm = self.get_llm(llm_type)
            # Simple test prompt
            test_prompt = "Hello, how are you?"
            response = await llm.generate(test_prompt, max_tokens=10)
            return len(response) > 0
        except Exception as e:
            logger.error(f"LLM test failed for {llm_type.value}: {e}")
            return False

    def get_llm_status(self) -> Dict[str, Any]:
        """Get status of all LLM instances."""
        status = {}

        for llm_type in [LLMType.CHARACTER_CREATION, LLMType.RUNTIME]:
            try:
                llm = self.get_llm(llm_type)
                capabilities = llm.get_capabilities()
                connection_ok = self.test_llm_connection(llm_type)

                status[llm_type.value] = {
                    "provider": capabilities.get("provider", "unknown"),
                    "connection_ok": connection_ok,
                    "capabilities": capabilities,
                    "cost_per_token": capabilities.get("cost_per_token", 0.0),
                    "requires_internet": capabilities.get("requires_internet", False),
                }
            except Exception as e:
                status[llm_type.value] = {"error": str(e), "connection_ok": False}

        return status

    def get_cost_estimate(self, llm_type: LLMType, tokens: int) -> float:
        """Get cost estimate for token usage."""
        config = self.config_manager.get_config(llm_type)
        provider = self._map_config_provider(config.provider)
        return self.config_manager.get_cost_estimate(provider, tokens)

    def get_recommended_models(self, llm_type: LLMType) -> list:
        """Get recommended models for LLM type."""
        if llm_type == LLMType.CHARACTER_CREATION:
            return self.model_manager.get_recommended_models("character_creation")
        elif llm_type == LLMType.RUNTIME:
            return self.model_manager.get_recommended_models("runtime")
        return []  # type: ignore

    def switch_llm_provider(
        self,
        llm_type: LLMType,
        new_provider: LLMProvider,
        new_model: Optional[str] = None,
    ) -> None:
        """Switch LLM provider and model."""
        config = self.config_manager.get_config(llm_type)
        config.provider = new_provider

        if new_model:
            config.model = new_model

        # Refresh the LLM instance
        self.refresh_llm(llm_type)

        logger.info(
            f"Switched {llm_type.value} to {new_provider.value} with model {config.model}"
        )

    def get_available_providers(self) -> Dict[str, list]:
        """Get available providers and their models."""
        providers: Dict[str, List[str]] = {
            "local": [],
            "ollama": [],
            "openai": [],
            "anthropic": [],
        }

        # Local models (installed)
        installed_models = self.model_manager.list_installed_models()
        providers["local"] = installed_models

        # Ollama models (from config)
        providers["ollama"] = self.config_manager.providers.ollama_models

        # OpenAI models (from config)
        providers["openai"] = self.config_manager.providers.openai_models

        # Anthropic models (from config)
        providers["anthropic"] = self.config_manager.providers.anthropic_models

        return providers

    def validate_configuration(self) -> Dict[str, Any]:
        """Validate LLM configuration."""
        validation: Dict[str, Any] = {"valid": True, "errors": [], "warnings": []}

        # Check character creation LLM
        try:
            self.get_character_creation_llm()
            if not self.test_llm_connection(LLMType.CHARACTER_CREATION):
                validation["errors"].append("Character creation LLM connection failed")
                validation["valid"] = False
        except Exception as e:
            validation["errors"].append(f"Character creation LLM error: {e}")
            validation["valid"] = False

        # Check runtime LLM
        try:
            self.get_runtime_llm()
            if not self.test_llm_connection(LLMType.RUNTIME):
                validation["errors"].append("Runtime LLM connection failed")
                validation["valid"] = False
        except Exception as e:
            validation["errors"].append(f"Runtime LLM error: {e}")
            validation["valid"] = False

        # Check API keys for token-based providers
        cc_config = self.config_manager.get_config(LLMType.CHARACTER_CREATION)
        rt_config = self.config_manager.get_config(LLMType.RUNTIME)

        for llm_type, config in [
            (LLMType.CHARACTER_CREATION, cc_config),
            (LLMType.RUNTIME, rt_config),
        ]:
            if config.provider in [LLMProvider.OPENAI, LLMProvider.ANTHROPIC]:
                api_key = self.config_manager._get_api_key(config.provider)
                if not api_key:
                    validation["warnings"].append(
                        f"API key missing for {llm_type.value} provider {config.provider.value}"
                    )

        return validation
