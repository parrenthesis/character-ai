"""Engine lifecycle management for real-time interaction."""

import asyncio
import json
from pathlib import Path
from typing import Any, Optional

from ...algorithms.conversational_ai.memory.hybrid_memory import (
    HybridMemorySystem,
    MemoryConfig,
)
from ...algorithms.conversational_ai.memory.session_memory import SessionMemory
from ...algorithms.conversational_ai.utils.text_normalizer import TextNormalizer
from ...characters import (
    Character,
    CharacterResponseFilter,
    CharacterService,
    ChildSafetyFilter,
)
from ...characters.voices import SchemaVoiceService
from ...core.audio_io.voice_activity_detection import VADConfig
from ...core.caching import ResponseCache
from ...core.config import Config
from ...core.llm.template_prompt_builder import TemplatePromptBuilder
from ...core.resource_manager import ResourceManager
from ...hardware.toy_hardware_manager import ToyHardwareManager
from ...observability import get_logger
from ...services import HardwareProfileService
from ...services.edge_optimizer import EdgeModelOptimizer

logger = get_logger(__name__)


class EngineLifecycle:
    """Manages engine initialization, shutdown, and session state."""

    def __init__(
        self,
        hardware_manager: ToyHardwareManager,
        text_normalizer: Optional[Any] = None,
        prompt_builder: Optional[Any] = None,
        resource_manager: Optional[Any] = None,
        hardware_profile: Optional[str] = None,
    ):
        self.hardware_manager: Optional[ToyHardwareManager] = hardware_manager
        self.character_manager: Optional[CharacterService] = CharacterService()

        # Use HardwareProfileService for unified hardware detection
        hardware_service = HardwareProfileService()
        self.hardware_profile, self.hardware_config = hardware_service.load_or_detect(
            hardware_profile
        )

        # Create or inject services
        if text_normalizer is None:
            self.text_normalizer = TextNormalizer()
        else:
            self.text_normalizer = text_normalizer

        if prompt_builder is None:
            self.prompt_builder = TemplatePromptBuilder()
        else:
            self.prompt_builder = prompt_builder

        # Character-specific response filter (initialized per character)
        self.character_response_filter: Optional[CharacterResponseFilter] = None

        if resource_manager is None:
            # edge_optimizer will be set later, so pass None for now
            # Pass hardware config to ResourceManager for proper device selection
            self.resource_manager = ResourceManager(
                Config(), None, self.hardware_config
            )
        else:
            self.resource_manager = resource_manager

        # Phase 2: Apply CPU limiting from hardware config (AFTER Config() creation to override main config)
        self._apply_hardware_cpu_limits()

        # Initialize safety filter and voice manager directly
        self.safety_filter: Optional[ChildSafetyFilter] = ChildSafetyFilter()
        self.voice_manager: Optional[SchemaVoiceService] = SchemaVoiceService()
        self.edge_optimizer: Optional[EdgeModelOptimizer] = (
            EdgeModelOptimizer(hardware_manager.constraints)
            if hardware_manager
            else None
        )

        # Update ResourceManager with edge_optimizer if it was created
        if self.resource_manager and self.edge_optimizer:
            self.resource_manager.edge_optimizer = self.edge_optimizer

        # Session memory management
        self.session_memory = SessionMemory()

        # Response cache
        self.response_cache = ResponseCache()

        # Load device ID for user identification
        self.device_id = self._load_device_id()

        # Initialize hybrid memory system
        self.hybrid_memory: Optional[HybridMemorySystem] = None

        # Active character
        self.active_character: Optional[Character] = None

        # VAD configuration
        self.vad_config = VADConfig()

        # Real-time session state
        self.is_realtime_session_active = False
        self.realtime_session_task: Optional[asyncio.Task] = None

    def _load_device_id(self) -> str:
        """Load device ID from configs/device_id.json."""
        try:
            device_id_path = Path("configs/device_id.json")
            if device_id_path.exists():
                with open(device_id_path, "r") as f:
                    data = json.load(f)
                device_id = data.get("device_id", "default_device")
                logger.info(f"Loaded device ID: {device_id}")
                return str(device_id)
            else:
                logger.warning("Device ID file not found, using default")
                return "default_device"
        except Exception as e:
            logger.error(f"Failed to load device ID: {e}")
            return "default_device"

    def _apply_hardware_cpu_limits(self) -> None:
        """Apply CPU thread limits from hardware config."""
        import os

        # Check if user explicitly set CPU limiting via env var (testing override)
        if os.environ.get("CAI_ENABLE_CPU_LIMITING", "false").lower() == "true":
            max_threads = os.environ.get("CAI_MAX_CPU_THREADS")
            if max_threads:
                logger.info(
                    f"⚠️  Using env var override: {max_threads} threads (ignoring hardware config)"
                )
                return  # Don't apply hardware config, respect user's test override

        # Normal operation: use hardware config
        if not self.hardware_config:
            return

        # Get n_threads from hardware config
        llm_config = self.hardware_config.get("optimizations", {}).get("llm", {})
        n_threads = llm_config.get("n_threads")

        # Debug logging to see what's being loaded
        logger.debug(
            f"Hardware config optimizations: {self.hardware_config.get('optimizations', {})}"
        )
        logger.debug(f"LLM config: {llm_config}")
        logger.debug(f"n_threads from config: {n_threads}")

        if n_threads is None:
            logger.debug("No n_threads in hardware config, skipping CPU limiting")
            return

        # Set environment variables for all threading libraries
        os.environ["OPENBLAS_NUM_THREADS"] = str(n_threads)
        os.environ["MKL_NUM_THREADS"] = str(n_threads)
        os.environ["OMP_NUM_THREADS"] = str(n_threads)
        os.environ["NUMEXPR_NUM_THREADS"] = str(n_threads)
        os.environ["VECLIB_MAXIMUM_THREADS"] = str(n_threads)

        # Set PyTorch threads
        try:
            import torch

            torch.set_num_threads(n_threads)

            # Also set PyTorch threading at the module level
            try:
                torch.set_num_interop_threads(n_threads)
            except RuntimeError:
                # Interop threads already set in torch_init.py - this is expected
                pass

            logger.info(
                f"✅ CPU limiting applied: {n_threads} threads (from {self.hardware_profile} profile)"
            )
            # Console echo for test visibility
            try:
                if os.getenv("CAI_ENVIRONMENT", "").lower() == "testing":
                    import click

                    click.echo(
                        f"✅ CPU limiting applied: {n_threads} threads (from {self.hardware_profile} profile)"
                    )
            except Exception:
                pass
        except ImportError:
            logger.debug("PyTorch not available, skipping torch thread limiting")

    async def initialize(self) -> None:
        """Initialize the real-time interaction engine."""
        try:
            logger.info("Initializing real-time interaction engine...")

            # Initialize character manager
            if self.character_manager:
                await self.character_manager.initialize()

            # Initialize safety filter
            if self.safety_filter:
                await self.safety_filter.initialize()

            # Initialize voice manager
            if self.voice_manager:
                await self.voice_manager.initialize()

            # Initialize hybrid memory system
            await self._initialize_hybrid_memory()

            logger.info("Real-time interaction engine initialized successfully")
        except Exception as e:
            logger.error(f"Failed to initialize real-time engine: {e}")
            raise

    async def _initialize_hybrid_memory(self) -> None:
        """Initialize hybrid memory system."""
        try:
            # Create memory config from runtime config
            memory_config = self._create_memory_config()

            # Get LLM provider for summarization
            llm_provider = None
            if self.resource_manager:
                llm_processor = self.resource_manager.get_llm_processor()
                if llm_processor:
                    # Create a simple LLM provider wrapper
                    class LLMProviderWrapper:
                        def __init__(self, processor: Any) -> None:
                            self.processor = processor

                        def generate_response(
                            self, prompt: str, max_tokens: int = 150
                        ) -> str:
                            """Generate response using the LLM processor."""
                            if not self.processor:
                                logger.debug(
                                    "No LLM processor available for summarization"
                                )
                                return "Conversation summary placeholder"

                            try:
                                import asyncio

                                # Simple approach: run in new event loop
                                try:
                                    result = asyncio.run(
                                        self.processor.process_text(prompt)
                                    )
                                    return (
                                        result.text
                                        if result and result.text
                                        else "Conversation summary"
                                    )
                                except Exception as e:
                                    logger.debug(f"LLM summarization failed: {e}")
                                    return "Conversation summary placeholder"
                            except Exception as e:
                                logger.debug(f"LLM summarization error: {e}")
                                return "Conversation summary placeholder"

                    llm_provider = LLMProviderWrapper(llm_processor)

            # Initialize hybrid memory system
            self.hybrid_memory = HybridMemorySystem(
                llm_provider=llm_provider,
                config=memory_config,
                device_id=self.device_id,
            )

            logger.info("Hybrid memory system initialized")

        except Exception as e:
            logger.warning(f"Failed to initialize hybrid memory system: {e}")
            logger.info("Falling back to SessionMemory only")
            self.hybrid_memory = None

    def _create_memory_config(self) -> MemoryConfig:
        """Create memory configuration from runtime config."""
        try:
            # Get memory system config from runtime config
            if not self.resource_manager:
                logger.warning("No resource manager available, using defaults")
                return MemoryConfig()

            runtime_config = self.resource_manager.config
            if not hasattr(runtime_config, "memory_system"):
                logger.warning("No memory_system config found, using defaults")
                return MemoryConfig()

            mem_config = getattr(runtime_config, "memory_system")
            return MemoryConfig(
                enabled=getattr(mem_config, "enabled", True),
                data_directory=getattr(mem_config, "data_directory", "data"),
                preferences_enabled=getattr(mem_config, "preferences_enabled", True),
                preferences_storage_path=getattr(
                    mem_config, "preferences_storage_path", "data/user_preferences.json"
                ),
                storage_enabled=getattr(mem_config, "storage_enabled", True),
                storage_db_path=getattr(
                    mem_config, "storage_db_path", "data/conversations.db"
                ),
                max_age_days=getattr(mem_config, "max_age_days", 30),
                summarization_enabled=getattr(
                    mem_config, "summarization_enabled", True
                ),
                summarize_every_n_turns=getattr(
                    mem_config, "summarize_every_n_turns", 10
                ),
                keep_recent_turns=getattr(mem_config, "keep_recent_turns", 5),
                max_summary_tokens=getattr(mem_config, "max_summary_tokens", 150),
                include_recent_turns=getattr(mem_config, "include_recent_turns", 3),
                include_summaries=getattr(mem_config, "include_summaries", True),
                include_preferences=getattr(mem_config, "include_preferences", True),
            )

        except Exception as e:
            logger.warning(f"Failed to create memory config: {e}")
            return MemoryConfig()

    async def shutdown(self) -> None:
        """Shutdown the engine and clean up resources."""
        try:
            logger.info("Shutting down real-time interaction engine...")

            # Stop real-time session if active
            if self.is_realtime_session_active:
                await self._stop_realtime_session()

            # Shutdown services
            if self.resource_manager:
                # Unload all models
                await self.resource_manager.unload_idle_models()

            # Shutdown character manager
            if self.character_manager:
                await self.character_manager.shutdown()

            # Shutdown safety filter
            if self.safety_filter:
                await self.safety_filter.shutdown()

            # Shutdown voice manager
            if self.voice_manager:
                await self.voice_manager.shutdown()

            # End hybrid memory session if active
            if self.hybrid_memory:
                try:
                    self.hybrid_memory.end_session()
                    logger.info("Ended hybrid memory session")
                except Exception as e:
                    logger.warning(f"Failed to end hybrid memory session: {e}")

            logger.info("Real-time interaction engine shutdown complete")
        except Exception as e:
            logger.error(f"Error during engine shutdown: {e}")

    async def _stop_realtime_session(self) -> None:
        """Stop the real-time session."""
        if self.realtime_session_task and not self.realtime_session_task.done():
            self.realtime_session_task.cancel()
            try:
                await self.realtime_session_task
            except asyncio.CancelledError:
                pass
        self.is_realtime_session_active = False
        self.realtime_session_task = None
        logger.info("Real-time session stopped")

    def set_active_character(self, character: Character) -> None:
        """Set the active character for interaction."""
        self.active_character = character
        logger.info(f"Active character set to: {character.name}")

        # Start a hybrid memory session if available
        if self.hybrid_memory:
            try:
                session_id = self.hybrid_memory.start_session(character.name)
                logger.info(f"Started hybrid memory session: {session_id}")
            except Exception as e:
                logger.warning(f"Failed to start hybrid memory session: {e}")

    def get_active_character(self) -> Optional[Character]:
        """Get the currently active character."""
        return self.active_character
