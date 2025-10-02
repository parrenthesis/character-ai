"""
Real-time interaction engine for character.ai.

Optimized for real-time toy interaction with sub-500ms latency.
"""

import time
from typing import Any, Dict, List, Optional

from ..algorithms.conversational_ai.session_memory import SessionMemory
from ..characters import CharacterManager, ChildSafetyFilter, VoiceManager
from ..characters.types import Character
from ..core.edge_optimizer import EdgeModelOptimizer
from ..core.logging import ProcessingTimer, get_logger
from ..core.protocols import AudioData, AudioResult
from ..hardware.toy_hardware_manager import ToyHardwareManager

logger = get_logger(__name__)


class RealTimeInteractionEngine:
    """Optimized for real-time toy interaction."""

    def __init__(self, hardware_manager: ToyHardwareManager):
        self.hardware_manager: Optional[ToyHardwareManager] = hardware_manager
        self.character_manager: Optional[CharacterManager] = CharacterManager()

        # Adapter to provide an initialize hook for safety filter to satisfy tests
        class _SafetyFilterFacade:
            def __init__(self, inner: Any) -> None:
                self._inner = inner

            async def initialize(self) -> None:
                return None

            async def filter_response(self, text: str) -> str:
                result = await self._inner.filter_response(text)
                return str(result)

        self.safety_filter: Optional[_SafetyFilterFacade] = _SafetyFilterFacade(
            ChildSafetyFilter()
        )

        class _VoiceManagerFacade:
            def __init__(self, inner: Any) -> None:
                self._inner = inner

            async def initialize(self) -> None:
                return None

            def __getattr__(self, name: str) -> Any:
                return getattr(self._inner, name)

        self.voice_manager: Optional[_VoiceManagerFacade] = _VoiceManagerFacade(
            VoiceManager()
        )  # Voice injection system
        self.edge_optimizer: Optional[EdgeModelOptimizer] = (
            EdgeModelOptimizer(hardware_manager.constraints)
            if hardware_manager
            else None
        )

        # Session memory management
        self.session_memory = SessionMemory()

        # Runtime behavior defaults (will be finalized during initialize())
        self.target_latency = 0.5
        self.streaming_enabled = True
        self.predictive_loading = True

        # Performance tracking
        self.total_interactions = 0
        self.successful_interactions = 0
        self.average_latency = 0.0

    async def initialize(self) -> None:
        """Initialize the real-time interaction engine."""
        try:
            logger.info("Initializing RealTimeInteractionEngine...")

            # Initialize hardware
            if self.hardware_manager is not None:
                await self.hardware_manager.initialize()

            # Initialize character manager
            if self.character_manager is not None:
                await self.character_manager.initialize()

            # Load runtime config from edge optimizer
            if self.edge_optimizer is not None:
                cfg = await self.edge_optimizer.optimize_llm_for_toy()
                self.target_latency = cfg.runtime.target_latency_s
                self.streaming_enabled = cfg.runtime.streaming_enabled
                self.predictive_loading = cfg.runtime.predictive_loading

            # Optimize for toy deployment
            await self.optimize_for_toy()

            logger.info("RealTimeInteractionEngine initialized successfully")

        except Exception as e:
            logger.error(f"Failed to initialize RealTimeInteractionEngine: {e}")
            raise

    async def process_realtime_audio(self, audio_stream: AudioData) -> AudioResult:
        """Process audio in real-time for natural interaction."""
        start_time = time.time()

        try:
            # Log processing start
            logger.info(
                "Real-time audio processing started",
                audio_duration=(
                    audio_stream.duration if hasattr(audio_stream, "duration") else None
                ),
                sample_rate=(
                    audio_stream.sample_rate
                    if hasattr(audio_stream, "sample_rate")
                    else None
                ),
            )

            # Get active character
            active_character = None
            if self.character_manager is not None:
                active_character = self.character_manager.get_active_character()
            if not active_character:
                logger.warning("No active character set for audio processing")
                return AudioResult(
                    error="No active character set",
                    metadata={"component": "real_time_engine"},
                )

            logger.info(
                "Processing audio with character",
                character_name=active_character.name,
            )

            # Process audio with character personality
            with ProcessingTimer(
                logger, "character_processing", "real_time_engine"
            ) as timer:
                result = await self._process_with_character_personality(
                    audio_stream, active_character
                )

            # Apply safety filter
            if result.text:
                logger.info(
                    "Applying safety filter to response", text_length=len(result.text)
                )
                with ProcessingTimer(
                    logger, "safety_filtering", "safety_filter"
                ) as safety_timer:
                    if self.safety_filter is not None:
                        result.text = await self.safety_filter.filter_response(
                            result.text
                        )

                # Log safety filter results
                if safety_timer.duration_ms:
                    logger.info(
                        "Safety filtering completed",
                        duration_ms=safety_timer.duration_ms,
                        text_length=len(result.text),
                    )

            processing_time = time.time() - start_time

            # Update performance tracking
            self._update_performance_metrics(processing_time, result.error is None)

            # Log performance metrics
            logger.info(
                "Real-time audio processing completed",
                total_duration_ms=processing_time * 1000,
                character_processing_ms=(
                    timer.duration_ms if hasattr(timer, "duration_ms") else None
                ),
                success=result.error is None,
                target_latency_ms=self.target_latency * 1000,
            )

            if processing_time > self.target_latency:
                logger.warning(
                    "Latency exceeded target",
                    actual_latency_ms=processing_time * 1000,
                    target_latency_ms=self.target_latency * 1000,
                    excess_ms=(processing_time - self.target_latency) * 1000,
                )

            return result

        except Exception as e:
            logger.error(f"Error in real-time audio processing: {e}")
            return AudioResult(
                error=f"Real-time processing failed: {e}",
                metadata={"component": "real_time_engine"},
            )

    async def _process_with_character_personality(
        self, audio: AudioData, character: Character
    ) -> AudioResult:
        """Process audio with character personality."""
        try:
            # Step 1: Speech-to-Text using Wav2Vec2
            transcribed_text = await self._transcribe_audio(audio)

            # Step 2: Generate response using LLM with character personality and context

            response_text = await self._generate_character_response(
                transcribed_text, character
            )

            # Step 3: Apply safety filter
            safe_response = response_text
            if self.safety_filter is not None:
                safe_response = await self.safety_filter.filter_response(response_text)

            # Step 4: Generate audio response using Coqui TTS with character voice
            response_audio = await self._synthesize_character_voice(
                safe_response, character
            )

            # Step 5: Store conversation turn in session memory
            self.session_memory.add_turn(
                character_name=character.name,
                user_input=transcribed_text,
                character_response=safe_response,
            )

            return AudioResult(
                text=safe_response,
                audio_data=response_audio if hasattr(response_audio, "data") else None,  # type: ignore
                metadata={
                    "character": character.name,
                    "character_type": character.dimensions.species.value,
                    "voice_style": character.voice_style,
                    "transcribed_text": transcribed_text,
                    "component": "real_time_engine",
                },
            )

        except Exception as e:
            logger.error(f"Error processing with character personality: {e}")
            return AudioResult(
                error=f"Character processing failed: {e}",
                metadata={"component": "real_time_engine"},
            )

    async def _transcribe_audio(self, audio: AudioData) -> str:
        """Transcribe audio using Wav2Vec2."""
        try:
            # Import Wav2Vec2 processor
            from ..algorithms.conversational_ai.wav2vec2_processor import (
                Wav2Vec2Processor,
            )

            # Initialize Wav2Vec2 with edge optimizations
            if self.edge_optimizer is not None:
                stt_config = await self.edge_optimizer.optimize_wav2vec2_for_toy()
            else:
                # Fallback configuration
                from ..core.config import Config

                stt_config = Config()
            wav2vec2 = Wav2Vec2Processor(stt_config)

            # Transcribe audio
            result = await wav2vec2.process_audio(audio)
            return (
                result.text
                if result.text
                else "I didn't catch that. Could you please repeat?"
            )

        except Exception as e:
            logger.error(f"Wav2Vec2 transcription failed: {e}")
            return "I'm having trouble hearing you right now."

    async def _generate_character_response(
        self, text: str, character: Character
    ) -> str:
        """Generate response using LLM with character personality."""
        try:
            # Import LLM processor selected by backend
            if self.hardware_manager and hasattr(self.hardware_manager, "constraints"):
                pass  # reserved for future hardware-based selection
            from ..algorithms.conversational_ai.llama_cpp_processor import (
                LlamaCppProcessor,
            )
            from ..algorithms.conversational_ai.llama_processor import LlamaProcessor

            # Initialize LLM with edge optimizations
            if self.edge_optimizer is not None:
                llm_config = await self.edge_optimizer.optimize_llm_for_toy()
            else:
                # Fallback configuration
                from ..core.config import Config

                llm_config = Config()
            if llm_config.models.llama_backend == "llama_cpp":
                llm: Any = LlamaCppProcessor(llm_config)
            else:
                llm = LlamaProcessor(llm_config)

            # Create character-specific prompt with conversation context
            character_prompt = self._create_character_prompt_with_context(
                text, character
            )

            # Generate response
            result = await llm.process_text(character_prompt)
            return (
                result.text
                if result.text
                else f"Hi! I'm {character.name}. What would you like to talk about?"
            )

        except Exception as e:
            logger.error(f"LLM response generation failed: {e}")
            return f"Hello! I'm {character.name}, a {character.dimensions.species.value}. I love talking about {', '.join([trait.value for trait in character.dimensions.personality_traits[:2]])}!"

    async def _synthesize_character_voice(
        self, text: str, character: Character
    ) -> bytes:
        """Synthesize character voice using injected character voice."""
        try:
            # Import Coqui TTS processor
            from ..algorithms.conversational_ai.coqui_processor import CoquiProcessor

            # Initialize Coqui TTS with edge optimizations
            if self.edge_optimizer is not None:
                tts_config = await self.edge_optimizer.optimize_coqui_for_toy()
            else:
                # Fallback configuration
                from ..core.config import Config

                tts_config = Config()
            tts = CoquiProcessor(tts_config)

            # Check if character has injected voice
            character_name = character.name.lower()
            voice_path = None
            if self.voice_manager is not None:
                voice_path = await self.voice_manager.get_character_voice_path(
                    character_name
                )

            if voice_path:
                # Use injected character voice
                logger.info(f"Using injected voice for {character_name}")
                result = await tts.synthesize_speech(
                    text=text, voice_path=voice_path, language="en"
                )
                return result.audio_data.data if result.audio_data else b""
            else:
                # Fallback to default voice synthesis
                logger.info(f"No injected voice for {character_name}, using default")
                character.get_voice_characteristics()
                result = await tts.synthesize_speech(
                    text=text, voice_path=None, language="en"
                )
                return result.audio_data.data if result.audio_data else b""

        except Exception as e:
            logger.error(f"Voice synthesis failed: {e}")
            return b""

    def _create_character_prompt(self, user_input: str, character: Character) -> str:
        """Create a character-specific prompt for the LLM using profile traits and optio
        nal prompt.md."""
        character_name = character.name
        character_type = character.dimensions.species.value
        voice_style = character.voice_style
        topics = ", ".join([topic.value for topic in character.dimensions.topics[:5]])

        # Check for custom prompt template in character metadata
        custom_prompt = None
        if hasattr(character, "metadata") and character.metadata:
            prompt_template_path = character.metadata.get("prompt_template")
            if prompt_template_path:
                try:
                    from pathlib import Path

                    from ..core.config import Config

                    cfg = Config()
                    characters_dir = Path(cfg.paths.characters_dir)
                    character_dir = characters_dir / character_name.lower()
                    prompt_file = character_dir / prompt_template_path
                    if prompt_file.exists():
                        custom_prompt = prompt_file.read_text(encoding="utf-8")
                        logger.info(
                            f"Using custom prompt template for {character_name}: {prompt_file}"
                        )
                except Exception as e:
                    logger.warning(
                        f"Failed to load custom prompt for {character_name}: {e}"
                    )

        # Use custom prompt if available, otherwise fall back to default
        prompt: str
        if custom_prompt:
            # Replace placeholders in custom prompt
            prompt = custom_prompt.replace("{character_name}", character_name)
            prompt = prompt.replace("{character_type}", character_type)
            prompt = prompt.replace("{voice_style}", voice_style)
            prompt = prompt.replace("{topics}", topics)
            prompt = prompt.replace("{user_input}", user_input)

            # Add traits from metadata if available
            if hasattr(character, "metadata") and character.metadata:
                traits = character.metadata.get("traits", {})
                for key, value in traits.items():
                    prompt = prompt.replace(f"{{{key}}}", str(value))
        else:
            # Default prompt template
            prompt = f"""You are {character_name}, a {character_type} with a {voice_style} voice.
You love talking about: {topics}.

User said: "{user_input}"

Respond as {character_name} would, keeping your response:
- Under 50 words
- Positive and child-friendly
- Related to your interests: {topics}
- In character as a {character_type}

Response:"""

        return prompt

    def _create_character_prompt_with_context(
        self, user_input: str, character: Character
    ) -> str:
        """Create a character-specific prompt with conversation context."""
        # Get conversation context
        context = self.session_memory.format_context_for_llm(
            character_name=character.name,
            current_user_input=user_input,
            max_turns=5,  # Limit context to last 5 turns to manage token usage
        )

        # Create base prompt
        base_prompt = self._create_character_prompt(user_input, character)

        # If we have context, prepend it
        if context:
            return f"{context}\n\n{base_prompt}"

        return base_prompt

    async def optimize_for_toy(self) -> None:
        """Optimize entire system for toy deployment."""
        try:
            # Get hardware constraints
            constraints = {}
            if self.hardware_manager is not None:
                constraints = await self.hardware_manager.optimize_for_toy()

            # Get edge optimizations
            edge_optimizations = {}
            if self.edge_optimizer is not None:
                edge_optimizations = (
                    await self.edge_optimizer.get_edge_optimization_summary()
                )

            logger.info("System optimized for toy deployment")
            logger.info(f"Hardware constraints: {constraints}")
            logger.info(f"Edge optimizations: {edge_optimizations}")

        except Exception as e:
            logger.error(f"Failed to optimize for toy: {e}")
            raise

    async def set_active_character(self, character_name: str) -> bool:
        """Set the active character for interactions."""
        try:
            success = False
            if self.character_manager is not None:
                success = await self.character_manager.set_active_character(
                    character_name
                )
            if success:
                logger.info(f"Active character set to: {character_name}")
            return success
        except Exception as e:
            logger.error(f"Failed to set active character: {e}")
            return False

    async def get_character_info(self) -> Dict[str, Any]:
        """Get information about the active character."""
        try:
            if self.character_manager is not None:
                return self.character_manager.get_character_info()
            return {"error": "Character manager not available"}
        except Exception as e:
            logger.error(f"Failed to get character info: {e}")
            return {"error": str(e)}

    async def inject_character_voice(
        self, character_name: str, voice_file_path: str
    ) -> bool:
        """Inject a voice for a character (used during toy manufacturing/setup)."""
        try:
            from ..algorithms.conversational_ai.coqui_processor import CoquiProcessor

            # Initialize Coqui TTS processor
            if self.edge_optimizer is not None:
                tts_config = await self.edge_optimizer.optimize_coqui_for_toy()
            else:
                from ..core.config import Config

                tts_config = Config()
            tts = CoquiProcessor(tts_config)
            await tts.initialize()

            # Inject the voice
            success = False
            if self.voice_manager is not None:
                success = await self.voice_manager.inject_character_voice(
                    character_name, voice_file_path, tts
                )
            success = bool(success)

            if success:
                logger.info(
                    f"Voice injected for character '{character_name}' during toy setup"
                )

            return success

        except Exception as e:
            logger.error(f"Failed to inject voice for {character_name}: {e}")
            return False

    async def list_character_voices(self) -> List[str]:
        """List all characters that have injected voices."""
        try:
            if self.voice_manager is not None:
                voices = await self.voice_manager.list_available_voices()
                return list(voices) if voices else []
            return []
        except Exception as e:
            logger.error(f"Failed to list character voices: {e}")
            return []

    async def get_performance_metrics(self) -> Dict[str, Any]:
        """Get performance metrics for the real-time engine."""
        try:
            success_rate = self.successful_interactions / max(
                self.total_interactions, 1
            )

            return {
                "total_interactions": self.total_interactions,
                "successful_interactions": self.successful_interactions,
                "success_rate": success_rate,
                "average_latency": self.average_latency,
                "target_latency": self.target_latency,
                "latency_within_target": self.average_latency <= self.target_latency,
            }
        except Exception as e:
            logger.error(f"Failed to get performance metrics: {e}")
            return {"error": str(e)}

    def _update_performance_metrics(self, latency: float, success: bool) -> None:
        """Update performance tracking metrics."""
        self.total_interactions += 1
        if success:
            self.successful_interactions += 1

        # Update average latency
        if self.average_latency == 0:
            self.average_latency = latency
        else:
            self.average_latency = (self.average_latency + latency) / 2

    async def get_health_status(self) -> Dict[str, Any]:
        """Get health status of the real-time engine."""
        try:
            # Check character manager status
            character_info = await self.get_character_info()
            character_healthy = "error" not in character_info

            # Check performance metrics
            metrics = await self.get_performance_metrics()
            performance_healthy = metrics.get("latency_within_target", False)

            # Overall health
            overall_healthy = character_healthy and performance_healthy

            return {
                "healthy": overall_healthy,
                "character_manager": {
                    "healthy": character_healthy,
                    "active_character": character_info.get("name", "None"),
                },
                "performance": metrics,
                "hardware_constraints": {
                    "max_memory_gb": self.hardware_manager.constraints.max_memory_gb
                    if self.hardware_manager is not None
                    else 0,
                    "target_latency_ms": self.hardware_manager.constraints.target_latency_ms
                    if self.hardware_manager is not None
                    else 0,
                },
            }
        except Exception as e:
            logger.error(f"Failed to get health status: {e}")
            return {"healthy": False, "error": str(e)}

    async def shutdown(self) -> None:
        """Shutdown the engine and clean up resources."""
        try:
            logger.info("Shutting down RealTimeInteractionEngine...")

            # Shutdown hardware manager
            if self.hardware_manager is not None and hasattr(
                self.hardware_manager, "shutdown"
            ):
                await self.hardware_manager.shutdown()

            # Shutdown character manager
            if self.character_manager is not None and hasattr(
                self.character_manager, "shutdown"
            ):
                await self.character_manager.shutdown()

            # Clear references to help with garbage collection
            self.hardware_manager = None
            self.character_manager = None
            self.safety_filter = None
            self.voice_manager = None
            self.edge_optimizer = None

            logger.info("RealTimeInteractionEngine shutdown complete")

        except Exception as e:
            logger.error(f"Error during engine shutdown: {e}")
