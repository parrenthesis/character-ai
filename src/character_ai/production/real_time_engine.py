"""
Real-time interaction engine for character.ai.

Optimized for real-time toy interaction with sub-500ms latency.
"""

# CRITICAL: Import torch_init FIRST to set environment variables before any torch imports
# isort: off
from ..core import torch_init  # noqa: F401

# isort: on

import asyncio
import time
import traceback
from typing import Any, Dict, List, Optional

from ..algorithms.conversational_ai.session_memory import SessionMemory
from ..algorithms.conversational_ai.text_normalizer import TextNormalizer
from ..characters import CharacterManager, ChildSafetyFilter
from ..characters.schema_voice_manager import SchemaVoiceManager
from ..characters.types import Character
from ..core.audio_io.device_selector import AudioDeviceSelector
from ..core.audio_io.factory import AudioComponentFactory
from ..core.audio_io.vad_session import VADSessionManager
from ..core.config import Config
from ..core.edge_optimizer import EdgeModelOptimizer
from ..core.hardware_profile import HardwareProfileManager
from ..core.llm.prompt_builder import LLMPromptBuilder
from ..core.logging import ProcessingTimer, get_logger
from ..core.protocols import AudioData, AudioResult
from ..core.resource_manager import ResourceManager
from ..core.response_cache import ResponseCache
from ..core.voice_activity_detection import VADConfig
from ..hardware.toy_hardware_manager import ToyHardwareManager

logger = get_logger(__name__)


class RealTimeInteractionEngine:
    """Optimized for real-time toy interaction."""

    def __init__(
        self,
        hardware_manager: ToyHardwareManager,
        # Optional dependency injection for new services
        text_normalizer: Optional[Any] = None,
        prompt_builder: Optional[Any] = None,
        resource_manager: Optional[Any] = None,
        hardware_profile: Optional[str] = None,
    ):
        self.hardware_manager: Optional[ToyHardwareManager] = hardware_manager
        self.character_manager: Optional[CharacterManager] = CharacterManager()

        # Load hardware profile if specified, otherwise auto-detect
        self.hardware_profile = hardware_profile
        self.hardware_config: Optional[Dict[str, Any]] = None
        if hardware_profile:
            profile_manager = HardwareProfileManager()
            self.hardware_config = profile_manager.load_profile(hardware_profile)
            logger.info(f"Loaded hardware profile: {hardware_profile}")
        else:
            # Auto-detect hardware profile
            profile_manager = HardwareProfileManager()
            detected_profile = profile_manager.detect_hardware()
            self.hardware_profile = detected_profile
            self.hardware_config = profile_manager.load_profile(detected_profile)
            logger.info(f"Auto-detected hardware profile: {detected_profile}")

        # Create or inject services
        if text_normalizer is None:
            self.text_normalizer = TextNormalizer()
        else:
            self.text_normalizer = text_normalizer

        if prompt_builder is None:
            self.prompt_builder = LLMPromptBuilder()
        else:
            self.prompt_builder = prompt_builder

        if resource_manager is None:
            # edge_optimizer will be set later, so pass None for now
            # Pass hardware config to ResourceManager for proper device selection
            self.resource_manager = ResourceManager(
                Config(), None, self.hardware_config
            )
        else:
            self.resource_manager = resource_manager

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
            SchemaVoiceManager()
        )  # Voice injection system
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

        # Response caching for frequent interactions
        self.response_cache = ResponseCache()

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

            # Pre-load models for real-time performance
            logger.info("Pre-loading models for real-time interaction...")

            # Preload all models including TTS for optimal performance
            # Try loading TTS first to avoid CUDA context conflicts
            preload_results = await self.resource_manager.preload_models(
                ["tts", "stt", "llm"]
            )
            loaded_count = sum(1 for success in preload_results.values() if success)
            logger.info(
                f"Pre-loaded {loaded_count}/{len(preload_results)} models: {preload_results}"
            )

            # Warm up models with dummy inference
            logger.info("Warming up models...")
            warmup_results = await self.resource_manager.warmup_all_models()
            warmup_count = sum(1 for success in warmup_results.values() if success)
            logger.info(f"✅ Warmed up {warmup_count}/{len(warmup_results)} models")

            # Pin models during realtime sessions to keep them hot
            self.resource_manager.pin_models(True)

            logger.info("RealTimeInteractionEngine initialized successfully")

        except Exception as e:
            logger.error(f"Failed to initialize RealTimeInteractionEngine: {e}")
            raise

    async def preload_models(self) -> Dict[str, bool]:
        """Pre-load all models for real-time interaction."""
        # Delegate to ResourceManager
        results = await self.resource_manager.preload_models(["stt", "llm", "tts"])

        # Log results
        loaded_count = sum(1 for success in results.values() if success)
        logger.info(f"Pre-loaded {loaded_count}/{len(results)} models: {results}")

        return results

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

            # Process audio with character personality (optimized version)
            with ProcessingTimer(
                logger, "character_processing", "real_time_engine"
            ) as timer:
                result = await self._process_with_character_personality_optimized(
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

    async def process_audio_file(
        self, input_file: str, character_name: str, output_file: Optional[str] = None
    ) -> Dict[str, Any]:
        """Process an audio file through the full pipeline.

        Args:
            input_file: Path to input audio file
            character_name: Name of character to use
            output_file: Optional path to save output audio

        Returns:
            Dict with transcription, response, and success status
        """
        try:
            # Load audio file using soundfile (avoids scipy conflicts)
            try:
                import soundfile as sf

                audio_data, sample_rate = sf.read(input_file)
            except Exception as e:
                error_details = f"{type(e).__name__}: {str(e)}"
                logger.error(f"Failed to load audio file {input_file}: {error_details}")
                logger.error(f"Traceback: {traceback.format_exc()}")
                return {
                    "success": False,
                    "error": f"Failed to load audio file: {error_details}",
                    "transcription": None,
                    "response": None,
                    "output_file": None,
                }

            # Create AudioData object
            audio_obj = AudioData(
                data=audio_data,
                sample_rate=sample_rate,
                duration=len(audio_data) / sample_rate,
                channels=1 if len(audio_data.shape) == 1 else audio_data.shape[1],
            )

            # Get character
            character = None
            if self.character_manager is not None:
                character = self.character_manager.get_active_character()
                if not character or character.name.lower() != character_name.lower():
                    # Try to get character by name
                    characters = self.character_manager.characters
                    for char in characters.values():
                        if char.name.lower() == character_name.lower():
                            character = char
                            break

            if not character:
                return {
                    "success": False,
                    "error": f"Character '{character_name}' not found",
                    "transcription": None,
                    "response": None,
                }

            # Debug: Check character metadata
            logger.info(f"Loaded character: {character.name}")
            logger.info(f"Character metadata: {character.metadata}")

            # Process through pipeline
            result = await self._process_with_character_personality(
                audio_obj, character
            )

            # Save outputs if requested
            output_files = {}
            if output_file and result.audio_data:
                # Save TTS audio output
                with open(output_file, "wb") as f:
                    f.write(result.audio_data.data)
                output_files["audio"] = output_file

                # Save STT transcription
                transcription = (
                    result.metadata.get("transcribed_text", "")
                    if result.metadata
                    else ""
                )
                if transcription:
                    stt_file = output_file.replace("_response.wav", "_stt_output.txt")
                    with open(stt_file, "w", encoding="utf-8") as f:
                        f.write(transcription)
                    output_files["stt"] = stt_file

                # Save LLM response
                if result.text:
                    llm_file = output_file.replace("_response.wav", "_llm_response.txt")
                    with open(llm_file, "w", encoding="utf-8") as f:
                        f.write(result.text)
                    output_files["llm"] = llm_file

            return {
                "success": bool(result.text) and bool(result.audio_data),
                "transcription": result.metadata.get("transcribed_text", "")
                if result.metadata
                else "",
                "response": result.text,
                "output_file": output_file if result.audio_data else None,
                "output_files": output_files,
            }

        except Exception as e:
            logger.error(f"Error processing audio file: {e}")
            return {
                "success": False,
                "error": str(e),
                "transcription": None,
                "response": None,
            }

    async def _process_with_character_personality(
        self, audio: AudioData, character: Character
    ) -> AudioResult:
        """Process audio with character personality."""
        try:
            pipeline_start = time.time()

            # Step 1: Speech-to-Text
            stt_start = time.time()
            transcribed_text = await self._transcribe_audio(audio)
            stt_time = time.time() - stt_start
            logger.info(f"⏱️  STT: {stt_time:.2f}s - '{transcribed_text[:50]}...'")

            # Step 2: Generate response
            llm_start = time.time()
            response_text = await self._generate_character_response(
                transcribed_text, character
            )
            llm_time = time.time() - llm_start
            logger.info(f"⏱️  LLM: {llm_time:.2f}s - '{response_text[:50]}...'")

            # Step 3: Safety filter
            safety_start = time.time()
            safe_response = response_text
            if self.safety_filter:
                safe_response = await self.safety_filter.filter_response(response_text)
            safety_time = time.time() - safety_start
            if safety_time > 0.1:  # Only log if significant
                logger.info(f"⏱️  Safety: {safety_time:.2f}s")

            # Step 4: Generate audio response
            tts_start = time.time()
            response_audio = await self._synthesize_character_voice(
                safe_response, character
            )
            tts_time = time.time() - tts_start
            logger.info(f"⏱️  TTS: {tts_time:.2f}s - {len(response_audio)} bytes")

            # Total pipeline time
            total_time = time.time() - pipeline_start
            logger.info(f"⏱️  TOTAL PIPELINE: {total_time:.2f}s (target: <5s)")

            if total_time > 5.0:
                logger.warning(f"⚠️  Pipeline exceeded 5s target: {total_time:.2f}s")
                logger.warning(
                    f"   Breakdown: STT={stt_time:.2f}s, LLM={llm_time:.2f}s, TTS={tts_time:.2f}s"
                )

            # Step 5: Store conversation turn in session memory
            self.session_memory.add_turn(
                character_name=character.name,
                user_input=transcribed_text,
                character_response=safe_response,
            )

            # Create AudioData object from TTS response
            audio_data_obj = None
            if response_audio:
                audio_data_obj = AudioData(
                    data=response_audio,
                    sample_rate=22050,  # Default TTS sample rate
                    duration=len(response_audio)
                    / (22050 * 2),  # 2 bytes per sample for int16
                    channels=1,
                )

            return AudioResult(
                text=safe_response,
                audio_data=audio_data_obj,
                metadata={
                    "character": character.name,
                    "character_type": character.dimensions.species.value,
                    "voice_style": character.voice_style,
                    "transcribed_text": transcribed_text,
                    "component": "real_time_engine",
                    "timing": {
                        "stt_time": stt_time,
                        "llm_time": llm_time,
                        "tts_time": tts_time,
                        "total_time": total_time,
                    },
                },
            )

        except Exception as e:
            logger.error(f"Error processing with character personality: {e}")
            return AudioResult(
                error=f"Character processing failed: {e}",
                metadata={"component": "real_time_engine"},
            )

    async def _process_with_character_personality_optimized(
        self, audio: AudioData, character: Character
    ) -> AudioResult:
        """Optimized pipeline with parallel processing and caching."""
        try:
            pipeline_start = time.time()

            # Step 1: STT
            stt_start = time.time()
            transcribed_text = await self._transcribe_audio(audio)
            stt_time = time.time() - stt_start
            logger.info(f"⏱️  STT: {stt_time:.2f}s - '{transcribed_text[:50]}...'")

            # Check cache
            cached_response = self.response_cache.get(transcribed_text, character.name)
            if cached_response:
                logger.info("💾 Cache hit! Skipping LLM generation")
                safe_response = cached_response
                llm_time = 0.0
            else:
                # Step 2: LLM with TTS warm-start (parallel)
                tts_warmup_task = asyncio.create_task(self._warmup_tts())

                llm_start = time.time()
                response_text = await self._generate_character_response(
                    transcribed_text, character
                )
                llm_time = time.time() - llm_start
                logger.info(f"⏱️  LLM: {llm_time:.2f}s - '{response_text[:50]}...'")
                logger.debug(f"Full LLM response for TTS: '{response_text}'")

                # Safety filter
                safe_response = response_text
                if self.safety_filter:
                    safe_response = await self.safety_filter.filter_response(
                        response_text
                    )

                # Cache the response
                self.response_cache.set(transcribed_text, character.name, safe_response)

                # Wait for TTS warmup
                await tts_warmup_task

            # Step 3: TTS (warmed up)
            tts_start = time.time()
            response_audio = await self._synthesize_character_voice(
                safe_response, character
            )
            tts_time = time.time() - tts_start
            logger.info(f"⏱️  TTS: {tts_time:.2f}s")

            total_time = time.time() - pipeline_start
            logger.info(f"⏱️  TOTAL: {total_time:.2f}s (target: <5s)")

            if total_time > 5.0:
                logger.warning(f"⚠️  Exceeded 5s target: {total_time:.2f}s")

            # Store in session memory
            self.session_memory.add_turn(
                character.name, transcribed_text, safe_response
            )

            # Create AudioData object from TTS response
            audio_data_obj = None
            if response_audio:
                audio_data_obj = AudioData(
                    data=response_audio,
                    sample_rate=22050,  # Default TTS sample rate
                    duration=len(response_audio)
                    / (22050 * 2),  # 2 bytes per sample for int16
                    channels=1,
                )

            return AudioResult(
                text=safe_response,
                audio_data=audio_data_obj,
                metadata={
                    "character": character.name,
                    "transcribed_text": transcribed_text,
                    "timing": {
                        "stt_time": stt_time,
                        "llm_time": llm_time,
                        "tts_time": tts_time,
                        "total_time": total_time,
                    },
                    "cache_hit": llm_time == 0.0,
                },
            )

        except Exception as e:
            logger.error(f"Error in optimized character processing: {e}")
            return AudioResult(
                error=f"Optimized character processing failed: {e}",
                metadata={"component": "real_time_engine"},
            )

    async def _warmup_tts(self) -> None:
        """Pre-warm TTS model for faster synthesis."""
        if not self.resource_manager.get_tts_processor():
            await self.resource_manager.preload_models(["tts"])

    async def _transcribe_audio(self, audio: AudioData) -> str:
        """Transcribe audio using ResourceManager's STT processor."""
        try:
            # Get processor from ResourceManager
            stt_processor = self.resource_manager.get_stt_processor()
            if stt_processor is None:
                # Use ResourceManager's preload_models to create processor
                await self.resource_manager.preload_models(["stt"])
                stt_processor = self.resource_manager.get_stt_processor()

            if stt_processor is None:
                raise RuntimeError("Failed to initialize STT processor")

            # Use processor
            result = await stt_processor.process_audio(audio)
            transcribed_text = (
                result.text
                if result.text
                else "I didn't catch that. Could you please repeat?"
            )

            # Mark model as used
            self.resource_manager.mark_model_used("stt")
            return transcribed_text
        except Exception as e:
            logger.error(f"Wav2Vec2 transcription failed: {e}")
            return "I'm having trouble hearing you right now."

    async def _generate_character_response(
        self, text: str, character: Character
    ) -> str:
        """Generate response using ResourceManager's LLM processor."""
        try:
            # Get processor from ResourceManager
            llm_processor = self.resource_manager.get_llm_processor()
            if llm_processor is None:
                # Use ResourceManager's preload_models to create processor
                await self.resource_manager.preload_models(["llm"])
                llm_processor = self.resource_manager.get_llm_processor()

            if llm_processor is None:
                raise RuntimeError("Failed to initialize LLM processor")

            # Use prompt builder instead of _create_character_prompt
            character_name = (
                character.name if hasattr(character, "name") else str(character)
            )
            character_prompt = self.prompt_builder.build_prompt(
                user_input=text,
                character=character,
                conversation_context=self.session_memory.format_context_for_llm(
                    character_name=character_name, current_user_input=text, max_turns=5
                ),
            )

            # Use processor
            result = await llm_processor.process_text(character_prompt)
            response_text = (
                result.text if result.text else f"Hi! I'm {character_name}..."
            )

            # Debug logging to see raw LLM output before cleaning
            logger.debug(f"Raw LLM output before normalization: '{response_text}'")

            # Use text normalizer instead of _clean_llm_response
            response_text = self.text_normalizer.clean_llm_response(response_text)

            logger.debug(f"After text normalization: '{response_text}'")

            # Mark model as used
            self.resource_manager.mark_model_used("llm")
            return response_text
        except Exception as e:
            logger.error(f"LLM response generation failed: {e}")
            char_name = character.name if hasattr(character, "name") else str(character)
            return f"Hello! I'm {char_name}..."

    async def _synthesize_character_voice(
        self, text: str, character: Character
    ) -> bytes:
        """Synthesize voice using ResourceManager's TTS processor."""
        try:
            # Get TTS config from character metadata
            speed = 1.0  # Default speed
            tts_model_name = None
            if hasattr(character.metadata, "get") and character.metadata:
                tts_config = character.metadata.get("tts_config", {})
                speed = (
                    tts_config.get("speed", 1.0)
                    if isinstance(tts_config, dict)
                    else 1.0
                )
                tts_model_name = (
                    tts_config.get("model") if isinstance(tts_config, dict) else None
                )

            # Load character-specific TTS model if specified
            if tts_model_name:
                logger.info(f"Loading character-specific TTS model: {tts_model_name}")
                await self.resource_manager.preload_models_with_config(
                    {"tts": tts_model_name}
                )

            # Get processor from ResourceManager
            tts_processor = self.resource_manager.get_tts_processor()
            if tts_processor is None:
                # Fallback to default TTS model
                await self.resource_manager.preload_models(["tts"])
                tts_processor = self.resource_manager.get_tts_processor()

            if tts_processor is None:
                raise RuntimeError("Failed to initialize TTS processor")

            # Get voice configuration
            character_name = (
                character.name.lower()
                if hasattr(character, "name")
                else str(character).lower()
            )
            voice_path = None
            if self.voice_manager:
                # Get franchise from character metadata or use character name as fallback
                franchise = (
                    getattr(character, "franchise", None)
                    or (
                        character.metadata.get("franchise")
                        if hasattr(character, "metadata") and character.metadata
                        else None
                    )
                    or character_name
                )
                voice_info = await self.voice_manager.get_character_voice_path(
                    character_name, franchise
                )
                voice_path = voice_info.get("voice_file_path") if voice_info else None

            # Use text normalizer to prepare for TTS
            tts_text = self.text_normalizer.prepare_for_tts(text)

            # Use processor
            logger.info(
                f"TTS synthesis: text='{tts_text}', voice_path='{voice_path}', speed={speed}"
            )
            result = await tts_processor.synthesize_speech(
                text=tts_text, voice_path=voice_path, language="en", speed=speed
            )
            logger.info(
                f"TTS result: audio_data={result.audio_data is not None}, error={result.error}"
            )
            if result.audio_data:
                logger.info(f"TTS audio data size: {len(result.audio_data.data)} bytes")
            else:
                logger.warning("TTS result has no audio_data!")

            # Mark model as used
            self.resource_manager.mark_model_used("tts")
            return result.audio_data.data if result.audio_data else b""
        except Exception as e:
            logger.error(f"Voice synthesis failed: {e}")
            return b""

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
            # Get TTS processor from ResourceManager
            tts_processor = self.resource_manager.get_tts_processor()
            if tts_processor is None:
                await self.resource_manager.preload_models(["tts"])
                tts_processor = self.resource_manager.get_tts_processor()

            if tts_processor is None:
                raise RuntimeError(
                    "Failed to initialize TTS processor for voice injection"
                )

            # Inject the voice
            success = False
            if self.voice_manager is not None:
                # Use the correct method name from SchemaVoiceManager
                success = await self.voice_manager.clone_character_voice(
                    character_name, voice_file_path, tts_processor=tts_processor
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
                # Use the correct method name from SchemaVoiceManager
                voices_data = await self.voice_manager.list_characters_with_voice()
                # Extract character names from the returned data
                voices = [
                    voice.get("name", "") for voice in voices_data if voice.get("name")
                ]
                return voices
            return []
        except Exception as e:
            logger.error(f"Failed to list character voices: {e}")
            return []

    async def start_realtime_session(
        self,
        character: Character,
        duration: int,
        device_pattern: Optional[str] = None,
        vad_config: Optional[Any] = None,
    ) -> Dict[str, Any]:
        """Start real-time voice interaction session."""

        logger.info(f"Starting real-time session with {character.name} for {duration}s")

        # Get wake word config from hardware profile
        wake_word_config = None
        hardware_vad_settings = {}
        if self.hardware_config and "vad" in self.hardware_config:
            hardware_vad_settings = self.hardware_config["vad"]
            wake_word_config = hardware_vad_settings.get("wake_word")

        # Load character wake words if wake word is enabled
        character_wake_words = None
        if wake_word_config and wake_word_config.get("enabled"):
            # Character wake words come from character config
            # In production, these should be loaded from character.wake_words or similar
            # For now, just pass None and let VADSessionManager handle defaults
            pass

        # Initialize services
        device_selector = AudioDeviceSelector()
        vad_manager = VADSessionManager(
            vad_config=vad_config or VADConfig.for_toy_interaction(),
            hardware_vad_settings=hardware_vad_settings,
            wake_word_config=wake_word_config,
            character_wake_words=character_wake_words,
        )
        audio_factory = AudioComponentFactory()

        # Find compatible audio device
        device_pattern = device_pattern or "audiobox"
        audio_device = device_selector.get_compatible_device(
            device_pattern, fallback_to_default=True
        )
        if not audio_device:
            raise RuntimeError(
                f"No compatible audio device found for pattern: {device_pattern}"
            )

        logger.info(f"Using audio device: {audio_device.name}")

        # Initialize audio capture
        audio_capture = audio_factory.create_audio_capture()

        # Test sample rates and start capture
        sample_rates = [44100, 48000, 16000]
        compatible_rate = device_selector.test_sample_rates(audio_device, sample_rates)
        if not compatible_rate:
            compatible_rate = 44100  # Default fallback

        try:
            await audio_capture.start_capture(
                audio_device, sample_rate=compatible_rate, channels=1, chunk_size=512
            )
        except Exception:
            logger.warning(f"Failed with {compatible_rate}Hz, trying fallback rates")
            for rate in [48000, 16000]:
                try:
                    await audio_capture.start_capture(
                        audio_device, sample_rate=rate, channels=1, chunk_size=512
                    )
                    compatible_rate = rate
                    break
                except Exception:
                    continue
            else:
                raise RuntimeError("No compatible sample rates found for audio device")

        logger.info(f"Audio capture started at {compatible_rate}Hz")

        # Session state
        interaction_count = 0
        start_time = time.time()
        end_time = start_time + duration
        is_processing = False
        total_processing_time = 0.0

        logger.info("🎙️  Listening for speech... (speak naturally)")

        try:
            while time.time() < end_time:
                # Skip processing if already processing to prevent overflow
                if is_processing:
                    await asyncio.sleep(0.1)
                    continue

                # Read audio chunk
                audio_chunk = await audio_capture.read_audio_chunk()
                if audio_chunk is None:
                    await asyncio.sleep(0.01)
                    continue

                # Process audio chunk through VAD
                vad_manager.process_audio_chunk(audio_chunk)

                # Handle speech end
                if vad_manager.should_end_speech():
                    is_processing = True
                    vad_manager.set_processing_state()

                    # Get combined speech audio
                    speech_audio = vad_manager.get_combined_speech_audio()
                    if speech_audio is not None:
                        # Process the speech segment
                        process_start = time.time()
                        await self.process_speech_segment(speech_audio, character)
                        process_time = time.time() - process_start

                        total_processing_time += process_time
                        interaction_count += 1

                        logger.info(
                            f"Processed interaction {interaction_count} in {process_time:.2f}s"
                        )

                    # Reset VAD session
                    vad_manager.reset_session()
                    is_processing = False
                    logger.info("🎙️  Listening for speech... (speak naturally)")

        except Exception as e:
            logger.error(f"Error during real-time session: {e}")
            raise
        finally:
            # Cleanup
            try:
                await audio_capture.stop_capture()
            except Exception as e:
                logger.warning(f"Error stopping audio capture: {e}")

        # Calculate statistics
        session_duration = time.time() - start_time
        average_processing_time = total_processing_time / max(interaction_count, 1)

        logger.info(
            f"Session completed: {interaction_count} interactions in {session_duration:.1f}s"
        )

        return {
            "interaction_count": interaction_count,
            "session_duration": session_duration,
            "average_processing_time": average_processing_time,
            "total_processing_time": total_processing_time,
            "device_used": audio_device.name,
            "sample_rate": compatible_rate,
            "vad_statistics": vad_manager.get_session_statistics(),
            "status": "completed",
        }

    async def process_speech_segment(
        self, audio_array: Any, character: Character
    ) -> AudioResult:
        """Process a detected speech segment."""

        logger.info(f"Processing speech segment for {character.name}")

        # Convert audio array to numpy if needed
        import numpy as np

        if not isinstance(audio_array, np.ndarray):
            audio_array = np.array(audio_array, dtype=np.float32)

        # Ensure audio is in the right format
        if audio_array.ndim > 1:
            audio_array = audio_array.flatten()

        # Check audio quality
        audio_level = np.max(np.abs(audio_array))
        if audio_level < 1e-6:
            logger.warning("Audio level too low, skipping processing")
            return AudioResult(text="Audio too quiet to process", audio_data=None)

        # Process through the character personality pipeline
        try:
            result = await self._process_with_character_personality(
                audio_array, character
            )

            # Log processing statistics
            if result.metadata:
                processing_time = result.metadata.get("processing_time", 0.0)
                logger.info(f"Processed speech in {processing_time:.2f}s")

            return result

        except Exception as e:
            logger.error(f"Error processing speech segment: {e}")
            return AudioResult(
                text=f"Error processing speech: {str(e)}", audio_data=None
            )

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
