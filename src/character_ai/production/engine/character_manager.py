"""
Character management module for real-time interaction engine.

Handles character selection, voice injection, and real-time session management.
"""

import asyncio
import time
from typing import Any, Dict, List, Optional

from ...characters import Character, CharacterService
from ...characters.voices import SchemaVoiceService
from ...core.audio_io.device_selector import AudioDeviceSelector
from ...core.audio_io.factory import AudioComponentFactory
from ...core.audio_io.vad_session import VADSessionManager
from ...core.audio_io.voice_activity_detection import VADConfig
from ...core.resource_manager import ResourceManager
from ...observability import get_logger

logger = get_logger(__name__)


class CharacterInteractionController:
    """Manages character interactions and real-time sessions."""

    def __init__(
        self,
        character_manager: Optional[CharacterService] = None,
        voice_manager: Optional[SchemaVoiceService] = None,
        resource_manager: Optional[ResourceManager] = None,
        hardware_config: Optional[Dict[str, Any]] = None,
    ):
        self.character_manager = character_manager
        self.voice_manager = voice_manager
        self.resource_manager = resource_manager
        self.hardware_config = hardware_config or {}
        self.active_character: Optional[Character] = None

    def get_character(self, character_name: str) -> Optional[Character]:
        """Get a character by name."""
        if self.character_manager is not None:
            return self.character_manager.get_character(character_name)
        return None

    def get_available_characters(self) -> List[str]:
        """Get list of available character names."""
        if self.character_manager is not None:
            return self.character_manager.get_available_characters()
        return []

    def get_character_info(self, character_name: str) -> Optional[Dict[str, Any]]:
        """Get character information."""
        if self.character_manager is not None:
            return self.character_manager.get_character_info(character_name)
        return None

    def get_active_character(self) -> Optional[Character]:
        """Get the currently active character."""
        return self.active_character

    async def create_custom_character(
        self, name: str, character_type: Any, custom_topics: List[str]
    ) -> bool:
        """Create a custom character."""
        if self.character_manager is not None:
            return await self.character_manager.create_custom_character(
                name, character_type, custom_topics
            )
        return False

    async def reload_profiles(self) -> bool:
        """Reload character profiles."""
        if self.character_manager is not None:
            return await self.character_manager.reload_profiles()
        return False

    async def set_active_character(self, character_name: str) -> bool:
        """Set the active character for interactions."""
        try:
            success = False
            if self.character_manager is not None:
                success = await self.character_manager.set_active_character(
                    character_name
                )
                if success:
                    self.active_character = self.character_manager.get_character(
                        character_name
                    )
            if success:
                logger.info(f"Active character set to: {character_name}")
            return success
        except Exception as e:
            logger.error(f"Failed to set active character: {e}")
            return False

    async def get_active_character_info(self) -> Dict[str, Any]:
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
            if not self.resource_manager:
                raise ValueError("Resource manager not available")

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
                # Use the correct method name from SchemaVoiceService
                success = await self.voice_manager.clone_character_voice(
                    character_name, character_name.lower(), voice_file_path
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
                # Use the correct method name from SchemaVoiceService
                voices = self.voice_manager.list_characters_with_voice()
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
        audio_processor: Optional[Any] = None,
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
                    if speech_audio is not None and audio_processor:
                        # Process the speech segment
                        process_start = time.time()
                        await audio_processor.process_speech_segment(
                            speech_audio, character
                        )
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

    async def load_character(self, character_name: str) -> Optional[Character]:
        """Load a character by name."""
        try:
            if self.character_manager:
                character = self.character_manager.get_character(character_name)
                if character:
                    self.active_character = character
                return character
            return None
        except Exception as e:
            logger.error(f"Failed to load character {character_name}: {e}")
            return None
