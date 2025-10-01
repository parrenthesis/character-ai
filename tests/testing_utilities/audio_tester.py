"""
Audio testing utilities for character.ai.

Provides audio file testing when no physical microphone is available.
"""

import logging
import wave
from pathlib import Path
from typing import Any, Dict

from character_ai.core.protocols import AudioData
from character_ai.hardware.toy_hardware_manager import (
    HardwareConstraints,
    ToyHardwareManager,
)
from character_ai.production.real_time_engine import RealTimeInteractionEngine

logger = logging.getLogger(__name__)


class AudioTester:
    """Test the character.ai with audio files."""

    def __init__(self, test_audio_dir: str = None):
        if test_audio_dir is None:
            # Use temporary directory to avoid creating files in project root
            import tempfile
            self.test_audio_dir = Path(tempfile.mkdtemp(prefix="test_audio_"))
        else:
            self.test_audio_dir = Path(test_audio_dir)
            self.test_audio_dir.mkdir(exist_ok=True)

        # Lazy initialization to avoid import-time side effects
        self._constraints = HardwareConstraints()
        self._hardware_manager = None
        self._engine = None

    @property
    def hardware_manager(self):
        if self._hardware_manager is None:
            self._hardware_manager = ToyHardwareManager(self._constraints)
        return self._hardware_manager

    @property
    def engine(self):
        if self._engine is None:
            self._engine = RealTimeInteractionEngine(self.hardware_manager)
        return self._engine

    async def initialize(self) -> None:
        """Initialize the testing system."""
        try:
            await self.engine.initialize()
            logger.info("Audio tester initialized successfully")
        except Exception as e:
            logger.error(f"Failed to initialize audio tester: {e}")
            raise

    async def create_test_audio_file(
        self, text: str, filename: str = "test_input.wav"
    ) -> str:
        """Create a test audio file with the given text."""
        try:
            # Create a simple test audio file (silence for now)
            # In a real implementation, this would use TTS to generate audio
            file_path = self.test_audio_dir / filename

            # Create a simple WAV file with silence
            from character_ai.core.config import Config

            cfg = Config()
            sample_rate = cfg.interaction.sample_rate
            duration = 2.0  # 2 seconds
            samples = int(sample_rate * duration)

            with wave.open(str(file_path), "w") as wav_file:
                wav_file.setnchannels(1)  # Mono
                wav_file.setsampwidth(2)  # 16-bit
                wav_file.setframerate(sample_rate)
                wav_file.writeframes(b"\x00" * samples * 2)  # Silence

            logger.info(f"Created test audio file: {file_path}")
            return str(file_path)

        except Exception as e:
            logger.error(f"Failed to create test audio file: {e}")
            raise

    async def test_with_audio_file(
        self, audio_file_path: str, character_name: str = "sparkle"
    ) -> Dict[str, Any]:
        """Test the system with an audio file."""
        try:
            # Set active character
            await self.engine.set_active_character(character_name)

            # Read audio file
            with open(audio_file_path, "rb") as f:
                audio_data = f.read()

            # Create AudioData object
            from character_ai.core.config import Config

            cfg = Config()
            audio = AudioData(
                data=audio_data,
                sample_rate=cfg.interaction.sample_rate,
                channels=cfg.interaction.channels,
                format="wav",
            )

            # Process audio
            result = await self.engine.process_realtime_audio(audio)

            return {
                "success": result.error is None,
                "response_text": result.text,
                "character": result.metadata.get("character", "Unknown"),
                "error": result.error,
                "metadata": result.metadata,
            }

        except Exception as e:
            logger.error(f"Failed to test with audio file: {e}")
            return {
                "success": False,
                "error": str(e),
                "response_text": None,
                "character": None,
                "metadata": {},
            }

    async def test_character_switching(self) -> Dict[str, Any]:
        """Test character switching functionality."""
        try:
            results = {}

            # Test each character
            characters = ["sparkle", "bumblebee", "flame"]

            for character in characters:
                success = await self.engine.set_active_character(character)
                if success:
                    character_info = await self.engine.get_character_info()
                    results[character] = {"success": True, "info": character_info}
                else:
                    results[character] = {
                        "success": False,
                        "error": f"Failed to set character {character}",
                    }

            return results

        except Exception as e:
            logger.error(f"Failed to test character switching: {e}")
            return {"error": str(e)}

    async def test_performance(self, num_tests: int = 5) -> Dict[str, Any]:
        """Test system performance."""
        try:
            import time

            # Create test audio file
            test_file = await self.create_test_audio_file("Hello, how are you?")

            latencies = []
            successes = 0

            for i in range(num_tests):
                start_time = time.time()
                result = await self.test_with_audio_file(test_file)
                latency = time.time() - start_time

                latencies.append(latency)
                if result["success"]:
                    successes += 1

            # Calculate metrics
            avg_latency = sum(latencies) / len(latencies)
            success_rate = successes / num_tests

            return {
                "average_latency": avg_latency,
                "success_rate": success_rate,
                "total_tests": num_tests,
                "successful_tests": successes,
                "latencies": latencies,
            }

        except Exception as e:
            logger.error(f"Failed to test performance: {e}")
            return {"error": str(e)}

    async def run_full_test_suite(self) -> Dict[str, Any]:
        """Run the complete test suite."""
        try:
            logger.info("Running full test suite...")

            results = {
                "character_switching": await self.test_character_switching(),
                "performance": await self.test_performance(),
                "audio_processing": {},
            }

            # Test audio processing with different characters
            test_file = await self.create_test_audio_file("Hello, what's your name?")

            for character in ["sparkle", "bumblebee", "flame"]:
                audio_result = await self.test_with_audio_file(test_file, character)
                results["audio_processing"][character] = audio_result

            # Get system health
            results["system_health"] = await self.engine.get_health_status()

            logger.info("Full test suite completed")
            return results

        except Exception as e:
            logger.error(f"Failed to run full test suite: {e}")
            return {"error": str(e)}

    def cleanup_test_files(self) -> None:
        """Clean up test audio files."""
        try:
            if self.test_audio_dir.exists():
                for file in self.test_audio_dir.glob("*.wav"):
                    file.unlink()
                logger.info("Cleaned up test audio files")
        except Exception as e:
            logger.error(f"Failed to cleanup test files: {e}")
