"""
Voice Pipeline Tester for Character AI Platform.

Provides comprehensive testing for the complete voice pipeline: STT → LLM → TTS
with franchise-based organization and auto-creation of directories.
"""

import logging
from pathlib import Path
from typing import Any, Dict, List, Optional

from character_ai.algorithms.conversational_ai.coqui_processor import CoquiProcessor
from character_ai.algorithms.conversational_ai.wav2vec2_processor import (
    Wav2Vec2Processor,
)
from character_ai.characters.schema_voice_manager import SchemaVoiceManager
from character_ai.core.config import Config
from character_ai.core.llm.config import LLMConfigManager
from character_ai.core.llm.factory import LLMFactory
from character_ai.core.llm.manager import OpenModelManager

logger = logging.getLogger(__name__)


class VoicePipelineTester:
    """Test the complete voice pipeline: STT → LLM → TTS with franchise organization."""

    def __init__(self, test_dir: str = "tests_dev"):
        self.test_dir = Path(test_dir)
        self.audio_samples_dir = self.test_dir / "audio_samples"

        # Create base directory if it doesn't exist
        self.audio_samples_dir.mkdir(parents=True, exist_ok=True)

        # Initialize components
        self._stt_processor: Optional[Wav2Vec2Processor] = None
        self._tts_processor: Optional[CoquiProcessor] = None
        self._voice_manager: Optional[SchemaVoiceManager] = None
        self._llm_factory: Optional[LLMFactory] = None

    def get_character_directories(
        self, character: str, franchise: str
    ) -> Dict[str, Path]:
        """Get character-specific directories, creating them if needed."""

        # Create franchise directory
        franchise_dir = self.audio_samples_dir / franchise
        franchise_dir.mkdir(parents=True, exist_ok=True)

        # Create character directory
        character_dir = franchise_dir / character
        character_dir.mkdir(parents=True, exist_ok=True)

        # Create input and output directories
        input_dir = character_dir / "input"
        output_dir = character_dir / "output"

        input_dir.mkdir(parents=True, exist_ok=True)
        output_dir.mkdir(parents=True, exist_ok=True)

        return {
            "franchise": franchise_dir,
            "character": character_dir,
            "input": input_dir,
            "output": output_dir,
        }

    def get_default_output_paths(
        self, character: str, franchise: str
    ) -> Dict[str, str]:
        """Get default output paths for a character."""
        dirs = self.get_character_directories(character, franchise)
        output_dir = dirs["output"]

        return {
            "stt": str(output_dir / f"{character}_stt_output.txt"),
            "llm": str(output_dir / f"{character}_llm_response.txt"),
            "tts": str(output_dir / f"{character}_response.wav"),
        }

    def list_available_inputs(self, character: str, franchise: str) -> List[str]:
        """List available input audio files for a character."""
        dirs = self.get_character_directories(character, franchise)
        input_dir = dirs["input"]

        # Find all .wav files in the input directory
        audio_files = []
        if input_dir.exists():
            for file_path in input_dir.glob("*.wav"):
                audio_files.append(file_path.name)

        return sorted(audio_files)

    def get_output_paths_for_input(
        self, input_file: str, character: str, franchise: str
    ) -> Dict[str, str]:
        """Get output paths for a specific input file."""
        dirs = self.get_character_directories(character, franchise)
        output_dir = dirs["output"]

        # Extract base name without extension
        base_name = Path(input_file).stem

        return {
            "stt": str(output_dir / f"{base_name}_stt_output.txt"),
            "llm": str(output_dir / f"{base_name}_llm_response.txt"),
            "tts": str(output_dir / f"{base_name}_response.wav"),
        }

    async def initialize_components(self) -> None:
        """Initialize STT, TTS, and LLM components."""
        try:
            # Initialize STT processor
            config = Config()
            self._stt_processor = Wav2Vec2Processor(config)
            await self._stt_processor.initialize()
            logger.info("STT processor initialized")

            # Initialize TTS processor using production component
            from character_ai.core.multilingual_audio import MultiLanguageAudioManager

            self._tts_manager = MultiLanguageAudioManager(config)
            await self._tts_manager.initialize()
            if self._tts_manager.tts_manager is None:
                raise RuntimeError("TTS manager not initialized")
            self._tts_processor = self._tts_manager.tts_manager.tts_processor
            logger.info("TTS processor initialized")

            # Initialize voice manager
            self._voice_manager = SchemaVoiceManager()
            logger.info("Voice manager initialized")

            # Test voice manager
            try:
                test_voice_info = await self._voice_manager.get_character_voice_info(
                    "data"
                )
                logger.info(f"Test voice info for 'data': {test_voice_info}")
            except Exception as e:
                logger.error(f"Error testing voice manager: {e}")

            # Initialize LLM factory
            config_manager = LLMConfigManager()
            model_manager = OpenModelManager()
            self._llm_factory = LLMFactory(config_manager, model_manager)
            logger.info("LLM factory initialized")

        except Exception as e:
            logger.error(f"Failed to initialize components: {e}")
            raise

    async def test_stt_only(
        self,
        input_file: str,
        character: str,
        franchise: str,
        output_file: Optional[str] = None,
    ) -> Dict[str, Any]:
        """Test STT component only."""
        if not self._stt_processor:
            await self.initialize_components()

        if output_file is None:
            # Use input file name for output file naming
            if Path(input_file).is_absolute() or "/" in input_file:
                # Full path provided, extract filename
                input_filename = Path(input_file).name
            else:
                # Just filename provided
                input_filename = input_file
            output_paths = self.get_output_paths_for_input(
                input_filename, character, franchise
            )
            output_file = output_paths["stt"]

        try:
            # Process audio file with STT
            if self._stt_processor is None:
                raise RuntimeError("STT processor not initialized")

            # Load audio file
            import soundfile as sf

            from character_ai.core.protocols import AudioData

            audio_data, sample_rate = sf.read(input_file)
            audio = AudioData(
                data=audio_data,
                sample_rate=sample_rate,
                duration=len(audio_data) / sample_rate,
                channels=1 if len(audio_data.shape) == 1 else audio_data.shape[1],
            )

            result = await self._stt_processor.process_audio(audio)
            transcription = result.text if result.text else "STT processing failed"

            # Save transcription to file
            with open(output_file, "w") as f:
                f.write(transcription or "")

            logger.info(f"STT transcription saved to: {output_file}")

            return {
                "success": True,
                "transcription": transcription,
                "output_file": output_file,
                "character": character,
            }

        except Exception as e:
            logger.error(f"STT test failed: {e}")
            return {"success": False, "error": str(e), "character": character}

    async def test_llm_only(
        self,
        input_text: str,
        character: str,
        franchise: str,
        output_file: Optional[str] = None,
        input_filename: Optional[str] = None,
    ) -> Dict[str, Any]:
        """Test LLM component only."""
        if not self._llm_factory:
            await self.initialize_components()

        if output_file is None:
            if input_filename:
                output_paths = self.get_output_paths_for_input(
                    input_filename, character, franchise
                )
                output_file = output_paths["llm"]
            else:
                output_paths = self.get_default_output_paths(character, franchise)
                output_file = output_paths["llm"]

        try:
            # Get LLM instance
            if self._llm_factory is None:
                raise RuntimeError("LLM factory not initialized")
            llm = self._llm_factory.get_runtime_llm()

            # Load character prompts
            character_dir = Path(f"configs/characters/{franchise}/{character}")
            prompts_file = character_dir / "prompts.yaml"

            system_prompt = ""
            if prompts_file.exists():
                import yaml

                with open(prompts_file, "r") as f:
                    prompts_config = yaml.safe_load(f)
                system_prompt = prompts_config.get("system_prompt", "")

            # Create full prompt
            if system_prompt:
                full_prompt = f"{system_prompt}\n\nUser: {input_text}\n{character}:"
            else:
                full_prompt = input_text

            # Generate response
            response = await llm.generate(full_prompt)

            # Save response to file
            with open(output_file, "w") as f:
                f.write(response)

            logger.info(f"LLM response saved to: {output_file}")

            return {
                "success": True,
                "response": response,
                "output_file": output_file,
                "character": character,
            }

        except Exception as e:
            logger.error(f"LLM test failed: {e}")
            return {"success": False, "error": str(e), "character": character}

    async def test_tts_only(
        self,
        input_text: str,
        character: str,
        franchise: str,
        output_file: Optional[str] = None,
        input_filename: Optional[str] = None,
    ) -> Dict[str, Any]:
        """Test TTS component only."""
        logger.info(
            f"TTS test called with text: '{input_text}', character: {character}, franchise: {franchise}"
        )
        if not self._tts_processor:
            logger.info("TTS processor not initialized, initializing components...")
            await self.initialize_components()

        if output_file is None:
            if input_filename:
                output_paths = self.get_output_paths_for_input(
                    input_filename, character, franchise
                )
                output_file = output_paths["tts"]
            else:
                output_paths = self.get_default_output_paths(character, franchise)
                output_file = output_paths["tts"]

        try:
            # Get character voice info
            if self._voice_manager:
                voice_info = await self._voice_manager.get_character_voice_info(
                    character
                )
                if not voice_info or not voice_info.get("available"):
                    logger.warning(
                        f"No voice available for character {character}, using default TTS"
                    )

            # Generate speech
            if self._tts_processor is None:
                raise RuntimeError("TTS processor not initialized")

            # Get character voice info for voice cloning (XTTS v2 requires a reference voice)
            voice_path = None
            if self._voice_manager:
                voice_info = await self._voice_manager.get_character_voice_info(
                    character
                )
                if voice_info and voice_info.get("available"):
                    # Use processed voice file for XTTS v2
                    voice_path = voice_info.get("processed_voice_path")
                    if not voice_path:
                        # Fallback to regular voice file
                        voice_path = voice_info.get("voice_file_path")
                    # Convert to absolute path
                    if voice_path and not Path(voice_path).is_absolute():
                        voice_path = str(Path(voice_path).resolve())
                    logger.info(f"Using cloned voice for {character}: {voice_path}")
                else:
                    logger.warning(
                        f"No voice available for character {character}, XTTS v2 requires a reference voice"
                    )
                    raise RuntimeError(
                        f"XTTS v2 requires a reference voice, but none found for {character}"
                    )

            # Synthesize speech
            logger.info(
                f"Synthesizing speech for text: '{input_text}' with voice_path: {voice_path}"
            )
            result = await self._tts_processor.synthesize_speech(
                text=input_text, voice_path=voice_path
            )

            logger.info(f"TTS synthesis result: {result}")
            logger.info(f"Audio data type: {type(result.audio_data)}")
            logger.info(
                f"Audio data length: {len(result.audio_data.data) if result.audio_data else 'None'}"
            )

            # Save audio to file
            if result.audio_data:
                with open(output_file, "wb") as f:
                    f.write(result.audio_data.data)
                logger.info(f"TTS output saved to: {output_file}")
            else:
                raise RuntimeError("TTS synthesis failed - no audio data generated")

            return {"success": True, "output_file": output_file, "character": character}

        except Exception as e:
            logger.error(f"TTS test failed: {e}")
            return {"success": False, "error": str(e), "character": character}

    async def test_full_pipeline(
        self,
        input_file: str,
        character: str,
        franchise: str,
        output_dir: Optional[str] = None,
    ) -> Dict[str, Any]:
        """Test complete pipeline with all intermediate outputs."""
        try:
            # Get output paths
            if output_dir is None:
                # Use input file name for output file naming
                if Path(input_file).is_absolute() or "/" in input_file:
                    # Full path provided, extract filename
                    input_filename = Path(input_file).name
                else:
                    # Just filename provided
                    input_filename = input_file
                output_paths = self.get_output_paths_for_input(
                    input_filename, character, franchise
                )
            else:
                # Use character name for output file naming (legacy behavior)
                output_paths = {
                    "stt": f"{output_dir}/{character}_stt_output.txt",
                    "llm": f"{output_dir}/{character}_llm_response.txt",
                    "tts": f"{output_dir}/{character}_response.wav",
                }

            # Test STT
            logger.info("Testing STT component...")
            stt_result = await self.test_stt_only(
                input_file, character, franchise, output_paths["stt"]
            )

            if not stt_result["success"]:
                return {
                    "success": False,
                    "error": "STT failed",
                    "stt_result": stt_result,
                }

            # Test LLM
            logger.info("Testing LLM component...")
            input_filename = Path(input_file).name if "/" in input_file else input_file
            llm_result = await self.test_llm_only(
                stt_result["transcription"],
                character,
                franchise,
                output_paths["llm"],
                input_filename,
            )

            if not llm_result["success"]:
                return {
                    "success": False,
                    "error": "LLM failed",
                    "stt_result": stt_result,
                    "llm_result": llm_result,
                }

            # Test TTS
            logger.info("Testing TTS component...")
            tts_result = await self.test_tts_only(
                llm_result["response"],
                character,
                franchise,
                output_paths["tts"],
                input_filename,
            )

            if not tts_result["success"]:
                return {
                    "success": False,
                    "error": "TTS failed",
                    "stt_result": stt_result,
                    "llm_result": llm_result,
                    "tts_result": tts_result,
                }

            # All components successful
            logger.info("Full pipeline test completed successfully")
            return {
                "success": True,
                "stt_result": stt_result,
                "llm_result": llm_result,
                "tts_result": tts_result,
                "output_paths": output_paths,
            }

        except Exception as e:
            logger.error(f"Full pipeline test failed: {e}")
            return {"success": False, "error": str(e), "character": character}

    async def test_all_inputs(self, character: str, franchise: str) -> Dict[str, Any]:
        """Test all available input files for a character."""
        try:
            # Get list of available input files
            input_files = self.list_available_inputs(character, franchise)

            if not input_files:
                return {
                    "success": False,
                    "error": f"No input files found for character {character} in franchise {franchise}",
                    "character": character,
                }

            results = {}
            successful_tests = 0

            for input_file in input_files:
                logger.info(f"Testing input file: {input_file}")

                # Get full path to input file
                dirs = self.get_character_directories(character, franchise)
                input_path = dirs["input"] / input_file

                # Run full pipeline test
                result = await self.test_full_pipeline(
                    str(input_path), character, franchise
                )
                results[input_file] = result

                if result.get("success"):
                    successful_tests += 1
                    logger.info(f"✅ Test passed for {input_file}")
                else:
                    logger.error(
                        f"❌ Test failed for {input_file}: {result.get('error')}"
                    )

            return {
                "success": successful_tests > 0,
                "total_tests": len(input_files),
                "successful_tests": successful_tests,
                "failed_tests": len(input_files) - successful_tests,
                "results": results,
                "character": character,
            }

        except Exception as e:
            logger.error(f"Failed to test all inputs: {e}")
            return {"success": False, "error": str(e), "character": character}

    async def cleanup_test_files(self, character: str, franchise: str) -> None:
        """Clean up test files for a character."""
        try:
            dirs = self.get_character_directories(character, franchise)
            output_dir = dirs["output"]

            # Remove output files
            for file_path in output_dir.glob("*"):
                if file_path.is_file():
                    file_path.unlink()

            logger.info(f"Cleaned up test files for character: {character}")

        except Exception as e:
            logger.error(f"Failed to cleanup test files: {e}")

    async def get_test_summary(self, character: str, franchise: str) -> Dict[str, Any]:
        """Get summary of test results for a character."""
        try:
            dirs = self.get_character_directories(character, franchise)
            output_dir = dirs["output"]

            files_list: List[Dict[str, Any]] = []
            summary = {
                "character": character,
                "franchise": franchise,
                "output_directory": str(output_dir),
                "files": files_list,
            }

            # List all output files
            for file_path in output_dir.glob("*"):
                if file_path.is_file():
                    files_list.append(
                        {
                            "name": file_path.name,
                            "size": file_path.stat().st_size,
                            "modified": file_path.stat().st_mtime,
                        }
                    )

            return summary

        except Exception as e:
            logger.error(f"Failed to get test summary: {e}")
            return {"error": str(e)}
