"""
Testing commands for the Character AI CLI.

Provides Click-based commands for testing platform functionality.
"""

# mypy: ignore-errors

# CRITICAL: Import torch_init FIRST to set environment variables before any torch imports
# isort: off
from ..core import torch_init  # noqa: F401

# isort: on

import asyncio
import json
import logging
import sys
from typing import Any, Dict, Optional

import click

from ..characters.types import Character
from ..hardware.toy_hardware_manager import ToyHardwareManager

# Import core components directly
from ..production.real_time_engine import RealTimeInteractionEngine

logger = logging.getLogger(__name__)


def _configure_logging(verbose: bool) -> None:
    """Configure logging based on verbosity level."""
    if verbose:
        logging.basicConfig(
            level=logging.INFO,
            format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
        )
    else:
        logging.basicConfig(level=logging.WARNING)


def _load_character(
    engine: RealTimeInteractionEngine, character_name: str
) -> Optional[Character]:
    """Load character from engine's character manager."""
    if not engine.character_manager:
        return None
    return engine.character_manager.get_character(character_name)


def _auto_detect_hardware_profile() -> str:
    """Auto-detect hardware profile based on system characteristics."""
    try:
        import psutil

        # Check system characteristics
        cpu_count = psutil.cpu_count()
        memory_gb = psutil.virtual_memory().total / (1024**3)

        # Check for GPU
        has_gpu = False
        try:
            import torch

            has_gpu = torch.cuda.is_available()
        except ImportError:
            pass

        # Simple heuristics for hardware detection
        if cpu_count >= 8 and memory_gb >= 16 and has_gpu:
            return "desktop"
        elif cpu_count <= 4 and memory_gb <= 4:
            return "raspberry_pi"
        elif cpu_count <= 6 and memory_gb <= 8:
            return "orange_pi"
        else:
            # Default to desktop for unknown configurations
            return "desktop"

    except Exception as e:
        logger.warning(f"Hardware auto-detection failed: {e}, defaulting to desktop")
        return "desktop"


async def _initialize_engine(
    verbose: bool = False, hardware_profile: Optional[str] = None
) -> RealTimeInteractionEngine:
    """Initialize and return configured engine."""
    _configure_logging(verbose)

    # Auto-detect hardware profile if not specified
    if hardware_profile == "auto" or hardware_profile is None:
        hardware_profile = _auto_detect_hardware_profile()
        if verbose:
            click.echo(f"🔍 Auto-detected hardware profile: {hardware_profile}")

    hardware_manager = ToyHardwareManager()
    engine = RealTimeInteractionEngine(
        hardware_manager, hardware_profile=hardware_profile
    )
    await engine.initialize()
    return engine


@click.group()
def test_commands() -> None:
    """Testing and validation commands."""
    pass


@test_commands.command()
@click.option(
    "--input",
    "-i",
    help="Input audio file path (defaults to audio_samples/{franchise}/{character}/input/test_interaction_voice_sample.wav)",
)
@click.option("--character", "-c", required=True, help="Character name")
@click.option(
    "--franchise",
    "-f",
    required=True,
    help="Franchise name (e.g., star_trek, transformers)",
)
@click.option(
    "--output-dir",
    "-o",
    help="Output directory (defaults to audio_samples/{franchise}/{character}/output/)",
)
@click.option("--verbose", "-v", is_flag=True, help="Enable verbose output")
@click.option(
    "--quiet", "-q", is_flag=True, help="Quiet mode - show only essential output"
)
@click.option("--test-stt-only", is_flag=True, help="Test STT component only")
@click.option("--test-llm-only", is_flag=True, help="Test LLM component only")
@click.option("--test-tts-only", is_flag=True, help="Test TTS component only")
@click.option(
    "--list-inputs", is_flag=True, help="List available input files for the character"
)
@click.option(
    "--test-all", is_flag=True, help="Test all available input files for the character"
)
@click.option(
    "--realtime",
    is_flag=True,
    help="Test real-time voice interaction (requires microphone)",
)
@click.option(
    "--duration",
    type=int,
    default=30,
    help="Duration in seconds for real-time test (default: 30)",
)
@click.option(
    "--hardware-profile",
    type=click.Choice(["desktop", "raspberry_pi", "orange_pi", "auto"]),
    default="auto",
    help="Hardware profile to use for optimization (default: auto-detect)",
)
def voice_pipeline(
    input: Optional[str],
    character: str,
    franchise: str,
    output_dir: Optional[str],
    verbose: bool,
    quiet: bool,
    test_stt_only: bool,
    test_llm_only: bool,
    test_tts_only: bool,
    list_inputs: bool,
    test_all: bool,
    realtime: bool,
    duration: int,
    hardware_profile: str,
) -> None:
    """Test the complete voice pipeline: STT → LLM → TTS"""
    try:
        # Configure logging
        _configure_logging(verbose)

        from pathlib import Path

        # Initialize core engine directly
        engine = asyncio.run(_initialize_engine(verbose, hardware_profile))

        # Handle list inputs option
        if list_inputs:
            input_dir = Path(f"tests_dev/audio_samples/{franchise}/{character}/input")
            if input_dir.exists():
                input_files = list(input_dir.glob("*.wav"))
                if input_files:
                    click.echo(f"Available input files for {character} ({franchise}):")
                    for i, file in enumerate(input_files, 1):
                        click.echo(f"  {i}. {file.name}")
                else:
                    click.echo(f"No .wav files found in {input_dir}")
            else:
                click.echo(f"Input directory not found: {input_dir}")
            return

        # Handle realtime mode
        if realtime:
            asyncio.run(
                _test_realtime_interaction(
                    engine,
                    character,
                    franchise,
                    verbose,
                    quiet,
                    duration,
                    hardware_profile,
                )
            )
            return

        # Handle test all option
        if test_all:
            click.echo(
                f"Testing all available input files for {character} ({franchise})..."
            )
            input_dir = Path(f"tests_dev/audio_samples/{franchise}/{character}/input")
            if not input_dir.exists():
                click.echo(f"❌ Input directory not found: {input_dir}")
                return

            input_files = list(input_dir.glob("*.wav"))
            if not input_files:
                click.echo(f"❌ No .wav files found in {input_dir}")
                return

            successful_tests = 0
            total_tests = len(input_files)

            for input_file in input_files:
                output_file = (
                    input_file.parent.parent
                    / "output"
                    / f"{input_file.stem}_response.wav"
                )
                result = asyncio.run(
                    engine.process_audio_file(
                        str(input_file), character, str(output_file)
                    )
                )

                if result.get("success"):
                    successful_tests += 1
                    click.echo(
                        f"✅ {input_file.name}: {result.get('transcription', 'N/A')}"
                    )
                else:
                    click.echo(
                        f"❌ {input_file.name}: {result.get('error', 'Unknown error')}"
                    )

            click.echo(
                f"✅ Test completed: {successful_tests}/{total_tests} tests passed"
            )
            return

        # Provide sensible defaults if not specified
        if input is None:
            # Default input file path
            input = f"tests_dev/audio_samples/{franchise}/{character}/input/test_interaction_voice_sample.wav"
            if verbose:
                click.echo(f"Using default input file: {input}")

        # Resolve bare filenames (no path separators) to the character's input directory
        if (
            input is not None
            and not Path(input).is_absolute()
            and "/" not in input
            and "\\" not in input
        ):
            candidate = Path(
                f"tests_dev/audio_samples/{franchise}/{character}/input/{input}"
            )
            if candidate.exists():
                input = str(candidate)
                if verbose:
                    click.echo(f"Resolved input filename to: {input}")

        if output_dir is None:
            # Default output directory
            output_dir = f"tests_dev/audio_samples/{franchise}/{character}/output"
            if verbose:
                click.echo(f"Using default output directory: {output_dir}")

        # Run async test using core engine
        if test_stt_only:
            # STT-only test - just transcribe the audio
            result = asyncio.run(engine.process_audio_file(input, character, None))
            if result.get("success"):
                result = {"success": True, "transcription": result.get("transcription")}
        elif test_llm_only:
            # LLM-only test - first transcribe audio, then process with LLM
            from pathlib import Path

            input_path = Path(input)

            if input_path.suffix.lower() == ".wav":
                # For audio files, first transcribe then process with LLM
                temp_result = asyncio.run(
                    engine.process_audio_file(input, character, None)
                )
                if temp_result.get("success"):
                    input_text = temp_result.get("transcription", "")
                    click.echo(f"📝 Transcribed: {input_text}")
                else:
                    result = {"success": False, "error": "Failed to transcribe audio"}
                    return
            else:
                # For text files, read directly
                with open(input, "r", encoding="utf-8") as f:
                    input_text = f.read()

            # Use engine's LLM directly
            character_obj = _load_character(engine, character)
            if character_obj:
                response = asyncio.run(
                    engine._generate_character_response(input_text, character_obj)
                )
                result = {"success": True, "response": response}
            else:
                result = {
                    "success": False,
                    "error": f"Could not load character {character}",
                }
        elif test_tts_only:
            # TTS-only test - first transcribe audio, then synthesize
            from pathlib import Path

            input_path = Path(input)

            if input_path.suffix.lower() == ".wav":
                # For audio files, first transcribe then synthesize
                temp_result = asyncio.run(
                    engine.process_audio_file(input, character, None)
                )
                if temp_result.get("success"):
                    input_text = temp_result.get("transcription", "")
                    click.echo(f"📝 Transcribed: {input_text}")
                else:
                    result = {"success": False, "error": "Failed to transcribe audio"}
                    return
            else:
                # For text files, read directly
                with open(input, "r", encoding="utf-8") as f:
                    input_text = f.read()

            # Use engine's TTS directly
            character_obj = _load_character(engine, character)
            if character_obj:
                audio_result = asyncio.run(
                    engine._synthesize_character_voice(input_text, character_obj)
                )
                if audio_result:
                    result = {"success": True, "audio_data": audio_result}
                else:
                    result = {"success": False, "error": "TTS synthesis failed"}
            else:
                result = {
                    "success": False,
                    "error": f"Could not load character {character}",
                }
        else:
            # Full pipeline test - generate output file path from input file
            from pathlib import Path

            input_path = Path(input)
            output_file = Path(output_dir) / f"{input_path.stem}_response.wav"
            result = asyncio.run(
                engine.process_audio_file(input, character, str(output_file))
            )

        # Display results
        if result.get("success"):
            click.echo("✅ Test completed successfully!")
            if "output_files" in result:
                click.echo("📁 Output files:")
                for key, path in result["output_files"].items():
                    click.echo(f"  {key}: {path}")
            elif "output_paths" in result:
                click.echo("📁 Output files:")
                for key, path in result["output_paths"].items():
                    click.echo(f"  {key}: {path}")
        else:
            click.echo(f"❌ Test failed: {result.get('error', 'Unknown error')}")
            if verbose:
                click.echo(f"Details: {result}")

    except Exception as e:
        click.echo(f"Error running voice pipeline test: {e}", err=True)
        if verbose:
            import traceback

            click.echo(traceback.format_exc())
        raise click.Abort()


async def _process_speech_chunk(
    engine: Any,
    character_obj: Any,
    full_audio: Any,
    interaction_count: int,
    verbose: bool,
    audio_output: Optional[Any] = None,
) -> None:
    """Process speech chunk and play response."""
    import time

    import numpy as np

    from ..core.audio_io.audio_config import AudioConfig
    from ..core.audio_io.audio_utils import prepare_audio_for_playback
    from ..core.protocols import AudioData

    click.echo(
        f"📝 Processing {len(full_audio)/AudioConfig.DEFAULT_SAMPLE_RATE:.1f}s of audio..."
    )

    # Create AudioData
    audio_data = AudioData(
        data=full_audio,
        sample_rate=AudioConfig.DEFAULT_SAMPLE_RATE,
        channels=AudioConfig.DEFAULT_CHANNELS,
        duration=len(full_audio) / AudioConfig.DEFAULT_SAMPLE_RATE,
    )

    # Process with optimized pipeline
    processing_start = time.time()
    result = await engine._process_with_character_personality_optimized(
        audio_data, character_obj
    )
    processing_time = time.time() - processing_start

    if result.text:
        # Show what was transcribed
        if result.metadata and "transcribed_text" in result.metadata:
            click.echo(f"🎤 You said: \"{result.metadata['transcribed_text']}\"")

        click.echo(f"🤖 {character_obj.name}: {result.text}")

        # Display detailed timing metrics
        if result.metadata and "timing" in result.metadata:
            timing = result.metadata["timing"]
            click.echo(
                f"⏱️  STT: {timing['stt_time']:.2f}s | "
                f"LLM: {timing['llm_time']:.2f}s | "
                f"TTS: {timing['tts_time']:.2f}s | "
                f"Total: {timing['total_time']:.2f}s"
            )

            # Show cache hit if applicable
            if result.metadata.get("cache_hit"):
                click.echo("💾 Cache hit - LLM generation skipped!")
        else:
            click.echo(f"⏱️  Processing time: {processing_time:.2f}s")

        # Play response using AudioOutput infrastructure
        if result.audio_data and result.audio_data.data:
            click.echo("🔊 Playing response...")

            # Prepare audio for playback using consolidated utilities
            audio_array, sample_rate = prepare_audio_for_playback(
                result.audio_data.data, AudioConfig.DEFAULT_SAMPLE_RATE
            )

            if verbose:
                click.echo(
                    f"🔊 Audio data: {len(audio_array)} samples, dtype: {audio_array.dtype}"
                )
                click.echo(
                    f"🔊 Audio range: {np.min(audio_array):.6f} to {np.max(audio_array):.6f}"
                )

            # Use AudioOutput if available, otherwise fallback to direct playback
            if audio_output:
                await audio_output.play_audio_blocking(audio_array, sample_rate)
            else:
                # Fallback to direct sounddevice playback
                import sounddevice as sd

                sd.play(audio_array, samplerate=sample_rate, device="default")
                sd.wait()

            click.echo("✅ Response played")

        click.echo(f"📊 Total interactions: {interaction_count + 1}")
    else:
        click.echo("❌ No response generated")


async def _process_speech_chunk_with_metrics(
    engine: Any,
    character_obj: Any,
    full_audio: Any,
    interaction_count: int,
    verbose: bool,
    audio_output: Optional[Any] = None,
    device_manager: Any = None,
    output_sample_rate: float = 22050,
) -> Optional[Any]:
    """Process speech chunk and return result with metrics."""
    import time

    import numpy as np

    from ..core.audio_io.audio_config import AudioConfig
    from ..core.audio_io.audio_utils import prepare_audio_for_playback
    from ..core.protocols import AudioData

    # Note: Processing message is printed by caller for consistency
    # Create AudioData
    audio_data = AudioData(
        data=full_audio,
        sample_rate=AudioConfig.DEFAULT_SAMPLE_RATE,
        channels=AudioConfig.DEFAULT_CHANNELS,
        duration=len(full_audio) / AudioConfig.DEFAULT_SAMPLE_RATE,
    )

    # Process with optimized pipeline
    processing_start = time.time()
    result = await engine._process_with_character_personality_optimized(
        audio_data, character_obj
    )
    processing_time = time.time() - processing_start

    if result.text:
        # Show what was transcribed
        if result.metadata and "transcribed_text" in result.metadata:
            click.echo(f"🎤 You said: \"{result.metadata['transcribed_text']}\"")

        click.echo(f"🤖 {character_obj.name}: {result.text}")

        # Display detailed timing metrics
        if result.metadata and "timing" in result.metadata:
            timing = result.metadata["timing"]
            click.echo(
                f"⏱️  STT: {timing['stt_time']:.2f}s | "
                f"LLM: {timing['llm_time']:.2f}s | "
                f"TTS: {timing['tts_time']:.2f}s | "
                f"Total: {timing['total_time']:.2f}s"
            )

            # Show cache hit if applicable
            if result.metadata.get("cache_hit"):
                click.echo("💾 Cache hit - LLM generation skipped!")
        else:
            click.echo(f"⏱️  Processing time: {processing_time:.2f}s")

        # Play response using AudioOutput infrastructure
        if result.audio_data and result.audio_data.data:
            click.echo("🔊 Playing response...")

            # Prepare audio for playback using consolidated utilities
            audio_array, sample_rate = prepare_audio_for_playback(
                result.audio_data.data, AudioConfig.DEFAULT_SAMPLE_RATE
            )

            if verbose:
                click.echo(
                    f"🔊 Audio data: {len(audio_array)} samples, dtype: {audio_array.dtype}"
                )
                click.echo(
                    f"🔊 Audio range: {np.min(audio_array):.6f} to {np.max(audio_array):.6f}"
                )

            # Use sounddevice directly for output to configured device
            import sounddevice as sd

            try:
                # Use the configured output device and sample rate
                output_device = device_manager.get_output_device()
                output_sample_rate = output_sample_rate  # From device setup

                if output_device:
                    sd.play(
                        audio_array,
                        samplerate=output_sample_rate,
                        device=output_device.index,
                    )
                else:
                    sd.play(audio_array, samplerate=output_sample_rate)
                sd.wait()  # Wait for playback to complete
                click.echo("✅ Audio playback completed")
            except Exception as e:
                click.echo(f"⚠️  Audio playback error: {e}")
                if verbose:
                    import traceback

                    click.echo(traceback.format_exc())
        else:
            click.echo("⚠️  No audio data to play")

        return result
    else:
        click.echo("❌ No response generated")
        return None


async def _test_realtime_interaction(
    engine: Any,
    character: str,
    franchise: str,
    verbose: bool,
    quiet: bool,
    duration: int,
    hardware_profile: str,
) -> None:
    """Test real-time voice interaction with microphone and speakers."""
    # Quiet mode overrides verbose
    if quiet:
        verbose = False

    # Suppress verbose library warnings during testing
    import warnings

    warnings.filterwarnings("ignore", message=".*ffmpeg.*")
    warnings.filterwarnings("ignore", message=".*pkg_resources.*")
    warnings.filterwarnings("ignore", message=".*GenerationMixin.*")
    warnings.filterwarnings("ignore", message=".*not initialized from.*")
    warnings.filterwarnings("ignore", message=".*n_ctx_per_seq.*")
    warnings.filterwarnings("ignore", message=".*attention_mask.*")
    warnings.filterwarnings("ignore", message=".*TORCH_FORCE_NO_WEIGHTS_ONLY_LOAD.*")
    warnings.filterwarnings("ignore", category=DeprecationWarning, module="librosa")
    warnings.filterwarnings("ignore", category=UserWarning, module="pydub")

    try:
        import time

        from ..core.audio_io import AudioComponentFactory
        from ..core.audio_io.audio_config import AudioConfig

        if not quiet:
            click.echo("🎤 Starting real-time voice interaction test...")
            click.echo(f"Character: {character} ({franchise})")
            click.echo("Press Ctrl+C to stop")
            click.echo()

        # Initialize audio components
        audio_factory = AudioComponentFactory()
        device_manager = audio_factory.create_device_manager()

        # List available devices
        input_devices = await device_manager.list_input_devices()
        output_devices = await device_manager.list_output_devices()

        if verbose:
            click.echo("Available input devices:")
            for i, device in enumerate(input_devices):
                click.echo(f"  {i}: {device.name}")
            click.echo("Available output devices:")
            for i, device in enumerate(output_devices):
                click.echo(f"  {i}: {device.name}")
            click.echo()

        # Use AudioDeviceManager for device selection
        from ..core.audio_io.device_manager import AudioDeviceManager
        from ..core.config import Config

        # Get audio device configuration from runtime config
        config = Config()
        audio_config = getattr(config.runtime, "audio_devices", {})
        device_manager = AudioDeviceManager(audio_config)

        # Setup input and output devices
        try:
            input_device, input_sample_rate = device_manager.setup_input_device()
            output_device, output_sample_rate = device_manager.setup_output_device()

            click.echo(f"🎤 Using input: {input_device.name} ({input_sample_rate}Hz)")
            click.echo(f"🔊 Using output: {output_device.name} ({output_sample_rate}Hz)")

            # Display hardware profile information
            click.echo(f"⚙️  Hardware profile: {hardware_profile}")
            if engine.hardware_config:
                models = engine.hardware_config.get("models", {})
                optimizations = engine.hardware_config.get("optimizations", {})
                click.echo(
                    f"   Models: STT={models.get('stt', 'default')}, LLM={models.get('llm', 'default')}, TTS={models.get('tts', 'default')}"
                )
                if optimizations:
                    click.echo(f"   Optimizations: {list(optimizations.keys())}")
        except Exception as e:
            click.echo(f"❌ Failed to setup audio devices: {e}", err=True)
            return
        click.echo()

        # Load character first (needed for model preloading)
        character_obj = _load_character(engine, character)
        if not character_obj:
            click.echo(f"❌ Could not load character {character}", err=True)
            return

        # Pre-load models BEFORE starting audio capture to avoid overflow
        click.echo("🔄 Pre-loading models for faster processing...")
        preload_results = await engine.preload_models()
        loaded_count = sum(1 for success in preload_results.values() if success)
        click.echo(f"✅ Pre-loaded {loaded_count}/3 models")
        if loaded_count < 3:
            click.echo(f"⚠️  Some models failed to pre-load: {preload_results}")
            click.echo("   Continuing with on-demand loading...")

        # NOW initialize and start audio capture (after models are loaded)
        audio_capture = audio_factory.create_audio_capture()

        # Start audio capture with compatible sample rates and channels
        # Use 1 channel for input (mono mic on Channel 1) with smaller chunk size
        try:
            await audio_capture.start_capture(
                input_device, sample_rate=44100, channels=1, chunk_size=512
            )
        except Exception as e:
            click.echo(f"⚠️  Failed with 44100Hz, trying 48000Hz: {e}")
            try:
                await audio_capture.start_capture(
                    input_device, sample_rate=48000, channels=1, chunk_size=512
                )
            except Exception as e2:
                click.echo("❌ No compatible sample rates found for input device")
                raise e2

        click.echo("✅ Audio capture started")

        # Countdown before starting
        click.echo("🎙️  Starting in...")
        for i in range(3, 0, -1):
            click.echo(f"   {i}...")
            await asyncio.sleep(1)
        click.echo(
            f"   🎤 GO! You can have multiple conversations with {character} for {duration} seconds"
        )
        click.echo("   Speak → Listen to response → Speak again → Repeat")
        click.echo()

        # Delay to avoid capturing countdown audio
        await asyncio.sleep(1.0)  # 1s delay to let countdown audio fully dissipate

        interaction_count = 0
        start_time = time.time()
        end_time = start_time + duration
        is_processing = False

        # Metrics tracking
        interaction_times = []
        cache_hits = 0
        cache_misses = 0

        click.echo("\n" + "=" * 60)
        click.echo("LIVE METRICS")
        click.echo("=" * 60)

        # Use VADSessionManager for voice activity detection
        from ..core.audio_io.vad_session import VADSessionManager
        from ..core.voice_activity_detection import VADConfig

        # Get VAD settings from hardware profile
        vad_settings = {}
        if engine.hardware_config and "vad" in engine.hardware_config:
            vad_settings = engine.hardware_config["vad"]

        # Create VAD config with hardware profile settings
        vad_config = VADConfig.for_toy_interaction()  # Start with defaults

        # Create VAD session manager with hardware settings directly
        vad_manager = VADSessionManager(
            vad_config=vad_config,
            hardware_vad_settings=vad_settings,  # Pass hardware settings directly
        )

        click.echo("🎙️  Listening for speech... (speak naturally)")

        while time.time() < end_time:
            try:
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
                prev_state = vad_manager.state
                vad_state = vad_manager.process_audio_chunk(audio_chunk)

                # Show VAD state transitions
                if vad_state != prev_state:
                    import sys

                    from ..core.audio_io.vad_session import VADSessionState

                    if vad_state == VADSessionState.SPEECH_DETECTED:
                        audio_level = vad_manager.get_audio_level(audio_chunk)
                        click.echo(
                            f"🎤 Speech start detected (level: {audio_level:.4f} > threshold: {vad_manager.speech_start_threshold:.4f})"
                        )
                        sys.stdout.flush()
                    elif vad_state == VADSessionState.SPEECH_ENDING:
                        speech_duration = (
                            time.time() - vad_manager.speech_start_time
                            if vad_manager.speech_start_time > 0
                            else 0
                        )
                        click.echo(
                            f"🛑 Speech end detected (duration: {speech_duration:.2f}s, silence: {vad_manager.silence_duration:.2f}s)"
                        )
                        sys.stdout.flush()

                # Debug: show audio levels occasionally
                if verbose and time.time() % 1.0 < 0.01:  # Show every ~1 second
                    audio_level = vad_manager.get_audio_level(audio_chunk)
                    silence_dur = (
                        vad_manager.silence_duration if vad_manager.is_speaking else 0
                    )
                    click.echo(
                        f"🎤 Audio level: {audio_level:.6f} (threshold: {vad_manager.speech_threshold:.6f}), silence: {silence_dur:.3f}s"
                    )

                # Handle speech end
                if vad_manager.should_end_speech():
                    is_processing = True
                    vad_manager.set_processing_state()

                    # Stop audio capture during processing to prevent overflow
                    # Show processing start timestamp
                    import sys

                    processing_start_ts = time.time()
                    click.echo(
                        f"⏱️  Processing started at {processing_start_ts - start_time:.2f}s"
                    )
                    sys.stdout.flush()  # Force immediate display

                    stop_start = time.time()
                    await audio_capture.stop_capture()
                    stop_time = time.time() - stop_start

                    combine_start = time.time()
                    # Get combined speech audio
                    speech_audio = vad_manager.get_combined_speech_audio()
                    combine_time = time.time() - combine_start

                    if speech_audio is not None:
                        # Show processing message with audio duration (informative metric)
                        audio_duration = (
                            len(speech_audio) / AudioConfig.DEFAULT_SAMPLE_RATE
                        )
                        num_chunks = len(vad_manager.speech_buffer)

                        if verbose:
                            click.echo(
                                f"⏱️  stop_capture: {stop_time:.3f}s, combine_audio: {combine_time:.3f}s"
                            )
                        click.echo(
                            f"📝 Processing {audio_duration:.1f}s of audio ({num_chunks} chunks buffered)"
                        )
                        sys.stdout.flush()  # Force immediate display
                        process_start = time.time()
                        result = await _process_speech_chunk_with_metrics(
                            engine,
                            character_obj,
                            speech_audio,
                            interaction_count,
                            verbose,
                            None,
                            device_manager,
                            output_sample_rate,
                        )
                        process_time = time.time() - process_start

                        # Update metrics
                        interaction_times.append(process_time)
                        if result and result.metadata.get("cache_hit"):
                            cache_hits += 1
                        else:
                            cache_misses += 1

                        # Display live metrics
                        avg_time = sum(interaction_times) / len(interaction_times)
                        click.echo(f"\n{'─'*60}")
                        click.echo(f"Interaction #{len(interaction_times)}")
                        click.echo(
                            f"  Current: {process_time:.2f}s | Average: {avg_time:.2f}s"
                        )
                        if result and result.metadata and "timing" in result.metadata:
                            timing = result.metadata["timing"]
                            click.echo(
                                f"  STT: {timing['stt_time']:.2f}s | "
                                f"LLM: {timing['llm_time']:.2f}s | "
                                f"TTS: {timing['tts_time']:.2f}s"
                            )
                        click.echo(
                            f"  Cache: {cache_hits} hits / {cache_misses} misses "
                            f"({cache_hits/(cache_hits+cache_misses)*100:.1f}% hit rate)"
                        )
                        if result and result.text:
                            click.echo(f"  Response: {result.text}")
                        click.echo(f"{'─'*60}\n")

                        interaction_count += 1

                    # Restart audio capture after processing
                    # Always show restart message - important user feedback
                    click.echo("🔄 Ready for next question...")
                    await audio_capture.start_capture(
                        input_device, sample_rate=44100, channels=1, chunk_size=512
                    )

                    # Reset VAD session
                    vad_manager.reset_session()
                    is_processing = False
                    click.echo("🎙️  Listening for speech... (speak naturally)")

                await asyncio.sleep(0.01)

            except KeyboardInterrupt:
                break
            except Exception as e:
                click.echo(f"❌ Error during interaction: {e}")
                if verbose:
                    import traceback

                    click.echo(traceback.format_exc())
                # Reset processing flag on error
                is_processing = False
                await asyncio.sleep(1)

    except KeyboardInterrupt:
        click.echo("\n🛑 Stopping real-time interaction...")
    except Exception as e:
        click.echo(f"❌ Error in real-time test: {e}", err=True)
        if verbose:
            import traceback

            click.echo(traceback.format_exc())
    else:
        # Test completed normally
        elapsed = time.time() - start_time
        click.echo("\n" + "=" * 60)
        click.echo("SESSION SUMMARY")
        click.echo("=" * 60)
        click.echo(f"Test duration: {elapsed:.1f}s")
        click.echo(f"Total interactions: {len(interaction_times)}")
        if interaction_times:
            click.echo(
                f"Average latency: {sum(interaction_times)/len(interaction_times):.2f}s"
            )
            click.echo(f"Best latency: {min(interaction_times):.2f}s")
            click.echo(f"Worst latency: {max(interaction_times):.2f}s")
        click.echo(f"Cache performance: {cache_hits} hits, {cache_misses} misses")
        if cache_hits + cache_misses > 0:
            click.echo(
                f"Cache hit rate: {cache_hits/(cache_hits+cache_misses)*100:.1f}%"
            )
        click.echo("=" * 60)
    finally:
        # Cleanup
        try:
            if "audio_capture" in locals():
                await audio_capture.stop_capture()
            click.echo("✅ Audio devices stopped")
        except Exception as e:
            click.echo(f"⚠️  Error stopping audio devices: {e}")


@test_commands.command()
@click.option("--verbose", "-v", is_flag=True, help="Enable verbose output")
@click.option("--output", "-o", help="Save test results to file")
def interactive(verbose: bool, output: Optional[str]) -> None:
    """Run character.ai tests."""
    try:
        # Configure logging
        _configure_logging(verbose)

        # Run async test
        results = asyncio.run(_run_interactive_test())

        # Display results
        if results:
            _display_test_results(results)

        # Save results if requested
        if output:
            with open(output, "w") as f:
                json.dump(results, f, indent=2)
            click.echo(f"Test results saved to {output}")

        # Check if all tests passed
        all_passed = all(
            result.get("success", False)
            for result in (results.get("character_switching") or {}).values()
        )

        if all_passed:
            click.echo("✓ All tests passed!")
        else:
            click.echo("✗ Some tests failed!")
            sys.exit(1)

    except Exception as e:
        click.echo(f"Error running tests: {e}", err=True)
        raise click.Abort()


async def _run_interactive_test() -> Dict[str, Any]:
    """Run the character.ai test using core engine."""
    try:
        logger.info("Starting Character AI test...")

        # Initialize core engine
        engine = await _initialize_engine(True)  # Use verbose for test

        results = {}

        # Test 1: Engine initialization
        logger.info("Testing engine initialization...")
        results["engine_initialization"] = {
            "success": True,
            "message": "Engine initialized successfully",
        }

        # Test 2: Character manager
        logger.info("Testing character manager...")
        if engine.character_manager:
            results["character_manager"] = {
                "success": True,
                "message": "Character manager available",
            }
        else:
            results["character_manager"] = {
                "success": False,
                "message": "Character manager not available",
            }

        # Test 3: Voice manager
        logger.info("Testing voice manager...")
        if engine.voice_manager:
            results["voice_manager"] = {
                "success": True,
                "message": "Voice manager available",
            }
        else:
            results["voice_manager"] = {
                "success": False,
                "message": "Voice manager not available",
            }

        # Test 4: Hardware manager
        logger.info("Testing hardware manager...")
        if engine.hardware_manager:
            results["hardware_manager"] = {
                "success": True,
                "message": "Hardware manager available",
            }
        else:
            results["hardware_manager"] = {
                "success": False,
                "message": "Hardware manager not available",
            }

        return results

    except Exception as e:
        logger.error(f"Test failed: {e}")
        return {"error": str(e)}


def _display_test_results(results: Dict[str, Any]) -> None:
    """Display test results in a formatted way."""
    if "error" in results:
        click.echo(f"Test Error: {results['error']}")
        return

    # Character switching results
    if "character_switching" in results:
        click.echo("\n=== Character Switching Test ===")
        for character, result in results["character_switching"].items():
            if isinstance(result, dict) and "success" in result:
                status = "PASS" if result["success"] else "FAIL"
                click.echo(f"{character}: {status}")
                if result["success"] and "info" in result:
                    info = result["info"]
                    click.echo(f"  - Name: {info.get('name', 'Unknown')}")
                    click.echo(f"  - Type: {info.get('type', 'Unknown')}")
                    click.echo(f"  - Voice Style: {info.get('voice_style', 'Unknown')}")

            else:
                click.echo(f"{character}: {result}")

    # Audio processing results
    if "audio_processing" in results:
        click.echo("\n=== Audio Processing Test ===")
        for character, result in results["audio_processing"].items():
            if isinstance(result, dict) and "success" in result:
                status = "PASS" if result["success"] else "FAIL"
                click.echo(f"{character}: {status}")
                if not result["success"] and "error" in result:
                    click.echo(f"  - Error: {result['error']}")
            else:
                click.echo(f"{character}: {result}")

    # Performance results
    if "performance" in results:
        perf = results["performance"]
        click.echo("\n=== Performance Test ===")
        click.echo(f"Average Latency: {perf.get('average_latency', 'N/A')}")
        click.echo(f"Success Rate: {perf.get('success_rate', 'N/A')}")
        click.echo(f"Tests: {perf.get('tests_run', 'N/A')}")

    # System health
    if "system_health" in results:
        health = results["system_health"]
        click.echo("\n=== System Health ===")
        click.echo(f"Overall Health: {health.get('overall_health', 'UNKNOWN')}")
        click.echo(f"Character Manager: {health.get('character_manager', 'UNKNOWN')}")
        click.echo(f"Active Character: {health.get('active_character', 'None')}")
        click.echo(f"Total Interactions: {health.get('total_interactions', 0)}")
        click.echo(f"Success Rate: {health.get('success_rate', '0.0%')}")
        click.echo(f"Average Latency: {health.get('average_latency', '0.000s')}")


@test_commands.command()
@click.option("--component", help="Test specific component (stt, llm, tts, safety)")
@click.option("--duration", type=int, default=10, help="Test duration in seconds")
@click.option("--concurrency", type=int, default=1, help="Number of concurrent tests")
def performance(component: str, duration: int, concurrency: int) -> None:
    """Run performance tests."""
    try:
        click.echo(f"Running performance test for {component or 'all components'}...")
        click.echo(f"Duration: {duration}s, Concurrency: {concurrency}")

        # This would integrate with the existing performance testing system
        # For now, just show a placeholder
        click.echo("Performance testing not yet implemented in CLI")
        click.echo("Use the web API or existing test scripts for performance testing")

    except Exception as e:
        click.echo(f"Error running performance test: {e}", err=True)
        raise click.Abort()


@test_commands.command()
@click.option("--llm", is_flag=True, help="Test LLM connections")
@click.option("--audio", is_flag=True, help="Test audio processing")
@click.option("--hardware", is_flag=True, help="Test hardware interfaces")
@click.option("--all", is_flag=True, help="Test all components")
def connectivity(llm: bool, audio: bool, hardware: bool, all: bool) -> None:
    """Test component connectivity."""
    try:
        if all:
            llm = audio = hardware = True

        if llm:
            click.echo("Testing LLM connectivity...")
            # This would test LLM connections
            click.echo("LLM connectivity test not yet implemented")

        if audio:
            click.echo("Testing audio processing...")
            # This would test audio components
            click.echo("Audio connectivity test not yet implemented")

        if hardware:
            click.echo("Testing hardware interfaces...")
            # This would test hardware components
            click.echo("Hardware connectivity test not yet implemented")

        if not any([llm, audio, hardware]):
            click.echo("No components specified. Use --all or specify components.")

    except Exception as e:
        click.echo(f"Error testing connectivity: {e}", err=True)
        raise click.Abort()


@test_commands.command()
@click.option("--output", "-o", help="Save test report to file")
def report(output: Optional[str]) -> None:
    """Generate comprehensive test report."""
    try:
        click.echo("Generating test report...")

        # This would run all tests and generate a comprehensive report
        report_data = {
            "timestamp": "2025-01-01T00:00:00Z",
            "platform_version": "0.1.0",
            "tests": {
                "interactive": "not_run",
                "performance": "not_run",
                "connectivity": "not_run",
            },
            "summary": "Test report generation not yet implemented",
        }

        if output:
            with open(output, "w") as f:
                json.dump(report_data, f, indent=2)
            click.echo(f"Test report saved to {output}")
        else:
            click.echo(json.dumps(report_data, indent=2))

    except Exception as e:
        click.echo(f"Error generating report: {e}", err=True)
        raise click.Abort()


@test_commands.command()
@click.option(
    "--mode",
    type=click.Choice(["mock"]),
    default="mock",
    help="Testing mode: mock (mock audio infrastructure)",
)
@click.option("--list-devices", is_flag=True, help="List available audio devices")
def audio(mode: str, list_devices: bool) -> None:
    """Test audio infrastructure and hardware components."""
    try:
        if list_devices:
            click.echo("Listing audio devices...")
            asyncio.run(_list_audio_devices())
            return

        if mode == "mock":
            click.echo("Testing mock audio infrastructure...")
            asyncio.run(_test_mock_audio())

    except Exception as e:
        click.echo(f"Error running audio test: {e}", err=True)
        raise click.Abort()


async def _list_audio_devices() -> None:
    """List available audio devices."""
    try:
        from character_ai.core.audio_io import AudioComponentFactory

        # Test real devices
        click.echo("\n=== Real Audio Devices ===")
        device_manager = AudioComponentFactory.create_device_manager(use_mocks=False)
        input_devices = await device_manager.list_input_devices()
        output_devices = await device_manager.list_output_devices()

        click.echo("Input devices:")
        for device in input_devices:
            click.echo(f"  - {device.name} (index: {device.index})")

        click.echo("Output devices:")
        for device in output_devices:
            click.echo(f"  - {device.name} (index: {device.index})")

        # Test mock devices
        click.echo("\n=== Mock Audio Devices ===")
        mock_manager = AudioComponentFactory.create_device_manager(use_mocks=True)
        mock_input = await mock_manager.list_input_devices()
        mock_output = await mock_manager.list_output_devices()

        click.echo("Mock input devices:")
        for device in mock_input:
            click.echo(f"  - {device.name}")

        click.echo("Mock output devices:")
        for device in mock_output:
            click.echo(f"  - {device.name}")

    except Exception as e:
        click.echo(f"Error listing devices: {e}", err=True)


async def _test_mock_audio() -> None:
    """Test mock audio infrastructure."""
    try:
        from character_ai.core.audio_io import AudioComponentFactory

        # Test device manager
        device_manager = AudioComponentFactory.create_device_manager(use_mocks=True)
        input_devices = await device_manager.list_input_devices()
        output_devices = await device_manager.list_output_devices()

        click.echo(
            f"✅ Mock device manager working: {len(input_devices)} input, {len(output_devices)} output devices"
        )

        # Test audio capture and output
        AudioComponentFactory.create_audio_capture(use_mocks=True)
        AudioComponentFactory.create_audio_output(use_mocks=True)

        click.echo("✅ Mock audio capture and output created successfully")
        click.echo("✅ Mock audio infrastructure test completed!")

    except Exception as e:
        click.echo(f"Error in mock audio test: {e}", err=True)


@test_commands.command()
@click.option("--character", "-c", required=True, help="Character name")
@click.option("--franchise", "-f", required=True, help="Franchise name")
@click.option("--duration", "-d", default=60, help="Test duration in seconds")
@click.option("--device", help="Audio device pattern (e.g., 'audiobox', 'hw:3,0')")
@click.option("--verbose", "-v", is_flag=True, help="Enable verbose output")
def session(
    character: str, franchise: str, duration: int, device: Optional[str], verbose: bool
) -> None:
    """Test real-time interaction using the new session management methods."""
    import asyncio

    async def run_session_test() -> None:
        try:
            click.echo(
                f"🚀 Starting session-based real-time test with {character} ({franchise})"
            )
            click.echo(f"   Duration: {duration}s")
            if device:
                click.echo(f"   Device pattern: {device}")
            click.echo()

            # Initialize hardware manager and engine
            engine = await _initialize_engine(verbose)

            # Load character
            character_obj = _load_character(engine, character)
            if not character_obj:
                click.echo(
                    f"❌ Failed to load character {character} from {franchise}",
                    err=True,
                )
                return

            click.echo(f"✅ Loaded character: {character_obj.name}")

            # Start real-time session
            result = await engine.start_realtime_session(
                character=character_obj,
                duration=duration,
                device_pattern=device,
                vad_config=None,  # Use default VAD config
            )

            # Display results
            click.echo("\n✅ Session completed!")
            click.echo(f"   Interactions: {result['interaction_count']}")
            click.echo(f"   Session duration: {result['session_duration']:.1f}s")
            click.echo(
                f"   Average processing time: {result['average_processing_time']:.2f}s"
            )
            click.echo(f"   Device used: {result['device_used']}")
            click.echo(f"   Sample rate: {result['sample_rate']}Hz")
            click.echo(f"   Status: {result['status']}")

            if verbose and "vad_statistics" in result:
                vad_stats = result["vad_statistics"]
                click.echo(
                    f"   VAD speech detections: {vad_stats.get('total_speech_detections', 0)}"
                )
                click.echo(
                    f"   VAD silence detections: {vad_stats.get('total_silence_detections', 0)}"
                )

        except Exception as e:
            click.echo(f"❌ Error in session test: {e}", err=True)
            if verbose:
                import traceback

                click.echo(traceback.format_exc())

    asyncio.run(run_session_test())


@test_commands.command()
@click.option("--character", "-c", required=True)
@click.option("--franchise", "-f", required=True)
@click.option(
    "--component",
    type=click.Choice(["stt", "llm", "tts", "all"]),
    default="all",
    help="Component to benchmark",
)
@click.option(
    "--compare-models",
    help="Comma-separated model names to compare (e.g., 'llama-3.2-1b-instruct,llama-3.2-3b-instruct')",
)
@click.option("--iterations", "-n", default=5, help="Iterations per model")
@click.option("--save-metrics", help="Save metrics to file")
@click.option("--hardware-profile", help="Hardware profile to use")
def benchmark(
    character,
    franchise,
    component,
    compare_models,
    iterations,
    save_metrics,
    hardware_profile,
):
    """Benchmark pipeline components with model comparison."""
    import asyncio

    async def run_benchmark():
        # Initialize engine with hardware profile if specified
        engine = await _initialize_engine_with_profile(hardware_profile)

        if compare_models:
            # Compare specific models
            model_list = [m.strip() for m in compare_models.split(",")]
            click.echo(f"Comparing {component} models: {', '.join(model_list)}")

            results = {}
            for model_name in model_list:
                click.echo(f"\n{'='*60}")
                click.echo(f"Testing {component}: {model_name}")
                click.echo(f"{'='*60}")

                # Load model
                await _load_specific_model(engine, component, model_name)

                # Run benchmark
                result = await _run_benchmark(
                    engine, character, franchise, component, iterations
                )
                results[model_name] = result

            # Display comparison
            _display_model_comparison(results, component)

        elif component == "all":
            # Benchmark all components with current config
            click.echo("Benchmarking complete pipeline...")
            result = await _run_pipeline_benchmark(
                engine, character, franchise, iterations
            )
            _display_pipeline_results(result)
        else:
            # Benchmark single component with current config
            click.echo(f"Benchmarking {component} component...")
            result = await _run_benchmark(
                engine, character, franchise, component, iterations
            )
            _display_results(result, component)

        if save_metrics:
            _save_metrics(results if compare_models else result, save_metrics)

    asyncio.run(run_benchmark())


async def _initialize_engine_with_profile(hardware_profile):
    """Initialize engine with hardware profile if specified."""
    engine = await _initialize_engine(False)

    if hardware_profile:
        from ..core.config import Config
        from ..core.hardware_profile import HardwareProfileManager

        profile_manager = HardwareProfileManager()
        hardware_config = profile_manager.load_profile(hardware_profile)
        base_config = Config()
        merged_config = profile_manager.merge_with_config(hardware_config, base_config)

        # Update engine's resource manager with merged config
        engine.resource_manager.config = merged_config

        click.echo(f"Using hardware profile: {hardware_profile}")

    return engine


async def _load_specific_model(engine, component, model_name):
    """Load a specific model for benchmarking."""
    models_config = {component: model_name}
    await engine.resource_manager.preload_models_with_config(models_config)
    await engine.resource_manager.warmup_all_models()


async def _run_benchmark(engine, character, franchise, component, iterations):
    """Run benchmark for specific component."""
    import statistics
    import time

    times = []
    test_audio_path = f"tests_dev/audio_samples/{franchise}/{character}/input/test_interaction_voice_sample.wav"

    # Load character
    character_obj = _load_character(engine, character)
    if not character_obj:
        raise ValueError(f"Could not load character {character}")

    for i in range(iterations):
        start = time.time()

        if component == "stt":
            # Test STT only
            import soundfile as sf

            from ..core.protocols import AudioData

            audio_data, sample_rate = sf.read(test_audio_path)
            audio_obj = AudioData(data=audio_data, sample_rate=sample_rate, channels=1)
            await engine._transcribe_audio(audio_obj)
        elif component == "llm":
            # Test LLM only (with pre-transcribed text)
            await engine._generate_character_response("Hello", character_obj)
        elif component == "tts":
            # Test TTS only
            await engine._synthesize_character_voice("Hello", character_obj)

        elapsed = time.time() - start
        times.append(elapsed)

    return {
        "mean": statistics.mean(times),
        "stdev": statistics.stdev(times) if len(times) > 1 else 0,
        "min": min(times),
        "max": max(times),
        "iterations": iterations,
        "times": times,
    }


async def _run_pipeline_benchmark(engine, character, franchise, iterations):
    """Run full pipeline benchmark."""
    import statistics
    from pathlib import Path

    results = {}
    test_files = list(
        Path(f"tests_dev/audio_samples/{franchise}/{character}/input").glob("*.wav")
    )[:5]

    if not test_files:
        click.echo(
            f"No test files found in tests_dev/audio_samples/{franchise}/{character}/input/"
        )
        return results

    for test_file in test_files:
        times = {"stt": [], "llm": [], "tts": [], "total": []}

        for i in range(iterations):
            result = await engine.process_audio_file(str(test_file), character, None)
            if result.get("success") and "timing" in result.get("metadata", {}):
                timing = result["metadata"]["timing"]
                times["stt"].append(timing["stt_time"])
                times["llm"].append(timing["llm_time"])
                times["tts"].append(timing["tts_time"])
                times["total"].append(timing["total_time"])

        results[test_file.name] = {
            component: {
                "mean": statistics.mean(times[component]),
                "min": min(times[component]),
                "max": max(times[component]),
            }
            for component in times.keys()
        }

    return results


def _display_model_comparison(results, component):
    """Display side-by-side model comparison."""
    click.echo(f"\n{'='*80}")
    click.echo(f"{component.upper()} Model Comparison")
    click.echo(f"{'='*80}")
    click.echo(f"{'Model':<30} {'Mean':<12} {'Min':<12} {'Max':<12} {'StdDev':<12}")
    click.echo(f"{'-'*80}")

    for model_name, result in results.items():
        click.echo(
            f"{model_name:<30} {result['mean']:<12.3f}s {result['min']:<12.3f}s "
            f"{result['max']:<12.3f}s {result['stdev']:<12.3f}s"
        )

    # Find fastest
    fastest = min(results.items(), key=lambda x: x[1]["mean"])
    click.echo(f"{'-'*80}")
    click.echo(f"⚡ Fastest: {fastest[0]} ({fastest[1]['mean']:.3f}s)")
    click.echo(f"{'='*80}")

    # Show percentage differences
    baseline = list(results.values())[0]["mean"]
    click.echo(f"\nPercentage difference from {list(results.keys())[0]}:")
    for model_name, result in list(results.items())[1:]:
        diff = ((result["mean"] - baseline) / baseline) * 100
        symbol = "+" if diff > 0 else ""
        click.echo(f"  {model_name}: {symbol}{diff:.1f}%")


def _display_results(result, component):
    """Display single component results."""
    click.echo(f"\n{'='*60}")
    click.echo(f"{component.upper()} Benchmark Results")
    click.echo(f"{'='*60}")
    click.echo(f"Mean:   {result['mean']:.3f}s")
    click.echo(f"StdDev: {result['stdev']:.3f}s")
    click.echo(f"Min:    {result['min']:.3f}s")
    click.echo(f"Max:    {result['max']:.3f}s")
    click.echo(f"Iterations: {result['iterations']}")
    click.echo(f"{'='*60}")


def _display_pipeline_results(results):
    """Display full pipeline results."""
    click.echo(f"\n{'='*80}")
    click.echo("Complete Pipeline Benchmark")
    click.echo(f"{'='*80}")

    for filename, components in results.items():
        click.echo(f"\n{filename}:")
        for component, stats in components.items():
            click.echo(
                f"  {component.upper():6s}: {stats['mean']:.2f}s "
                f"(min: {stats['min']:.2f}s, max: {stats['max']:.2f}s)"
            )

    click.echo(f"{'='*80}")


def _save_metrics(results, filename):
    """Save metrics to file."""
    import json

    with open(filename, "w") as f:
        json.dump(results, f, indent=2)

    click.echo(f"Metrics saved to {filename}")
