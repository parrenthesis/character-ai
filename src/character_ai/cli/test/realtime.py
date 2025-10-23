"""
Real-time testing functionality for the Character AI CLI.
"""

import logging
import time
import warnings
from typing import Any

import click

from .helpers import _load_character

logger = logging.getLogger(__name__)


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
        from ...core.audio_io import AudioComponentFactory
        from ...core.audio_io.audio_config import AudioConfig

        if not quiet:
            click.echo("🎤 Starting real-time voice interaction test...")
            click.echo(f"Character: {character} ({franchise})")
            click.echo("Press Ctrl+C to stop")
            click.echo()

        # Initialize audio components
        audio_factory = AudioComponentFactory()
        interface_device_manager = audio_factory.create_device_manager()

        # List available devices
        input_devices = await interface_device_manager.list_input_devices()
        output_devices = await interface_device_manager.list_output_devices()

        if verbose:
            click.echo("Available input devices:")
            for i, device in enumerate(input_devices):
                click.echo(f"  {i}: {device.name}")
            click.echo("Available output devices:")
            for i, device in enumerate(output_devices):
                click.echo(f"  {i}: {device.name}")
            click.echo()

        # Use AudioDeviceManager for device selection
        from ...core.audio_io.device_manager import AudioDeviceManager
        from ...core.config import Config

        # Get audio device configuration from runtime config
        config = Config()
        audio_config = getattr(config.runtime, "audio_devices", {})
        device_manager: AudioDeviceManager = AudioDeviceManager(audio_config)

        # Setup input and output devices
        try:
            input_device, input_sample_rate = device_manager.setup_input_device()
            output_device, output_sample_rate = device_manager.setup_output_device()

            click.echo(f"🎤 Using input: {input_device.name} ({input_sample_rate}Hz)")
            click.echo(f"🔊 Using output: {output_device.name} ({output_sample_rate}Hz)")

            # Display hardware profile information
            click.echo(f"⚙️  Hardware profile: {hardware_profile}")
            if engine.core_engine.lifecycle.hardware_config:
                models = engine.core_engine.lifecycle.hardware_config.get("models", {})
                optimizations = engine.core_engine.lifecycle.hardware_config.get(
                    "optimizations", {}
                )
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

        # Set active character (this starts the hybrid memory session)
        engine.core_engine.lifecycle.set_active_character(character_obj)

        # Pre-load models BEFORE starting audio capture to avoid overflow
        click.echo("🔄 Pre-loading models for faster processing...")
        preload_results = await engine.preload_models()
        loaded_count = sum(1 for success in preload_results.values() if success)
        click.echo(f"✅ Pre-loaded {loaded_count}/3 models")
        if loaded_count < 3:
            click.echo(
                "⚠️  Some models failed to pre-load - performance may be affected"
            )

        # Warm up models with dummy inference
        click.echo("🔥 Warming up models...")
        warmup_results = (
            await engine.core_engine.lifecycle.resource_manager.warmup_all_models(
                character_obj
            )
        )
        warmup_count = sum(1 for success in warmup_results.values() if success)
        click.echo(f"✅ Warmed up {warmup_count}/{len(warmup_results)} models")

        click.echo("✅ Audio capture started")
        click.echo("🎙️  Starting in...")
        for i in range(3, 0, -1):
            click.echo(f"   {i}...")
            time.sleep(1)
        click.echo(
            f"   🎤 GO! You can have multiple conversations with {character} for {duration} seconds"
        )
        click.echo("   Speak → Listen to response → Speak again → Repeat")
        click.echo()

        # Import required modules for real-time processing
        import asyncio
        import sys

        from ...core.audio_io import AudioComponentFactory
        from ...core.audio_io.vad_session import VADConfig, VADSessionManager
        from .helpers import _process_speech_chunk_with_metrics

        # Get VAD settings from hardware profile
        vad_settings = {}
        if (
            engine.core_engine.lifecycle.hardware_config
            and "vad" in engine.core_engine.lifecycle.hardware_config
        ):
            vad_settings = engine.core_engine.lifecycle.hardware_config["vad"]

        # Initialize VAD session manager with hardware settings
        vad_config = VADConfig.for_toy_interaction()
        vad_manager = VADSessionManager(
            vad_config=vad_config,
            hardware_vad_settings=vad_settings,  # Pass hardware settings directly
        )

        # Initialize audio capture using factory
        audio_capture = AudioComponentFactory.create_audio_capture(use_mocks=False)

        # Start audio capture
        await audio_capture.start_capture(
            device=input_device,  # type: ignore[arg-type]
            sample_rate=int(input_sample_rate),
            channels=1,
            chunk_size=AudioConfig.DEFAULT_CHUNK_SIZE,
        )

        # Real-time processing loop
        start_time = time.time()
        interaction_count = 0
        interaction_times: list[float] = []
        cache_hits = 0
        cache_misses = 0

        try:
            while time.time() - start_time < duration:
                # Get audio chunk
                audio_chunk = await audio_capture.read_audio_chunk()
                if audio_chunk is None:
                    await asyncio.sleep(0.01)
                    continue

                # Process audio chunk through VAD
                prev_state = vad_manager.state
                vad_state = vad_manager.process_audio_chunk(audio_chunk)

                # Show VAD state transitions
                if vad_state != prev_state:
                    from ...core.audio_io.vad_session import VADSessionState

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

                # Handle speech end
                if vad_manager.should_end_speech():
                    vad_manager.set_processing_state()

                    # Stop audio capture during processing
                    processing_start_ts = time.time()
                    click.echo(
                        f"⏱️  Processing started at {processing_start_ts - start_time:.2f}s"
                    )
                    sys.stdout.flush()

                    await audio_capture.stop_capture()

                    # Get combined speech audio
                    speech_audio = vad_manager.get_combined_speech_audio()

                    if speech_audio is not None:
                        # Show processing message
                        audio_duration = (
                            len(speech_audio) / AudioConfig.DEFAULT_SAMPLE_RATE
                        )
                        num_chunks = len(vad_manager.speech_buffer)

                        click.echo(
                            f"📝 Processing {audio_duration:.1f}s of audio ({num_chunks} chunks buffered)"
                        )
                        sys.stdout.flush()

                        # Process speech chunk
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

                        # Update metrics
                        interaction_count += 1
                        if result and result.metadata.get("cache_hit"):
                            cache_hits += 1
                        else:
                            cache_misses += 1

                        # Display live metrics
                        if result and result.metadata and "timing" in result.metadata:
                            timing = result.metadata["timing"]
                            click.echo(
                                f"⚠️  Exceeded 5s target: {timing['total_time']:.2f}s"
                            )
                            click.echo(
                                f"   Breakdown: STT={timing['stt_time']:.2f}s, LLM={timing['llm_time']:.2f}s, TTS={timing['tts_time']:.2f}s"
                            )

                    # Reset VAD for next interaction
                    vad_manager.reset_session()

                    # Restart audio capture
                    await audio_capture.start_capture(
                        device=input_device,  # type: ignore[arg-type]
                        sample_rate=int(input_sample_rate),
                        channels=1,
                        chunk_size=AudioConfig.DEFAULT_CHUNK_SIZE,
                    )

        finally:
            # Cleanup
            await audio_capture.stop_capture()

        # Display final results
        click.echo("\n✅ Session completed!")
        click.echo(f"   Interactions: {interaction_count}")
        if interaction_times:
            avg_time = sum(interaction_times) / len(interaction_times)
            click.echo(f"   Average latency: {avg_time:.2f}s")
        if cache_hits + cache_misses > 0:
            cache_hit_rate = cache_hits / (cache_hits + cache_misses) * 100
            click.echo(f"   Cache hit rate: {cache_hit_rate:.1f}%")

    except KeyboardInterrupt:
        click.echo("\n🛑 Test interrupted by user")
    except Exception as e:
        click.echo(f"❌ Error in real-time test: {e}", err=True)
        logger.exception("Real-time test error")
        raise click.Abort()
