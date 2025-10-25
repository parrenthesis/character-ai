"""
Shared helper functions for test CLI commands.
"""

import logging
from typing import Any, Dict, Optional

import click

from ...characters import Character
from ...hardware.toy_hardware_manager import ToyHardwareManager
from ...observability.logging import configure_logging
from ...production.real_time_engine import RealTimeInteractionEngine

logger = logging.getLogger(__name__)


def _configure_logging(verbose: bool) -> None:
    """Configure logging for tests using observability's structured logger."""
    import os

    # Determine level from env or verbosity
    level = os.getenv("CAI_LOG_LEVEL", "INFO" if verbose else "WARNING").upper()
    # Use console-friendly (non-JSON) rendering during tests
    try:
        configure_logging(level=level, json_format=False)
    except Exception:
        # Fallback to basicConfig if observability fails for any reason
        logging.basicConfig(
            level=getattr(logging, level, logging.INFO),
            format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
        )


def _load_character(
    engine: RealTimeInteractionEngine, character_name: str
) -> Optional[Character]:
    """Load character from engine."""
    character = engine.character_manager.get_character(character_name)
    if character:
        logger.info(
            f"Loaded character: {character.name}, metadata: {character.metadata}"
        )
    else:
        logger.warning(f"Failed to load character: {character_name}")
    return character


async def _initialize_engine(
    verbose: bool = False, hardware_profile: Optional[str] = None
) -> RealTimeInteractionEngine:
    """Initialize and return configured engine."""
    _configure_logging(verbose)

    # Hardware profile auto-detection is now handled by RealTimeInteractionEngine
    # via HardwareProfileService (pass None or "auto" to trigger auto-detection)
    if hardware_profile == "auto":
        hardware_profile = None  # Let engine auto-detect

    if verbose and (hardware_profile is None):
        # Engine will auto-detect, inform user
        click.echo("🔍 Auto-detecting hardware profile...")

    hardware_manager = ToyHardwareManager()
    engine = RealTimeInteractionEngine(
        hardware_manager, hardware_profile=hardware_profile
    )
    await engine.initialize()
    return engine


def _display_test_results(results: Dict[str, Any]) -> None:
    """Display test results in a formatted way."""
    click.echo("\n" + "=" * 60)
    click.echo("TEST RESULTS")
    click.echo("=" * 60)

    for test_name, result in results.items():
        if isinstance(result, dict):
            if "success" in result:
                status = "✅ PASS" if result["success"] else "❌ FAIL"
                click.echo(f"{status} {test_name}: {result.get('message', '')}")
            elif "error" in result:
                click.echo(f"❌ ERROR {test_name}: {result['error']}")
            else:
                click.echo(f"ℹ️  {test_name}: {result}")
        else:
            click.echo(f"ℹ️  {test_name}: {result}")

    click.echo("=" * 60)


async def _initialize_engine_with_profile(hardware_profile: str) -> Any:
    """Initialize engine with hardware profile if specified."""
    if hardware_profile and hardware_profile != "auto":
        click.echo(f"🔧 Using hardware profile: {hardware_profile}")
    else:
        click.echo("🔍 Auto-detecting hardware profile...")

    return await _initialize_engine(True, hardware_profile)


async def _load_specific_model(engine: Any, component: str, model_name: str) -> None:
    """Load a specific model for benchmarking."""
    click.echo(f"Loading {component} model: {model_name}")
    # This would load the specific model
    # For now, just warmup all models
    await engine.resource_manager.warmup_all_models()


def _display_model_comparison(results: dict[str, Any], component: str) -> None:
    """Display side-by-side model comparison."""
    click.echo(f"\n{component.upper()} Model Comparison:")
    click.echo("-" * 50)

    for model_name, result in results.items():
        if isinstance(result, dict) and "mean" in result:
            click.echo(f"{model_name}:")
            click.echo(f"  Mean: {result['mean']:.3f}s")
            click.echo(f"  StdDev: {result['stdev']:.3f}s")
            click.echo(f"  Min: {result['min']:.3f}s")
            click.echo(f"  Max: {result['max']:.3f}s")
            click.echo()


def _display_results(result: dict[str, Any], component: str) -> None:
    """Display single component results."""
    click.echo(f"\n{component.upper()} Results:")
    click.echo("-" * 30)
    click.echo(f"Mean: {result['mean']:.3f}s")
    click.echo(f"StdDev: {result['stdev']:.3f}s")
    click.echo(f"Min: {result['min']:.3f}s")
    click.echo(f"Max: {result['max']:.3f}s")
    click.echo(f"Iterations: {result['iterations']}")


def _display_pipeline_results(results: dict[str, Any]) -> None:
    """Display full pipeline results."""
    click.echo("\nFull Pipeline Results:")
    click.echo("-" * 30)
    click.echo(f"Mean: {results['mean']:.3f}s")
    click.echo(f"StdDev: {results['stdev']:.3f}s")
    click.echo(f"Min: {results['min']:.3f}s")
    click.echo(f"Max: {results['max']:.3f}s")
    click.echo(f"Iterations: {results['iterations']}")


def _save_metrics(results: dict[str, Any], filename: str) -> None:
    """Save metrics to file."""
    import json

    with open(filename, "w") as f:
        json.dump(results, f, indent=2)
    click.echo(f"Metrics saved to {filename}")


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

    from ...core.audio_io.audio_config import AudioConfig
    from ...core.audio_io.audio_utils import prepare_audio_for_playback
    from ...core.protocols import AudioData

    # Create AudioData
    audio_data = AudioData(
        data=full_audio,
        sample_rate=AudioConfig.DEFAULT_SAMPLE_RATE,
        channels=AudioConfig.DEFAULT_CHANNELS,
        duration=len(full_audio) / AudioConfig.DEFAULT_SAMPLE_RATE,
    )

    # Process with optimized pipeline
    processing_start = time.time()
    result = await engine.process_audio_with_character(audio_data, character_obj)
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
        if (
            result.audio_data
            and result.audio_data.data is not None
            and len(result.audio_data.data) > 0
        ):
            click.echo("🔊 Playing response...")

            # Prepare audio for playback using consolidated utilities
            # Convert from TTS sample rate (24000 Hz) to device sample rate (48000 Hz)
            audio_array, sample_rate = prepare_audio_for_playback(
                result.audio_data.data, int(output_sample_rate)
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
            click.echo("⚠️  No audio data in response")

        # Update interaction count
        click.echo(f"📊 Total interactions: {interaction_count + 1}")
    else:
        click.echo("❌ No response generated")

    return result
