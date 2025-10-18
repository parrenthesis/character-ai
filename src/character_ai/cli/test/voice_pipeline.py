"""
Voice pipeline testing commands for the Character AI CLI.
"""

import asyncio
import logging
from pathlib import Path
from typing import Optional

import click

from .helpers import _configure_logging, _initialize_engine
from .realtime import _test_realtime_interaction

logger = logging.getLogger(__name__)


@click.command()
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
@click.option(
    "--streaming",
    is_flag=True,
    default=False,
    help="Enable streaming TTS synthesis (sentence-based)",
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
    streaming: bool,
) -> None:
    """Test the complete voice pipeline: STT → LLM → TTS"""
    try:
        # Configure logging
        _configure_logging(verbose)

        # Initialize core engine directly
        engine = asyncio.run(_initialize_engine(verbose, hardware_profile))

        # Enable streaming if flag set
        if streaming:
            # Note: Streaming configuration is handled internally by the engine
            if verbose:
                click.echo("🔊 Streaming TTS enabled via --streaming flag")

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

            click.echo(f"\n📊 Test Results: {successful_tests}/{total_tests} successful")
            return

        # Default single file test
        if not input:
            input = f"tests_dev/audio_samples/{franchise}/{character}/input/test_interaction_voice_sample.wav"

        if not output_dir:
            output_dir = f"tests_dev/audio_samples/{franchise}/{character}/output/"

        # Ensure output directory exists
        Path(output_dir).mkdir(parents=True, exist_ok=True)

        # Test individual components
        if test_stt_only:
            click.echo("🎤 Testing STT component only...")
            result = asyncio.run(engine.transcribe_audio_file(input))
            if result.get("success"):
                click.echo(f"✅ STT Success: {result.get('transcription', 'N/A')}")
            else:
                click.echo(f"❌ STT Failed: {result.get('error', 'Unknown error')}")
            return

        if test_llm_only:
            click.echo("🧠 Testing LLM component only...")
            # Load character
            character_obj = engine.character_manager.get_character(character)
            if not character_obj:
                click.echo(f"❌ Character '{character}' not found")
                return

            result = asyncio.run(
                engine.generate_response("Hello, how are you?", character_obj)
            )
            if result.get("success"):
                click.echo(f"✅ LLM Success: {result.get('response', 'N/A')}")
            else:
                click.echo(f"❌ LLM Failed: {result.get('error', 'Unknown error')}")
            return

        if test_tts_only:
            click.echo("🔊 Testing TTS component only...")
            # Load character
            character_obj = engine.character_manager.get_character(character)
            if not character_obj:
                click.echo(f"❌ Character '{character}' not found")
                return

            output_file = Path(output_dir) / "tts_test.wav"
            result = asyncio.run(
                engine.synthesize_voice(
                    "Hello, this is a test.", character_obj, str(output_file)
                )
            )
            if result.get("success"):
                click.echo(f"✅ TTS Success: Audio saved to {output_file}")
            else:
                click.echo(f"❌ TTS Failed: {result.get('error', 'Unknown error')}")
            return

        # Full pipeline test
        click.echo("🎯 Testing full voice pipeline...")
        click.echo(f"Input: {input}")
        click.echo(f"Character: {character}")
        click.echo(f"Output: {output_dir}")

        # Use input filename for output files
        input_path = Path(input)
        output_file = Path(output_dir) / f"{input_path.stem}_response.wav"
        result = asyncio.run(
            engine.process_audio_file(input, character, str(output_file))
        )

        if result.get("success"):
            click.echo("✅ Voice pipeline test successful!")
            click.echo(f"Transcription: {result.get('transcription', 'N/A')}")
            click.echo(f"Response: {result.get('response', 'N/A')}")
            click.echo(f"Audio saved to: {output_file}")
        else:
            click.echo(
                f"❌ Voice pipeline test failed: {result.get('error', 'Unknown error')}"
            )

    except Exception as e:
        click.echo(f"Error in voice pipeline test: {e}", err=True)
        raise click.Abort()
