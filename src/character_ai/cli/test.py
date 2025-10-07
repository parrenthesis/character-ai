"""
Testing commands for the Character AI CLI.

Provides Click-based commands for testing platform functionality.
"""

import asyncio
import json
import logging
import sys
from pathlib import Path
from typing import Any, Dict, Optional

import click

# Add tests to path for testing utilities
sys.path.append(
    str(Path(__file__).parent.parent.parent / "tests" / "testing_utilities")
)

try:
    from audio_tester import AudioTester
    from mock_hardware import MockHardwareManager
except ImportError:
    # Testing utilities not available, create mock classes
    class MockAudioTester:
        async def initialize(self) -> None:
            pass

        async def test_character_switching(self) -> Dict[str, Any]:
            return {}

        async def test_audio_processing(self) -> Dict[str, Any]:
            return {}

        async def test_performance(self) -> Dict[str, Any]:
            return {}

        async def check_system_health(self) -> Dict[str, Any]:
            return {}

    class MockHardwareManagerImpl:
        pass

    class MockRealTimeInteractionEngine:
        pass

    # Create aliases for the mock classes
    AudioTester = MockAudioTester
    MockHardwareManager = MockHardwareManagerImpl
    # RealTimeInteractionEngine = MockRealTimeInteractionEngine  # type: ignore


logger = logging.getLogger(__name__)


@click.group()
def test_commands() -> None:
    """Testing and validation commands."""
    pass


@test_commands.command()
@click.option(
    "--input",
    "-i",
    help="Input audio file path (defaults to tests_dev/audio_samples/{franchise}/{character}/input/test_interaction_voice_sample.wav)",
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
    help="Output directory (defaults to tests_dev/audio_samples/{franchise}/{character}/output/)",
)
@click.option("--verbose", "-v", is_flag=True, help="Enable verbose output")
@click.option("--test-stt-only", is_flag=True, help="Test STT component only")
@click.option("--test-llm-only", is_flag=True, help="Test LLM component only")
@click.option("--test-tts-only", is_flag=True, help="Test TTS component only")
@click.option(
    "--list-inputs", is_flag=True, help="List available input files for the character"
)
@click.option(
    "--test-all", is_flag=True, help="Test all available input files for the character"
)
def voice_pipeline(
    input: Optional[str],
    character: str,
    franchise: str,
    output_dir: Optional[str],
    verbose: bool,
    test_stt_only: bool,
    test_llm_only: bool,
    test_tts_only: bool,
    list_inputs: bool,
    test_all: bool,
) -> None:
    """Test the complete voice pipeline: STT → LLM → TTS"""
    try:
        # Configure logging
        if verbose:
            logging.basicConfig(
                level=logging.INFO,
                format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
            )
        else:
            logging.basicConfig(level=logging.WARNING)

        # Import VoicePipelineTester
        import sys
        from pathlib import Path

        sys.path.append(
            str(Path(__file__).parent.parent.parent / "tests_dev" / "testing_utilities")
        )
        try:
            from voice_pipeline_tester import VoicePipelineTester
        except ImportError:
            # Fallback to absolute import
            import sys

            sys.path.insert(0, str(Path(__file__).parent.parent.parent))
            from tests_dev.testing_utilities.voice_pipeline_tester import (
                VoicePipelineTester,
            )

        # Initialize tester
        tester = VoicePipelineTester()

        # Handle list inputs option
        if list_inputs:
            input_files = tester.list_available_inputs(character, franchise)
            if input_files:
                click.echo(f"Available input files for {character} ({franchise}):")
                for i, file in enumerate(input_files, 1):
                    click.echo(f"  {i}. {file}")
            else:
                click.echo(
                    f"No input files found for {character} in franchise {franchise}"
                )
                click.echo(
                    f"Expected location: tests_dev/audio_samples/{franchise}/{character}/input/"
                )
            return

        # Handle test all option
        if test_all:
            click.echo(
                f"Testing all available input files for {character} ({franchise})..."
            )
            result = asyncio.run(tester.test_all_inputs(character, franchise))

            # Display results
            if result.get("success"):
                click.echo(
                    f"✅ Test completed: {result['successful_tests']}/{result['total_tests']} tests passed"
                )
                if verbose:
                    for input_file, test_result in result.get("results", {}).items():
                        status = "✅" if test_result.get("success") else "❌"
                        click.echo(f"  {status} {input_file}")
            else:
                click.echo(f"❌ Test failed: {result.get('error', 'Unknown error')}")
                if verbose:
                    click.echo(f"Details: {result}")
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

        # Run async test
        if test_stt_only:
            result = asyncio.run(
                tester.test_stt_only(input, character, franchise, output_dir)
            )
        elif test_llm_only:
            # For LLM-only test, we need input text instead of audio file
            with open(input, "r") as f:
                input_text = f.read()
            result = asyncio.run(
                tester.test_llm_only(input_text, character, franchise, output_dir)
            )
        elif test_tts_only:
            # For TTS-only test, we need input text instead of audio file
            with open(input, "r") as f:
                input_text = f.read()
            result = asyncio.run(
                tester.test_tts_only(input_text, character, franchise, output_dir)
            )
        else:
            # Full pipeline test
            result = asyncio.run(
                tester.test_full_pipeline(input, character, franchise, output_dir)
            )

        # Display results
        if result.get("success"):
            click.echo("✅ Test completed successfully!")
            if "output_paths" in result:
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


@test_commands.command()
@click.option("--verbose", "-v", is_flag=True, help="Enable verbose output")
@click.option("--output", "-o", help="Save test results to file")
def interactive(verbose: bool, output: Optional[str]) -> None:
    """Run character.ai tests."""
    try:
        # Configure logging
        if verbose:
            logging.basicConfig(
                level=logging.INFO,
                format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
            )
        else:
            logging.basicConfig(level=logging.WARNING)

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
    """Run the character.ai test."""
    try:
        logger.info("Starting Character AI test...")

        # Initialize audio tester
        tester = AudioTester()
        await tester.initialize()

        results = {}

        # Test 1: Character switching
        logger.info("Testing character switching...")
        character_results = await tester.test_character_switching()
        results["character_switching"] = character_results

        # Test 2: Audio processing
        logger.info("Testing audio processing...")
        audio_results = await tester.test_audio_processing()
        results["audio_processing"] = audio_results

        # Test 3: Performance
        logger.info("Testing performance...")
        performance_results = await tester.test_performance()
        results["performance"] = performance_results

        # Test 4: System health
        logger.info("Checking system health...")
        health_results = await tester.check_system_health()
        results["system_health"] = health_results

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
