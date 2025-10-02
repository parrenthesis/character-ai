#!/usr/bin/env python3
"""
Test script for Character AI.

Demonstrates how to test the system with audio files when no physical hardware is available.
"""

import asyncio
import logging
import sys
from pathlib import Path

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

# Import testing utilities directly
sys.path.append(str(Path(__file__).parent.parent / "tests" / "testing_utilities"))
from audio_tester import AudioTester  # noqa: E402
from mock_hardware import MockHardwareManager  # noqa: E402

from character_ai.production.real_time_engine import (  # noqa: E402
    RealTimeInteractionEngine,
)

# Configure logging
logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)


async def test_character_platform():
    """Test the character.ai."""
    try:
        logger.info("Starting Character AI test...")

        # Initialize audio tester
        tester = AudioTester()
        await tester.initialize()

        # Test 1: Character switching
        logger.info("Testing character switching...")
        character_results = await tester.test_character_switching()
        print("\n=== Character Switching Test ===")
        for character, result in character_results.items():
            if isinstance(result, dict) and "success" in result:
                status = "PASS" if result["success"] else "FAIL"
                print(f"{character}: {status}")
                if result["success"] and "info" in result:
                    info = result["info"]
                    print(f"  - Name: {info.get('name', 'Unknown')}")
                    print(f"  - Type: {info.get('type', 'Unknown')}")
                    print(f"  - Voice Style: {info.get('voice_style', 'Unknown')}")
            else:
                print(f"{character}: FAIL - {result}")

        # Test 2: Audio processing
        logger.info("Testing audio processing...")
        test_file = await tester.create_test_audio_file("Hello, what's your name?")

        print("\n=== Audio Processing Test ===")
        for character in ["sparkle", "bumblebee", "flame"]:
            result = await tester.test_with_audio_file(test_file, character)
            status = "PASS" if result["success"] else "FAIL"
            print(f"{character}: {status}")
            if result["success"]:
                print(f"  - Response: {result['response_text']}")
                print(f"  - Character: {result['character']}")
            else:
                print(f"  - Error: {result['error']}")

        # Test 3: Performance
        logger.info("Testing performance...")
        perf_results = await tester.test_performance(num_tests=3)

        print("\n=== Performance Test ===")
        if "error" not in perf_results:
            print(f"Average Latency: {perf_results['average_latency']:.3f}s")
            print(f"Success Rate: {perf_results['success_rate']:.1%}")
            print(
                f"Tests: {perf_results['successful_tests']}/{perf_results['total_tests']}"
            )
        else:
            print(f"Performance test failed: {perf_results['error']}")

        # Test 4: System health
        logger.info("Checking system health...")
        health = await tester.engine.get_health_status()

        print("\n=== System Health ===")
        print(f"Overall Health: {' HEALTHY' if health['healthy'] else ' UNHEALTHY'}")
        if "character_manager" in health:
            char_mgr = health["character_manager"]
            print(
                f"Character Manager: {' HEALTHY' if char_mgr['healthy'] else ' UNHEALTHY'}"
            )
            print(f"Active Character: {char_mgr.get('active_character', 'None')}")

        if "performance" in health:
            perf = health["performance"]
            print(f"Total Interactions: {perf.get('total_interactions', 0)}")
            print(f"Success Rate: {perf.get('success_rate', 0):.1%}")
            print(f"Average Latency: {perf.get('average_latency', 0):.3f}s")

        # Cleanup
        tester.cleanup_test_files()

        logger.info("Character AI test completed successfully!")

    except Exception as e:
        logger.error(f"Test failed: {e}")
        raise


async def demo_character_interaction():
    """Demonstrate character interaction."""
    try:
        logger.info("Starting character interaction demo...")

        # Initialize with mock hardware
        from character_ai.hardware import HardwareConstraints

        constraints = HardwareConstraints()
        mock_hardware = MockHardwareManager(constraints)

        # Initialize real-time engine with mock hardware
        engine = RealTimeInteractionEngine(mock_hardware)
        await engine.initialize()

        # Set active character
        await engine.set_active_character("sparkle")

        # Get character info
        character_info = await engine.get_character_info()
        print("\n=== Active Character ===")
        print(f"Name: {character_info['name']}")
        print(f"Type: {character_info['type']}")
        print(f"Voice Style: {character_info['voice_style']}")
        print(f"Description: {character_info['description']}")

        # Simulate some interactions
        print("\n=== Simulated Interactions ===")

        # Create test audio data
        from character_ai.core.protocols import AudioData

        test_audio = AudioData(
            data=b"test_audio_data", sample_rate=16000, channels=1, format="wav"
        )

        # Process audio
        result = await engine.process_realtime_audio(test_audio)

        if result.error:
            print(f" Error: {result.error}")
        else:
            print(f" Response: {result.text}")
            print(f"Character: {result.metadata.get('character', 'Unknown')}")
            print(f"Voice Style: {result.metadata.get('voice_style', 'Unknown')}")

        # Test character switching
        print("\n=== Character Switching Demo ===")
        characters = ["sparkle", "bumblebee", "flame"]

        for char in characters:
            success = await engine.set_active_character(char)
            if success:
                info = await engine.get_character_info()
                print(f" Switched to {info['name']} ({info['type']})")
            else:
                print(f" Failed to switch to {char}")

        logger.info("Character interaction demo completed!")

    except Exception as e:
        logger.error(f"Demo failed: {e}")
        raise


async def main():
    """Main function."""
    print(" Character AI Test Suite")
    print("=" * 50)

    try:
        # Run tests
        await test_character_platform()

        print("\n" + "=" * 50)
        print(" Character Interaction Demo")
        print("=" * 50)

        # Run demo
        await demo_character_interaction()

        print("\n" + "=" * 50)
        print(" All tests completed successfully!")
        print(" Character AI is ready for production!")

    except Exception as e:
        print(f"\n Test suite failed: {e}")
        sys.exit(1)


if __name__ == "__main__":
    asyncio.run(main())
