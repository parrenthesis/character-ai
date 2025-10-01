#!/usr/bin/env python3
"""
Demo script showing how a toy works in practice.

This demonstrates the complete toy lifecycle from manufacturing to child interaction.
"""

import asyncio
import sys
from pathlib import Path

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from character_ai.production.toy_setup import ToySetup
from character_ai.production.real_time_engine import RealTimeInteractionEngine
from character_ai.hardware.toy_hardware_manager import ToyHardwareManager, HardwareConstraints
from character_ai.core.protocols import AudioData

async def demo_manufacturing_process():
    """Demo: How voice injection happens during manufacturing."""
    print("TOY MANUFACTURING PROCESS")
    print("=" * 50)
    
    # Step 1: Factory setup
    print("\n1. Factory initializing toy...")
    setup = ToySetup()
    await setup.initialize_toy()
    
    # Step 2: Manufacturer provides character voice files
    print("\n2. Manufacturer provides character voice files...")
    character_voice_files = {
        "sparkle": "factory_voices/sparkle_voice.wav",  # Manufacturer provides this
        "bumblebee": "factory_voices/bumblebee_voice.wav",  # Manufacturer provides this
        "flame": "factory_voices/flame_voice.wav"  # Manufacturer provides this
    }
    
    # Step 3: Factory injects voices into toy
    print("\n3. Factory injecting voices into toy...")
    results = await setup.inject_character_voices(character_voice_files)
    
    for character, success in results.items():
        status = "SUCCESS" if success else "FAILED"
        print(f"   {character}: {status}")
    
    # Step 4: Set default character for this toy type
    print("\n4. Setting default character (My Little Pony toy)...")
    await setup.engine.set_active_character("sparkle")
    
    # Step 5: Verify setup
    print("\n5. Verifying toy setup...")
    verification = await setup.verify_toy_setup()
    
    if verification["setup_complete"]:
        print("   Toy setup COMPLETE - ready for shipping!")
        character = verification["character"]
        print(f"   Active character: {character['name']} ({character['type']})")
        print(f"   Available voices: {verification['available_voices']}")
    else:
        print("   Toy setup FAILED")
    
    return setup.engine

async def demo_child_interaction(engine: RealTimeInteractionEngine):
    """Demo: How child interacts with the toy."""
    print("\n\nCHILD INTERACTION WITH TOY")
    print("=" * 50)
    
    # Child turns on toy
    print("\n1. Child turns on toy...")
    print("   *startup sound*")
    print("   LEDs light up")
    
    # Toy says hello
    print("\n2. Toy introduces itself...")
    character_info = await engine.get_character_info()
    print(f"   \"Hi! I'm {character_info['name']}! {character_info['description']}\"")
    
    # Child speaks to toy
    print("\n3. Child speaks to toy...")
    print("   \"Hi Sparkle! What's your favorite color?\"")
    
    # Create fake audio input
    fake_audio = AudioData(
        data=b"fake_audio_data_from_child",
        sample_rate=16000,
        channels=1,
        format="wav"
    )
    
    # Toy processes and responds
    print("\n4. Toy processes speech and responds...")
    result = await engine.process_realtime_audio(fake_audio)
    
    if result.error:
        print(f"   Error: {result.error}")
    else:
        print(f"   \"{result.text}\"")
        print(f"   *plays audio response in {result.metadata.get('voice_style', 'default')} voice*")
        print(f"   Response time: {result.metadata.get('processing_time', 0):.2f}s")
    
    # Show performance metrics
    print("\n5. Toy performance metrics...")
    metrics = await engine.get_performance_metrics()
    print(f"   Total interactions: {metrics['total_interactions']}")
    print(f"   Success rate: {metrics['success_rate']:.1%}")
    print(f"   Average latency: {metrics['average_latency']:.3f}s")
    print(f"   Within target (<500ms): {metrics['latency_within_target']}")

async def demo_toy_lifecycle():
    """Complete toy lifecycle demo."""
    print("INTERACTIVE CHARACTER TOY - COMPLETE LIFECYCLE DEMO")
    print("=" * 70)
    
    # Manufacturing process
    engine = await demo_manufacturing_process()
    
    # Child interaction
    await demo_child_interaction(engine)
    
    print("\n\nDEMO COMPLETE!")
    print("=" * 30)
    print("This shows how:")
    print("- Manufacturer provides voice files during manufacturing")
    print("- Factory injects voices into toys")
    print("- Children interact with toys in real-time")
    print("- Toys respond with character personalities")
    print("- Everything works within 500ms latency")

if __name__ == "__main__":
    asyncio.run(demo_toy_lifecycle())
