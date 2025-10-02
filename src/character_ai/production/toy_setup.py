"""
Toy setup and manufacturing configuration.

This is where voice injection happens during toy manufacturing or first boot.
"""

import logging
from pathlib import Path
from typing import Any, Dict, Optional

from ..hardware.toy_hardware_manager import HardwareConstraints, ToyHardwareManager
from .real_time_engine import RealTimeInteractionEngine

logger = logging.getLogger(__name__)


class ToySetup:
    """Handles toy manufacturing setup and voice injection."""

    def __init__(self) -> None:
        self.constraints = HardwareConstraints()
        self._hardware_manager: Optional[ToyHardwareManager] = None
        self._engine: Optional[RealTimeInteractionEngine] = None
        self.initialized = False
        # Allow lazy engine auto-creation until an explicit shutdown occurs
        self._allow_engine_autocreate = True

    @property
    def hardware_manager(self) -> ToyHardwareManager:
        if self._hardware_manager is None:
            self._hardware_manager = ToyHardwareManager(self.constraints)
        return self._hardware_manager

    @hardware_manager.setter
    def hardware_manager(self, value: ToyHardwareManager) -> None:
        self._hardware_manager = value

    @property
    def engine(self) -> Optional[RealTimeInteractionEngine]:
        if self._engine is None:
            if not getattr(self, "_allow_engine_autocreate", True):
                return None
            self._engine = RealTimeInteractionEngine(self.hardware_manager)
        return self._engine

    @engine.setter
    def engine(self, value: Optional[RealTimeInteractionEngine]) -> None:
        self._engine = value

    async def initialize_toy(self) -> bool:
        """Initialize toy for first boot."""
        try:
            logger.info("Initializing toy for first boot...")

            # Initialize hardware and engine
            await self.hardware_manager.initialize()
            if self.engine is not None:
                await self.engine.initialize()

            logger.info("Toy initialized successfully")
            self.initialized = True
            return True

        except Exception as e:
            logger.error(f"Failed to initialize toy: {e}")
            return False

    async def inject_character_voices(
        self, voice_files: Dict[str, str]
    ) -> Dict[str, Any]:
        """
        Inject character voices during manufacturing.

        Args:
            voice_files: Dict mapping character_name -> voice_file_path
                        e.g., {"sparkle": "/factory/voices/sparkle.wav"}

        Returns:
            Dict mapping character_name -> success_status
        """
        if not getattr(self, "initialized", False):
            return {"success": False, "error": "ToySetup not initialized"}

        results: Dict[str, bool] = {}

        for character_name, voice_file_path in voice_files.items():
            try:
                logger.info(
                    f"Injecting voice for {character_name} from {voice_file_path}"
                )

                # Validate file exists
                import os

                if not os.path.exists(voice_file_path):
                    return {
                        "success": False,
                        "error": f"Voice file not found: {voice_file_path}",
                    }

                if self.engine is not None:
                    success = await self.engine.inject_character_voice(
                        character_name, voice_file_path
                    )
                else:
                    success = False
                results[character_name] = bool(success)

                if success:
                    logger.info(f"Voice injected for {character_name}")
                else:
                    logger.error(f"Failed to inject voice for {character_name}")

            except Exception as e:
                logger.error(f"Error injecting voice for {character_name}: {e}")
                return {"success": False, "error": str(e)}

        return {
            "success": all(results.values()),
            "injected_count": sum(1 for v in results.values() if v),
            "injected_voices": list(results.keys()),
        }

    async def initialize(self) -> None:
        """Initialize ToySetup dependencies."""
        # Access properties to trigger lazy initialization
        _ = self.hardware_manager
        _ = self.engine
        if self.engine is not None:
            await self.engine.initialize()
        self.initialized = True

    async def validate_character_voice(self, voice_file_path: str) -> Dict[str, Any]:
        """Validate a character voice file path."""
        from os.path import exists, getsize

        if not exists(voice_file_path):
            return {"valid": False, "error": "File not found"}
        if not voice_file_path.lower().endswith(".wav"):
            return {"valid": False, "error": "Invalid format"}
        return {"valid": True, "file_size": getsize(voice_file_path), "format": "wav"}

    async def create_character_manifest(
        self, voice_files: Dict[str, str]
    ) -> Dict[str, Any]:
        """Create a simple manifest for character voices."""
        import hashlib
        import os

        manifest: Dict[str, Any] = {"characters": {}, "total_files": 0}
        for name, path in voice_files.items():
            size = 0
            try:
                size = os.path.getsize(path)
                with open(path, "rb") as f:
                    checksum = hashlib.sha256(f.read()).hexdigest()
            except Exception:
                checksum = ""
            manifest["characters"][name] = {
                "path": path,
                "size": size,
                "checksum": checksum,
            }
            manifest["total_files"] += 1
        return manifest

    async def setup_toy_environment(self) -> Dict[str, Any]:
        """Create directories and set permissions for toy environment."""
        import os

        try:
            import tempfile

            base = Path(tempfile.gettempdir()) / "icp_toy_env"
            os.makedirs(base, exist_ok=True)
            os.chmod(base, 0o700)  # More restrictive permissions
            return {"success": True, "directories_created": [str(base)]}
        except Exception as e:
            return {"success": False, "error": str(e)}

    async def configure_toy_settings(self, settings: Dict[str, Any]) -> Dict[str, Any]:
        """Write settings to a json file."""
        import json

        try:
            import tempfile

            settings_path = Path(tempfile.gettempdir()) / "icp_toy_settings.json"
            with open(settings_path, "w") as f:
                json.dump(settings, f)
            return {"success": True, "settings_file": str(settings_path)}
        except Exception as e:
            return {"success": False, "error": str(e)}

    async def shutdown(self) -> None:
        """Shutdown engine and mark uninitialized."""
        try:
            if self._engine is not None:
                await self._engine.shutdown()
        finally:
            self._engine = None
            self._allow_engine_autocreate = False
            self.initialized = False

    async def setup_my_little_pony_toy(self) -> bool:
        """Setup a My Little Pony toy with Sparkle's voice."""
        try:
            # During manufacturing, manufacturer would provide these files
            voice_files = {"sparkle": "/factory/character_voices/sparkle_voice.wav"}

            # Inject voices
            results = await self.inject_character_voices(voice_files)

            # Set default character
            character_info = None
            if self.engine is not None:
                await self.engine.set_active_character("sparkle")

                # Verify setup
                character_info = await self.engine.get_character_info()
            if character_info:
                logger.info(
                    f"My Little Pony toy setup complete: {character_info['name']}"
                )
            else:
                logger.info("My Little Pony toy setup complete")

            return all(results.values())

        except Exception as e:
            logger.error(f"Failed to setup My Little Pony toy: {e}")
            return False

    async def setup_transformers_toy(self) -> bool:
        """Setup a Transformers toy with Bumblebee's voice."""
        try:
            voice_files = {"bumblebee": "/factory/character_voices/bumblebee_voice.wav"}

            results = await self.inject_character_voices(voice_files)
            character_info = None
            if self.engine is not None:
                await self.engine.set_active_character("bumblebee")
                character_info = await self.engine.get_character_info()
            if character_info:
                logger.info(
                    f"Transformers toy setup complete: {character_info['name']}"
                )
            else:
                logger.info("Transformers toy setup complete")

            return all(results.values())

        except Exception as e:
            logger.error(f"Failed to setup Transformers toy: {e}")
            return False

    async def setup_dnd_dragon_toy(self) -> bool:
        """Setup a D&D Dragon toy with Flame's voice."""
        try:
            voice_files = {"flame": "/factory/character_voices/flame_voice.wav"}

            results = await self.inject_character_voices(voice_files)
            character_info = None
            if self.engine is not None:
                await self.engine.set_active_character("flame")
                character_info = await self.engine.get_character_info()
            if character_info:
                logger.info(f"D&D Dragon toy setup complete: {character_info['name']}")
            else:
                logger.info("D&D Dragon toy setup complete")

            return all(results.values())

        except Exception as e:
            logger.error(f"Failed to setup D&D Dragon toy: {e}")
            return False

    async def verify_toy_setup(self) -> Dict[str, Any]:
        """Verify toy is properly set up."""
        try:
            # Get character info
            character_info = None
            voices = []
            health = {}
            if self.engine is not None:
                character_info = await self.engine.get_character_info()
                # Get available voices
                voices = await self.engine.list_character_voices()
                # Get health status
                health = await self.engine.get_health_status()

            return {
                "character": character_info,
                "available_voices": voices,
                "health": health,
                "setup_complete": health["healthy"] and len(voices) > 0,
            }

        except Exception as e:
            logger.error(f"Failed to verify toy setup: {e}")
            return {"error": str(e), "setup_complete": False}


# Factory setup functions for different toy lines
async def setup_toy_for_production(toy_type: str) -> bool:
    """Main factory setup function."""
    setup = ToySetup()
    await setup.initialize_toy()

    if toy_type == "my_little_pony":
        return await setup.setup_my_little_pony_toy()
    elif toy_type == "transformers":
        return await setup.setup_transformers_toy()
    elif toy_type == "dnd_dragon":
        return await setup.setup_dnd_dragon_toy()
    else:
        logger.error(f"Unknown toy type: {toy_type}")
        return False
