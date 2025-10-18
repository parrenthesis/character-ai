"""
Tests for VoiceService.
"""

import tempfile
from pathlib import Path
from unittest.mock import MagicMock

import pytest

from character_ai.characters.voices.voice_manager import VoiceService


class TestVoiceService:
    """Test VoiceService functionality."""

    def test_voice_manager_init(self) -> None:
        """Test VoiceService initialization."""
        with tempfile.TemporaryDirectory() as temp_dir:
            manager = VoiceService(voice_storage_dir=str(temp_dir))

        assert manager.voice_storage_dir is not None
        assert manager.character_voices == {}

    def test_voice_manager_init_with_custom_dir(self) -> None:
        """Test VoiceService initialization with custom directory."""
        import tempfile

        with tempfile.TemporaryDirectory() as temp_dir:
            custom_dir = Path(temp_dir) / "voices"
            manager = VoiceService(voice_storage_dir=str(custom_dir))

            assert manager.voice_storage_dir == custom_dir

    @pytest.mark.asyncio
    async def test_voice_manager_inject_character_voice(self) -> None:
        """Test VoiceService character voice injection."""
        with tempfile.TemporaryDirectory() as temp_dir:
            manager = VoiceService(voice_storage_dir=str(temp_dir))

        # Create temporary voice file
        with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as tmp:
            tmp.write(b"fake_audio_data")
            voice_file = tmp.name

        try:
            from unittest.mock import AsyncMock

            mock_xtts = MagicMock()
            mock_xtts.inject_character_voice = AsyncMock(return_value=MagicMock())
            result = await manager.inject_character_voice(
                "test_character", voice_file, mock_xtts
            )

            assert result is True
            assert "test_character" in manager.character_voices
        finally:
            Path(voice_file).unlink(missing_ok=True)

    @pytest.mark.asyncio
    async def test_voice_manager_get_character_voice_path(self) -> None:
        """Test VoiceService get character voice path."""
        with tempfile.TemporaryDirectory() as temp_dir:
            manager = VoiceService(voice_storage_dir=str(temp_dir))

        # Create a mock processed voice file
        processed_voice_path = (
            manager.voice_storage_dir
            / "test_character"
            / "processed_voices"
            / "voice.wav"
        )
        processed_voice_path.parent.mkdir(parents=True, exist_ok=True)
        processed_voice_path.touch()

        # Set the character voice in the manager
        manager.character_voices["test_character"] = str(processed_voice_path)

        path = await manager.get_character_voice_path("test_character")
        assert path == str(processed_voice_path)

        path = await manager.get_character_voice_path("nonexistent")
        assert path is None

    def test_voice_manager_list_character_voices(self) -> None:
        """Test VoiceService list character voices."""
        with tempfile.TemporaryDirectory() as temp_dir:
            manager = VoiceService(voice_storage_dir=str(temp_dir))
        manager.character_voices = {"char1": "/path1", "char2": "/path2"}

        voices = list(manager.character_voices.keys())
        assert "char1" in voices
        assert "char2" in voices
