"""Speech-to-Text service wrapper."""

import logging
from typing import TYPE_CHECKING, Any

from ..core.exceptions import handle_audio_error
from ..core.protocols import AudioData
from .base_service import BaseService
from .error_messages import ServiceErrorMessages

if TYPE_CHECKING:
    from ..core.resource_manager import ResourceManager

logger = logging.getLogger(__name__)


class STTService(BaseService):
    """Speech-to-text service wrapper.

    Thin wrapper around ResourceManager's STT processor functionality.
    """

    def __init__(self, resource_manager: "ResourceManager"):
        super().__init__(resource_manager)

    def _get_processor(self, model_type: str) -> Any:
        """Get STT processor from resource manager."""
        return self.resource_manager.get_stt_processor()

    @handle_audio_error
    async def transcribe(self, audio: AudioData) -> str:
        """Transcribe audio to text.

        Args:
            audio: Audio data to transcribe

        Returns:
            Transcribed text string
        """
        # Use common processor initialization pattern
        stt_processor = await self.get_or_create_processor("stt")

        # Use processor
        result = await stt_processor.process_audio(audio)
        transcribed_text = (
            result.text
            if result.text
            else ServiceErrorMessages.get_stt_fallback(has_audio=True)
        )

        return transcribed_text
