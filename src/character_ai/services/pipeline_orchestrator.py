"""Pipeline orchestration for STT->LLM->TTS flow.

Consolidates duplicate pipeline logic with support for:
- Optimized mode (caching, parallel warmup)
- Streaming TTS
- Safety filtering
- Session memory
"""

import asyncio
import logging
import os
import time
from typing import TYPE_CHECKING, Any, Optional

from ..characters import Character
from ..core.protocols import AudioData, AudioResult
from .llm_service import LLMService
from .stt_service import STTService
from .tts_service import TTSService

if TYPE_CHECKING:
    from ..algorithms.conversational_ai.session_memory import SessionMemory
    from ..core.response_cache import ResponseCache

logger = logging.getLogger(__name__)


class PipelineOrchestrator:
    """Orchestrates the full STT->LLM->TTS pipeline.

    Single source of truth for processing pipeline, eliminating duplication.
    """

    def __init__(
        self,
        stt_service: STTService,
        llm_service: LLMService,
        tts_service: TTSService,
        session_memory: "SessionMemory",
        response_cache: "ResponseCache",
        safety_filter: Optional[Any] = None,
        streaming_config: Optional[Any] = None,
    ):
        self.stt_service = stt_service
        self.llm_service = llm_service
        self.tts_service = tts_service
        self.session_memory = session_memory
        self.response_cache = response_cache
        self.safety_filter = safety_filter
        self.streaming_config = streaming_config

    async def process_pipeline(
        self,
        audio: AudioData,
        character: Character,
        optimized: bool = False,
    ) -> AudioResult:
        """Run full STT->LLM->TTS pipeline.

        Args:
            audio: Input audio data
            character: Character for personality/voice
            optimized: Enable cache + parallel warmup

        Returns:
            AudioResult with transcription, response, and audio
        """
        try:
            pipeline_start = time.time()

            # Step 1: STT
            stt_start = time.time()
            transcribed_text = await self.stt_service.transcribe(audio)
            stt_time = time.time() - stt_start
            logger.info(f"⏱️  STT: {stt_time:.2f}s - '{transcribed_text[:50]}...'")

            # Step 2: LLM (with optional caching + parallel warmup)
            safe_response, llm_time = await self._process_llm(
                transcribed_text, character, optimized
            )

            # Step 3: TTS (with optional streaming)
            response_audio, tts_time = await self._process_tts(safe_response, character)

            # Total timing
            total_time = time.time() - pipeline_start
            logger.info(f"⏱️  TOTAL: {total_time:.2f}s (target: <5s)")

            if total_time > 5.0:
                logger.warning(f"⚠️  Exceeded 5s target: {total_time:.2f}s")
                logger.warning(
                    f"   Breakdown: STT={stt_time:.2f}s, LLM={llm_time:.2f}s, TTS={tts_time:.2f}s"
                )

            # Store in session memory
            self.session_memory.add_turn(
                character_name=character.name,
                user_input=transcribed_text,
                character_response=safe_response,
            )

            # response_audio is already an AudioData object from TTS service
            audio_data_obj = response_audio

            return AudioResult(
                text=safe_response,
                audio_data=audio_data_obj,
                metadata={
                    "character": character.name,
                    "character_type": getattr(
                        getattr(character, "dimensions", None), "species", None
                    ),
                    "voice_style": getattr(character, "voice_style", None),
                    "transcribed_text": transcribed_text,
                    "component": "pipeline_orchestrator",
                    "timing": {
                        "stt_time": stt_time,
                        "llm_time": llm_time,
                        "tts_time": tts_time,
                        "total_time": total_time,
                    },
                    "cache_hit": llm_time == 0.0 if optimized else False,
                },
            )

        except Exception as e:
            logger.error(f"Pipeline processing failed: {e}")
            return AudioResult(
                error=f"Pipeline processing failed: {e}",
                metadata={"component": "pipeline_orchestrator"},
            )

    async def _process_llm(
        self, transcribed_text: str, character: Character, optimized: bool
    ) -> tuple[str, float]:
        """Process LLM with optional caching and parallel warmup.

        Returns:
            Tuple of (safe_response, llm_time)
        """
        # Check cache if optimized
        if optimized:
            cached_response = self.response_cache.get(transcribed_text, character.name)
            if cached_response:
                logger.info("💾 Cache hit! Skipping LLM generation")
                return (cached_response, 0.0)

        # Start TTS warmup in parallel if optimized
        tts_warmup_task = None
        if optimized:
            tts_warmup_task = asyncio.create_task(self.tts_service.warmup())

        # Generate response
        llm_start = time.time()
        response_text = await self.llm_service.generate_response(
            transcribed_text, character
        )
        llm_time = time.time() - llm_start
        logger.info(f"⏱️  LLM: {llm_time:.2f}s - '{response_text[:50]}...'")
        logger.debug(f"Full LLM response for TTS: '{response_text}'")

        # Safety filter
        safe_response = response_text
        if self.safety_filter:
            safety_start = time.time()
            safe_response = await self.safety_filter.filter_response(response_text)
            safety_time = time.time() - safety_start
            if safety_time > 0.1:  # Only log if significant
                logger.info(f"⏱️  Safety: {safety_time:.2f}s")

        # Cache the response if optimized
        if optimized:
            self.response_cache.set(transcribed_text, character.name, safe_response)
            # Wait for TTS warmup
            if tts_warmup_task:
                await tts_warmup_task

        return (safe_response, llm_time)

    async def _process_tts(self, text: str, character: Character) -> tuple[Any, float]:
        """Process TTS with optional streaming.

        Returns:
            Tuple of (response_audio, tts_time)
        """
        quiet_mode = os.getenv("CAI_QUIET_MODE") == "1"

        # Check if streaming is enabled
        streaming_enabled = False
        if self.streaming_config and hasattr(self.streaming_config, "enabled"):
            streaming_enabled = getattr(self.streaming_config, "enabled", False)

        if not quiet_mode:
            print(f"\n{'='*60}\nDEBUG: Checking streaming TTS configuration\n{'='*60}")
            print(f"DEBUG: Streaming enabled = {streaming_enabled}")

        logger.info(f"🔊 TTS Streaming: enabled={streaming_enabled}")

        tts_start = time.time()
        response_audio = None

        # Get sample rate from TTS service config
        tts_sample_rate = getattr(
            self.tts_service.resource_manager.config.tts,
            "voice_cloning_sample_rate",
            22050,
        )

        if streaming_enabled:
            # Streaming TTS synthesis
            if not quiet_mode:
                print("DEBUG: Entering streaming TTS path")
            logger.info("Using streaming TTS synthesis")

            try:
                if not quiet_mode:
                    print("DEBUG: About to call synthesize_streaming")

                # Collect audio chunks for return value
                audio_chunks: list[Any] = []
                async for chunk in self.tts_service.synthesize_streaming(
                    text, character
                ):
                    if not quiet_mode:
                        chunk_size = (
                            len(chunk.data)
                            if hasattr(chunk, "data")
                            else len(chunk)
                            if isinstance(chunk, bytes)
                            else "unknown"
                        )
                        print(
                            f"DEBUG: Received chunk {len(audio_chunks)+1}, size={chunk_size} bytes"
                        )
                    audio_chunks.append(chunk.data if hasattr(chunk, "data") else chunk)

                if not quiet_mode:
                    print(f"DEBUG: Finished collecting {len(audio_chunks)} chunks")

                # Concatenate all chunks for return value
                if audio_chunks:
                    response_audio = self._concatenate_audio_chunks(audio_chunks)

                tts_time = time.time() - tts_start
                if response_audio is not None:
                    logger.info(
                        f"⏱️  Streaming TTS: {tts_time:.2f}s ({len(audio_chunks)} chunks) - {len(response_audio.data)} samples"
                    )
                else:
                    logger.info(
                        f"⏱️  Streaming TTS: {tts_time:.2f}s ({len(audio_chunks)} chunks) - no audio"
                    )

            except Exception as e:
                if not quiet_mode:
                    print(f"DEBUG: Streaming TTS EXCEPTION: {e}")
                logger.error(f"Streaming TTS failed: {e}")

                # Fallback to blocking synthesis
                fallback_enabled = True
                if self.streaming_config and hasattr(
                    self.streaming_config, "fallback_to_blocking"
                ):
                    fallback_enabled = getattr(
                        self.streaming_config, "fallback_to_blocking", True
                    )

                if fallback_enabled:
                    logger.info("Falling back to blocking TTS")
                    audio_bytes = await self.tts_service.synthesize_blocking(
                        text, character
                    )
                    tts_time = time.time() - tts_start
                    if audio_bytes and len(audio_bytes) > 0:
                        from ..core.protocols import AudioData

                        # TTS already returns WAV bytes, just wrap in AudioData
                        response_audio = AudioData(
                            data=audio_bytes,
                            sample_rate=tts_sample_rate,
                            duration=len(audio_bytes) / tts_sample_rate,
                            channels=1,
                        )
                        logger.info(
                            f"⏱️  TTS (fallback): {tts_time:.2f}s - {len(audio_bytes)} bytes"
                        )
                    else:
                        response_audio = None
                        logger.info(f"⏱️  TTS (fallback): {tts_time:.2f}s - no audio")
                else:
                    raise
        else:
            # Original blocking TTS
            if not quiet_mode:
                print("DEBUG: Using blocking TTS (streaming disabled)")
            audio_bytes = await self.tts_service.synthesize_blocking(text, character)
            print(
                f"DEBUG: TTS returned - type: {type(audio_bytes)}, len: {len(audio_bytes) if hasattr(audio_bytes, '__len__') else 'no len'}"
            )
            if audio_bytes and len(audio_bytes) > 0:
                from ..core.protocols import AudioData

                # Create AudioData with bytes (as per protocol definition)
                response_audio = AudioData(
                    data=audio_bytes,
                    sample_rate=tts_sample_rate,
                    duration=len(audio_bytes) / tts_sample_rate,
                    channels=1,
                )
            else:
                response_audio = None
            tts_time = time.time() - tts_start
            if response_audio is not None:
                logger.info(
                    f"⏱️  TTS: {tts_time:.2f}s - {len(response_audio.data)} bytes"
                )
            else:
                logger.info(f"⏱️  TTS: {tts_time:.2f}s - no audio")

        return (
            (response_audio, tts_time)
            if response_audio is not None
            else (None, tts_time)
        )

    def _concatenate_audio_chunks(self, chunks: list[Any]) -> AudioData:
        """Combine multiple audio chunks into single AudioData.

        Args:
            chunks: List of audio chunks (bytes or numpy arrays)

        Returns:
            Combined AudioData object
        """
        if not chunks:
            from ..core.protocols import AudioData

            return AudioData(data=b"", sample_rate=44100, duration=0.0, channels=1)

        # Delegate to audio_io utilities for proper audio processing
        from ..core.audio_io.audio_utils import concatenate_audio_chunks

        result = concatenate_audio_chunks(chunks, target_sample_rate=44100)
        if result is None:
            from ..core.protocols import AudioData

            return AudioData(data=b"", sample_rate=44100, duration=0.0, channels=1)
        return result
