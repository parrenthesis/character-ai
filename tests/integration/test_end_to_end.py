import io
import os
from pathlib import Path

import numpy as np
import pytest
import soundfile as sf

from character_ai.core.protocols import AudioData
from character_ai.hardware.toy_hardware_manager import ToyHardwareManager
from character_ai.production.real_time_engine import RealTimeInteractionEngine


def _require_env_var(name: str) -> str:
    value = os.getenv(name)
    if not value:
        pytest.skip(f"skipping integration test: env var {name} not set")
    return value


def _require_file(path: str) -> None:
    if not Path(path).exists():
        pytest.skip(f"skipping integration test: required file missing: {path}")


def _make_tone_wav_bytes(sample_rate: int = 16000, duration_s: float = 0.5) -> bytes:
    t = np.linspace(0, duration_s, int(sample_rate * duration_s), endpoint=False)
    audio_array = (0.2 * np.sin(2 * np.pi * 440 * t)).astype(np.float32)
    bio = io.BytesIO()
    sf.write(bio, audio_array, sample_rate, format="WAV")
    return bio.getvalue()


@pytest.mark.asyncio
@pytest.mark.integration
async def test_engine_e2e_whisper_llama_cpp_text_only():
    # Gate on explicit opt-in
    if os.getenv("CAI_RUN_INTEGRATION") != "1":
        pytest.skip("skipping integration test: CAI_RUN_INTEGRATION!=1")

    gguf_path = _require_env_var("CAI_LLAMA_GGUF")
    _require_file(gguf_path)

    # Configure engine via env (Config reads CAI_* overrides)
    os.environ["CAI_MODELS__LLAMA_BACKEND"] = "llama_cpp"
    os.environ["CAI_MODELS__LLAMA_GGUF_PATH"] = gguf_path
    os.environ["CAI_INTERACTION__STT_LANGUAGE"] = "en"
    os.environ["CAI_INTERACTION__SAMPLE_RATE"] = "16000"
    os.environ["CAI_INTERACTION__CHANNELS"] = "1"

    try:
        hw = ToyHardwareManager()
        engine = RealTimeInteractionEngine(hw)
        await engine.initialize()

        audio = AudioData(
            data=_make_tone_wav_bytes(),
            sample_rate=16000,
            channels=1,
            duration=0.5,
            format="wav",
        )
        res = await engine.process_realtime_audio(audio)

        # We only require a non-error and some text; exact content is model-dependent
        assert res is not None
        assert res.error is None
        assert isinstance(res.text, str)
    finally:
        # Clean up to prevent memory issues
        if "engine" in locals():
            await engine.shutdown()
        if "hw" in locals():
            await hw.shutdown()


@pytest.mark.asyncio
@pytest.mark.integration
async def test_engine_e2e_with_tts_audio_optional():
    # Separate gate for TTS, since it can be heavy
    if os.getenv("CAI_RUN_TTS") != "1":
        pytest.skip("skipping integration TTS test: CAI_RUN_TTS!=1")

    gguf_path = _require_env_var("CAI_LLAMA_GGUF")
    _require_file(gguf_path)

    os.environ["CAI_MODELS__LLAMA_BACKEND"] = "llama_cpp"
    os.environ["CAI_MODELS__LLAMA_GGUF_PATH"] = gguf_path
    os.environ["CAI_INTERACTION__STT_LANGUAGE"] = "en"
    os.environ["CAI_INTERACTION__SAMPLE_RATE"] = "16000"
    os.environ["CAI_INTERACTION__CHANNELS"] = "1"

    try:
        hw = ToyHardwareManager()
        engine = RealTimeInteractionEngine(hw)
        await engine.initialize()

        audio = AudioData(
            data=_make_tone_wav_bytes(),
            sample_rate=16000,
            channels=1,
            duration=0.5,
            format="wav",
        )
        res = await engine.process_realtime_audio(audio)

        # For TTS path, require audio bytes to be present
        assert res is not None
        assert res.error is None
        assert res.audio_data is not None
        assert isinstance(res.audio_data.data, (bytes, bytearray))
        assert len(res.audio_data.data) > 0
    finally:
        # Clean up to prevent memory issues
        if "engine" in locals():
            await engine.shutdown()
        if "hw" in locals():
            await hw.shutdown()
