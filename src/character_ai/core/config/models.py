"""
Model configuration for Character AI Platform.

Contains all model-related configuration classes.
"""

from dataclasses import dataclass
from typing import Optional


@dataclass
class RepresentationLearningConfig:
    """Representation learning model configuration."""

    wav2vec2_model: str = "facebook/wav2vec2-base-960h"
    clap_model: str = "laion/larger_clap_music_and_speech"
    emotion_model: str = "facebook/wav2vec2-base-960h"
    max_audio_length: float = 30.0
    sample_rate: int = 16000


@dataclass
class VoiceCloningConfig:
    """Voice cloning model configuration."""

    coqui_model: str = "tts_models/en/ljspeech/tacotron2-DDC"
    bark_model: str = "suno/bark"
    sovits_model: str = "so-vits-svc"
    max_audio_length: float = 30.0
    sample_rate: int = 22050


@dataclass
class MultimodalFusionConfig:
    """Multimodal fusion model configuration."""

    # Model selection with fallbacks
    mistral_model: str = "mistralai/Mistral-7B-v0.1"  # Gated model (requires HF token)
    phi_model: str = (
        "microsoft/Phi-3-mini-4k-instruct"  # Open-source model (no auth required)
    )
    fallback_model: str = "gpt2"  # Dev fallback
    clap_model: str = "laion/larger_clap_music_and_speech"
    max_context_length: int = 2048
    temperature: float = 0.7

    # Authentication
    huggingface_token: Optional[str] = None
    use_gated_models: bool = True


@dataclass
class ConversationalAIConfig:
    """Conversational AI model configuration."""

    wav2vec2_model: str = "facebook/wav2vec2-base"
    coqui_model: str = "tts_models/en/ljspeech/tacotron2-DDC"
    llama_model: str = "meta-llama/Llama-2-7b-hf"
    max_audio_length: float = 30.0
    sample_rate: int = 16000


@dataclass
class ModelConfig:
    """Model configuration settings."""

    # Conversational AI
    wav2vec2_model: str = "facebook/wav2vec2-base"  # Wav2Vec2 model for STT
    coqui_model: str = "tts_models/en/ljspeech/tacotron2-DDC"  # Coqui TTS model
    llama_model: str = "llama-2-7b-chat"
    llama_quantization: str = "4bit"  # 4bit, 8bit, fp16
    llama_backend: str = "transformers"  # transformers | llama_cpp
    llama_gguf_path: str = "models/llm/tinyllama-1.1b-q4_k_m.gguf"
    llama_n_ctx: int = 2048  # Context window size

    # Representation Learning
    clap_model: str = "laion/larger_clap_music_and_speech"
    bark_model: str = "suno/bark"
    so_vits_model: str = "so-vits-svc"

    # Performance
    max_audio_length: float = 30.0  # seconds
    sample_rate: int = 16000
    max_concurrent_requests: int = 10
    request_timeout: float = 30.0

    # Nested configurations for specific components
    representation_learning: Optional[RepresentationLearningConfig] = None
    voice_cloning: Optional[VoiceCloningConfig] = None
    multimodal_fusion: Optional[MultimodalFusionConfig] = None
    conversational_ai: Optional[ConversationalAIConfig] = None

    def __post_init__(self) -> None:
        """Validate model configuration."""
        if self.llama_quantization not in ["4bit", "8bit", "fp16"]:
            raise ValueError(f"Invalid Llama quantization: {self.llama_quantization}")
