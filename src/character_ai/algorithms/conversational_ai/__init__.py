"""
Conversational AI algorithms and processors.

Organization:
- Processors: Model interfaces (LLM, TTS, STT)
  - llama_cpp_processor: LLM text generation
  - coqui_processor: TTS with voice cloning
  - wav2vec2_processor: STT speech recognition

- Memory System: Conversation persistence and tracking
  - hybrid_memory: Orchestrates preferences, storage, summarization
  - optimized_conversation_storage: Persistent SQLite storage
  - session_memory: In-memory session tracking
  - preference_extractor: User preference extraction
  - conversation_summarizer: LLM-based history compression

- Utilities:
  - text_normalizer: Text preprocessing for LLM/TTS
"""

from typing import List

from .coqui_processor import CoquiProcessor
from .hybrid_memory import HybridMemorySystem

# Import commonly used classes for convenience
from .llama_cpp_processor import LlamaCppProcessor
from .text_normalizer import TextNormalizer
from .wav2vec2_processor import Wav2Vec2Processor

__all__: List[str] = [
    "LlamaCppProcessor",
    "CoquiProcessor",
    "Wav2Vec2Processor",
    "HybridMemorySystem",
    "TextNormalizer",
]
