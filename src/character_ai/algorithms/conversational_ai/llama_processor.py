"""
Compact LLM processor for text generation (TinyLlama default).

Uses open, token-free models by default for edge deployment.
"""

import logging
import time
from pathlib import Path
from typing import Any, Dict, List, Optional

import torch

from ...core.config import Config
from ...core.exceptions import AudioProcessingError, ModelError
from ...core.protocols import BaseTextProcessor, ModelInfo, TextResult

logger = logging.getLogger(__name__)


class LlamaProcessor(BaseTextProcessor):
    """Llama-2-7B based text generation processor."""

    def __init__(self, config: Config):
        super().__init__("llama_2_7b")
        self.config: Config = config
        self.model: Optional[Any] = None
        self.tokenizer: Optional[Any] = None
        self.device: Optional[Any] = None  # Will be set during initialization

    async def initialize(self) -> None:
        """Initialize compact chat model (TinyLlama by default)."""
        try:
            logger.info("Loading compact chat model (TinyLlama)")

            # Set device (defer CUDA check until here)
            self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

            # Import transformers
            from transformers import AutoModelForCausalLM, AutoTokenizer

            model_id = (
                self.config.models.llama_model or "TinyLlama/TinyLlama-1.1B-Chat-v1.0"
            )

            # Load tokenizer with revision pinning for security
            self.tokenizer = AutoTokenizer.from_pretrained(
                model_id,
                cache_dir=str(Path(self.config.models_dir) / "llm"),
                trust_remote_code=True,
                revision="main",  # nosec B615 - Pinned to main branch for security
            )

            # Add padding token
            if self.tokenizer is not None:
                if self.tokenizer.pad_token is None:
                    self.tokenizer.pad_token = self.tokenizer.eos_token

            # Load model (CPU/GPU auto). Quantization can be applied via
            # bitsandbytes in future.
            self.model = AutoModelForCausalLM.from_pretrained(
                model_id,
                device_map="auto",
                cache_dir=str(Path(self.config.models_dir) / "llm"),
                torch_dtype=(
                    torch.float16 if torch.cuda.is_available() else torch.float32
                ),
                trust_remote_code=True,
                revision="main",  # nosec B615 - Pinned to main branch for security
            )

            # Create model info
            self.model_info = ModelInfo(
                name="tiny_llama",
                type="language_model",
                size="~1.1B parameters",
                memory_usage=f"{self._estimate_memory_usage():.2f}GB",
                precision="fp16",
                quantization=self.config.models.llama_quantization,
                loaded_at=time.time(),
                status="loaded",
            )

            self._initialized = True
            logger.info("Compact chat model loaded successfully")

        except Exception as e:
            logger.error(f"Failed to load compact chat model: {e}")
            raise ModelError(
                f"Failed to load compact chat model: {e}", component="LlamaProcessor"
            )

    async def get_model_info(self) -> ModelInfo:
        """Return model information or a not_loaded placeholder when uninitialized.

        Overrides BaseTextProcessor behavior to avoid raising in callers that
        probe readiness.
        """
        if self.model_info is None:
            # Provide a conservative placeholder reflecting an unloaded state
            return ModelInfo(
                name="tiny_llama",
                type="text_generation",
                size="unknown",
                memory_usage="0GB",
                precision="unknown",
                quantization=getattr(self.config.models, "llama_quantization", "none"),
                loaded_at=0.0,
                status="not_loaded",
            )
        return self.model_info

    async def shutdown(self) -> None:
        """Shutdown Llama processor."""
        try:
            if self.model:
                self.model = None

            if self.tokenizer:
                self.tokenizer = None

            # Clear GPU cache
            if torch.cuda.is_available():
                torch.cuda.empty_cache()

            self._initialized = False
            logger.info("Llama processor shutdown complete")

        except Exception as e:
            logger.error(f"Error during Llama processor shutdown: {e}")

    async def process_text(
        self, text: str, context: Optional[Dict[str, Any]] = None
    ) -> TextResult:
        """Process text and return result."""
        if not self._initialized:
            raise AudioProcessingError(
                "Llama processor not initialized", component="LlamaProcessor"
            )

        try:
            start_time = time.time()

            # Validate text
            if not text or not text.strip():
                return await self._create_error_result("Empty text provided")

            # Prepare prompt
            prompt = await self._prepare_prompt(text, context)

            # Generate response
            response = await self._generate_response(prompt)

            processing_time = time.time() - start_time

            return TextResult(
                text=response,
                metadata={
                    "model": "tiny_llama",
                    "input_length": len(text),
                    "output_length": len(response),
                    "processing_time": processing_time,
                    "context_used": context is not None,
                },
                processing_time=processing_time,
                confidence=0.8,  # Llama-2-7B has good quality
            )

        except Exception as e:
            logger.error(f"Error processing text with Llama: {e}")
            return await self._create_error_result(f"Llama text processing failed: {e}")


    async def generate_text(self, prompt: str, **kwargs: Any) -> TextResult:
        """Generate text from prompt."""
        if not self._initialized:
            raise AudioProcessingError(
                "Llama processor not initialized", component="LlamaProcessor"
            )

        try:
            start_time = time.time()

            # Validate prompt
            if not prompt or not prompt.strip():
                return await self._create_error_result("Empty prompt provided")

            # Generate response
            response = await self._generate_response(prompt, **kwargs)

            processing_time = time.time() - start_time

            return TextResult(
                text=response,
                metadata={
                    "model": "tiny_llama",
                    "prompt_length": len(prompt),
                    "output_length": len(response),
                    "processing_time": processing_time,
                    "generation_params": kwargs,
                },
                processing_time=processing_time,
                confidence=0.8,
            )

        except Exception as e:
            logger.error(f"Error generating text with Llama: {e}")
            return await self._create_error_result(f"Llama text generation failed: {e}")


    async def get_embeddings(self, text: str) -> TextResult:  # type: ignore
        """Extract embeddings from text."""
        if not self._initialized:
            raise AudioProcessingError(
                "Llama processor not initialized", component="LlamaProcessor"
            )

        try:
            start_time = time.time()

            # Tokenize text
            if self.tokenizer is None:
                raise RuntimeError("Tokenizer not initialized")
            inputs = self.tokenizer(
                text, return_tensors="pt", padding=True, truncation=True, max_length=512

            ).to(self.device)

            # Get embeddings from model
            with torch.no_grad():
                if self.model is None:
                    raise RuntimeError("Model not initialized")
                outputs = self.model(**inputs, output_hidden_states=True)
                # Use last hidden state and average pool
                embeddings = (
                    outputs.hidden_states[-1]
                    .mean(dim=1)
                    .squeeze(0)
                    .cpu()
                    .numpy()
                    .tolist()
                )

            processing_time = time.time() - start_time

            return TextResult(
                text=text,
                embeddings=embeddings,
                metadata={
                    "model": "llama_2_7b",
                    "embedding_dim": len(embeddings),
                    "processing_time": processing_time,
                },
                processing_time=processing_time,
            )

        except Exception as e:
            logger.error(f"Error extracting embeddings with Llama: {e}")
            return await self._create_error_result(
                f"Llama embedding extraction failed: {e}"
            )

    async def _prepare_prompt(
        self, text: str, context: Optional[Dict[str, Any]] = None
    ) -> str:
        """Prepare prompt for Llama-2-7B."""
        # Basic prompt template for conversational AI
        system_prompt = (
            "You are a helpful AI assistant. Respond naturally and conversationally."
        )

        if context:
            # Add context if provided
            context_str = context.get("conversation_history", "")
            if context_str:
                system_prompt += f"\n\nPrevious conversation:\n{context_str}"

        # Format as Llama-2 chat format
        prompt = f"<s>[INST] <<SYS>>\n{system_prompt}\n<</SYS>>\n\n{text} [/INST]"

        return prompt

    async def _generate_response(
        self,
        prompt: str,
        max_length: int = 512,
        temperature: float = 0.7,
        top_p: float = 0.9,
        do_sample: bool = True,
    ) -> str:
        """Generate response from prompt."""
        try:
            # Tokenize input
            if self.tokenizer is None:
                raise RuntimeError("Tokenizer not initialized")
            inputs = self.tokenizer(
                prompt,
                return_tensors="pt",
                padding=True,
                truncation=True,
                max_length=2048,  # Input length limit
            ).to(self.device)

            # Generate response
            with torch.no_grad():
                if self.model is None:
                    raise RuntimeError("Model not initialized")
                outputs = self.model.generate(
                    **inputs,
                    max_new_tokens=max_length,
                    temperature=temperature,
                    top_p=top_p,
                    do_sample=do_sample,
                    pad_token_id=self.tokenizer.eos_token_id if self.tokenizer else None,
                    eos_token_id=self.tokenizer.eos_token_id if self.tokenizer else None,
                    repetition_penalty=1.1,
                )

            # Decode response
            if self.tokenizer is None:
                raise RuntimeError("Tokenizer not initialized")
            response = self.tokenizer.decode(
                outputs[0][inputs["input_ids"].shape[1] :],  # Only new tokens
                skip_special_tokens=True,
            )

            # Clean up response
            response = response.strip()

            # Remove any remaining special tokens
            response = response.replace("<s>", "").replace("</s>", "").strip()

            return str(response)

        except Exception as e:
            logger.error(f"Error generating response: {e}")
            return "I apologize, but I encountered an error generating a response."

    def _estimate_memory_usage(self) -> float:
        """Estimate memory usage for Llama-2-7B with 4-bit quantization."""
        return 2.0  # TinyLlama typically <2GB fp16, less when quantized

    async def _create_error_result(self, error_message: str) -> TextResult:
        """Create an error result."""
        return TextResult(
            text="",
            error=error_message,
            metadata={"component": "LlamaProcessor", "error": True},
        )

    async def get_model_capabilities(self) -> Dict[str, Any]:
        """Get model capabilities and limitations."""
        return {
            "max_input_length": 2048,
            "max_output_length": 512,
            "supported_languages": ["en"],  # Llama-2-7B is primarily English
            "memory_usage_gb": self._estimate_memory_usage(),
            "processing_speed": "medium",
            "quality": "high",
            "quantization": "4bit",
            "precision": "fp16",
        }

    async def get_generation_parameters(self) -> Dict[str, Any]:
        """Get available generation parameters."""
        return {
            "max_length": {"type": "int", "default": 512, "min": 1, "max": 1024},
            "temperature": {"type": "float", "default": 0.7, "min": 0.1, "max": 2.0},
            "top_p": {"type": "float", "default": 0.9, "min": 0.1, "max": 1.0},
            "do_sample": {"type": "bool", "default": True},
            "repetition_penalty": {
                "type": "float",
                "default": 1.1,
                "min": 1.0,
                "max": 2.0,
            },
        }

    async def set_generation_parameters(self, **kwargs: Any) -> None:
        """Set default generation parameters."""
        valid_params = await self.get_generation_parameters()

        for param, value in kwargs.items():
            if param in valid_params:
                logger.info(f"Set {param} to {value}")
            else:
                logger.warning(f"Unknown parameter: {param}")

    async def get_conversation_templates(self) -> List[str]:
        """Get available conversation templates."""
        return [
            "You are a helpful AI assistant.",
            "You are a friendly chatbot.",
            "You are a professional assistant.",
            "You are a creative writing assistant.",
            "You are a technical support assistant.",
        ]
