"""
LLM provider implementations for the Character AI.

Supports both open (local) and token-based (cloud) LLM providers.
"""

import asyncio
import logging
from abc import ABC, abstractmethod
from enum import Enum
from typing import Any, AsyncGenerator, Dict, Optional

logger = logging.getLogger(__name__)


class LLMProvider(Enum):
    """Supported LLM providers."""

    LOCAL = "local"
    OLLAMA = "ollama"
    OPENAI = "openai"
    ANTHROPIC = "anthropic"


class LLMInterface(ABC):
    """Abstract interface for LLM providers."""

    @abstractmethod
    async def generate(self, prompt: str, **kwargs: Any) -> str:
        """Generate a response from the LLM."""
        pass

    @abstractmethod
    def stream(self, prompt: str, **kwargs: Any) -> AsyncGenerator[str, None]:
        """Stream a response from the LLM."""
        pass

    @abstractmethod
    def get_capabilities(self) -> Dict[str, Any]:
        """Get provider capabilities."""
        pass


class LocalLLMProvider(LLMInterface):
    """Local LLM provider using llama.cpp or transformers."""

    def __init__(self, config: Dict[str, Any]):
        self.model_path = config["model_path"]
        self.device = config.get("device", "cpu")
        self.quantization = config.get("quantization", "q4_k_m")
        self._model: Optional[Any] = None
        self._tokenizer: Optional[Any] = None
        self.provider = config.get("provider", "llama_cpp")

    async def _load_model(self) -> None:
        """Load local model (llama.cpp, transformers, etc.)"""
        if self.provider == "llama_cpp":
            try:
                from llama_cpp import Llama

                self._model = Llama(
                    model_path=self.model_path,
                    n_ctx=2048,
                    n_gpu_layers=0 if self.device == "cpu" else -1,
                    verbose=False,
                )
            except ImportError:
                logger.warning(
                    "llama-cpp-python not available, falling back to transformers"
                )
                self.provider = "transformers"

        if self.provider == "transformers":
            try:
                import torch
                from transformers import AutoModelForCausalLM, AutoTokenizer

                # Pin to specific revision for security
                self._tokenizer = AutoTokenizer.from_pretrained(
                    self.model_path,
                    revision="main",  # nosec B615 - Using main branch for stability
                    trust_remote_code=False,  # Disable remote code execution
                )
                self._model = AutoModelForCausalLM.from_pretrained(
                    self.model_path,
                    revision="main",  # nosec B615 - Using main branch for stability
                    trust_remote_code=False,  # Disable remote code execution
                    device_map="auto" if self.device == "cuda" else "cpu",
                    torch_dtype=(
                        torch.float16 if self.device == "cuda" else torch.float32
                    ),
                )
            except ImportError:
                logger.error("transformers not available for local LLM")
                raise

    async def generate(self, prompt: str, **kwargs: Any) -> str:
        """Generate response using local model"""
        if not self._model:
            await self._load_model()

        if self.provider == "llama_cpp" and self._model is not None:
            # Compose stop sequences: built-ins + caller-provided + generic role labels
            caller_stop = kwargs.get("stop", []) or []
            stop_sequences = [
                "</s>",
                "\n\n",
                "\n",
                "user:",
                "User:",
                "assistant:",
                "Assistant:",
                "system:",
                "System:",
            ]
            if isinstance(caller_stop, list):
                stop_sequences = stop_sequences + caller_stop

            response = self._model(
                prompt,
                max_tokens=kwargs.get("max_tokens", 150),
                temperature=kwargs.get("temperature", 0.7),
                stop=stop_sequences,
            )
            text = response["choices"][0]["text"]

            # Enforce single-line, no role labels, no leading stage directions
            try:
                import re

                line = (text or "").strip().splitlines()[0] if text else ""
                # remove leading role labels like "user:" or "assistant:" (case-insensitive)
                line = re.sub(
                    r"^(user|assistant|system)\s*:\s*", "", line, flags=re.IGNORECASE
                )
                # remove leading parenthetical stage directions e.g., "(smiling) "
                line = re.sub(r"^\([^)]*\)\s*", "", line)
                return line
            except Exception:
                return (text or "").strip()
        elif (
            self.provider == "transformers"
            and self._model is not None
            and self._tokenizer is not None
        ):
            import torch

            inputs = self._tokenizer(prompt, return_tensors="pt")
            with torch.no_grad():
                outputs = self._model.generate(
                    **inputs,
                    max_new_tokens=kwargs.get("max_tokens", 150),
                    temperature=kwargs.get("temperature", 0.7),
                    do_sample=True,
                    pad_token_id=self._tokenizer.eos_token_id,
                )
            return self._tokenizer.decode(outputs[0], skip_special_tokens=True)  # type: ignore
        else:
            # Fallback for unknown providers
            return "Error: Unknown provider"

    def stream(self, prompt: str, **kwargs: Any) -> AsyncGenerator[str, None]:
        """Stream response from local model"""

        async def _stream() -> AsyncGenerator[str, None]:
            # For now, generate full response and yield word by word
            response = await self.generate(prompt, **kwargs)
            words = response.split()

            for word in words:
                yield word + " "
                await asyncio.sleep(0.05)  # Simulate streaming delay

        return _stream()

    def get_capabilities(self) -> Dict[str, Any]:
        """Get local LLM capabilities"""
        return {
            "provider": "local",
            "supports_streaming": True,
            "supports_temperature": True,
            "supports_max_tokens": True,
            "cost_per_token": 0.0,
            "requires_internet": False,
        }


class OllamaProvider(LLMInterface):
    """Ollama provider for open LLMs."""

    def __init__(self, config: Dict[str, Any]):
        self.model = config["model"]
        self.base_url = config.get("base_url", "http://localhost:11434")
        self.client: Optional[Any] = None

    async def _get_client(self) -> Any:
        """Get async HTTP client"""
        if not self.client:
            try:
                from httpx import AsyncClient

                self.client = AsyncClient(base_url=self.base_url)
            except ImportError:
                logger.error("httpx not available for Ollama provider")
                raise

    async def generate(self, prompt: str, **kwargs: Any) -> str:
        """Generate response using Ollama"""
        await self._get_client()

        if self.client is None:
            raise RuntimeError("Client not initialized")

        response = await self.client.post(
            "/api/generate",
            json={
                "model": self.model,
                "prompt": prompt,
                "stream": False,
                "options": {
                    "temperature": kwargs.get("temperature", 0.7),
                    "num_predict": kwargs.get("max_tokens", 150),
                },
            },
        )
        return str(response.json()["response"])

    def stream(self, prompt: str, **kwargs: Any) -> AsyncGenerator[str, None]:
        """Stream response from Ollama"""

        async def _stream() -> AsyncGenerator[str, None]:
            await self._get_client()

            if self.client is None:
                raise RuntimeError("Client not initialized")

            async with self.client.stream(
                "POST",
                "/api/generate",
                json={
                    "model": self.model,
                    "prompt": prompt,
                    "stream": True,
                    "options": {
                        "temperature": kwargs.get("temperature", 0.7),
                        "num_predict": kwargs.get("max_tokens", 150),
                    },
                },
            ) as response:
                async for line in response.aiter_lines():
                    if line.strip():
                        try:
                            import json

                            data = json.loads(line)
                            if "response" in data:
                                yield data["response"]
                        except json.JSONDecodeError:
                            continue

        return _stream()

    def get_capabilities(self) -> Dict[str, Any]:
        """Get Ollama capabilities"""
        return {
            "provider": "ollama",
            "supports_streaming": True,
            "supports_temperature": True,
            "supports_max_tokens": True,
            "cost_per_token": 0.0,
            "requires_internet": False,
        }


class OpenAIProvider(LLMInterface):
    """OpenAI provider for token-based LLMs."""

    def __init__(self, config: Dict[str, Any]):
        self.api_key = config["api_key"]
        self.model = config["model"]
        self.temperature = config.get("temperature", 0.7)
        self.max_tokens = config.get("max_tokens", 150)
        self.client: Optional[Any] = None

    async def _get_client(self) -> Any:
        """Get OpenAI client"""
        if not self.client:
            try:
                from openai import AsyncOpenAI

                self.client = AsyncOpenAI(api_key=self.api_key)
            except ImportError:
                logger.error("openai not available for OpenAI provider")
                raise

    async def generate(self, prompt: str, **kwargs: Any) -> str:
        """Generate response using OpenAI API"""
        await self._get_client()

        if self.client is None:
            raise RuntimeError("Client not initialized")

        response = await self.client.chat.completions.create(
            model=self.model,
            messages=[{"role": "user", "content": prompt}],
            temperature=kwargs.get("temperature", self.temperature),
            max_tokens=kwargs.get("max_tokens", self.max_tokens),
        )
        return str(response.choices[0].message.content)

    def stream(self, prompt: str, **kwargs: Any) -> AsyncGenerator[str, None]:
        """Stream response from OpenAI"""

        async def _stream() -> AsyncGenerator[str, None]:
            await self._get_client()

            if self.client is None:
                raise RuntimeError("Client not initialized")

            stream = await self.client.chat.completions.create(
                model=self.model,
                messages=[{"role": "user", "content": prompt}],
                temperature=kwargs.get("temperature", self.temperature),
                max_tokens=kwargs.get("max_tokens", self.max_tokens),
                stream=True,
            )

            async for chunk in stream:
                if chunk.choices[0].delta.content:
                    yield chunk.choices[0].delta.content

        return _stream()

    def get_capabilities(self) -> Dict[str, Any]:
        """Get OpenAI capabilities"""
        return {
            "provider": "openai",
            "supports_streaming": True,
            "supports_temperature": True,
            "supports_max_tokens": True,
            "cost_per_token": 0.15,  # Approximate cost per 1K tokens
            "requires_internet": True,
        }


class AnthropicProvider(LLMInterface):
    """Anthropic provider for token-based LLMs."""

    def __init__(self, config: Dict[str, Any]):
        self.api_key = config["api_key"]
        self.model = config["model"]
        self.temperature = config.get("temperature", 0.7)
        self.max_tokens = config.get("max_tokens", 150)
        self.client: Optional[Any] = None

    async def _get_client(self) -> Any:
        """Get Anthropic client"""
        if not self.client:
            try:
                from anthropic import AsyncAnthropic

                self.client = AsyncAnthropic(api_key=self.api_key)
            except ImportError:
                logger.error("anthropic not available for Anthropic provider")
                raise

    async def generate(self, prompt: str, **kwargs: Any) -> str:
        """Generate response using Anthropic API"""
        await self._get_client()

        if self.client is None:
            raise RuntimeError("Client not initialized")

        response = await self.client.messages.create(
            model=self.model,
            max_tokens=kwargs.get("max_tokens", self.max_tokens),
            temperature=kwargs.get("temperature", self.temperature),
            messages=[{"role": "user", "content": prompt}],
        )
        return str(response.content[0].text)

    def stream(self, prompt: str, **kwargs: Any) -> AsyncGenerator[str, None]:
        """Stream response from Anthropic"""

        async def _stream() -> AsyncGenerator[str, None]:
            await self._get_client()

            if self.client is None:
                raise RuntimeError("Client not initialized")

            stream = await self.client.messages.create(
                model=self.model,
                max_tokens=kwargs.get("max_tokens", self.max_tokens),
                temperature=kwargs.get("temperature", self.temperature),
                messages=[{"role": "user", "content": prompt}],
                stream=True,
            )

            async for chunk in stream:
                if chunk.type == "content_block_delta":
                    yield chunk.delta.text

        return _stream()

    def get_capabilities(self) -> Dict[str, Any]:
        """Get Anthropic capabilities"""
        return {
            "provider": "anthropic",
            "supports_streaming": True,
            "supports_temperature": True,
            "supports_max_tokens": True,
            "cost_per_token": 0.10,  # Approximate cost per 1K tokens
            "requires_internet": True,
        }
