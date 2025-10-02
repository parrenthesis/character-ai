"""
Pytest configuration and fixtures for Character AI.
Only external/integration deps are mocked per policy.
"""

import asyncio
import io
import os
import shutil
import sys
import tempfile
import types
import warnings
from pathlib import Path
from typing import Generator
from unittest.mock import MagicMock, patch

import numpy as np
import pytest
import soundfile as sf

# Suppress PyTorch RuntimeError about docstring already existing
try:
    pass
except RuntimeError as e:
    if "already has a docstring" in str(e):
        # This is a known PyTorch bug, ignore it
        pass
    else:
        raise

from character_ai.core.config import Config, Environment
from character_ai.core.protocols import AudioData

# Set up test environment to suppress expected warnings
os.environ["CUDA_VISIBLE_DEVICES"] = ""
os.environ[
    "PYTHONWARNINGS"
] = "ignore::RuntimeWarning:sys:1,ignore::RuntimeWarning:unittest.mock,ignore::RuntimeWarning:tracemalloc"

# Suppress expected warnings globally
warnings.filterwarnings("ignore", category=UserWarning, message="CUDA.*unknown error")
warnings.filterwarnings(
    "ignore", category=RuntimeWarning, message="coroutine.*was never awaited"
)
warnings.filterwarnings(
    "ignore", category=RuntimeWarning, message=".*found in sys.modules.*"
)
warnings.filterwarnings(
    "ignore",
    category=PendingDeprecationWarning,
    message="Please use.*import python_multipart",
)
warnings.filterwarnings("ignore", category=RuntimeWarning, message=".*AsyncMockMixin.*")
warnings.filterwarnings(
    "ignore", category=RuntimeWarning, message=".*_execute_mock_call.*"
)
warnings.filterwarnings(
    "ignore", category=RuntimeWarning, message=".*add_relationship.*"
)
warnings.filterwarnings(
    "ignore", category=RuntimeWarning, message=".*add_localization.*"
)
warnings.filterwarnings("ignore", category=RuntimeWarning, message=".*set_licensing.*")

# Suppress all RuntimeWarnings from sys module
warnings.filterwarnings("ignore", category=RuntimeWarning, module="sys")
warnings.filterwarnings("ignore", category=RuntimeWarning, message=".*coroutine.*")
warnings.filterwarnings("ignore", category=RuntimeWarning, message=".*AsyncMockMixin.*")
warnings.filterwarnings(
    "ignore", category=RuntimeWarning, message=".*_execute_mock_call.*"
)

# Suppress specific async mock warnings from unittest.mock
warnings.filterwarnings(
    "ignore", category=RuntimeWarning, message=".*coroutine.*was never awaited"
)
warnings.filterwarnings(
    "ignore", category=RuntimeWarning, message=".*AsyncMockMixin.*_execute_mock_call.*"
)
warnings.filterwarnings(
    "ignore", category=RuntimeWarning, message=".*add_relationship.*_add_relationship.*"
)
warnings.filterwarnings(
    "ignore", category=RuntimeWarning, message=".*add_localization.*_add_localization.*"
)
warnings.filterwarnings(
    "ignore", category=RuntimeWarning, message=".*set_licensing.*_set_licensing.*"
)


# Suppress expected warnings
warnings.filterwarnings("ignore", category=RuntimeWarning, module="character_ai.cli")
warnings.filterwarnings("ignore", category=UserWarning, message="CUDA.*unknown error")
warnings.filterwarnings(
    "ignore", category=RuntimeWarning, message="coroutine.*was never awaited"
)


@pytest.fixture(scope="session")
def event_loop() -> Generator[asyncio.AbstractEventLoop, None, None]:
    loop = asyncio.get_event_loop_policy().new_event_loop()
    yield loop
    loop.close()


@pytest.fixture
def temp_dir() -> Generator[Path, None, None]:
    path = Path(tempfile.mkdtemp())
    yield path
    shutil.rmtree(path, ignore_errors=True)


@pytest.fixture
def test_config(temp_dir: Path) -> Config:
    cfg = Config(environment=Environment.TESTING)
    cfg.data_dir = temp_dir / "data"
    cfg.cache_dir = temp_dir / "cache"
    cfg.logs_dir = temp_dir / "logs"
    cfg.paths.models_dir = temp_dir / "models"
    cfg.paths.voices_dir = temp_dir / "voices"
    return cfg


@pytest.fixture
def sample_audio_data() -> AudioData:
    sample_rate = 16000
    duration = 1.0
    t = np.linspace(0, duration, int(sample_rate * duration), endpoint=False)
    audio_array = np.sin(2 * np.pi * 440 * t).astype(np.float32)
    bio = io.BytesIO()
    sf.write(bio, audio_array, sample_rate, format="WAV")
    return AudioData(
        data=bio.getvalue(),
        sample_rate=sample_rate,
        channels=1,
        duration=duration,
        format="wav",
    )


@pytest.fixture(autouse=True)
def setup_test_environment() -> Generator[None, None, None]:
    """Set up test environment to suppress expected warnings."""
    # Environment is already set up globally, but this ensures it's applied per test
    yield


@pytest.fixture(autouse=True)
def suppress_async_mock_warnings() -> Generator[None, None, None]:
    """Suppress async mock warnings that are expected in test environment."""
    import warnings

    # Create a custom warning filter that catches all the specific patterns
    def custom_warning_filter(
        message: str,
        category: type,
        filename: str,
        lineno: int,
        file: str | None = None,
        line: str | None = None,
    ) -> bool:
        """Custom warning filter that suppresses async mock warnings."""
        if category is RuntimeWarning:
            msg_str = str(message)
            # Suppress specific async mock patterns
            if any(
                pattern in msg_str
                for pattern in [
                    "coroutine 'AsyncMockMixin._execute_mock_call' was never awaited",
                    "coroutine 'add_relationship.<locals>._add_relationship' was never awaited",
                    "coroutine 'add_localization.<locals>._add_localization' was never awaited",
                    "coroutine 'set_licensing.<locals>._set_licensing' was never awaited",
                    "AsyncMockMixin",
                    "_execute_mock_call",
                ]
            ):
                return True

        # Call the original warning handler for other warnings
        if hasattr(warnings, "_showwarning_orig"):
            warnings._showwarning_orig(message, category, filename, lineno, file, line)
        return False

    # Store original warning handler
    if not hasattr(warnings, "_showwarning_orig"):
        warnings._showwarning_orig = warnings.showwarning  # type: ignore
    warnings.showwarning = custom_warning_filter  # type: ignore

    yield

    # Restore original warning handler
    if hasattr(warnings, "_showwarning_orig"):
        warnings.showwarning = warnings._showwarning_orig


@pytest.fixture(autouse=True)
def mock_log_aggregator() -> Generator[MagicMock, None, None]:
    """Mock get_log_aggregator to prevent logs directory creation."""
    with patch(
        "character_ai.core.log_aggregation.get_log_aggregator"
    ) as mock_get_log_aggregator:
        mock_log_aggregator = MagicMock()
        mock_get_log_aggregator.return_value = mock_log_aggregator
        yield mock_get_log_aggregator


@pytest.fixture(autouse=True)
def mock_external_deps(request: pytest.FixtureRequest) -> Generator[None, None, None]:
    # Allow real-component integration tests to bypass mocks
    if request.node.get_closest_marker("integration"):
        yield
        return

    # TTS (Coqui) - Mock the entire Coqui TTS import chain to avoid dependency issues
    def create_mock_module(name: str) -> MagicMock:
        mock = MagicMock()
        mock.__name__ = name
        mock.__spec__ = MagicMock()
        mock.__spec__.name = name
        return mock

    def create_async_mock() -> MagicMock:
        """Create a proper async mock that handles coroutines correctly."""
        mock = MagicMock()
        mock.__aenter__ = MagicMock(return_value=mock)
        mock.__aexit__ = MagicMock(return_value=None)

        # Ensure async methods return proper coroutines
        async def async_method(*args: object, **kwargs: object) -> MagicMock:
            return MagicMock()

        mock.async_method = async_method
        return mock

    with patch.dict(
        "sys.modules",
        {
            "TTS": create_mock_module("TTS"),
            "TTS.api": create_mock_module("TTS.api"),
            "gruut": create_mock_module("gruut"),
            "pycrfsuite": create_mock_module("pycrfsuite"),
            "regex": create_mock_module("regex"),
            "matplotlib": create_mock_module("matplotlib"),
            "pandas": create_mock_module("pandas"),
            "transformers": create_mock_module("transformers"),
            "transformers.AutoTokenizer": create_mock_module(
                "transformers.AutoTokenizer"
            ),
            "transformers.AutoModelForCausalLM": create_mock_module(
                "transformers.AutoModelForCausalLM"
            ),
            "safetensors": create_mock_module("safetensors"),
            "safetensors._safetensors_rust": create_mock_module(
                "safetensors._safetensors_rust"
            ),
            "numba": create_mock_module("numba"),
        },
    ):
        # Create a mock Coqui TTS class that can be instantiated
        mock_coqui_class = MagicMock()
        coqui_inst = MagicMock()
        # Return a small array for audio
        coqui_inst.tts.return_value = np.zeros(22050, dtype=np.float32)
        mock_coqui_class.return_value = coqui_inst

        # Set up the mock Coqui TTS module structure
        import TTS

        TTS.api.TTS = mock_coqui_class

        # Mock torch to avoid docstring conflicts and CUDA issues
        if "torch" not in sys.modules:
            sys.modules["torch"] = types.ModuleType("torch")
            torch = sys.modules["torch"]
            torch.cuda = MagicMock()  # type: ignore
            torch.cuda.is_available = MagicMock(return_value=False)
            torch.cuda.device_count = MagicMock(return_value=0)
            torch.Tensor = MagicMock()  # type: ignore
            torch.nn = MagicMock()  # type: ignore
            torch.optim = MagicMock()  # type: ignore

        # Transformers (TinyLlama via Auto classes)
        with (
            patch("transformers.AutoTokenizer") as mock_tok,
            patch("transformers.AutoModelForCausalLM") as mock_lm,
        ):
            tok = MagicMock()
            tok.eos_token_id = 2
            tok.decode.return_value = "hi"
            mock_tok.from_pretrained.return_value = tok
            model = MagicMock()
            model.generate.return_value = [[1, 2, 3]]
            mock_lm.from_pretrained.return_value = model
            # Ensure llama_cpp module path exists for patching
            if "llama_cpp" not in sys.modules:
                sys.modules["llama_cpp"] = types.ModuleType("llama_cpp")
                # llama.cpp
                with patch("llama_cpp.Llama", create=True) as mock_llama:
                    ll = MagicMock()
                    ll.return_value = {"choices": [{"text": "hi there"}]}
                    mock_llama.return_value = ll
                    yield
            else:
                yield
