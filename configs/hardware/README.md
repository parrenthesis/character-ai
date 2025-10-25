# Hardware Configuration Profiles

This directory contains hardware-specific configuration profiles that optimize the character AI platform for different hardware platforms.

## Overview

Each hardware profile configures:
- **Model Selection**: Which models to use for STT, LLM, and TTS
- **Hardware Constraints**: Memory, CPU, GPU availability
- **Performance Targets**: Expected latency and optimization goals
- **Runtime Settings**: Audio, VAD, power management
- **Optimizations**: Model-specific tuning for the hardware

## Available Profiles

### Desktop / Server (`desktop.yaml`)

**Target Hardware**: High-performance desktop or server with NVIDIA GPU
- **CPU**: 16+ cores
- **Memory**: 32GB+
- **GPU**: NVIDIA RTX 3090 or equivalent
- **Models**:
  - STT: `wav2vec2-base`
  - LLM: `llama-3.2-1b-instruct`
  - TTS: `coqui-xtts-v2` (GPU-accelerated)
- **Performance Target**: <2s total latency
- **GPU Settings**:
  - LLM: 20 GPU layers
  - TTS: FP16 half-precision
  - STT: GPU-accelerated

**Expected Performance**:
- STT: 30-50ms
- LLM: 800ms-1.2s
- TTS: 500ms (GPU-accelerated with PyTorch 2.9+)

**Note**: GPU acceleration is enabled by default with PyTorch 2.9.0+cu128. For optimal performance, ensure CUDA is available and models are loaded on GPU.

### Orange Pi 5 (`orange_pi.yaml`)

**Target Hardware**: Orange Pi 5 with Mali GPU
- **CPU**: 8 cores (ARM)
- **Memory**: 8GB
- **GPU**: Mali G610 MP4
- **Models**:
  - STT: `wav2vec2-base`
  - LLM: `tinyllama-1.1b`
  - TTS: `coqui-xtts-v2`
- **Performance Target**: <1.5s total latency
- **GPU Settings**:
  - LLM: 10 GPU layers (conservative for Mali)
  - TTS: FP16 half-precision
- **VAD**: 0.4s silence delay (faster response)

**Note**: Mali GPU compatibility needs testing. May require CPU fallback.

### Raspberry Pi 4/5 (`raspberry_pi.yaml`)

**Target Hardware**: Raspberry Pi 4 or 5 (CPU-only)
- **CPU**: 4 cores (ARM)
- **Memory**: 4GB
- **GPU**: None (CPU-only mode)
- **Models**:
  - STT: `wav2vec2-base`
  - LLM: `tinyllama-1.1b`
  - TTS: `coqui-xtts-v2` (CPU-only)
- **Performance Target**: <2s total latency (best-effort)
- **Optimizations**:
  - Aggressive power saving
  - Larger TTS chunks (512) for efficiency
  - Reduced threading (2 threads)
- **VAD**: 0.6s silence delay

**Expected Performance**:
- STT: 100-200ms
- LLM: 2-3s
- TTS: 20-30s (CPU-only voice cloning)

**Note**: Voice cloning on Raspberry Pi is slow but functional. For production use, consider pre-generating common responses.

## Voice Cloning Requirement

**All profiles use XTTS v2 for voice cloning.** This is a core platform requirement that cannot be compromised. Alternative TTS models without voice cloning support (e.g., Tacotron2) are not acceptable.

## VAD (Voice Activity Detection) Configuration

All profiles include VAD settings to control speech detection:

```yaml
vad:
  threshold: 0.5-0.7          # Speech detection sensitivity (higher = more strict)
  min_speech_duration_s: 0.1-0.3  # Minimum duration to register as speech
  max_silence_duration_s: 0.3-0.6  # How long to wait after silence before stopping
```

**Tuning Guidelines**:
- **threshold**: Increase in noisy environments, decrease for quiet environments
- **min_speech_duration_s**: Increase to filter out noise, decrease for quick responses
- **max_silence_duration_s**: **Lower values = faster response** but may cut off natural pauses

Desktop uses 0.3s, Orange Pi uses 0.4s, Raspberry Pi uses 0.6s silence delays.

## Performance Optimization Notes

### XTTS v2 CPU Performance

**Profiling Results** (desktop CPU, 8 threads):
- Average synthesis time: 8-10s for 4s audio
- Real-time factor: 2.1-2.8x (inherent model limitation)
- Attempts: torch.compile() (incompatible), int8 quantization (not available)

**Conclusion**: XTTS v2 on CPU has a ~2.1x minimum real-time factor due to model architecture. **GPU acceleration is required to meet <2s performance targets.**

### XTTS v2 GPU Performance

**Current Status**: XTTS v2 GPU acceleration works with PyTorch 2.9.0+cu128:
- GPU synthesis: ~500ms for 4s audio
- Real-time factor: ~0.125x (8x faster than real-time)
- Half-precision (FP16) supported for 2x speed improvement

**Optimization Notes**:
- Ensure CUDA is available: `torch.cuda.is_available()`
- Check GPU memory allocation for large models
- Monitor for CUDA context conflicts between models

## Creating Custom Profiles

To create a custom hardware profile:

1. Copy an existing profile that matches your hardware closest
2. Update `constraints` section with your hardware specs
3. Adjust `models` based on available resources
4. Tune `optimizations` for your specific hardware
5. Test and measure actual performance
6. Adjust `target_latency_s` to realistic expectations

## Model Selection Guidelines

### STT (Speech-to-Text)
- **wav2vec2-base**: Good balance of speed and accuracy
- Runs well on CPU or GPU

### LLM (Large Language Model)
- **tinyllama-1.1b**: Fastest, good for limited hardware
- **llama-3.2-1b-instruct**: Better quality, still fast
- **llama-3.2-3b-instruct**: Best quality, slower

Use `n_gpu_layers` to offload layers to GPU. Start conservative and increase based on VRAM availability.

### TTS (Text-to-Speech)
- **coqui-xtts-v2**: Required for voice cloning
- No alternatives - voice cloning is platform core feature

## Testing Your Profile

```bash
# Test with specific profile
make test-voice-pipeline-realtime-desktop   # Desktop profile
make test-voice-pipeline-realtime-pi        # Raspberry Pi profile

# Or specify profile directly
poetry run cai test voice-pipeline \\
  --character data --franchise star_trek \\
  --realtime --duration 10 \\
  --hardware-profile desktop --quiet
```

## torch_init Import Pattern

**Critical**: All processor files must import `torch_init` FIRST before any PyTorch imports:

```python
# CRITICAL: Import torch_init FIRST to set environment variables
# isort: off
from ...core import torch_init  # noqa: F401
# isort: on

import torch  # Now safe to import torch
```

The `isort: off/on` comments prevent linters from reordering imports, which would break the environment variable setup (`TORCH_FORCE_NO_WEIGHTS_ONLY_LOAD=1`).

**Breaking this pattern causes**:
- Segmentation faults during model loading
- XTTS v2 weight loading failures
- CUDA initialization issues

## Troubleshooting

### Segmentation Faults
- Ensure `torch_init` is imported first in all processor files
- Run with `TORCH_FORCE_NO_WEIGHTS_ONLY_LOAD=1` environment variable
- Check model load order (TTS → STT → LLM)

### CUDA Errors
- Try CPU fallback by setting `use_half_precision: false` for TTS
- Check PyTorch and CUDA version compatibility
- Monitor GPU memory usage

### Slow Performance
- Verify GPU layers are being used (`n_gpu_layers > 0`)
- Check if models are actually loading on GPU (look for device logs)
- Consider smaller/faster models
- Reduce `max_new_tokens` in runtime config

### Audio Issues
- Verify VAD threshold is appropriate for your environment
- Check audio device sample rates match config
- Test with different microphones/speakers

## Support

For issues or questions about hardware profiles:
1. Check this README for configuration guidelines
2. Review the plan documentation in project root
3. Test with verbose mode: `--verbose` flag
4. Check logs for hardware detection and model loading

## Future Work

- Add automated performance benchmarks per profile
- Create profiles for additional hardware (Jetson, Mac M1/M2, etc.)
- Implement model quantization for faster CPU inference
- Add streaming synthesis for reduced latency perception
- Optimize CUDA context management for multi-model GPU usage
