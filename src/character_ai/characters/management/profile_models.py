from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, List, Optional

from pydantic import BaseModel, Field, validator

from ...core.config.yaml_loader import YAMLConfigLoader


class ModerationConfig(BaseModel):
    toxicity: bool = True
    pii: bool = True


class SafetyConfig(BaseModel):
    allowlist: List[str] = Field(default_factory=list)
    denylist: List[str] = Field(default_factory=list)
    moderation: ModerationConfig = Field(default_factory=ModerationConfig)


class LLMConfig(BaseModel):
    backend: Optional[str] = None  # "llama_cpp" | "transformers"
    max_tokens: int = 64
    temperature: float = 0.7
    prompt_template: Optional[str] = None  # path to prompt.md inside character dir


class STTConfig(BaseModel):
    preferred_model: str = "base"
    sample_rate: int = 16000


class TTSConfig(BaseModel):
    engine: str = "coqui_tts"
    voice_artifact: Optional[str] = None  # voice.wav or voice_emb.npz
    watermark: bool = True


class ConsentMetadata(BaseModel):
    subject: str
    method: str
    timestamp: str
    scope: List[str]
    provenance: Optional[str] = None
    retention_days: int = 365


class CharacterProfile(BaseModel):
    schema_version: int = 1
    id: str
    display_name: Optional[str] = None
    character_type: Optional[str] = None  # maps to CharacterType (fallback to robot)
    language: str = "en"
    traits: Dict[str, Any] = Field(default_factory=dict)
    voice_style: str = "neutral"
    topics: List[str] = Field(default_factory=list)
    safety: SafetyConfig = Field(default_factory=SafetyConfig)
    llm: LLMConfig = Field(default_factory=LLMConfig)
    stt: STTConfig = Field(default_factory=STTConfig)
    tts: TTSConfig = Field(default_factory=TTSConfig)
    consent: Optional[Dict[str, Any]] = None  # may reference file path via profile

    @validator("schema_version")
    def _check_schema(cls, v: int) -> int:
        if v != 1:
            raise ValueError("Unsupported schema_version")
        return v


def load_consent(consent_path: Path) -> ConsentMetadata:
    data = YAMLConfigLoader.load_yaml(consent_path)
    return ConsentMetadata(**data)


def load_profile(profile_path: Path) -> CharacterProfile:
    data = YAMLConfigLoader.load_yaml(profile_path)
    return CharacterProfile(**data)


def load_profile_dir(char_dir: Path) -> Dict[str, Any]:
    """Load and validate a character profile folder.

    Returns a dict with: id, name, character_type, voice_style, topics, metadata,
    voice_path
    """
    profile_path = char_dir / "profile.yaml"
    if not profile_path.exists():
        raise FileNotFoundError(f"profile.yaml not found in {char_dir}")
    prof = load_profile(profile_path)

    # Consent
    consent_meta: Optional[ConsentMetadata] = None
    if prof.consent and isinstance(prof.consent, dict) and "file" in prof.consent:
        cfile = char_dir / str(prof.consent.get("file"))
        if not cfile.exists():
            raise FileNotFoundError(f"consent file missing: {cfile}")
        consent_meta = load_consent(cfile)

    # Voice artifact
    voice_path: Optional[Path] = None
    if prof.tts.voice_artifact:
        candidate = char_dir / prof.tts.voice_artifact
        if not candidate.exists():
            raise FileNotFoundError(f"voice artifact missing: {candidate}")
        voice_path = candidate

    name = prof.display_name or prof.id
    meta = {
        "language": prof.language,
        "safety": prof.safety.dict(),
        "llm": prof.llm.dict(),
        "stt": prof.stt.dict(),
        "tts": prof.tts.dict(),
        "prompt_template": prof.llm.prompt_template,
        "profile_dir": str(char_dir),
        "consent": consent_meta.dict() if consent_meta else None,
    }

    return {
        "id": prof.id,
        "name": name,
        "character_type": prof.character_type or "robot",
        "voice_style": prof.voice_style,
        "topics": prof.topics,
        "metadata": meta,
        "voice_path": str(voice_path) if voice_path else None,
    }


# Optional: index schema for index.yaml
class CharacterIndexItem(BaseModel):
    id: str
    path: str


class CharactersIndex(BaseModel):
    schema_version: int = 1
    characters: List[CharacterIndexItem] = Field(default_factory=list)
    defaults: Dict[str, Any] = Field(default_factory=dict)

    @validator("schema_version")
    def _check_schema(cls, v: int) -> int:
        if v != 1:
            raise ValueError("Unsupported schema_version")
        return v


def load_index(index_path: Path) -> CharactersIndex:
    data = YAMLConfigLoader.load_yaml(index_path)
    return CharactersIndex(**data)


def export_json_schema() -> Dict[str, Any]:
    """Export JSON Schemas for profile and consent documents."""
    return {
        "profile": CharacterProfile.model_json_schema(),
        "consent": ConsentMetadata.model_json_schema(),
        "index": CharactersIndex.model_json_schema(),
    }
