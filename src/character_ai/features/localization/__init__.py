"""
Localization feature package.

Provides language detection, localization, and cultural adaptation capabilities.
"""

from .language_support import (
    CulturalRegion,
    LanguageCode,
    LanguageDetectionResult,
    LanguagePack,
    LocalizationService,
    create_localization_manager,
)

__all__ = [
    "CulturalRegion",
    "LanguageCode",
    "LanguageDetectionResult",
    "LanguagePack",
    "LocalizationService",
    "create_localization_manager",
]
