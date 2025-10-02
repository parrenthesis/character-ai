"""
Personalization system for adaptive character interactions.

Provides user preference learning, adaptive conversation styles, and personalized
character recommendations based on interaction history and cultural preferences.
"""

import json
import logging
import time
from collections import defaultdict
from dataclasses import dataclass, field
from enum import Enum
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np

from ..algorithms.conversational_ai.session_memory import ConversationTurn
from ..characters.types import Character, Species
from .language_support import LanguageCode, get_localization_manager

logger = logging.getLogger(__name__)


class PreferenceType(Enum):
    """Types of user preferences."""

    CONVERSATION_STYLE = "conversation_style"
    TOPIC_INTEREST = "topic_interest"
    CHARACTER_TYPE = "character_type"
    VOICE_PREFERENCE = "voice_preference"
    LANGUAGE_PREFERENCE = "language_preference"
    EMOTIONAL_TONE = "emotional_tone"
    FORMALITY_LEVEL = "formality_level"
    INTERACTION_FREQUENCY = "interaction_frequency"


class LearningSource(Enum):
    """Sources of preference learning."""

    EXPLICIT_FEEDBACK = "explicit_feedback"
    INTERACTION_PATTERNS = "interaction_patterns"
    CULTURAL_ADAPTATION = "cultural_adaptation"
    BEHAVIORAL_ANALYSIS = "behavioral_analysis"


@dataclass
class UserPreference:
    """A single user preference."""

    preference_type: PreferenceType
    value: Any
    confidence: float  # 0.0 to 1.0
    source: LearningSource
    last_updated: float
    interaction_count: int = 1

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for serialization."""
        return {
            "preference_type": self.preference_type.value,
            "value": self.value,
            "confidence": self.confidence,
            "source": self.source.value,
            "last_updated": self.last_updated,
            "interaction_count": self.interaction_count,
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "UserPreference":
        """Create from dictionary."""
        return cls(
            preference_type=PreferenceType(data["preference_type"]),
            value=data["value"],
            confidence=data["confidence"],
            source=LearningSource(data["source"]),
            last_updated=data["last_updated"],
            interaction_count=data.get("interaction_count", 1),
        )


@dataclass
class InteractionPattern:
    """Pattern in user interactions."""

    pattern_type: str
    frequency: int
    confidence: float
    examples: List[str] = field(default_factory=list)
    last_seen: float = field(default_factory=time.time)


@dataclass
class UserProfile:
    """Complete user profile with preferences and patterns."""

    user_id: str
    device_id: str
    created_at: float
    last_updated: float
    preferences: Dict[PreferenceType, UserPreference] = field(default_factory=dict)
    interaction_patterns: List[InteractionPattern] = field(default_factory=list)
    cultural_adaptations: Dict[str, Any] = field(default_factory=dict)
    language_preferences: Dict[LanguageCode, float] = field(default_factory=dict)
    character_affinities: Dict[str, float] = field(
        default_factory=dict
    )  # character_id -> affinity_score
    privacy_settings: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for serialization."""
        return {
            "user_id": self.user_id,
            "device_id": self.device_id,
            "created_at": self.created_at,
            "last_updated": self.last_updated,
            "preferences": {
                pref_type.value: pref.to_dict()
                for pref_type, pref in self.preferences.items()
            },
            "interaction_patterns": [
                {
                    "pattern_type": pattern.pattern_type,
                    "frequency": pattern.frequency,
                    "confidence": pattern.confidence,
                    "examples": pattern.examples,
                    "last_seen": pattern.last_seen,
                }
                for pattern in self.interaction_patterns
            ],
            "cultural_adaptations": self.cultural_adaptations,
            "language_preferences": {
                lang.value: score for lang, score in self.language_preferences.items()
            },
            "character_affinities": self.character_affinities,
            "privacy_settings": self.privacy_settings,
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "UserProfile":
        """Create from dictionary."""
        preferences = {}
        for pref_type_str, pref_data in data.get("preferences", {}).items():
            pref_type = PreferenceType(pref_type_str)
            preferences[pref_type] = UserPreference.from_dict(pref_data)

        interaction_patterns = [
            InteractionPattern(
                pattern_type=pattern["pattern_type"],
                frequency=pattern["frequency"],
                confidence=pattern["confidence"],
                examples=pattern.get("examples", []),
                last_seen=pattern.get("last_seen", time.time()),
            )
            for pattern in data.get("interaction_patterns", [])
        ]

        language_preferences = {}
        for lang_str, score in data.get("language_preferences", {}).items():
            language_preferences[LanguageCode(lang_str)] = score

        return cls(
            user_id=data["user_id"],
            device_id=data["device_id"],
            created_at=data["created_at"],
            last_updated=data["last_updated"],
            preferences=preferences,
            interaction_patterns=interaction_patterns,
            cultural_adaptations=data.get("cultural_adaptations", {}),
            language_preferences=language_preferences,
            character_affinities=data.get("character_affinities", {}),
            privacy_settings=data.get("privacy_settings", {}),
        )


class PreferenceLearner:
    """Learns user preferences from interaction patterns."""

    def __init__(self) -> None:
        self.localization_manager = get_localization_manager()
        self._conversation_style_keywords = {
            "formal": ["please", "thank you", "sir", "madam", "would you", "could you"],
            "casual": ["hey", "what's up", "cool", "awesome", "yeah", "sure"],
            "friendly": ["hi", "hello", "how are you", "nice to meet you", "great"],
            "professional": [
                "regarding",
                "furthermore",
                "however",
                "therefore",
                "consequently",
            ],
        }

        self._emotional_tone_keywords = {
            "positive": ["happy", "excited", "great", "wonderful", "amazing", "love"],
            "neutral": ["okay", "fine", "alright", "sure", "yes", "no"],
            "negative": [
                "sad",
                "angry",
                "frustrated",
                "disappointed",
                "upset",
                "worried",
            ],
        }

        self._topic_keywords = {
            "technology": [
                "computer",
                "software",
                "programming",
                "AI",
                "robot",
                "tech",
            ],
            "entertainment": ["movie", "music", "game", "book", "show", "fun"],
            "education": [
                "learn",
                "study",
                "school",
                "teacher",
                "student",
                "knowledge",
            ],
            "health": [
                "exercise",
                "doctor",
                "medicine",
                "healthy",
                "fitness",
                "wellness",
            ],
            "travel": ["vacation", "trip", "travel", "country", "city", "adventure"],
        }

    def analyze_conversation_style(self, user_inputs: List[str]) -> Tuple[str, float]:
        """Analyze conversation style from user inputs."""
        if not user_inputs:
            return "neutral", 0.0

        style_scores: Dict[str, float] = defaultdict(float)
        total_words = 0

        for input_text in user_inputs:
            words = input_text.lower().split()
            total_words += len(words)

            for style, keywords in self._conversation_style_keywords.items():
                for keyword in keywords:
                    if keyword in input_text.lower():
                        style_scores[style] += 1

        if not style_scores:
            return "neutral", 0.0

        # Normalize scores
        for style in style_scores:
            style_scores[style] /= total_words

        # Find dominant style
        dominant_style = max(style_scores.items(), key=lambda x: x[1])
        return dominant_style[0], min(dominant_style[1] * 10, 1.0)  # Scale confidence

    def analyze_emotional_tone(self, user_inputs: List[str]) -> Tuple[str, float]:
        """Analyze emotional tone from user inputs."""
        if not user_inputs:
            return "neutral", 0.0

        tone_scores: Dict[str, float] = defaultdict(float)
        total_words = 0

        for input_text in user_inputs:
            words = input_text.lower().split()
            total_words += len(words)

            for tone, keywords in self._emotional_tone_keywords.items():
                for keyword in keywords:
                    if keyword in input_text.lower():
                        tone_scores[tone] += 1

        if not tone_scores:
            return "neutral", 0.0

        # Normalize scores
        for tone in tone_scores:
            tone_scores[tone] /= total_words

        # Find dominant tone
        dominant_tone = max(tone_scores.items(), key=lambda x: x[1])
        return dominant_tone[0], min(dominant_tone[1] * 10, 1.0)  # Scale confidence

    def analyze_topic_interests(self, user_inputs: List[str]) -> Dict[str, float]:
        """Analyze topic interests from user inputs."""
        if not user_inputs:
            return {}

        topic_scores: Dict[str, float] = defaultdict(float)
        total_words = 0

        for input_text in user_inputs:
            words = input_text.lower().split()
            total_words += len(words)

            for topic, keywords in self._topic_keywords.items():
                for keyword in keywords:
                    if keyword in input_text.lower():
                        topic_scores[topic] += 1

        # Normalize scores
        if total_words > 0:
            for topic in topic_scores:
                topic_scores[topic] /= total_words

        # Return topics with significant interest
        return {topic: score for topic, score in topic_scores.items() if score > 0.01}

    def analyze_interaction_patterns(
        self, conversation_turns: List[ConversationTurn]
    ) -> List[InteractionPattern]:
        """Analyze patterns in user interactions."""
        patterns: List[InteractionPattern] = []

        if len(conversation_turns) < 3:
            return patterns

        # Analyze response time patterns
        response_times = []
        for i in range(1, len(conversation_turns)):
            time_diff = (
                conversation_turns[i].timestamp - conversation_turns[i - 1].timestamp
            )
            response_times.append(time_diff)

        if response_times:
            avg_response_time = np.mean(response_times)
            if avg_response_time < 5.0:  # Quick responses
                patterns.append(
                    InteractionPattern(
                        pattern_type="quick_responses",
                        frequency=len([t for t in response_times if t < 5.0]),
                        confidence=0.8,
                        examples=[
                            f"Response time: {t:.1f}s" for t in response_times[:3]
                        ],
                    )
                )
            elif avg_response_time > 30.0:  # Thoughtful responses
                patterns.append(
                    InteractionPattern(
                        pattern_type="thoughtful_responses",
                        frequency=len([t for t in response_times if t > 30.0]),
                        confidence=0.8,
                        examples=[
                            f"Response time: {t:.1f}s" for t in response_times[:3]
                        ],
                    )
                )

        # Analyze conversation length patterns
        input_lengths = [len(turn.user_input.split()) for turn in conversation_turns]
        if input_lengths:
            avg_length = np.mean(input_lengths)
            if avg_length < 5:  # Short inputs
                patterns.append(
                    InteractionPattern(
                        pattern_type="concise_communication",
                        frequency=len(
                            [length for length in input_lengths if length < 5]
                        ),
                        confidence=0.7,
                        examples=[
                            (
                                turn.user_input[:50] + "..."
                                if len(turn.user_input) > 50
                                else turn.user_input
                            )
                            for turn in conversation_turns[:3]
                        ],
                    )
                )
            elif avg_length > 20:  # Detailed inputs
                patterns.append(
                    InteractionPattern(
                        pattern_type="detailed_communication",
                        frequency=len(
                            [length for length in input_lengths if length > 20]
                        ),
                        confidence=0.7,
                        examples=[
                            (
                                turn.user_input[:50] + "..."
                                if len(turn.user_input) > 50
                                else turn.user_input
                            )
                            for turn in conversation_turns[:3]
                        ],
                    )
                )

        return patterns


class PersonalizationManager:
    """Manages user personalization and adaptive interactions."""

    def __init__(self, storage_path: Path = Path.cwd() / "data/personalization"):
        self.storage_path = storage_path
        # Don't create directory during instantiation - create when actually needed

        self.user_profiles: Dict[str, UserProfile] = {}
        self.preference_learner = PreferenceLearner()
        self.localization_manager = get_localization_manager()

        # Don't load data during instantiation - load when actually needed

        logger.info(
            "PersonalizationManager initialized",
            extra={"storage_path": str(storage_path)},
        )

    def _load_profiles(self) -> None:
        """Load user profiles from storage."""
        try:
            # Create directory if it doesn't exist
            if not self.storage_path.exists():
                self.storage_path.mkdir(parents=True, exist_ok=True)

            profiles_file = self.storage_path / "user_profiles.json"
            if profiles_file.exists():
                with open(profiles_file, "r") as f:
                    profiles_data = json.load(f)

                for user_id, profile_data in profiles_data.items():
                    self.user_profiles[user_id] = UserProfile.from_dict(profile_data)

                logger.info(f"Loaded {len(self.user_profiles)} user profiles")
        except Exception as e:
            logger.error(f"Failed to load user profiles: {e}")

    def _save_profiles(self) -> None:
        """Save user profiles to storage."""
        try:
            # Create directory if it doesn't exist
            if not self.storage_path.exists():
                self.storage_path.mkdir(parents=True, exist_ok=True)

            profiles_file = self.storage_path / "user_profiles.json"
            profiles_data = {
                user_id: profile.to_dict()
                for user_id, profile in self.user_profiles.items()
            }

            with open(profiles_file, "w") as f:
                json.dump(profiles_data, f, indent=2)

            logger.debug("Saved user profiles to storage")
        except Exception as e:
            logger.error(f"Failed to save user profiles: {e}")

    def get_or_create_profile(self, user_id: str, device_id: str) -> UserProfile:
        """Get existing profile or create new one."""
        if user_id not in self.user_profiles:
            self.user_profiles[user_id] = UserProfile(
                user_id=user_id,
                device_id=device_id,
                created_at=time.time(),
                last_updated=time.time(),
            )
            self._save_profiles()

        return self.user_profiles[user_id]

    def update_preference(
        self,
        user_id: str,
        preference_type: PreferenceType,
        value: Any,
        confidence: float,
        source: LearningSource,
    ) -> None:
        """Update a user preference."""
        profile = self.get_or_create_profile(user_id, "")

        if preference_type in profile.preferences:
            # Update existing preference
            existing = profile.preferences[preference_type]
            # Weighted average based on confidence
            total_confidence = existing.confidence + confidence
            if total_confidence > 0:
                existing.value = value  # For now, just update value
                existing.confidence = min(1.0, total_confidence)
                existing.interaction_count += 1
        else:
            # Create new preference
            profile.preferences[preference_type] = UserPreference(
                preference_type=preference_type,
                value=value,
                confidence=confidence,
                source=source,
                last_updated=time.time(),
            )

        profile.last_updated = time.time()
        self._save_profiles()

    def learn_from_conversation(
        self,
        user_id: str,
        conversation_turns: List[ConversationTurn],
        character_id: str,
    ) -> None:
        """Learn preferences from conversation history."""
        if not conversation_turns:
            return

        profile = self.get_or_create_profile(user_id, "")

        # Extract user inputs
        user_inputs = [turn.user_input for turn in conversation_turns]

        # Learn conversation style
        style, style_confidence = self.preference_learner.analyze_conversation_style(
            user_inputs
        )
        if style_confidence > 0.3:
            self.update_preference(
                user_id,
                PreferenceType.CONVERSATION_STYLE,
                style,
                style_confidence,
                LearningSource.INTERACTION_PATTERNS,
            )

        # Learn emotional tone
        tone, tone_confidence = self.preference_learner.analyze_emotional_tone(
            user_inputs
        )
        if tone_confidence > 0.3:
            self.update_preference(
                user_id,
                PreferenceType.EMOTIONAL_TONE,
                tone,
                tone_confidence,
                LearningSource.INTERACTION_PATTERNS,
            )

        # Learn topic interests
        topic_interests = self.preference_learner.analyze_topic_interests(user_inputs)
        for topic, interest_score in topic_interests.items():
            self.update_preference(
                user_id,
                PreferenceType.TOPIC_INTEREST,
                topic,
                interest_score,
                LearningSource.INTERACTION_PATTERNS,
            )

        # Learn interaction patterns
        patterns = self.preference_learner.analyze_interaction_patterns(
            conversation_turns
        )
        profile.interaction_patterns.extend(patterns)

        # Update character affinity
        if character_id not in profile.character_affinities:
            profile.character_affinities[character_id] = 0.0
        profile.character_affinities[character_id] += 0.1  # Increase affinity
        profile.character_affinities[character_id] = min(
            1.0, profile.character_affinities[character_id]
        )

        profile.last_updated = time.time()
        self._save_profiles()

    def get_personalized_character_recommendations(
        self,
        user_id: str,
        available_characters: List[Character],
        max_recommendations: int = 5,
    ) -> List[Tuple[Character, float]]:
        """Get personalized character recommendations."""
        profile = self.get_or_create_profile(user_id, "")

        recommendations = []

        for character in available_characters:
            score = 0.0

            # Base score from character affinity
            if character.name.lower() in profile.character_affinities:
                score += profile.character_affinities[character.name.lower()] * 0.4

            # Boost based on conversation style match
            if PreferenceType.CONVERSATION_STYLE in profile.preferences:
                user_style = profile.preferences[
                    PreferenceType.CONVERSATION_STYLE
                ].value
                if (
                    user_style == "formal"
                    and character.dimensions.species == Species.ROBOT
                ):
                    score += 0.2
                elif user_style == "casual" and character.dimensions.species in [
                    Species.DRAGON,
                    Species.ROBOT,
                ]:
                    score += 0.2

            # Boost based on topic interests
            if PreferenceType.TOPIC_INTEREST in profile.preferences:
                user_topics = profile.preferences[PreferenceType.TOPIC_INTEREST].value
                if isinstance(user_topics, dict):
                    for topic, interest in user_topics.items():
                        if topic in [t.value for t in character.dimensions.topics]:
                            score += interest * 0.3

            # Boost based on emotional tone
            if PreferenceType.EMOTIONAL_TONE in profile.preferences:
                user_tone = profile.preferences[PreferenceType.EMOTIONAL_TONE].value
                if (
                    user_tone == "positive"
                    and character.dimensions.species == Species.DRAGON
                ):
                    score += 0.1
                elif (
                    user_tone == "neutral"
                    and character.dimensions.species == Species.ROBOT
                ):
                    score += 0.1

            recommendations.append((character, score))

        # Sort by score and return top recommendations
        recommendations.sort(key=lambda x: x[1], reverse=True)
        return recommendations[:max_recommendations]

    def get_adaptive_conversation_style(
        self, user_id: str, character: Character
    ) -> Dict[str, Any]:
        """Get adaptive conversation style for a character based on user preferences."""

        profile = self.get_or_create_profile(user_id, "")

        style_adaptations = {
            "formality_level": "neutral",
            "emotional_tone": "neutral",
            "response_length": "medium",
            "topic_focus": [],
            "cultural_adaptations": {},
        }

        # Apply conversation style preference
        if PreferenceType.CONVERSATION_STYLE in profile.preferences:
            user_style = profile.preferences[PreferenceType.CONVERSATION_STYLE].value
            if user_style == "formal":
                style_adaptations["formality_level"] = "high"
            elif user_style == "casual":
                style_adaptations["formality_level"] = "low"

        # Apply emotional tone preference
        if PreferenceType.EMOTIONAL_TONE in profile.preferences:
            user_tone = profile.preferences[PreferenceType.EMOTIONAL_TONE].value
            style_adaptations["emotional_tone"] = user_tone

        # Apply topic interests
        if PreferenceType.TOPIC_INTEREST in profile.preferences:
            user_topics = profile.preferences[PreferenceType.TOPIC_INTEREST].value
            if isinstance(user_topics, dict):
                style_adaptations["topic_focus"] = list(user_topics.keys())

        # Apply cultural adaptations
        if profile.cultural_adaptations:
            style_adaptations["cultural_adaptations"] = profile.cultural_adaptations

        return style_adaptations

    def get_user_insights(self, user_id: str) -> Dict[str, Any]:
        """Get insights about user preferences and patterns."""
        profile = self.get_or_create_profile(user_id, "")

        insights = {
            "user_id": user_id,
            "profile_age_days": (time.time() - profile.created_at) / 86400,
            "total_preferences": len(profile.preferences),
            "interaction_patterns": len(profile.interaction_patterns),
            "character_affinities": profile.character_affinities,
            "preferences_summary": {},
            "learning_sources": {},
            "privacy_compliance": True,
        }

        # Summarize preferences
        for pref_type, preference in profile.preferences.items():
            insights["preferences_summary"][pref_type.value] = {  # type: ignore
                "value": preference.value,
                "confidence": preference.confidence,
                "interaction_count": preference.interaction_count,
            }

            # Track learning sources
            source = preference.source.value
            if source not in insights["learning_sources"]:  # type: ignore
                insights["learning_sources"][source] = 0  # type: ignore
            insights["learning_sources"][source] += 1  # type: ignore

        return insights

    def clear_user_data(self, user_id: str) -> bool:
        """Clear all personalization data for a user (GDPR compliance)."""
        try:
            if user_id in self.user_profiles:
                del self.user_profiles[user_id]
                self._save_profiles()
                logger.info(f"Cleared personalization data for user {user_id}")
                return True
            return False
        except Exception as e:
            logger.error(f"Failed to clear user data for {user_id}: {e}")
            return False

    def export_user_data(self, user_id: str) -> Optional[Dict[str, Any]]:
        """Export all personalization data for a user (GDPR compliance)."""
        if user_id not in self.user_profiles:
            return None

        profile = self.user_profiles[user_id]
        return profile.to_dict()


# Global instance
_personalization_manager: Optional[PersonalizationManager] = None


def get_personalization_manager() -> PersonalizationManager:
    """Get the global personalization manager instance."""
    global _personalization_manager
    if _personalization_manager is None:
        _personalization_manager = PersonalizationManager()
    return _personalization_manager
