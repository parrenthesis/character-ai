"""
Preference extraction for user facts using pattern matching.

Extracts structured user preferences (name, interests, color, dislikes) from
conversation turns using regex patterns. No LLM required for basic extraction.
"""

import json
import logging
import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional, Set

# Optional fuzzy matching support
try:
    from difflib import SequenceMatcher

    FUZZY_AVAILABLE = True
except ImportError:
    FUZZY_AVAILABLE = False

logger = logging.getLogger(__name__)


@dataclass
class UserPreferences:
    """Structured user preferences extracted from conversations."""

    user_id: str
    name: Optional[str] = None
    interests: Set[str] = field(default_factory=set)
    favorite_color: Optional[str] = None
    dislikes: Set[str] = field(default_factory=set)
    age: Optional[int] = None
    location: Optional[str] = None
    occupation: Optional[str] = None
    custom_facts: Dict[str, Any] = field(default_factory=dict)
    last_updated: float = 0.0

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for serialization."""
        return {
            "user_id": self.user_id,
            "name": self.name,
            "interests": list(self.interests),
            "favorite_color": self.favorite_color,
            "dislikes": list(self.dislikes),
            "age": self.age,
            "location": self.location,
            "occupation": self.occupation,
            "custom_facts": self.custom_facts,
            "last_updated": self.last_updated,
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "UserPreferences":
        """Create from dictionary."""
        return cls(
            user_id=data["user_id"],
            name=data.get("name"),
            interests=set(data.get("interests", [])),
            favorite_color=data.get("favorite_color"),
            dislikes=set(data.get("dislikes", [])),
            age=data.get("age"),
            location=data.get("location"),
            occupation=data.get("occupation"),
            custom_facts=data.get("custom_facts", {}),
            last_updated=data.get("last_updated", 0.0),
        )


class PreferenceExtractor:
    """Extracts user preferences using pattern matching."""

    def __init__(self) -> None:
        """Initialize with regex patterns for common preference types."""
        self.patterns = {
            "age": [
                r"i am (\d+) years old",
                r"i'm (\d+) years old",
                r"(\d+) years old",
                r"age (\d+)",
                r"i am (\d+)(?:\s|$)",
                r"i'm (\d+)(?:\s|$)",
                r"i am ([a-z\s-]+) years old",
                r"i'm ([a-z\s-]+) years old",
                r"am (\d+) years old",
                r"(\d+) years old",
            ],
            "name": [
                # Direct name declarations - capture multi-word names
                r"my name is ([a-záéíóúñü]+(?: [a-záéíóúñü]+)*)",
                r"i'm ([a-záéíóúñü]+(?: [a-záéíóúñü]+)*?)(?:\s+and|[.,!?]|$)",
                r"i am ([a-záéíóúñü]+(?: [a-záéíóúñü]+)*?)(?:\s+and|[.,!?]|$)(?!\s+(?:a|an|the)\s+)",
                r"call me ([a-záéíóúñü]+(?: [a-záéíóúñü]+)*)",
                r"([a-záéíóúñü]+(?: [a-záéíóúñü]+)*) is my name",
                r"you can call me ([a-záéíóúñü]+(?: [a-záéíóúñü]+)*)",
                r"i go by ([a-záéíóúñü]+(?: [a-záéíóúñü]+)*)",
                # Character addressing user
                r"hello ([a-záéíóúñü]+(?: [a-záéíóúñü]+)*)",
                r"hi ([a-záéíóúñü]+(?: [a-záéíóúñü]+)*)",
                r"good (morning|afternoon|evening) ([a-záéíóúñü]+(?: [a-záéíóúñü]+)*)",
            ],
            "interests": [
                # Simple interests - capture until punctuation or conjunction
                r"i like ([\w\s]+?)(?=\s+and|[.,!?]|$)",
                r"i love ([\w\s]+?)(?=\s+and|[.,!?]|$)",
                r"i enjoy ([\w\s]+?)(?=\s+and|[.,!?]|$)",
                r"i'm interested in ([\w\s]+?)(?=\s+and|[.,!?]|$)",
                r"my hobby is ([\w\s]+?)(?=\s+and|[.,!?]|$)",
                r"i'm into ([\w\s]+?)(?=\s+and|[.,!?]|$)",
                r"i'm passionate about ([\w\s]+?)(?=\s+and|[.,!?]|$)",
                r"i'm a fan of ([\w\s]+?)(?=\s+and|[.,!?]|$)",
                # Academic subjects and fields
                r"i study ([\w\s]+?)(?=\s+and|[.,!?]|$)",
                r"i have a ([\w\s]+?)(?=\s+in\s+[\w\s]+|$)",
                r"i'm working on ([\w\s]+?)(?=\s+and|[.,!?]|$)",
                # Activities and sports
                r"i play ([\w\s]+?)(?=\s+and|[.,!?]|$)",
                r"i do ([\w\s]+?)(?=\s+and|[.,!?]|$)",
                r"i practice ([\w\s]+?)(?=\s+and|[.,!?]|$)",
                r"i'm learning ([\w\s]+?)(?=\s+and|[.,!?]|$)",
                # Food and entertainment
                r"i eat ([\w\s]+?)(?=\s+and|[.,!?]|$)",
                r"i watch ([\w\s]+?)(?=\s+and|[.,!?]|$)",
                r"i listen to ([\w\s]+?)(?=\s+and|[.,!?]|$)",
                r"i read ([\w\s]+?)(?=\s+and|[.,!?]|$)",
                # Work and studies
                r"i work in ([\w\s]+?)(?=\s+and|[.,!?]|$)",
                r"i study ([\w\s]+?)(?=\s+and|[.,!?]|$)",
            ],
            "color": [
                # Direct color preferences (handles both color/colour)
                r"my favorite colou?r is ([\w\s]+?)(?=\s*[.,!?]|\s+and|$)",
                r"my favourite colou?r is ([\w\s]+?)(?=\s*[.,!?]|\s+and|$)",
                r"i like ([\w\s]+?) colou?r",
                r"([\w\s]+?) is my favorite colou?r",
                r"([\w\s]+?) is my favourite colou?r",
                r"i prefer ([\w\s]+?)(?=\s*[.,!?]|\s+and|$)",
                r"i love ([\w\s]+?) colou?r",
                r"([\w\s]+?) is my colou?r",
                # Clothing and objects
                r"i wear ([\w\s]+?)(?=\s*[.,!?]|\s+and|$)",
                r"my car is ([\w\s]+?)(?=\s*[.,!?]|\s+and|$)",
            ],
            "dislikes": [
                # Direct dislikes
                r"i don't like ([\w\s]+?)(?=\s*[.,!?]|\s+and|$)",
                r"i hate ([\w\s]+?)(?=\s*[.,!?]|\s+and|$)",
                r"i dislike ([\w\s]+?)(?=\s*[.,!?]|\s+and|$)",
                r"([\w\s]+?) is not my thing",
                r"i can't stand ([\w\s]+?)(?=\s*[.,!?]|\s+and|$)",
                r"i'm not a fan of ([\w\s]+?)(?=\s*[.,!?]|\s+and|$)",
                r"i avoid ([\w\s]+?)(?=\s*[.,!?]|\s+and|$)",
                # Negative preferences
                r"i don't enjoy ([\w\s]+?)(?=\s*[.,!?]|\s+and|$)",
                r"i'm not into ([\w\s]+?)(?=\s*[.,!?]|\s+and|$)",
                r"([\w\s]+?) bothers me",
                # Fear and phobias
                r"i'm afraid of ([\w\s]+?)(?=\s*[.,!?]|\s+and|$)",
                r"i'm scared of ([\w\s]+?)(?=\s*[.,!?]|\s+and|$)",
                r"([\w\s]+?) scares me",
            ],
            "location": [
                r"i live in ([\w\s]+?)(?=\s*[.,!?]|\s+and|$)",
                r"i'm from ([\w\s]+?)(?=\s*[.,!?]|\s+and|$)",
                r"i'm originally from ([\w\s]+?)(?=\s*[.,!?]|\s+and|$)",
                r"i am originally from ([\w\s]+?)(?=\s*[.,!?]|\s+and|$)",
                r"i'm in ([\w\s]+?)(?=\s*[.,!?]|\s+and|$)",
                r"([\w\s]+?) is where i live",
                r"([\w\s]+?) is my hometown",
            ],
            "occupation": [
                r"i work as ([\w\s]+?)(?=\s*[.,!?]|\s+and|$)",
                r"i'm a ([\w\s]+?)(?=\s*[.,!?]|\s+and|$)",
                r"my job is ([\w\s]+?)(?=\s*[.,!?]|\s+and|$)",
                r"i do ([\w\s]+?) for work",
                r"i'm employed as ([\w\s]+?)(?=\s*[.,!?]|\s+and|$)",
                r"i am a ([\w\s]+?)(?=\s*[.,!?]|\s+and|$)",
                r"i am an ([\w\s]+?)(?=\s*[.,!?]|\s+and|$)",
            ],
        }

        # Compile patterns for efficiency
        self.compiled_patterns = {
            category: [re.compile(pattern, re.IGNORECASE) for pattern in patterns]
            for category, patterns in self.patterns.items()
        }

    def extract_preferences(self, text: str) -> Dict[str, List[str]]:
        """Extract preferences from a single text input."""
        extracted: Dict[str, List[str]] = {
            "name": [],
            "interests": [],
            "color": [],
            "dislikes": [],
            "age": [],
            "location": [],
            "occupation": [],
        }

        if not text or not text.strip():
            return extracted

        text_lower = text.lower().strip()

        for category, patterns in self.compiled_patterns.items():
            for pattern in patterns:
                matches = pattern.findall(text_lower)
                # Filter out empty matches and common false positives
                filtered_matches = self._filter_matches(category, matches)
                extracted[category].extend(filtered_matches)

        # Deduplicate results
        for category in extracted:
            extracted[category] = list(
                dict.fromkeys(extracted[category])
            )  # Preserve order, remove duplicates

        return extracted

    def _filter_matches(self, category: str, matches: List[Any]) -> List[str]:
        """Filter out false positives and clean up matches."""
        filtered = []

        for match in matches:
            # Handle both strings and tuples from regex groups
            if isinstance(match, tuple):
                # Take the last element (usually the main capture group)
                match_str = match[-1] if match else ""
            else:
                match_str = str(match)

            if not match_str or not match_str.strip():
                continue

            # Clean up the match
            cleaned = match_str.strip()

            # Category-specific filtering
            if category == "name":
                # Filter out common false positives
                if cleaned.lower() in [
                    "data",
                    "computer",
                    "ai",
                    "assistant",
                    "bot",
                    "robot",
                ]:
                    continue
                # Filter out age patterns
                if "years old" in cleaned.lower():
                    continue
                # Filter out location patterns
                if any(
                    loc in cleaned.lower()
                    for loc in ["originally from", "from", "live in"]
                ):
                    continue
                # Filter out articles and occupation patterns
                if cleaned.lower().startswith(("a ", "an ", "the ")):
                    continue
                # Must be reasonable length for a name
                if len(cleaned) < 2 or len(cleaned) > 50:
                    continue

            elif category == "age":
                # Handle both numeric and written ages
                try:
                    # Try to convert to int first
                    age = int(cleaned)
                    if age < 0 or age > 150:
                        continue
                except ValueError:
                    # Handle written numbers
                    written_numbers = {
                        "zero": 0,
                        "one": 1,
                        "two": 2,
                        "three": 3,
                        "four": 4,
                        "five": 5,
                        "six": 6,
                        "seven": 7,
                        "eight": 8,
                        "nine": 9,
                        "ten": 10,
                        "eleven": 11,
                        "twelve": 12,
                        "thirteen": 13,
                        "fourteen": 14,
                        "fifteen": 15,
                        "sixteen": 16,
                        "seventeen": 17,
                        "eighteen": 18,
                        "nineteen": 19,
                        "twenty": 20,
                        "twenty one": 21,
                        "twenty two": 22,
                        "twenty three": 23,
                        "twenty four": 24,
                        "twenty five": 25,
                        "twenty six": 26,
                        "twenty seven": 27,
                        "twenty eight": 28,
                        "twenty nine": 29,
                        "thirty": 30,
                        "thirty one": 31,
                        "thirty two": 32,
                        "thirty three": 33,
                        "thirty four": 34,
                        "thirty five": 35,
                        "thirty six": 36,
                        "thirty seven": 37,
                        "thirty eight": 38,
                        "thirty nine": 39,
                        "forty": 40,
                        "forty one": 41,
                        "forty two": 42,
                        "forty three": 43,
                        "forty four": 44,
                        "forty five": 45,
                        "forty six": 46,
                        "forty seven": 47,
                        "forty eight": 48,
                        "forty nine": 49,
                        "fifty": 50,
                    }
                    age_str = cleaned.lower().strip()
                    if age_str in written_numbers:
                        age = written_numbers[age_str]
                        if age < 0 or age > 150:
                            continue
                    else:
                        # If it's not a recognized written number, keep it as is
                        # (might be a valid age description)
                        pass

            elif category == "interests":
                # Filter out common false positives
                if cleaned.lower() in [
                    "like",
                    "love",
                    "enjoy",
                    "interested",
                    "hobby",
                    "into",
                    "passionate",
                    "fan",
                ]:
                    continue
                # Filter out negative phrases and incomplete thoughts
                if any(
                    neg in cleaned.lower()
                    for neg in [
                        "don't",
                        "hate",
                        "dislike",
                        "not",
                        "can't",
                        "stand",
                        "avoid",
                        "or not",
                    ]
                ):
                    continue
                # Must be reasonable length
                if len(cleaned) < 2 or len(cleaned) > 100:
                    continue

            elif category == "dislikes":
                # Filter out common false positives
                if cleaned.lower() in [
                    "don't",
                    "hate",
                    "dislike",
                    "not",
                    "can't",
                    "stand",
                    "avoid",
                ]:
                    continue
                # Must be reasonable length
                if len(cleaned) < 2 or len(cleaned) > 100:
                    continue

            elif category == "color":
                # Common color validation
                valid_colors = {
                    "red",
                    "blue",
                    "green",
                    "yellow",
                    "orange",
                    "purple",
                    "pink",
                    "brown",
                    "black",
                    "white",
                    "gray",
                    "grey",
                    "silver",
                    "gold",
                    "navy",
                    "maroon",
                    "teal",
                    "turquoise",
                    "lime",
                    "olive",
                    "coral",
                    "magenta",
                    "cyan",
                }
                if cleaned.lower() not in valid_colors and len(cleaned) > 20:
                    continue

            elif category in ["location", "occupation"]:
                # Must be reasonable length
                if len(cleaned) < 2 or len(cleaned) > 100:
                    continue

            filtered.append(cleaned)

        return filtered

    def _fuzzy_match(
        self, text: str, patterns: List[str], threshold: float = 0.8
    ) -> List[str]:
        """Use fuzzy matching to find approximate matches."""
        if not FUZZY_AVAILABLE:
            return []

        matches = []
        text_lower = text.lower()

        for pattern in patterns:
            pattern_lower = pattern.lower()

            # Check for exact substring match first
            if pattern_lower in text_lower:
                matches.append(pattern)
                continue

            # Use sliding window to find best partial match
            best_similarity = 0.0
            pattern_len = len(pattern_lower)

            # Try different window sizes around the pattern length
            for window_size in range(max(3, pattern_len - 2), pattern_len + 3):
                for i in range(len(text_lower) - window_size + 1):
                    window = text_lower[i : i + window_size]
                    similarity = SequenceMatcher(None, window, pattern_lower).ratio()
                    best_similarity = max(best_similarity, similarity)

            if best_similarity >= threshold:
                matches.append(pattern)

        return matches

    def _enhanced_extraction(self, text: str) -> Dict[str, List[str]]:
        """Enhanced extraction using both regex and fuzzy matching."""
        # First try regex patterns (fast and accurate)
        regex_results = self.extract_preferences(text)

        if not FUZZY_AVAILABLE:
            return regex_results

        # If regex found nothing, try fuzzy matching for common patterns
        enhanced_results = regex_results.copy()

        # Fuzzy matching for common STT errors and variations
        text_lower = text.lower().strip()

        # Age fuzzy matching
        if not enhanced_results["age"]:
            age_patterns = [
                "thirty five",
                "thirty-five",
                "35",
                "twenty five",
                "twenty-five",
                "25",
            ]
            fuzzy_ages = self._fuzzy_match(text_lower, age_patterns, threshold=0.7)
            if fuzzy_ages:
                enhanced_results["age"] = fuzzy_ages

        # Color fuzzy matching (only if no color found and text contains color-related words)
        if not enhanced_results["color"] and any(
            word in text_lower for word in ["color", "colour", "favorite", "favourite"]
        ):
            color_patterns = [
                "black",
                "white",
                "red",
                "blue",
                "green",
                "yellow",
                "orange",
                "purple",
                "pink",
            ]
            fuzzy_colors = self._fuzzy_match(text_lower, color_patterns, threshold=0.8)
            if fuzzy_colors:
                # Take the best match (highest similarity)
                best_color = max(
                    fuzzy_colors,
                    key=lambda c: SequenceMatcher(None, text_lower, c.lower()).ratio(),
                )
                enhanced_results["color"] = [best_color]

        # Occupation fuzzy matching (only if no occupation found and text contains work-related words)
        if not enhanced_results["occupation"] and any(
            word in text_lower
            for word in ["engineer", "teacher", "doctor", "work", "job"]
        ):
            occupation_patterns = [
                "software engineer",
                "engineer",
                "teacher",
                "doctor",
                "lawyer",
                "nurse",
            ]
            fuzzy_occupations = self._fuzzy_match(
                text_lower, occupation_patterns, threshold=0.7
            )
            if fuzzy_occupations:
                enhanced_results["occupation"] = fuzzy_occupations

        return enhanced_results

    def update_preferences(
        self,
        current_preferences: UserPreferences,
        new_extractions: Dict[str, List[str]],
    ) -> UserPreferences:
        """Update preferences with new extractions."""
        import time

        updated = UserPreferences(
            user_id=current_preferences.user_id,
            name=current_preferences.name,
            interests=current_preferences.interests.copy(),
            favorite_color=current_preferences.favorite_color,
            dislikes=current_preferences.dislikes.copy(),
            age=current_preferences.age,
            location=current_preferences.location,
            occupation=current_preferences.occupation,
            custom_facts=current_preferences.custom_facts.copy(),
            last_updated=time.time(),
        )

        # Update name (take first new name if none exists)
        if not updated.name and new_extractions["name"]:
            updated.name = new_extractions["name"][0].title()

        # Add interests
        for interest in new_extractions["interests"]:
            updated.interests.add(interest.lower())

        # Update favorite color (take most recent)
        if new_extractions["color"]:
            updated.favorite_color = new_extractions["color"][-1].lower()

        # Add dislikes
        for dislike in new_extractions["dislikes"]:
            updated.dislikes.add(dislike.lower())

        # Update age (take most recent)
        if new_extractions.get("age"):
            try:
                updated.age = int(new_extractions["age"][-1])
            except (ValueError, IndexError):
                pass

        # Update location (take most recent)
        if new_extractions.get("location"):
            updated.location = new_extractions["location"][-1].title()

        # Update occupation (take most recent)
        if new_extractions.get("occupation"):
            updated.occupation = new_extractions["occupation"][-1].title()

        return updated


class PreferenceStorage:
    """Persistent storage for user preferences with GDPR compliance."""

    def __init__(self, storage_path: str):
        """Initialize with storage path."""
        self.storage_path = Path(storage_path)
        self.storage_path.parent.mkdir(parents=True, exist_ok=True)

    def load_preferences(self, user_id: str) -> Optional[UserPreferences]:
        """Load preferences for a user."""
        try:
            if not self.storage_path.exists():
                return None

            with open(self.storage_path, "r") as f:
                data = json.load(f)

            if user_id in data:
                return UserPreferences.from_dict(data[user_id])

            return None

        except (json.JSONDecodeError, KeyError, FileNotFoundError) as e:
            logger.warning(f"Failed to load preferences for {user_id}: {e}")
            return None

    def save_preferences(self, preferences: UserPreferences) -> None:
        """Save preferences for a user."""
        try:
            # Load existing data
            data = {}
            if self.storage_path.exists():
                with open(self.storage_path, "r") as f:
                    data = json.load(f)

            # Update user's preferences
            data[preferences.user_id] = preferences.to_dict()

            # Save back to file
            with open(self.storage_path, "w") as f:
                json.dump(data, f, indent=2)

            logger.debug(f"Saved preferences for user {preferences.user_id}")

        except (json.JSONDecodeError, OSError) as e:
            logger.error(f"Failed to save preferences for {preferences.user_id}: {e}")
            raise

    def export_user_data(self, user_id: str) -> Dict[str, Any]:
        """Export all data for a user (GDPR compliance)."""
        try:
            if not self.storage_path.exists():
                return {}

            with open(self.storage_path, "r") as f:
                data = json.load(f)

            return data.get(user_id, {})  # type: ignore

        except (json.JSONDecodeError, FileNotFoundError) as e:
            logger.warning(f"Failed to export data for {user_id}: {e}")
            return {}

    def delete_user_data(self, user_id: str) -> bool:
        """Delete all data for a user (GDPR compliance)."""
        try:
            if not self.storage_path.exists():
                return True

            with open(self.storage_path, "r") as f:
                data = json.load(f)

            if user_id in data:
                del data[user_id]

                with open(self.storage_path, "w") as f:
                    json.dump(data, f, indent=2)

                logger.info(f"Deleted preferences for user {user_id}")
                return True

            return True

        except (json.JSONDecodeError, OSError) as e:
            logger.error(f"Failed to delete data for {user_id}: {e}")
            return False

    def get_all_users(self) -> List[str]:
        """Get list of all user IDs with stored preferences."""
        try:
            if not self.storage_path.exists():
                return []

            with open(self.storage_path, "r") as f:
                data = json.load(f)

            return list(data.keys())

        except (json.JSONDecodeError, FileNotFoundError) as e:
            logger.warning(f"Failed to get user list: {e}")
            return []
