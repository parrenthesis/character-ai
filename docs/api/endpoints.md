# API Endpoints Summary

Total endpoints: 123

### GET /characters
**Summary**: Root Characters

---

### GET /characters/active
**Summary**: Root Active Character

---

### GET /hardware/power
**Summary**: Root Hardware Power

---

### GET /hardware/sensors
**Summary**: Root Hardware Sensors

---

### GET /hardware/status
**Summary**: Root Hardware Status

---

### GET /health
**Summary**: Root Health

---

### GET /metrics
**Summary**: Root Metrics

---

### GET /models/info
**Summary**: Root Models Info

---

### GET /models/status
**Summary**: Root Models Status

---

### GET /voices
**Summary**: Root Voices

---

### POST /characters/active
**Summary**: Root Set Active Character

---

### POST /interact
**Summary**: Root Interact

---

### POST /voices/inject
**Summary**: Root Voices Inject

---


## Language Support

### GET /api/v1/language/packs
**Summary**: Get Language Packs

**Description**: Get information about available language packs.

---

### GET /api/v1/language/packs/{language_code}/cultural
**Summary**: Get Cultural Adaptations

**Description**: Get cultural adaptation information for a specific language.

---

### GET /api/v1/language/packs/{language_code}/is-rtl
**Summary**: Is Rtl Language

**Description**: Check if a language is right-to-left.

---

### GET /api/v1/language/packs/{language_code}/safety-patterns
**Summary**: Get Safety Patterns

**Description**: Get safety patterns for a specific language.

---

### GET /api/v1/language/packs/{language_code}/voice
**Summary**: Get Voice Characteristics

**Description**: Get voice characteristics for a specific language.

---

### GET /api/v1/language/status
**Summary**: Get Language Status

**Description**: Get current language status and available languages.

---

### POST /api/v1/language/detect
**Summary**: Detect Language

**Description**: Detect the language of input text.

---

### POST /api/v1/language/packs/create-template
**Summary**: Create Language Pack Template

**Description**: Create a template for a new language pack.

---

### POST /api/v1/language/safety/analyze
**Summary**: Analyze Safety Multilingual

**Description**: Analyze text for safety concerns with multi-language support.

---

### POST /api/v1/language/set
**Summary**: Set Language

**Description**: Set the current language.

---


## Multi-language Audio

### GET /api/v1/audio/health
**Summary**: Audio Health Check

**Description**: Health check for multi-language audio system.

---

### GET /api/v1/audio/languages
**Summary**: Get Supported Languages

**Description**: Get list of supported languages.

---

### GET /api/v1/audio/languages/{language_code}/capabilities
**Summary**: Get Language Capabilities

**Description**: Get capabilities for a specific language.

---

### GET /api/v1/audio/tts/stream/{text}
**Summary**: Stream Tts Audio

**Description**: Stream TTS audio for real-time playback.

---

### POST /api/v1/audio/stt
**Summary**: Transcribe Audio

**Description**: Transcribe audio with language detection.

---

### POST /api/v1/audio/tts
**Summary**: Synthesize Speech

**Description**: Synthesize speech with language-specific adaptations.

---

### POST /api/v1/audio/tts/batch
**Summary**: Batch Tts Synthesis

**Description**: Batch TTS synthesis for multiple texts.

---


## Parental Controls

### GET /api/v1/parental-controls/alerts
**Summary**: Get Safety Alerts

**Description**: Get safety alerts with optional filtering.

---

### GET /api/v1/parental-controls/dashboard/{parent_id}
**Summary**: Get Parent Dashboard

**Description**: Get parent dashboard with all children's data.

---

### GET /api/v1/parental-controls/health
**Summary**: Parental Controls Health Check

**Description**: Health check for parental controls system.

---

### GET /api/v1/parental-controls/profiles/{child_id}
**Summary**: Get Profile

**Description**: Get parental control profile for a child.

---

### GET /api/v1/parental-controls/usage-reports/{child_id}
**Summary**: Get Usage Report

**Description**: Get usage report for a child.

---

### POST /api/v1/parental-controls/content-safety
**Summary**: Check Content Safety

**Description**: Check if content is safe for a child.

---

### POST /api/v1/parental-controls/interactions
**Summary**: Record Interaction

**Description**: Record a child's interaction with a character.

---

### POST /api/v1/parental-controls/profiles
**Summary**: Create Profile

**Description**: Create a new parental control profile for a child.

---

### POST /api/v1/parental-controls/sessions/end
**Summary**: End Session

**Description**: End a monitoring session for a child.

---

### POST /api/v1/parental-controls/sessions/start
**Summary**: Start Session

**Description**: Start a monitoring session for a child.

---

### POST /api/v1/parental-controls/time-limits
**Summary**: Check Time Limits

**Description**: Check if child has exceeded time limits.

---

### PUT /api/v1/parental-controls/alerts/{alert_id}/resolve
**Summary**: Resolve Alert

**Description**: Mark a safety alert as resolved.

---

### PUT /api/v1/parental-controls/profiles/{child_id}
**Summary**: Update Profile

**Description**: Update parental control profile for a child.

---


## Personalization

### GET /api/v1/personalization/health
**Summary**: Personalization Health Check

**Description**: Health check for personalization system.

---

### GET /api/v1/personalization/insights
**Summary**: Get User Insights

**Description**: Get user personalization insights.

---

### GET /api/v1/personalization/preferences
**Summary**: Get User Preferences

**Description**: Get all user preferences.

---

### GET /api/v1/personalization/recommendations
**Summary**: Get Character Recommendations

**Description**: Get personalized character recommendations.

---

### POST /api/v1/personalization/conversation-style
**Summary**: Get Adaptive Conversation Style

**Description**: Get adaptive conversation style for a character.

---

### POST /api/v1/personalization/learn
**Summary**: Learn From Conversation

**Description**: Learn from conversation data (for internal use).

---

### POST /api/v1/personalization/preferences
**Summary**: Update Preference

**Description**: Update a user preference.

---

### POST /api/v1/personalization/privacy
**Summary**: Handle Privacy Request

**Description**: Handle privacy-related requests (export, clear, status).

---


## health

### GET /health/
**Summary**: Health Check

**Description**: Basic health check endpoint.

---

### GET /health/components
**Summary**: Component Health

**Description**: Get health status for all components.

---

### GET /health/crashes
**Summary**: Get Crashes

**Description**: Get crash reports with optional filtering.

---

### GET /health/crashes/{crash_id}
**Summary**: Get Crash Details

**Description**: Get detailed information for a specific crash.

---

### GET /health/detailed
**Summary**: Detailed Health Check

**Description**: Detailed health check with component status and system metrics.

---

### GET /health/metrics
**Summary**: Health Metrics

**Description**: Get health-related metrics for monitoring.

---

### GET /health/system
**Summary**: System Health

**Description**: Get system health metrics.

---

### POST /health/crashes/clear
**Summary**: Clear Crashes

**Description**: Clear old crash reports.

---

### POST /health/test-crash
**Summary**: Test Crash Reporting

**Description**: Test crash reporting system (for testing purposes only).

---


## log-search

### GET /api/v1/logs/errors/summary
**Summary**: Get Error Summary

**Description**: Get error summary for the last N hours.

