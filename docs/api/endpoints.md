# API Endpoints Summary

Total endpoints: 101

### GET /
**Summary**: Root

**Description**: Root endpoint with API information.

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


## character-auth

### GET /api/v1/character/auth/me
**Summary**: Get Current Device Info

**Description**: Get current device information.

