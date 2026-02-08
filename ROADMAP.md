# Voice Chat Roadmap

Goal: Transform push-to-talk voice chat into a natural, always-on voice assistant with wake word activation, streaming responses, and interruption support.

## Current State (v0.1 — Done)
- [x] Push-to-talk via mic button or spacebar
- [x] faster-whisper large-v3 STT on GPU
- [x] Kokoro ONNX TTS (local)
- [x] Routes through OpenClaw
- [x] Web UI with HTTPS
- [x] Git repo initialized

---

## Phase 1: Streaming TTS (v0.2) ✅
*Biggest perceived latency improvement for least effort.*

- [x] Split LLM response into sentences as they arrive
- [x] Generate TTS per sentence and send each chunk to the client immediately
- [x] Client plays audio chunks sequentially (queue-based)
- [x] Switch OpenClaw API call to streaming (`stream: true` SSE)
- [x] Accumulate streamed text, detect sentence boundaries, fire TTS per sentence
- [x] Show transcript updating in real-time as tokens arrive

**Completed:** 2026-02-03

---

## Phase 2: Voice Activity Detection — No More Button (v0.3) ✅
*Remove push-to-talk. System listens continuously and detects speech automatically.*

- [x] Add VAD to the frontend (@ricky0123/vad-web — Silero VAD in browser)
- [x] Auto-start recording when speech detected
- [x] Auto-stop and send to STT when silence detected
- [x] Visual indicator: idle → listening → processing → speaking
- [x] Keep the mic button as a manual fallback
- [x] Toggle button to enable/disable VAD mode

**Completed:** 2026-02-03

---

## Phase 3: Interruption Support (v0.4) ✅
*If you talk while Kismet is speaking, it stops and listens.*

- [x] Track playback state on the client (playing / idle)
- [x] Keep VAD active during TTS playback
- [x] On speech detected during playback:
  - Stop current audio immediately
  - Cancel any queued TTS chunks
  - Send cancel signal to server (abort in-flight LLM/TTS)
  - Capture new speech and send to STT
- [x] Server-side: handle cancel gracefully (abort streaming LLM call)
- [x] Interrupted messages shown with visual indicator

**Completed:** 2026-02-03

---

## Phase 4: Wake Word (v0.5) ✅
*Only activate after hearing "Hey Kismet." Low power idle state.*

- [x] OpenWakeWord engine (Python, CPU, TFLite)
- [x] Server-side detection — client streams audio chunks, server runs wake word on CPU
- [x] State machine: `sleeping → wake_word_detected → listening → processing → speaking → sleeping`
- [x] Visual UI states (sleeping indicator, active when awake)
- [x] Configurable timeout: return to sleep after N seconds of no interaction
- [x] Model: `hey_jarvis`, threshold: 0.5

**Completed:** 2026-02-08
**Branch:** `feat/wake-word`
**Depends on:** Phase 3 (interruption)

---

## Phase 5: Speaker Verification (v0.6) ✅
*Only respond to recognized voices. Reject strangers.*

- [x] SpeechBrain ECAPA-TDNN speaker verification module (`speaker_verify.py`)
  - Runs on CPU to keep GPU free for whisper + chatterbox
  - Embedding extraction, cosine similarity comparison
  - Enrollment: average multiple samples → save to `voices/ham_embedding.npy`
  - Verification: compare incoming audio against enrolled embedding
- [x] Enrollment flow via WebSocket (enroll_start → enroll_sample × N → enroll_complete)
- [x] Runtime verification gate in `process_audio()` — reject unrecognized speakers before STT
- [x] UI: Enrollment modal with guided prompts (3 sentences)
- [x] UI: Verification toggle, status badges, similarity scores
- [x] Configurable threshold via `SPEAKER_VERIFY_THRESHOLD` env var (default 0.65)
- [x] `SPEAKER_VERIFY` env var: "auto" (verify if enrolled), "true", "false"

**Completed:** 2026-02-08
**Branch:** `feat/speaker-verification`
**Depends on:** Phase 3 (interruption)

---

## Phase 6: Wake Word + Speaker Verification Combined (v0.7) ✅
*Wake word triggers listening, speaker verification gates processing.*

- [x] Integrate wake word (Phase 4) with speaker verification (Phase 5)
- [x] Flow: wake_word → record speech → verify speaker → if verified, transcribe + respond
- [x] On rejected speaker: return to sleep state (server + client)
- [ ] Reject unrecognized speakers with audio feedback ("I don't recognize your voice")
- [x] Configurable: wake word only, verification only, or both (via env vars)

**Completed:** 2026-02-08
**Depends on:** Phase 4 + Phase 5

---

## Phase 7: Polish & Hardening (v1.0)
*Production-quality touches.*

- [x] Reconnection handling (exponential backoff, state restoration)
- [x] Graceful error messages (toast notifications for all pipeline failures)
- [x] Audio level visualizer (green glow ring on mic button)
- [ ] Settings panel (voice selection, wake word toggle, VAD sensitivity)
- [ ] Mobile-friendly layout
- [ ] Conversation export (save transcript)
- [ ] Startup as a systemd service (optional)
- [ ] Performance profiling (GPU memory, latency benchmarks)
- [ ] Multi-speaker enrollment (recognize different users)
- [ ] Voice profile management UI

**Estimated effort:** 1-2 days
**Branch:** various `feat/*` and `fix/*`

---

## Architecture After All Phases

```
[Browser]
  │
  ├─ Wake word detection (idle, low power)
  ├─ VAD (speech start/end detection)
  ├─ Audio capture + streaming
  ├─ Audio playback (chunked, interruptible)
  │
  └─── WebSocket ───→ [Server on discovery:8765]
                         │
                         ├─ Speaker Verification (CPU) — ECAPA-TDNN
                         ├─ faster-whisper (GPU) — STT
                         ├─ OpenClaw API — LLM
                         └─ Chatterbox Turbo (GPU) — TTS
```

## Phase 8: Meeting Companion
*Passive transcription with diarization. Only responds to Ham's voice on command.*

- [ ] Passive transcription mode (always listening, transcribing in background)
- [ ] Speaker diarization with pyannote (who said what)
- [ ] Command-on-demand: only responds when Ham speaks a command
- [ ] Meeting notes / transcript export
- [ ] Multi-speaker labeling

**Estimated effort:** 2-3 days
**Depends on:** Phase 5 (speaker verification)

---

## Notes
- Each phase is a separate git branch, merged to main when stable
- Phases are incremental — each one works standalone on top of the previous
- GPU VRAM budget: ~4GB used (whisper 3GB + chatterbox), ~8GB headroom
- Wake word + speaker verification run on CPU to avoid competing with STT/TTS for GPU
- Speaker verification adds ~50-100ms latency per request (CPU embedding extraction)
