# Kismet Voice Agent

Real-time voice interface for [OpenClaw](https://github.com/openclaw/openclaw) agents. Talk to your agent using your voice — speech-to-text, text-to-speech, wake word detection, and speaker verification all run locally.

Built by [Kismet Labs](https://kismetlabs.com), an AI consulting firm in the Philippines.

## How It Works

```
Browser (mic) → WebSocket → Server
                              ├─ Wake Word: OpenWakeWord (CPU)
                              ├─ Speaker Verification: SpeechBrain ECAPA-TDNN (CPU)
                              ├─ STT: faster-whisper large-v3 (GPU)
                              ├─ LLM: OpenClaw /v1/chat/completions → your agent
                              └─ TTS: Chatterbox Turbo (GPU)
                            ← audio response
```

Everything except the LLM runs locally on your machine. No cloud STT/TTS APIs, no extra costs.

## Features

- **Wake Word** — say "Hey Jarvis" to activate (OpenWakeWord, runs on CPU)
- **Speaker Verification** — only responds to enrolled voices (SpeechBrain ECAPA-TDNN)
- **Streaming TTS** — responses spoken sentence-by-sentence as they arrive
- **Voice Activity Detection (VAD)** — hands-free, no button required
- **Interruption support** — talk over Kismet and it stops to listen
- **Voice Cloning** — Chatterbox TTS supports cloning from a reference audio file
- **Auto-Reconnection** — WebSocket reconnects with exponential backoff
- **Audio Level Visualizer** — mic input levels shown on the mic button
- **Local processing** — STT, TTS, wake word, and speaker verify all run locally

## Requirements

- **GPU:** NVIDIA with 6GB+ VRAM (tested on RTX 3060 12GB)
- **Python 3.11** (required for Chatterbox TTS)
- **OpenClaw** with chat completions endpoint enabled
- **CUDA** toolkit installed
- **conda** (recommended for environment management)

### Python Dependencies

```bash
pip install -r requirements.txt
```

Or manually:

```bash
pip install faster-whisper fastapi uvicorn numpy httpx chatterbox-tts speechbrain openwakeword
```

### Optional: Kokoro TTS (lighter alternative)

If you prefer Kokoro over Chatterbox (less VRAM, no voice cloning):

```bash
pip install kokoro-onnx
curl -L -o kokoro-v1.0.onnx https://github.com/thewh1teagle/kokoro-onnx/releases/download/model-files-v1.0/kokoro-v1.0.onnx
curl -L -o voices-v1.0.bin https://github.com/thewh1teagle/kokoro-onnx/releases/download/model-files-v1.0/voices-v1.0.bin
```

Set `TTS_ENGINE=kokoro` when running.

## OpenClaw Setup

Enable the chat completions endpoint in your OpenClaw config:

```json
{
  "gateway": {
    "http": {
      "endpoints": {
        "chatCompletions": { "enabled": true }
      }
    }
  }
}
```

## SSL Certificate

Browsers require HTTPS to access the microphone over a network. Generate a self-signed cert:

```bash
openssl req -x509 -newkey rsa:2048 -keyout key.pem -out cert.pem -days 365 -nodes -subj "/CN=localhost"
```

## Usage

### Quick Start (with Chatterbox TTS)

```bash
./start-chatterbox.sh
```

### Quick Start (with Kokoro TTS)

```bash
./start.sh
```

Open `https://<your-host>:8765` in your browser. Accept the self-signed cert warning.

**Controls:**
- **Wake word** — say "Hey Jarvis" to activate (when wake word mode is on)
- **Eye button** — toggle VAD (auto-listen mode)
- **Mic button** — hold to talk (manual mode), glows green to show audio levels
- **Spacebar** — hold to talk (when VAD is off)
- **Talk while Kismet speaks** — interrupts and listens to you
- **Enroll Voice** — register your voice for speaker verification

### Speaker Enrollment

1. Click "Enroll Voice" in the header
2. Record 3 guided sentences (hold to record each one)
3. Your voice embedding is saved to `~/.kismet/voices/`
4. Kismet will now only respond to your voice

### Configuration

| Variable | Default | Description |
|---|---|---|
| `TTS_ENGINE` | `chatterbox` | TTS engine (`chatterbox` or `kokoro`) |
| `CHATTERBOX_REF` | — | Reference audio for voice cloning |
| `WHISPER_MODEL` | `large-v3` | Whisper model size |
| `WHISPER_DEVICE` | `cuda` | Device for STT (`cuda`, `cpu`) |
| `KOKORO_VOICE` | `af_heart` | Kokoro voice ID (when using Kokoro) |
| `WAKE_WORD_ENABLED` | `true` | Enable wake word detection |
| `WAKE_WORD_MODEL` | `hey_jarvis` | Wake word model name |
| `WAKE_WORD_THRESHOLD` | `0.5` | Wake word detection threshold |
| `SPEAKER_VERIFY` | `auto` | Speaker verification (`auto`, `true`, `false`) |
| `SPEAKER_VERIFY_THRESHOLD` | `0.65` | Cosine similarity threshold |
| `IDLE_TIMEOUT_SEC` | `30` | Seconds before returning to sleep |
| `OPENCLAW_URL` | `http://127.0.0.1:18789/v1/chat/completions` | OpenClaw endpoint |
| `OPENCLAW_TOKEN` | — | Gateway auth token |
| `OPENCLAW_AGENT` | `main` | Agent ID to route to |
| `SYSTEM_PROMPT` | *(built-in)* | System prompt for voice responses |

## VRAM Usage

| Component | VRAM |
|---|---|
| faster-whisper large-v3 | ~3 GB |
| Chatterbox Turbo | ~2 GB |
| **Total** | **~5 GB** |

Wake word (OpenWakeWord) and speaker verification (SpeechBrain) run on CPU.

Use `TTS_ENGINE=kokoro` (~300MB) or `WHISPER_MODEL=medium` (~1.5GB) to reduce VRAM.

## Roadmap

See [ROADMAP.md](ROADMAP.md) for the full development plan:

- [x] **Phase 1:** Streaming TTS
- [x] **Phase 2:** Voice Activity Detection (VAD)
- [x] **Phase 3:** Interruption support
- [x] **Phase 4:** Wake word ("Hey Jarvis")
- [x] **Phase 5:** Speaker verification
- [x] **Phase 6:** Wake word + speaker verification combined
- [ ] **Phase 7:** Polish & hardening (in progress)
- [ ] **Phase 8:** Meeting companion (diarization)

## License

MIT
