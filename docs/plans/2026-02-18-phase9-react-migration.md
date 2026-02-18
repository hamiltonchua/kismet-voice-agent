# Phase 9: React UI Migration Implementation Plan

> **For Claude:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task.

**Goal:** Migrate the existing vanilla-JS `index.html` voice agent UI into a Vite + React + TypeScript + Tailwind CSS + shadcn/ui frontend, served by the existing FastAPI backend, preserving 100% of existing functionality.

**Architecture:** Scaffold `frontend/` as a separate Vite project. React hooks encapsulate WebSocket, audio, VAD, and wake-word logic. Components map to the existing HTML sections (header, chat, controls, modal). FastAPI serves the built `frontend/dist/` as static files, with fallback to the old `index.html`.

**Tech Stack:** Vite 6, React 19, TypeScript, Tailwind CSS v4, shadcn/ui (new-york, dark), lucide-react, sonner (toasts), @ricky0123/vad-web, onnxruntime-web

---

## Task 1: Scaffold Vite + React + TypeScript

**Files:**
- Create: `frontend/` (directory, via npm create vite)
- Create: `frontend/package.json`
- Create: `frontend/vite.config.ts`
- Create: `frontend/tsconfig.json`
- Create: `frontend/tsconfig.app.json`
- Create: `frontend/index.html`
- Create: `frontend/src/main.tsx`
- Create: `frontend/src/App.tsx` (placeholder)

**Step 1: Scaffold with Vite**

Run from the project root:
```bash
cd /Users/hamiltonchua/.openclaw/workspace/voice-chat
npm create vite@latest frontend -- --template react-ts
```

When prompted, confirm creating inside existing repo.

**Step 2: Install base dependencies**

```bash
cd frontend
npm install
```

**Step 3: Verify dev server starts**

```bash
cd frontend
npm run dev
```
Expected: Vite dev server starts on port 5173. Ctrl+C to stop.

**Step 4: Commit scaffold**

```bash
cd /Users/hamiltonchua/.openclaw/workspace/voice-chat
git add frontend/
git commit -m "feat: scaffold Vite+React+TS frontend"
```

---

## Task 2: Install and Configure Tailwind CSS v4

**Files:**
- Modify: `frontend/vite.config.ts`
- Modify: `frontend/src/index.css`
- Delete: `frontend/src/App.css` (if exists)

**Step 1: Install Tailwind v4**

```bash
cd /Users/hamiltonchua/.openclaw/workspace/voice-chat/frontend
npm install tailwindcss @tailwindcss/vite
```

**Step 2: Update vite.config.ts**

```typescript
import { defineConfig } from 'vite'
import react from '@vitejs/plugin-react'
import tailwindcss from '@tailwindcss/vite'
import path from 'path'

export default defineConfig({
  plugins: [
    tailwindcss(),
    react(),
  ],
  resolve: {
    alias: {
      '@': path.resolve(__dirname, './src'),
    },
  },
  server: {
    proxy: {
      '/ws': {
        target: 'ws://localhost:8765',
        ws: true,
      },
    },
  },
  build: {
    outDir: 'dist',
  },
})
```

**Step 3: Replace src/index.css with Tailwind import + CSS variables**

```css
@import "tailwindcss";

:root {
  --bg: #0f0f0f;
  --surface: #1a1a2e;
  --surface2: #16213e;
  --accent: #e94560;
  --accent2: #0f3460;
  --text: #eee;
  --text2: #999;
  --green: #00d97e;
  --purple: #9d4edd;
  --radius: 12px;
}

* { box-sizing: border-box; margin: 0; padding: 0; }

body {
  font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', sans-serif;
  background: var(--bg);
  color: var(--text);
  min-height: 100vh;
}
```

**Step 4: Delete App.css if it exists**

```bash
rm -f /Users/hamiltonchua/.openclaw/workspace/voice-chat/frontend/src/App.css
```

**Step 5: Install path types**

```bash
cd /Users/hamiltonchua/.openclaw/workspace/voice-chat/frontend
npm install -D @types/node
```

**Step 6: Verify build still works**

```bash
cd /Users/hamiltonchua/.openclaw/workspace/voice-chat/frontend
npm run build
```
Expected: Build succeeds, `frontend/dist/` created.

**Step 7: Commit**

```bash
cd /Users/hamiltonchua/.openclaw/workspace/voice-chat
git add frontend/
git commit -m "feat: configure Tailwind CSS v4"
```

---

## Task 3: Install and Configure shadcn/ui

**Files:**
- Create: `frontend/components.json`
- Create: `frontend/src/lib/utils.ts`
- Modify: `frontend/src/index.css` (shadcn CSS variables)

**Step 1: Initialize shadcn**

```bash
cd /Users/hamiltonchua/.openclaw/workspace/voice-chat/frontend
npx shadcn@latest init
```

When prompted:
- Style: **New York**
- Base color: **Neutral** (we'll override colors with our own CSS vars)
- CSS variables: **Yes**

**Step 2: Install the shadcn components we need**

```bash
cd /Users/hamiltonchua/.openclaw/workspace/voice-chat/frontend
npx shadcn@latest add button dialog badge toggle
npm install sonner
```

**Step 3: Update src/index.css to merge shadcn dark theme variables with our color scheme**

After shadcn init, the CSS file will have shadcn vars. We need to ensure our custom colors still work. Replace the `:root` and `.dark` blocks so our `--accent`, `--surface`, etc. are defined alongside shadcn vars. The shadcn vars use `--background`, `--foreground`, etc. — keep those. Also keep our `--accent`, `--surface`, etc. for component styling.

The key additions to ensure dark theme:
```css
/* At the top of the existing shadcn CSS variables block */
:root {
  color-scheme: dark;
}
```

**Step 4: Verify shadcn components exist**

```bash
ls /Users/hamiltonchua/.openclaw/workspace/voice-chat/frontend/src/components/ui/
```
Expected: `button.tsx`, `dialog.tsx`, `badge.tsx`, `toggle.tsx`

**Step 5: Commit**

```bash
cd /Users/hamiltonchua/.openclaw/workspace/voice-chat
git add frontend/
git commit -m "feat: install shadcn/ui with new-york dark theme"
```

---

## Task 4: Global Types and Constants

**Files:**
- Create: `frontend/src/types.ts`
- Create: `frontend/src/constants.ts`

**Step 1: Create types.ts**

```typescript
// frontend/src/types.ts

export type ConnectionStatus = 'connecting' | 'connected' | 'reconnecting' | 'disconnected'
export type ClientState = 'sleeping' | 'awake'
export type StatusClass = 'active' | 'listening' | 'sleeping' | ''

export interface ChatMessage {
  id: string
  role: 'user' | 'assistant' | 'system'
  text: string
  meta?: string
  streaming?: boolean
  interrupted?: boolean
  rejected?: boolean
}

export interface MeetingEntry {
  id: string
  time: string
  speaker: string
  text: string
  isOwner: boolean
  speakerType: 'ham' | 'kismet' | 'other'
}

export interface ServerReadyMsg {
  type: 'ready'
  wake_word_enabled: boolean
  wake_word: string | null
  idle_timeout: number
}

export interface TimingInfo {
  stt: number
  llm: number
  tts: number
  first_sentence?: number
}
```

**Step 2: Create constants.ts**

```typescript
// frontend/src/constants.ts

export const SAMPLE_RATE = 16000
export const WAKE_CHUNK_SIZE = 2048
export const MAX_RECONNECT_ATTEMPTS = 10
export const BASE_RECONNECT_MS = 1000

export const ENROLL_PROMPTS = [
  "The quick brown fox jumps over the lazy dog.",
  "How vexingly quick daft zebras jump.",
  "Pack my box with five dozen liquor jugs.",
  "She sells seashells by the seashore.",
  "A journey of a thousand miles begins with a single step.",
]

export const ENROLL_REQUIRED = 3
```

**Step 3: Commit**

```bash
cd /Users/hamiltonchua/.openclaw/workspace/voice-chat
git add frontend/src/types.ts frontend/src/constants.ts
git commit -m "feat: add shared types and constants"
```

---

## Task 5: WebSocket Hook (`useWebSocket`)

**Files:**
- Create: `frontend/src/hooks/useWebSocket.ts`

This hook manages: WS connection lifecycle, exponential backoff reconnect, all inbound message dispatch, outbound `send` helper.

**Step 1: Create the hook**

```typescript
// frontend/src/hooks/useWebSocket.ts
import { useRef, useCallback, useEffect } from 'react'
import { MAX_RECONNECT_ATTEMPTS, BASE_RECONNECT_MS } from '../constants'

export type WSMessage = Record<string, unknown>

export interface UseWebSocketOptions {
  onMessage: (msg: WSMessage) => void
  onOpen?: () => void
  onClose?: () => void
}

export function useWebSocket({ onMessage, onOpen, onClose }: UseWebSocketOptions) {
  const wsRef = useRef<WebSocket | null>(null)
  const reconnectAttemptsRef = useRef(0)
  const reconnectTimerRef = useRef<ReturnType<typeof setTimeout> | null>(null)
  const wasConnectedRef = useRef(false)

  const send = useCallback((data: WSMessage) => {
    if (wsRef.current?.readyState === WebSocket.OPEN) {
      wsRef.current.send(JSON.stringify(data))
    }
  }, [])

  const connect = useCallback(() => {
    if (reconnectTimerRef.current) {
      clearTimeout(reconnectTimerRef.current)
      reconnectTimerRef.current = null
    }

    const proto = location.protocol === 'https:' ? 'wss' : 'ws'
    const ws = new WebSocket(`${proto}://${location.host}/ws`)
    wsRef.current = ws

    ws.onopen = () => {
      console.log('[WS] Connected')
      reconnectAttemptsRef.current = 0
      onOpen?.()
      wasConnectedRef.current = true
    }

    ws.onmessage = (e) => {
      try {
        const msg = JSON.parse(e.data) as WSMessage
        onMessage(msg)
      } catch {
        console.error('[WS] Failed to parse message', e.data)
      }
    }

    ws.onclose = () => {
      console.log('[WS] Disconnected')
      onClose?.()
      wsRef.current = null

      if (reconnectAttemptsRef.current < MAX_RECONNECT_ATTEMPTS) {
        const delay = Math.min(
          BASE_RECONNECT_MS * Math.pow(2, reconnectAttemptsRef.current),
          30000
        )
        reconnectAttemptsRef.current++
        console.log(`[WS] Reconnecting in ${delay}ms (attempt ${reconnectAttemptsRef.current})`)
        reconnectTimerRef.current = setTimeout(connect, delay)
      }
    }

    ws.onerror = (err) => {
      console.error('[WS] Error:', err)
    }
  }, [onMessage, onOpen, onClose])

  const reconnect = useCallback(() => {
    reconnectAttemptsRef.current = 0
    connect()
  }, [connect])

  useEffect(() => {
    connect()
    return () => {
      if (reconnectTimerRef.current) clearTimeout(reconnectTimerRef.current)
      wsRef.current?.close()
    }
  }, []) // intentionally empty — connect is stable but we only want to run on mount

  return { send, reconnect, wsRef, reconnectAttemptsRef, wasConnectedRef }
}
```

**Step 2: Commit**

```bash
cd /Users/hamiltonchua/.openclaw/workspace/voice-chat
git add frontend/src/hooks/useWebSocket.ts
git commit -m "feat: add useWebSocket hook with exponential backoff reconnect"
```

---

## Task 6: Audio Hook (`useAudio`)

**Files:**
- Create: `frontend/src/hooks/useAudio.ts`

Manages: audio queue, sequential playback, interruption, audio level ref (updated via callback for performance — no re-render on every frame).

**Step 1: Create the hook**

```typescript
// frontend/src/hooks/useAudio.ts
import { useRef, useCallback } from 'react'

export function useAudio() {
  const audioQueueRef = useRef<string[]>([])
  const currentAudioRef = useRef<HTMLAudioElement | null>(null)
  const isSpeakingRef = useRef(false)

  const playNext = useCallback(() => {
    if (currentAudioRef.current || audioQueueRef.current.length === 0) return

    isSpeakingRef.current = true
    const b64 = audioQueueRef.current.shift()!
    const bytes = Uint8Array.from(atob(b64), c => c.charCodeAt(0))
    const blob = new Blob([bytes], { type: 'audio/wav' })
    const url = URL.createObjectURL(blob)
    const audio = new Audio(url)
    currentAudioRef.current = audio

    const cleanup = () => {
      URL.revokeObjectURL(url)
      currentAudioRef.current = null
      if (audioQueueRef.current.length === 0) {
        isSpeakingRef.current = false
      }
      playNext()
    }

    audio.onended = cleanup
    audio.onerror = cleanup
    audio.play().catch(cleanup)
  }, [])

  const enqueueAudio = useCallback((b64: string) => {
    audioQueueRef.current.push(b64)
    playNext()
  }, [playNext])

  const stopPlayback = useCallback(() => {
    if (currentAudioRef.current) {
      currentAudioRef.current.pause()
      currentAudioRef.current.currentTime = 0
      currentAudioRef.current = null
    }
    audioQueueRef.current = []
    isSpeakingRef.current = false
  }, [])

  // Legacy single audio playback
  const playAudio = useCallback((b64: string) => {
    const bytes = Uint8Array.from(atob(b64), c => c.charCodeAt(0))
    const blob = new Blob([bytes], { type: 'audio/wav' })
    const url = URL.createObjectURL(blob)
    const audio = new Audio(url)
    audio.play()
    audio.onended = () => URL.revokeObjectURL(url)
  }, [])

  return { enqueueAudio, stopPlayback, playAudio, isSpeakingRef, currentAudioRef }
}
```

**Step 2: Commit**

```bash
cd /Users/hamiltonchua/.openclaw/workspace/voice-chat
git add frontend/src/hooks/useAudio.ts
git commit -m "feat: add useAudio hook with queue and interruption"
```

---

## Task 7: Wake Word Streaming Hook (`useWakeWordStream`)

**Files:**
- Create: `frontend/src/hooks/useWakeWordStream.ts`

Manages: mic capture via ScriptProcessor, float32→int16 conversion, base64 encoding, sending `audio_stream` WS messages, audio level calculation.

**Step 1: Create the hook**

```typescript
// frontend/src/hooks/useWakeWordStream.ts
import { useRef, useCallback } from 'react'
import { SAMPLE_RATE, WAKE_CHUNK_SIZE } from '../constants'

function arrayBufferToBase64(buffer: ArrayBuffer): string {
  const bytes = new Uint8Array(buffer)
  let binary = ''
  for (let i = 0; i < bytes.length; i++) binary += String.fromCharCode(bytes[i])
  return btoa(binary)
}

interface UseWakeWordStreamOptions {
  onAudioLevel: (rms: number) => void
  shouldStream: () => boolean // called per chunk — returns true if should send to WS
  onSend: (b64: string) => void
}

export function useWakeWordStream({ onAudioLevel, shouldStream, onSend }: UseWakeWordStreamOptions) {
  const audioContextRef = useRef<AudioContext | null>(null)
  const processorRef = useRef<ScriptProcessorNode | null>(null)
  const sourceRef = useRef<MediaStreamAudioSourceNode | null>(null)
  const streamRef = useRef<MediaStream | null>(null)

  const start = useCallback(async () => {
    if (audioContextRef.current) return // Already running

    try {
      const stream = await navigator.mediaDevices.getUserMedia({
        audio: {
          sampleRate: SAMPLE_RATE,
          channelCount: 1,
          echoCancellation: true,
          noiseSuppression: true,
        },
      })
      streamRef.current = stream

      const ctx = new AudioContext({ sampleRate: SAMPLE_RATE })
      audioContextRef.current = ctx

      const source = ctx.createMediaStreamSource(stream)
      sourceRef.current = source

      const processor = ctx.createScriptProcessor(WAKE_CHUNK_SIZE, 1, 1)
      processor.onaudioprocess = (e) => {
        const float32 = e.inputBuffer.getChannelData(0)

        // Audio level
        let sum = 0
        for (let i = 0; i < float32.length; i++) sum += float32[i] * float32[i]
        onAudioLevel(Math.sqrt(sum / float32.length))

        if (!shouldStream()) return

        const int16 = new Int16Array(float32.length)
        for (let i = 0; i < float32.length; i++) {
          int16[i] = Math.max(-32768, Math.min(32767, Math.round(float32[i] * 32767)))
        }
        onSend(arrayBufferToBase64(int16.buffer))
      }

      source.connect(processor)
      processor.connect(ctx.destination)
      processorRef.current = processor

      console.log('[WakeWord] Audio streaming started')
    } catch (err: unknown) {
      console.error('[WakeWord] Failed to start streaming:', err)
      const error = err as { name?: string }
      if (error.name === 'NotAllowedError' || error.name === 'PermissionDeniedError') {
        throw new Error('MIC_DENIED')
      }
      throw err
    }
  }, [onAudioLevel, shouldStream, onSend])

  const stop = useCallback(() => {
    onAudioLevel(0)
    processorRef.current?.disconnect()
    processorRef.current = null
    sourceRef.current?.disconnect()
    sourceRef.current = null
    audioContextRef.current?.close()
    audioContextRef.current = null
    streamRef.current?.getTracks().forEach(t => t.stop())
    streamRef.current = null
    console.log('[WakeWord] Audio streaming stopped')
  }, [onAudioLevel])

  return { start, stop, isStreaming: () => !!audioContextRef.current }
}
```

**Step 2: Commit**

```bash
cd /Users/hamiltonchua/.openclaw/workspace/voice-chat
git add frontend/src/hooks/useWakeWordStream.ts
git commit -m "feat: add useWakeWordStream hook"
```

---

## Task 8: VAD Hook (`useVAD`)

**Files:**
- Create: `frontend/src/hooks/useVAD.ts`

Wraps `@ricky0123/vad-web` MicVAD. Manages lifecycle (init, start, pause). Exposes `onSpeechEnd` callback which the parent decides what to do with (send audio or discard).

**Step 1: Install VAD library**

```bash
cd /Users/hamiltonchua/.openclaw/workspace/voice-chat/frontend
npm install @ricky0123/vad-web onnxruntime-web
```

**Step 2: Create the hook**

```typescript
// frontend/src/hooks/useVAD.ts
import { useRef, useCallback } from 'react'
import { MicVAD } from '@ricky0123/vad-web'
import { SAMPLE_RATE } from '../constants'

interface UseVADOptions {
  onSpeechStart: () => void
  onSpeechEnd: (audio: Float32Array) => void
  onFrameProcessed?: (probs: { isSpeech: number }) => void
}

export function useVAD({ onSpeechStart, onSpeechEnd, onFrameProcessed }: UseVADOptions) {
  const vadRef = useRef<MicVAD | null>(null)
  const listeningRef = useRef(false)

  const init = useCallback(async () => {
    if (vadRef.current) return

    const vadStream = await navigator.mediaDevices.getUserMedia({
      audio: { echoCancellation: true, noiseSuppression: true, autoGainControl: true },
    })

    vadRef.current = await MicVAD.new({
      stream: vadStream,
      positiveSpeechThreshold: 0.9,
      negativeSpeechThreshold: 0.35,
      minSpeechFrames: 8,
      preSpeechPadFrames: 10,
      redemptionFrames: 20,
      onFrameProcessed: (probs) => {
        onFrameProcessed?.(probs)
      },
      onSpeechStart: () => {
        onSpeechStart()
      },
      onSpeechEnd: (audio) => {
        onSpeechEnd(audio)
      },
    })
    console.log('[VAD] Initialized')
  }, [onSpeechStart, onSpeechEnd, onFrameProcessed])

  const start = useCallback(async () => {
    if (!vadRef.current) await init()
    vadRef.current?.start()
    listeningRef.current = true
    console.log('[VAD] Started')
  }, [init])

  const pause = useCallback(() => {
    vadRef.current?.pause()
    listeningRef.current = false
    console.log('[VAD] Paused')
  }, [])

  const isListening = useCallback(() => listeningRef.current, [])

  return { start, pause, isListening, vadRef }
}
```

**Step 3: Commit**

```bash
cd /Users/hamiltonchua/.openclaw/workspace/voice-chat
git add frontend/src/hooks/useVAD.ts
git commit -m "feat: add useVAD hook wrapping @ricky0123/vad-web"
```

---

## Task 9: Manual Recording Hook (`useManualRecording`)

**Files:**
- Create: `frontend/src/hooks/useManualRecording.ts`

Manages: mic stream (reuse if possible), ScriptProcessor, chunk accumulation, float32→int16 PCM, merge and callback.

**Step 1: Create the hook**

```typescript
// frontend/src/hooks/useManualRecording.ts
import { useRef, useCallback } from 'react'
import { SAMPLE_RATE } from '../constants'

function arrayBufferToBase64(buffer: ArrayBuffer): string {
  const bytes = new Uint8Array(buffer)
  let binary = ''
  for (let i = 0; i < bytes.length; i++) binary += String.fromCharCode(bytes[i])
  return btoa(binary)
}

interface UseManualRecordingOptions {
  onAudioLevel: (rms: number) => void
  onRecordingStart: () => void
  onRecordingEnd: (float32: Float32Array) => void
  onError: (msg: string) => void
}

export function useManualRecording({
  onAudioLevel,
  onRecordingStart,
  onRecordingEnd,
  onError,
}: UseManualRecordingOptions) {
  const streamRef = useRef<MediaStream | null>(null)
  const ctxRef = useRef<AudioContext | null>(null)
  const recorderRef = useRef<{ proc: ScriptProcessorNode; source: MediaStreamAudioSourceNode } | null>(null)
  const chunksRef = useRef<Float32Array[]>([])

  const start = useCallback(async () => {
    if (!streamRef.current) {
      try {
        streamRef.current = await navigator.mediaDevices.getUserMedia({
          audio: { sampleRate: SAMPLE_RATE, channelCount: 1, echoCancellation: true, noiseSuppression: true },
        })
      } catch (err: unknown) {
        const error = err as { name?: string; message?: string }
        if (error.name === 'NotAllowedError' || error.name === 'PermissionDeniedError') {
          onError('Mic access denied — enable microphone in browser settings')
        } else {
          onError('Could not access microphone: ' + (error.message ?? ''))
        }
        return false
      }
    }

    const ctx = new AudioContext({ sampleRate: SAMPLE_RATE })
    ctxRef.current = ctx
    const source = ctx.createMediaStreamSource(streamRef.current)
    const proc = ctx.createScriptProcessor(4096, 1, 1)

    chunksRef.current = []
    proc.onaudioprocess = (e) => {
      const float32 = e.inputBuffer.getChannelData(0)
      chunksRef.current.push(new Float32Array(float32))
      let sum = 0
      for (let i = 0; i < float32.length; i++) sum += float32[i] * float32[i]
      onAudioLevel(Math.sqrt(sum / float32.length))
    }

    source.connect(proc)
    proc.connect(ctx.destination)
    recorderRef.current = { proc, source }

    onRecordingStart()
    return true
  }, [onAudioLevel, onRecordingStart, onError])

  const stop = useCallback(() => {
    if (!recorderRef.current) return

    const { proc, source } = recorderRef.current
    proc.disconnect()
    source.disconnect()
    ctxRef.current?.close()
    ctxRef.current = null
    recorderRef.current = null

    onAudioLevel(0)

    const chunks = chunksRef.current
    chunksRef.current = []

    const total = chunks.reduce((n, c) => n + c.length, 0)
    const merged = new Float32Array(total)
    let offset = 0
    for (const chunk of chunks) {
      merged.set(chunk, offset)
      offset += chunk.length
    }

    onRecordingEnd(merged)
  }, [onAudioLevel, onRecordingEnd])

  const isRecording = useCallback(() => !!recorderRef.current, [])

  return { start, stop, isRecording }
}

export function float32ToBase64(float32: Float32Array): string {
  const int16 = new Int16Array(float32.length)
  for (let i = 0; i < float32.length; i++) {
    int16[i] = Math.max(-32768, Math.min(32767, Math.round(float32[i] * 32767)))
  }
  const bytes = new Uint8Array(int16.buffer)
  let binary = ''
  for (let i = 0; i < bytes.length; i++) binary += String.fromCharCode(bytes[i])
  return btoa(binary)
}
```

**Step 2: Commit**

```bash
cd /Users/hamiltonchua/.openclaw/workspace/voice-chat
git add frontend/src/hooks/useManualRecording.ts
git commit -m "feat: add useManualRecording hook"
```

---

## Task 10: Audio Level Visualizer Hook (`useAudioLevel`)

**Files:**
- Create: `frontend/src/hooks/useAudioLevel.ts`

A simple hook that manages audio level state for the mic ring visualization. Uses a ref + RAF-friendly approach via a callback, not useState (avoids expensive re-renders).

**Step 1: Create the hook**

```typescript
// frontend/src/hooks/useAudioLevel.ts
import { useRef, useCallback } from 'react'

export function useAudioLevel(ringRef: React.RefObject<HTMLDivElement | null>) {
  const rmsRef = useRef(0)

  const updateAudioLevel = useCallback((rms: number) => {
    rmsRef.current = rms
    const el = ringRef.current
    if (!el) return

    if (rms > 0.01) {
      el.classList.add('active')
      const spread = Math.min(rms * 40, 12)
      const alpha = Math.min(rms * 3, 0.8)
      el.style.boxShadow = `0 0 ${spread}px rgba(0,217,126,${alpha})`
    } else {
      el.classList.remove('active')
      el.style.boxShadow = 'none'
    }
  }, [ringRef])

  return { updateAudioLevel }
}
```

**Step 2: Commit**

```bash
cd /Users/hamiltonchua/.openclaw/workspace/voice-chat
git add frontend/src/hooks/useAudioLevel.ts
git commit -m "feat: add useAudioLevel hook"
```

---

## Task 11: App State and Main App Component

**Files:**
- Create: `frontend/src/App.tsx` (main orchestration)

This is the central component that wires all hooks together and manages all application state. It renders the full UI tree. It's deliberately a large component — splitting state across too many contexts would introduce unnecessary complexity.

**Step 1: Create App.tsx**

This file is large. Write it in full, integrating all hooks:

```tsx
// frontend/src/App.tsx
import { useState, useRef, useCallback, useEffect } from 'react'
import { Toaster, toast } from 'sonner'
import { useWebSocket } from './hooks/useWebSocket'
import { useAudio } from './hooks/useAudio'
import { useWakeWordStream } from './hooks/useWakeWordStream'
import { useVAD } from './hooks/useVAD'
import { useManualRecording, float32ToBase64 } from './hooks/useManualRecording'
import { useAudioLevel } from './hooks/useAudioLevel'
import { SAMPLE_RATE, ENROLL_PROMPTS, ENROLL_REQUIRED } from './constants'
import type { ChatMessage, MeetingEntry, ClientState, TimingInfo } from './types'
import { Header } from './components/Header'
import { ChatDisplay } from './components/ChatDisplay'
import { Controls } from './components/Controls'
import { EnrollModal } from './components/EnrollModal'
import { ReconnectBanner } from './components/ReconnectBanner'
import { MeetingBanner } from './components/MeetingBanner'
import { StatusDisplay } from './components/StatusDisplay'

export default function App() {
  // ---- Connection state ----
  const [connectionStatus, setConnectionStatus] = useState<'connecting' | 'reconnecting' | 'disconnected' | 'connected'>('connecting')
  const [showReconnectBanner, setShowReconnectBanner] = useState(false)

  // ---- Server capabilities ----
  const [wakeWordEnabled, setWakeWordEnabled] = useState(false)
  const [wakeWordName, setWakeWordName] = useState<string | null>(null)
  const [idleTimeout, setIdleTimeout] = useState(30)

  // ---- Client mode state ----
  const [wakeWordActive, setWakeWordActive] = useState(false)
  const [clientState, setClientState] = useState<ClientState>('awake')
  const clientStateRef = useRef<ClientState>('awake')

  // ---- Processing state ----
  const [isProcessing, setIsProcessing] = useState(false)
  const isProcessingRef = useRef(false)

  // ---- Status display ----
  const [statusText, setStatusText] = useState('Connecting...')
  const [statusClass, setStatusClass] = useState<'active' | 'listening' | 'sleeping' | ''>('')

  // ---- Chat messages ----
  const [messages, setMessages] = useState<ChatMessage[]>([])
  const [streamingId, setStreamingId] = useState<string | null>(null)
  const streamingTextRef = useRef('')

  // ---- Speaker verification ----
  const [enrolled, setEnrolled] = useState(false)
  const [verifyEnabled, setVerifyEnabled] = useState(false)
  const [lastVerifyScore, setLastVerifyScore] = useState<number | null>(null)
  const [showEnrollModal, setShowEnrollModal] = useState(false)
  const [enrollStep, setEnrollStep] = useState(0)

  // ---- Meeting mode ----
  const [meetingMode, setMeetingMode] = useState(false)
  const meetingModeRef = useRef(false)
  const [meetingEntries, setMeetingEntries] = useState<MeetingEntry[]>([])
  const meetingCommandCaptureRef = useRef(false)

  // ---- Mic state for UI ----
  const [micState, setMicState] = useState<'idle' | 'recording' | 'vad-active' | 'sleeping'>('idle')

  // ---- Audio level ring ----
  const audioLevelRingRef = useRef<HTMLDivElement | null>(null)
  const { updateAudioLevel } = useAudioLevel(audioLevelRingRef)

  // ---- Reconnect state ----
  const savedWakeWordActiveRef = useRef(false)
  const wasConnectedRef2 = useRef(false)

  // ---- Refs for callbacks that need current state ----
  const wakeWordActiveRef = useRef(false)
  const wakeWordEnabledRef = useRef(false)
  const wakeWordNameRef = useRef<string | null>(null)
  const meetingEntriesRef = useRef<MeetingEntry[]>([])

  // Keep refs in sync
  useEffect(() => { clientStateRef.current = clientState }, [clientState])
  useEffect(() => { isProcessingRef.current = isProcessing }, [isProcessing])
  useEffect(() => { wakeWordActiveRef.current = wakeWordActive }, [wakeWordActive])
  useEffect(() => { wakeWordEnabledRef.current = wakeWordEnabled }, [wakeWordEnabled])
  useEffect(() => { wakeWordNameRef.current = wakeWordName }, [wakeWordName])
  useEffect(() => { meetingModeRef.current = meetingMode }, [meetingMode])

  const setStatus = useCallback((text: string, cls: 'active' | 'listening' | 'sleeping' | '' = '') => {
    setStatusText(text)
    setStatusClass(cls)
  }, [])

  // ---- Audio ----
  const { enqueueAudio, stopPlayback, playAudio, isSpeakingRef } = useAudio()

  // ---- Interrupt ----
  const interrupt = useCallback(() => {
    console.log('[Interrupt] User interrupted')
    stopPlayback()
    setMessages(prev => prev.map(m =>
      m.streaming ? { ...m, streaming: false, interrupted: true } : m
    ))
    setStreamingId(null)
    streamingTextRef.current = ''
    send({ type: 'cancel' })
    setIsProcessing(false)
    isProcessingRef.current = false
  }, [stopPlayback])

  // ---- Wake word streaming ----
  const shouldStreamWakeWord = useCallback(() => {
    return clientStateRef.current === 'sleeping' ||
      (meetingModeRef.current && !meetingCommandCaptureRef.current)
  }, [])

  const { start: startWakeStream, stop: stopWakeStream } = useWakeWordStream({
    onAudioLevel: updateAudioLevel,
    shouldStream: shouldStreamWakeWord,
    onSend: (b64) => send({ type: 'audio_stream', data: b64 }),
  })

  // ---- VAD ----
  const handleVADSpeechStart = useCallback(() => {
    if (!meetingModeRef.current && (isSpeakingRef.current || isProcessingRef.current)) {
      interrupt()
    }
    setMicState('recording')
    setStatus(meetingModeRef.current ? 'Meeting mode — speaking detected...' : 'Listening...', 'listening')
  }, [interrupt, isSpeakingRef, setStatus])

  const handleVADSpeechEnd = useCallback((audio: Float32Array) => {
    setMicState(wakeWordActiveRef.current ? 'vad-active' : 'idle')
    if (audio.length > SAMPLE_RATE * 0.3) {
      sendAudio(audio)
    } else {
      updateStatusForMode()
    }
  }, [])

  const { start: startVAD, pause: pauseVAD, isListening, vadRef } = useVAD({
    onSpeechStart: handleVADSpeechStart,
    onSpeechEnd: handleVADSpeechEnd,
    onFrameProcessed: (probs) => {
      updateAudioLevel(probs.isSpeech * 0.5)
    },
  })

  // sendAudio needs to be defined before VAD hooks use it in callbacks
  // Using a ref to avoid circular dependency
  const sendAudioRef = useRef<(float32: Float32Array) => void>(() => {})

  // ---- Manual recording ----
  const handleManualRecordingEnd = useCallback((float32: Float32Array) => {
    setMicState(wakeWordActiveRef.current && clientStateRef.current === 'awake' ? 'vad-active' : 'idle')
    updateAudioLevel(0)
    // Resume VAD if in awake wake-word mode
    if (clientStateRef.current === 'awake' && wakeWordActiveRef.current && vadRef.current) {
      vadRef.current.start()
    }
    if (float32.length < SAMPLE_RATE * 0.3) {
      setStatus('Too short — hold longer', 'active')
      return
    }
    sendAudioRef.current(float32)
  }, [updateAudioLevel, setStatus, vadRef])

  const { start: startManual, stop: stopManual, isRecording } = useManualRecording({
    onAudioLevel: updateAudioLevel,
    onRecordingStart: () => setMicState('recording'),
    onRecordingEnd: handleManualRecordingEnd,
    onError: (msg) => toast.error(msg),
  })

  // ---- Send audio helper ----
  const sendAudio = useCallback((float32: Float32Array) => {
    const b64 = float32ToBase64(float32)
    if (meetingModeRef.current && meetingCommandCaptureRef.current) {
      meetingCommandCaptureRef.current = false
      send({ type: 'meeting_command', data: b64 })
      setStatus('Processing command...', 'active')
      if (wakeWordEnabledRef.current) startWakeStream()
    } else if (meetingModeRef.current) {
      send({ type: 'meeting_audio', data: b64 })
      setStatus('Meeting mode — listening...', 'listening')
    } else {
      send({ type: 'audio', data: b64 })
      setIsProcessing(true)
      isProcessingRef.current = true
      setStatus('Processing...', 'active')
    }
  }, [setStatus, startWakeStream])

  // Keep sendAudioRef in sync
  useEffect(() => { sendAudioRef.current = sendAudio }, [sendAudio])

  const updateStatusForMode = useCallback(() => {
    if (wakeWordActiveRef.current) {
      if (clientStateRef.current === 'sleeping') {
        setStatus(`Say "${wakeWordNameRef.current}" to wake me`, 'sleeping')
      } else {
        setStatus('Listening — talk anytime', 'active')
      }
    } else if (isListening()) {
      setStatus('Listening...', 'listening')
    } else {
      setStatus('Ready — hold mic to talk', 'active')
    }
  }, [setStatus, isListening])

  // ---- Wake word mode ----
  const enableWakeWordMode = useCallback(async () => {
    setWakeWordActive(true)
    wakeWordActiveRef.current = true
    setClientState('sleeping')
    clientStateRef.current = 'sleeping'
    setMicState('sleeping')
    if (vadRef.current) pauseVAD()
    send({ type: 'set_state', state: 'sleeping' })
    try {
      await startWakeStream()
    } catch (e: unknown) {
      const err = e as Error
      if (err.message === 'MIC_DENIED') {
        toast.error('Mic access denied — enable microphone in browser settings', { duration: 6000 })
      }
    }
    setStatus(`Say "${wakeWordNameRef.current}" to wake me`, 'sleeping')
  }, [pauseVAD, startWakeStream, setStatus, vadRef])

  const disableWakeWordMode = useCallback(() => {
    setWakeWordActive(false)
    wakeWordActiveRef.current = false
    setClientState('awake')
    clientStateRef.current = 'awake'
    setMicState('idle')
    stopWakeStream()
    if (vadRef.current) pauseVAD()
    updateStatusForMode()
  }, [stopWakeStream, pauseVAD, updateStatusForMode, vadRef])

  const handleWake = useCallback(async () => {
    setClientState('awake')
    clientStateRef.current = 'awake'
    setMicState('vad-active')
    stopWakeStream()
    await new Promise(r => setTimeout(r, 200))
    setStatus('Listening...', 'listening')
    await startVAD()
  }, [stopWakeStream, setStatus, startVAD])

  const handleSleep = useCallback(() => {
    if (!wakeWordActiveRef.current) return
    setClientState('sleeping')
    clientStateRef.current = 'sleeping'
    setMicState('sleeping')
    if (vadRef.current) pauseVAD()
    startWakeStream()
    setStatus(`Say "${wakeWordNameRef.current}" to wake me`, 'sleeping')
  }, [pauseVAD, startWakeStream, setStatus, vadRef])

  // ---- Meeting mode ----
  const enterMeetingMode = useCallback(async () => {
    setMeetingMode(true)
    meetingModeRef.current = true
    setMessages([{ id: 'meeting-start', role: 'system', text: 'Meeting mode started. Transcribing all speakers... Say "Hey Friday" to ask a question.' }])
    setMeetingEntries([])
    send({ type: 'meeting_start' })
    await startVAD()
    setMicState('vad-active')
    if (wakeWordEnabledRef.current) startWakeStream()
    setStatus('Meeting mode — listening...', 'listening')
  }, [setStatus, startVAD, startWakeStream])

  const exitMeetingMode = useCallback(() => {
    setMeetingMode(false)
    meetingModeRef.current = false
    send({ type: 'meeting_stop' })
    pauseVAD()
    stopWakeStream()
    setMicState('idle')
    updateStatusForMode()
  }, [pauseVAD, stopWakeStream, updateStatusForMode])

  const handleMeetingWake = useCallback(async () => {
    meetingCommandCaptureRef.current = true
    stopWakeStream()
    try {
      const actx = new AudioContext()
      const osc = actx.createOscillator()
      const gain = actx.createGain()
      osc.frequency.value = 880
      gain.gain.value = 0.15
      osc.connect(gain).connect(actx.destination)
      osc.start()
      osc.stop(actx.currentTime + 0.15)
      osc.onended = () => actx.close()
    } catch {}
    setStatus('🎤 Listening for your command...', 'listening')
    if (vadRef.current) pauseVAD()
    await new Promise(r => setTimeout(r, 300))
    if (vadRef.current) vadRef.current.start()
  }, [stopWakeStream, setStatus, pauseVAD, vadRef])

  // ---- WebSocket message handler ----
  const handleWSMessage = useCallback((msg: Record<string, unknown>) => {
    const type = msg.type as string

    if (type === 'ready') {
      const wwe = msg.wake_word_enabled as boolean
      const wwn = msg.wake_word as string | null
      const it = (msg.idle_timeout as number) || 30
      setWakeWordEnabled(wwe)
      wakeWordEnabledRef.current = wwe
      setWakeWordName(wwn)
      wakeWordNameRef.current = wwn
      setIdleTimeout(it)
      setConnectionStatus('connected')
      setShowReconnectBanner(false)

      if (wwe) {
        if (savedWakeWordActiveRef.current || wakeWordActiveRef.current) {
          enableWakeWordMode()
        } else {
          updateStatusForMode()
        }
      } else {
        updateStatusForMode()
      }
      send({ type: 'check_enrollment' })

    } else if (type === 'error') {
      toast.error(msg.text as string)
      setIsProcessing(false)
      isProcessingRef.current = false
      updateStatusForMode()

    } else if (type === 'wake') {
      if (msg.meeting && meetingModeRef.current) {
        handleMeetingWake()
      } else {
        handleWake()
      }

    } else if (type === 'sleep') {
      handleSleep()

    } else if (type === 'status') {
      setStatus(msg.text as string, 'active')

    } else if (type === 'transcript') {
      const metaParts: string[] = []
      if (msg.time) metaParts.push(`${msg.time}s`)
      if (msg.role === 'user' && lastVerifyScore !== null) {
        const badge = lastVerifyScore >= 0.50 ? '✓' : '✗'
        metaParts.push(`speaker ${badge} ${lastVerifyScore.toFixed(2)}`)
      }
      setMessages(prev => [...prev, {
        id: crypto.randomUUID(),
        role: msg.role as 'user' | 'assistant',
        text: msg.text as string,
        meta: metaParts.join(' · ') || undefined,
      }])

    } else if (type === 'stream_start') {
      streamingTextRef.current = ''
      const id = crypto.randomUUID()
      setStreamingId(id)
      setMessages(prev => [...prev, {
        id,
        role: 'assistant',
        text: '',
        streaming: true,
      }])

    } else if (type === 'token') {
      streamingTextRef.current += (msg.text as string)
      const text = streamingTextRef.current
      setMessages(prev => prev.map(m =>
        m.streaming ? { ...m, text } : m
      ))

    } else if (type === 'audio_chunk') {
      enqueueAudio(msg.data as string)

    } else if (type === 'stream_end') {
      const t = msg.times as TimingInfo
      const parts = [`STT ${t.stt}s`, `LLM ${t.llm}s`, `TTS ${t.tts}s`]
      if (t.first_sentence) parts.push(`first audio ${t.first_sentence}s`)
      setMessages(prev => prev.map(m =>
        m.streaming ? { ...m, streaming: false, text: msg.text as string, meta: parts.join(' · ') } : m
      ))
      setStreamingId(null)
      streamingTextRef.current = ''
      setIsProcessing(false)
      isProcessingRef.current = false
      updateStatusForMode()

    } else if (type === 'meeting_transcript') {
      const entry: MeetingEntry = {
        id: crypto.randomUUID(),
        time: msg.time as string,
        speaker: msg.speaker as string,
        text: msg.text as string,
        isOwner: msg.is_owner as boolean,
        speakerType: msg.is_owner ? 'ham' : msg.speaker === 'Kismet' ? 'kismet' : 'other',
      }
      setMeetingEntries(prev => [...prev, entry])

    } else if (type === 'meeting_stopped') {
      const speakers = (msg.speakers as string[]).join(', ')
      setMessages(prev => [...prev, {
        id: crypto.randomUUID(),
        role: 'system',
        text: `Meeting ended. ${msg.entries} entries, ${(msg.speakers as string[]).length} speakers: ${speakers}`,
      }])

    } else if (type === 'cancelled') {
      setIsProcessing(false)
      isProcessingRef.current = false
      updateStatusForMode()

    } else if (type === 'audio') {
      playAudio(msg.data as string)
      const t = msg.times as TimingInfo
      setStatus(`STT ${t.stt}s · LLM ${t.llm}s · TTS ${t.tts}s`, 'active')
      setIsProcessing(false)
      isProcessingRef.current = false

    } else if (type === 'verified') {
      setLastVerifyScore(msg.score as number)

    } else if (type === 'rejected') {
      setLastVerifyScore(msg.score as number)
      setMessages(prev => [...prev, {
        id: crypto.randomUUID(),
        role: 'user',
        text: '[Voice not recognized]',
        meta: `Score: ${msg.score} · ${msg.time}s`,
        rejected: true,
      }])
      setIsProcessing(false)
      isProcessingRef.current = false
      updateStatusForMode()

    } else if (type === 'enroll_status') {
      if (msg.status === 'sample_received') {
        setEnrollStep(msg.samples as number)
      } else if (msg.status === 'complete') {
        setEnrolled(true)
        setVerifyEnabled(true)
        setShowEnrollModal(false)
      } else if (msg.status === 'error') {
        toast.error((msg.message as string) || 'Enrollment error')
      }

    } else if (type === 'enrollment_info') {
      setEnrolled(msg.enrolled as boolean)
      setVerifyEnabled(msg.verify_enabled as boolean)

    } else if (type === 'verify_toggled') {
      setVerifyEnabled(msg.enabled as boolean)
    }
  }, [
    enableWakeWordMode, updateStatusForMode, handleMeetingWake, handleWake, handleSleep,
    setStatus, enqueueAudio, playAudio, lastVerifyScore,
  ])

  // ---- WebSocket hook ----
  const { send, reconnect, reconnectAttemptsRef } = useWebSocket({
    onMessage: handleWSMessage,
    onOpen: () => {
      setConnectionStatus('connected')
      setShowReconnectBanner(false)
    },
    onClose: () => {
      stopWakeStream()
      stopPlayback()
      savedWakeWordActiveRef.current = wakeWordActiveRef.current
      if (reconnectAttemptsRef.current >= 10) {
        setShowReconnectBanner(true)
        setStatus('Connection lost', '')
      } else {
        setStatus(`Reconnecting (${reconnectAttemptsRef.current}/10)...`, '')
        setConnectionStatus('reconnecting')
      }
    },
  })

  // ---- Wake toggle ----
  const handleWakeToggle = useCallback(async () => {
    if (!wakeWordEnabledRef.current) {
      // Hands-free VAD toggle
      if (isListening()) {
        pauseVAD()
        setMicState('idle')
        setWakeWordActive(false)
        wakeWordActiveRef.current = false
        setStatus('Click mic to talk', '')
      } else {
        setClientState('awake')
        clientStateRef.current = 'awake'
        setMicState('vad-active')
        setWakeWordActive(true)
        wakeWordActiveRef.current = true
        setStatus('Listening...', 'listening')
        await startVAD()
      }
      return
    }
    if (wakeWordActiveRef.current) {
      disableWakeWordMode()
    } else {
      enableWakeWordMode()
    }
  }, [isListening, pauseVAD, setStatus, startVAD, disableWakeWordMode, enableWakeWordMode])

  // ---- Clear ----
  const handleClear = useCallback(() => {
    setMessages([])
    setMeetingEntries([])
    stopPlayback()
    send({ type: 'clear' })
  }, [stopPlayback, send])

  // ---- Manual mic button ----
  const handleMicDown = useCallback(async (e: React.MouseEvent | React.TouchEvent) => {
    e.preventDefault()
    if (wakeWordActiveRef.current && clientStateRef.current === 'sleeping') {
      handleWake()
      send({ type: 'set_state', state: 'awake' })
    }
    if (isSpeakingRef.current || isProcessingRef.current) interrupt()
    if (vadRef.current) pauseVAD()
    await startManual()
  }, [handleWake, interrupt, pauseVAD, startManual, isSpeakingRef, vadRef, send])

  const handleMicUp = useCallback((e: React.MouseEvent | React.TouchEvent) => {
    e.preventDefault()
    stopManual()
  }, [stopManual])

  // ---- Spacebar support ----
  useEffect(() => {
    const onKeyDown = (e: KeyboardEvent) => {
      if (e.code === 'Space' && !e.repeat && !isRecording()) {
        e.preventDefault()
        handleMicDown(e as unknown as React.MouseEvent)
      }
    }
    const onKeyUp = (e: KeyboardEvent) => {
      if (e.code === 'Space' && isRecording()) {
        e.preventDefault()
        stopManual()
      }
    }
    document.addEventListener('keydown', onKeyDown)
    document.addEventListener('keyup', onKeyUp)
    return () => {
      document.removeEventListener('keydown', onKeyDown)
      document.removeEventListener('keyup', onKeyUp)
    }
  }, [handleMicDown, stopManual, isRecording])

  // ---- Speaker verification ----
  const handleEnrollStart = useCallback(() => {
    setEnrollStep(0)
    setShowEnrollModal(true)
    send({ type: 'enroll_start' })
  }, [send])

  const handleEnrollCancel = useCallback(() => {
    send({ type: 'enroll_cancel' })
    setShowEnrollModal(false)
  }, [send])

  const handleVerifyToggle = useCallback(() => {
    const next = !verifyEnabled
    setVerifyEnabled(next)
    send({ type: 'toggle_verify', enabled: next })
  }, [verifyEnabled, send])

  return (
    <div className="flex flex-col items-center min-h-screen" style={{ background: 'var(--bg)', color: 'var(--text)' }}>
      <Toaster position="top-center" theme="dark" />

      <ReconnectBanner
        show={showReconnectBanner}
        onReconnect={() => {
          setShowReconnectBanner(false)
          setStatus('Reconnecting...', '')
          reconnect()
        }}
      />

      <Header
        enrolled={enrolled}
        verifyEnabled={verifyEnabled}
        meetingMode={meetingMode}
        onEnroll={handleEnrollStart}
        onVerifyToggle={handleVerifyToggle}
        onMeetingToggle={() => meetingMode ? exitMeetingMode() : enterMeetingMode()}
      />

      <MeetingBanner show={meetingMode} meetingCommandCapture={meetingCommandCaptureRef} />

      <StatusDisplay text={statusText} statusClass={statusClass} />

      <ChatDisplay
        messages={messages}
        meetingEntries={meetingEntries}
        meetingMode={meetingMode}
      />

      <Controls
        micState={micState}
        wakeWordActive={wakeWordActive}
        wakeWordEnabled={wakeWordEnabled}
        audioLevelRingRef={audioLevelRingRef}
        meetingMode={meetingMode}
        onMicDown={handleMicDown}
        onMicUp={handleMicUp}
        onMicLeave={() => { if (isRecording()) stopManual() }}
        onClear={handleClear}
        onWakeToggle={handleWakeToggle}
        onCommandStart={() => {/* handled in Controls */}}
        onCommandStop={() => {/* handled in Controls */}}
        send={send}
        vadRef={vadRef}
        meetingModeRef={meetingModeRef}
        wakeWordEnabledRef={wakeWordEnabledRef}
        startWakeStream={startWakeStream}
        setStatus={setStatus}
      />

      <EnrollModal
        show={showEnrollModal}
        step={enrollStep}
        onCancel={handleEnrollCancel}
        send={send}
      />
    </div>
  )
}
```

**Step 2: Commit**

```bash
cd /Users/hamiltonchua/.openclaw/workspace/voice-chat
git add frontend/src/App.tsx
git commit -m "feat: add main App component with all state and hook wiring"
```

---

## Task 12: UI Components

**Files:**
- Create: `frontend/src/components/Header.tsx`
- Create: `frontend/src/components/StatusDisplay.tsx`
- Create: `frontend/src/components/ChatDisplay.tsx`
- Create: `frontend/src/components/Controls.tsx`
- Create: `frontend/src/components/MeetingBanner.tsx`
- Create: `frontend/src/components/ReconnectBanner.tsx`
- Create: `frontend/src/components/EnrollModal.tsx`

**Step 1: Create Header.tsx**

```tsx
// frontend/src/components/Header.tsx
interface HeaderProps {
  enrolled: boolean
  verifyEnabled: boolean
  meetingMode: boolean
  onEnroll: () => void
  onVerifyToggle: () => void
  onMeetingToggle: () => void
}

export function Header({ enrolled, verifyEnabled, meetingMode, onEnroll, onVerifyToggle, onMeetingToggle }: HeaderProps) {
  return (
    <header className="py-6 text-center">
      <h1 className="text-2xl font-semibold mb-3">Kismet</h1>
      <div className="flex items-center gap-3 justify-center flex-wrap">
        <button
          className="header-btn"
          onClick={onEnroll}
          title="Enroll your voice for speaker verification"
        >
          🎤 {enrolled ? 'Re-enroll' : 'Enroll Voice'}
        </button>
        {enrolled && (
          <button
            className={`header-btn ${verifyEnabled ? 'active' : ''}`}
            onClick={onVerifyToggle}
            title="Toggle speaker verification"
          >
            {verifyEnabled ? '🔒 Verify: On' : '🔓 Verify: Off'}
          </button>
        )}
        <button
          className={`header-btn ${meetingMode ? 'active' : ''}`}
          onClick={onMeetingToggle}
          title="Toggle meeting companion mode"
        >
          📋 {meetingMode ? 'End Meeting' : 'Meeting'}
        </button>
      </div>
    </header>
  )
}
```

**Step 2: Create StatusDisplay.tsx**

```tsx
// frontend/src/components/StatusDisplay.tsx
interface StatusDisplayProps {
  text: string
  statusClass: 'active' | 'listening' | 'sleeping' | ''
}

export function StatusDisplay({ text, statusClass }: StatusDisplayProps) {
  return (
    <div id="status" className={statusClass} style={{ minHeight: 20 }}>
      {text}
    </div>
  )
}
```

**Step 3: Create MeetingBanner.tsx**

```tsx
// frontend/src/components/MeetingBanner.tsx
import { useRef } from 'react'

interface MeetingBannerProps {
  show: boolean
  meetingCommandCapture: React.RefObject<boolean>
}

export function MeetingBanner({ show, meetingCommandCapture }: MeetingBannerProps) {
  // Show different text depending on command capture state
  const text = meetingCommandCapture.current
    ? '🎤 LISTENING — Speak your command...'
    : '🔴 MEETING MODE — Transcribing • Say "Hey Friday" to ask Kismet'

  if (!show) return null

  return (
    <div className="meeting-banner show">
      {text}
    </div>
  )
}
```

**Step 4: Create ReconnectBanner.tsx**

```tsx
// frontend/src/components/ReconnectBanner.tsx
interface ReconnectBannerProps {
  show: boolean
  onReconnect: () => void
}

export function ReconnectBanner({ show, onReconnect }: ReconnectBannerProps) {
  if (!show) return null
  return (
    <div id="reconnectBanner" className="show" onClick={onReconnect}>
      Connection lost. Click to retry.
    </div>
  )
}
```

**Step 5: Create ChatDisplay.tsx**

```tsx
// frontend/src/components/ChatDisplay.tsx
import { useEffect, useRef } from 'react'
import type { ChatMessage, MeetingEntry } from '../types'

interface ChatDisplayProps {
  messages: ChatMessage[]
  meetingEntries: MeetingEntry[]
  meetingMode: boolean
}

export function ChatDisplay({ messages, meetingEntries, meetingMode }: ChatDisplayProps) {
  const chatRef = useRef<HTMLDivElement>(null)

  useEffect(() => {
    if (chatRef.current) {
      chatRef.current.scrollTop = chatRef.current.scrollHeight
    }
  }, [messages, meetingEntries])

  return (
    <div id="chat" ref={chatRef}>
      {messages.map(msg => (
        <div
          key={msg.id}
          className={[
            'msg',
            msg.role,
            msg.streaming ? 'streaming' : '',
            msg.interrupted ? 'interrupted' : '',
            msg.rejected ? 'rejected' : '',
          ].filter(Boolean).join(' ')}
        >
          {msg.text}
          {msg.meta && <div className="meta">{msg.meta}</div>}
        </div>
      ))}
      {meetingMode && meetingEntries.map(entry => (
        <div key={entry.id} className="meeting-entry">
          <span className="timestamp">{entry.time}</span>
          <span className={`speaker ${entry.speakerType}`}>{entry.speaker}:</span>
          {' '}{entry.text}
        </div>
      ))}
    </div>
  )
}
```

**Step 6: Create Controls.tsx**

```tsx
// frontend/src/components/Controls.tsx
import { useRef, useCallback } from 'react'
import { Trash2, Mic, MicOff } from 'lucide-react'
import { useManualRecording, float32ToBase64 } from '../hooks/useManualRecording'
import { SAMPLE_RATE } from '../constants'

interface ControlsProps {
  micState: 'idle' | 'recording' | 'vad-active' | 'sleeping'
  wakeWordActive: boolean
  wakeWordEnabled: boolean
  audioLevelRingRef: React.RefObject<HTMLDivElement | null>
  meetingMode: boolean
  onMicDown: (e: React.MouseEvent | React.TouchEvent) => void
  onMicUp: (e: React.MouseEvent | React.TouchEvent) => void
  onMicLeave: () => void
  onClear: () => void
  onWakeToggle: () => void
  onCommandStart: () => void
  onCommandStop: () => void
  send: (msg: Record<string, unknown>) => void
  vadRef: React.RefObject<import('@ricky0123/vad-web').MicVAD | null>
  meetingModeRef: React.RefObject<boolean>
  wakeWordEnabledRef: React.RefObject<boolean>
  startWakeStream: () => Promise<void>
  setStatus: (text: string, cls?: 'active' | 'listening' | 'sleeping' | '') => void
}

export function Controls({
  micState, wakeWordActive, wakeWordEnabled, audioLevelRingRef,
  meetingMode, onMicDown, onMicUp, onMicLeave,
  onClear, onWakeToggle, send, vadRef, meetingModeRef,
  wakeWordEnabledRef, startWakeStream, setStatus,
}: ControlsProps) {
  const commandRecordingRef = useRef(false)
  const commandChunksRef = useRef<Float32Array[]>([])
  const commandStreamRef = useRef<MediaStream | null>(null)
  const commandCtxRef = useRef<AudioContext | null>(null)
  const commandProcessorRef = useRef<ScriptProcessorNode | null>(null)

  const micBtnClass = [
    'btn',
    micState === 'recording' ? 'recording' : '',
    micState === 'vad-active' ? 'vad-active' : '',
    micState === 'sleeping' ? 'sleeping' : '',
    micState === 'vad-active' && micState === 'recording' ? 'recording' : '',
  ].filter(Boolean).join(' ')

  const startCommandRecording = useCallback(async () => {
    if (commandRecordingRef.current) return
    commandRecordingRef.current = true
    commandChunksRef.current = []
    if (vadRef.current) vadRef.current.pause()

    const stream = await navigator.mediaDevices.getUserMedia({
      audio: { sampleRate: 16000, channelCount: 1, echoCancellation: true },
    })
    commandStreamRef.current = stream
    const ctx = new AudioContext({ sampleRate: 16000 })
    commandCtxRef.current = ctx
    const source = ctx.createMediaStreamSource(stream)
    const processor = ctx.createScriptProcessor(4096, 1, 1)
    processor.onaudioprocess = (e) => {
      commandChunksRef.current.push(new Float32Array(e.inputBuffer.getChannelData(0)))
    }
    source.connect(processor)
    processor.connect(ctx.destination)
    commandProcessorRef.current = processor
    setStatus('Recording command...', 'listening')
  }, [vadRef, setStatus])

  const stopCommandRecording = useCallback(() => {
    if (!commandRecordingRef.current) return
    commandRecordingRef.current = false

    commandProcessorRef.current?.disconnect()
    commandCtxRef.current?.close()
    commandStreamRef.current?.getTracks().forEach(t => t.stop())
    commandStreamRef.current = null

    const chunks = commandChunksRef.current
    commandChunksRef.current = []
    const total = chunks.reduce((n, c) => n + c.length, 0)
    const merged = new Float32Array(total)
    let offset = 0
    for (const chunk of chunks) { merged.set(chunk, offset); offset += chunk.length }

    if (vadRef.current && meetingModeRef.current) vadRef.current.start()

    if (merged.length < SAMPLE_RATE * 0.3) {
      setStatus('Too short — hold longer', 'active')
      return
    }
    send({ type: 'meeting_command', data: float32ToBase64(merged) })
    setStatus('Processing command...', 'active')
  }, [vadRef, meetingModeRef, send, setStatus])

  const wakeToggleSvg = wakeWordEnabled ? (
    // Mic with slash (wake word available)
    <svg width="20" height="20" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2" strokeLinecap="round" strokeLinejoin="round">
      <path d="M12 2a3 3 0 0 0-3 3v4a3 3 0 0 0 6 0V5a3 3 0 0 0-3-3z"/>
      <path d="M19 10v1a7 7 0 0 1-14 0v-1"/>
      <path d="M12 19v2"/><path d="M8 21h8"/>
      {!wakeWordActive && <path d="M3 3l18 18" strokeWidth="2"/>}
    </svg>
  ) : (
    // Mic without slash (hands-free VAD)
    <svg width="20" height="20" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2" strokeLinecap="round" strokeLinejoin="round">
      <path d="M12 2a3 3 0 0 0-3 3v4a3 3 0 0 0 6 0V5a3 3 0 0 0-3-3z"/>
      <path d="M19 10v1a7 7 0 0 1-14 0v-1"/>
      <path d="M12 19v2"/><path d="M8 21h8"/>
    </svg>
  )

  return (
    <>
      {/* Meeting command button */}
      {meetingMode && (
        <button
          id="commandBtn"
          style={{ display: 'block' }}
          title="Hold to give Kismet a command"
          onMouseDown={(e) => { e.preventDefault(); startCommandRecording() }}
          onMouseUp={(e) => { e.preventDefault(); stopCommandRecording() }}
          onMouseLeave={() => { if (commandRecordingRef.current) stopCommandRecording() }}
          onTouchStart={(e) => { e.preventDefault(); startCommandRecording() }}
          onTouchEnd={(e) => { e.preventDefault(); stopCommandRecording() }}
        >
          💬
        </button>
      )}

      <div id="controls">
        {/* Clear button */}
        <button className="btn" id="clearBtn" title="Clear conversation" onClick={onClear}>
          <svg width="20" height="20" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2" strokeLinecap="round" strokeLinejoin="round">
            <polyline points="3 6 5 6 21 6"/>
            <path d="M19 6l-1 14a2 2 0 0 1-2 2H8a2 2 0 0 1-2-2L5 6"/>
            <path d="M10 11v6"/><path d="M14 11v6"/>
            <path d="M9 6V4a1 1 0 0 1 1-1h4a1 1 0 0 1 1 1v2"/>
          </svg>
        </button>

        {/* Mic button */}
        <button
          className={micBtnClass}
          id="micBtn"
          title="Hold to talk, or say the wake word"
          onMouseDown={onMicDown}
          onMouseUp={onMicUp}
          onMouseLeave={onMicLeave}
          onTouchStart={onMicDown}
          onTouchEnd={onMicUp}
        >
          <div id="audioLevel" ref={audioLevelRingRef} />
          <svg id="micIcon" width="32" height="32" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2" strokeLinecap="round" strokeLinejoin="round">
            <path d="M12 1a3 3 0 0 0-3 3v8a3 3 0 0 0 6 0V4a3 3 0 0 0-3-3z"/>
            <path d="M19 10v2a7 7 0 0 1-14 0v-2"/>
            <line x1="12" y1="19" x2="12" y2="23"/>
            <line x1="8" y1="23" x2="16" y2="23"/>
          </svg>
        </button>

        {/* Wake toggle */}
        <button
          className={`btn ${wakeWordActive ? 'active' : ''}`}
          id="wakeToggle"
          title={wakeWordEnabled ? 'Toggle wake word mode' : 'Toggle hands-free listening'}
          onClick={onWakeToggle}
        >
          {wakeToggleSvg}
        </button>
      </div>
    </>
  )
}
```

**Step 7: Create EnrollModal.tsx**

```tsx
// frontend/src/components/EnrollModal.tsx
import { useState, useRef, useCallback } from 'react'
import { ENROLL_PROMPTS, ENROLL_REQUIRED, SAMPLE_RATE } from '../constants'
import { float32ToBase64 } from '../hooks/useManualRecording'

interface EnrollModalProps {
  show: boolean
  step: number
  onCancel: () => void
  send: (msg: Record<string, unknown>) => void
}

export function EnrollModal({ show, step, onCancel, send }: EnrollModalProps) {
  const [recording, setRecording] = useState(false)
  const [btnText, setBtnText] = useState('🎤 Hold to Record')
  const streamRef = useRef<MediaStream | null>(null)
  const ctxRef = useRef<AudioContext | null>(null)
  const procRef = useRef<ScriptProcessorNode | null>(null)
  const sourceRef = useRef<MediaStreamAudioSourceNode | null>(null)
  const chunksRef = useRef<Float32Array[]>([])

  const startRecording = useCallback(async () => {
    if (!streamRef.current) {
      streamRef.current = await navigator.mediaDevices.getUserMedia({
        audio: { sampleRate: SAMPLE_RATE, channelCount: 1, echoCancellation: true, noiseSuppression: true },
      })
    }
    const ctx = new AudioContext({ sampleRate: SAMPLE_RATE })
    ctxRef.current = ctx
    const source = ctx.createMediaStreamSource(streamRef.current)
    sourceRef.current = source
    const proc = ctx.createScriptProcessor(4096, 1, 1)
    procRef.current = proc
    chunksRef.current = []
    proc.onaudioprocess = (e) => {
      chunksRef.current.push(new Float32Array(e.inputBuffer.getChannelData(0)))
    }
    source.connect(proc)
    proc.connect(ctx.destination)
    setRecording(true)
    setBtnText('⏹ Recording...')
  }, [])

  const stopRecording = useCallback(() => {
    if (!recording) return
    setRecording(false)
    procRef.current?.disconnect()
    sourceRef.current?.disconnect()
    ctxRef.current?.close()

    const chunks = chunksRef.current
    chunksRef.current = []
    const total = chunks.reduce((n, c) => n + c.length, 0)
    if (total < SAMPLE_RATE * 0.5) {
      setBtnText('🎤 Too short, try again')
      return
    }
    const merged = new Float32Array(total)
    let off = 0
    for (const c of chunks) { merged.set(c, off); off += c.length }
    send({ type: 'enroll_sample', data: float32ToBase64(merged) })
    setBtnText('🎤 Hold to Record')
  }, [recording, send])

  if (!show) return null

  const isDone = step >= ENROLL_REQUIRED
  const currentPrompt = isDone ? 'Processing enrollment...' : ENROLL_PROMPTS[step]
  const infoText = isDone ? '' : `Sample ${step + 1} of ${ENROLL_REQUIRED}`

  return (
    <div className="modal-overlay show">
      <div className="modal">
        <h2 id="enrollTitle">{isDone ? '✅ Enrolled!' : 'Voice Enrollment'}</h2>
        <p>Read each sentence aloud to enroll your voice. This lets Kismet verify it's you speaking.</p>
        <div className="prompt-text">"{currentPrompt}"</div>
        <div className="progress">
          {Array.from({ length: ENROLL_REQUIRED }, (_, i) => (
            <div
              key={i}
              className={`dot ${i < step ? 'done' : i === step ? 'active' : ''}`}
            />
          ))}
        </div>
        <p style={{ fontSize: '0.8rem' }}>{infoText}</p>
        <div>
          {!isDone && (
            <button
              className="modal-btn"
              onMouseDown={(e) => { e.preventDefault(); startRecording() }}
              onMouseUp={(e) => { e.preventDefault(); stopRecording() }}
              onMouseLeave={() => { if (recording) stopRecording() }}
              onTouchStart={(e) => { e.preventDefault(); startRecording() }}
              onTouchEnd={(e) => { e.preventDefault(); stopRecording() }}
            >
              {btnText}
            </button>
          )}
          <button className="modal-btn secondary" onClick={onCancel}>
            {isDone ? 'Close' : 'Cancel'}
          </button>
        </div>
      </div>
    </div>
  )
}
```

**Step 8: Commit all components**

```bash
cd /Users/hamiltonchua/.openclaw/workspace/voice-chat
git add frontend/src/components/
git commit -m "feat: add all UI components (Header, Chat, Controls, Modal, Banners)"
```

---

## Task 13: Global CSS — Port Styles from index.html

**Files:**
- Modify: `frontend/src/index.css`

Port ALL CSS from the original `index.html` `<style>` block (lines 7–380) into `frontend/src/index.css`. This includes: status colors, chat messages, controls, buttons, mic animations, meeting styles, enrollment modal, error toast, reconnect banner.

**Step 1: Append all vanilla CSS to index.css**

The CSS should be appended after the Tailwind `@import` and CSS variable declarations already in place. Copy everything from `#status` onwards through the media query at the end of the `<style>` block in index.html.

Key sections to port:
- `#status`, `#status.active`, `#status.listening`, `#status.sleeping`
- `#chat`, `.msg`, `.msg.user`, `.msg.assistant`, `.msg.streaming`, `.msg.interrupted`, `.msg.rejected`
- `@keyframes blink`, `.msg .meta`
- `#controls`, `.btn`, `.btn:hover`, `.btn:active`
- `#micBtn`, `#audioLevel`, `#audioLevel.active`
- `.error-toast`, `.error-toast.show`
- `#reconnectBanner`, `#reconnectBanner.show`
- `#micBtn.recording`, `#micBtn.vad-active`, `#micBtn.sleeping`
- `@keyframes pulse`, `@keyframes pulse-green`, `@keyframes pulse-sleep`
- `#clearBtn`, `#wakeToggle`, `#wakeToggle.active`
- `.header-btn`, `.header-btn:hover`, `.header-btn.active`
- `.verify-badge`, `.verify-badge.pass`, `.verify-badge.fail`
- `.meeting-banner`, `.meeting-banner.show`, `.meeting-entry`, `.speaker` classes
- `#commandBtn`, `#commandBtn.recording`
- `.modal-overlay`, `.modal-overlay.show`, `.modal`, `.modal-btn`, `.modal-btn.secondary`
- `.progress .dot`, `.dot.done`, `.dot.active`
- `@media (max-width: 480px)`

**Step 2: Commit**

```bash
cd /Users/hamiltonchua/.openclaw/workspace/voice-chat
git add frontend/src/index.css
git commit -m "feat: port all CSS from index.html into index.css"
```

---

## Task 14: Update main.tsx and Clean Up Boilerplate

**Files:**
- Modify: `frontend/src/main.tsx`
- Modify: `frontend/index.html` (title)
- Delete: any unused boilerplate (default Vite counter, assets, etc.)

**Step 1: Update main.tsx**

```tsx
// frontend/src/main.tsx
import { StrictMode } from 'react'
import { createRoot } from 'react-dom/client'
import './index.css'
import App from './App.tsx'

createRoot(document.getElementById('root')!).render(
  <StrictMode>
    <App />
  </StrictMode>,
)
```

**Step 2: Update frontend/index.html title**

Change `<title>Vite + React + TS</title>` to `<title>Kismet</title>`.

**Step 3: Remove boilerplate files**

```bash
rm -f /Users/hamiltonchua/.openclaw/workspace/voice-chat/frontend/src/assets/react.svg
rm -f /Users/hamiltonchua/.openclaw/workspace/voice-chat/public/vite.svg
```

**Step 4: Build and verify no errors**

```bash
cd /Users/hamiltonchua/.openclaw/workspace/voice-chat/frontend
npm run build 2>&1
```
Expected: Build succeeds with `frontend/dist/index.html` and `frontend/dist/assets/` created.

**Step 5: Commit**

```bash
cd /Users/hamiltonchua/.openclaw/workspace/voice-chat
git add frontend/
git commit -m "feat: finalize entry point and clean up boilerplate"
```

---

## Task 15: Update server.py to Serve React Build

**Files:**
- Modify: `server.py` (lines ~24-25 imports, lines ~537-540 route)

**Step 1: Add StaticFiles import to server.py**

The existing imports include `from fastapi.responses import HTMLResponse`. We need to also add:
```python
from fastapi.staticfiles import StaticFiles
```

**Step 2: Replace the `@app.get("/")` route and add static file mounting**

Find and replace the current route:
```python
@app.get("/")
async def index():
    html_path = Path(__file__).parent / "index.html"
    return HTMLResponse(html_path.read_text())
```

Replace with:
```python
# Serve React build output (frontend/dist/)
_DIST_DIR = Path(__file__).parent / "frontend" / "dist"
_DIST_INDEX = _DIST_DIR / "index.html"
_FALLBACK_HTML = Path(__file__).parent / "index.html"

if _DIST_DIR.exists() and (_DIST_DIR / "assets").exists():
    app.mount("/assets", StaticFiles(directory=str(_DIST_DIR / "assets")), name="assets")

@app.get("/")
async def index():
    if _DIST_INDEX.exists():
        return HTMLResponse(_DIST_INDEX.read_text())
    return HTMLResponse(_FALLBACK_HTML.read_text())
```

**Step 3: Verify server.py is syntactically valid**

```bash
cd /Users/hamiltonchua/.openclaw/workspace/voice-chat
python -c "import ast; ast.parse(open('server.py').read()); print('OK')"
```
Expected: `OK`

**Step 4: Commit**

```bash
cd /Users/hamiltonchua/.openclaw/workspace/voice-chat
git add server.py
git commit -m "feat: serve React build from frontend/dist/ with fallback to index.html"
```

---

## Task 16: End-to-End Verification

**Step 1: Build the frontend**

```bash
cd /Users/hamiltonchua/.openclaw/workspace/voice-chat/frontend
npm run build
```
Expected: `dist/index.html` and `dist/assets/` exist.

**Step 2: Verify build output structure**

```bash
ls /Users/hamiltonchua/.openclaw/workspace/voice-chat/frontend/dist/
ls /Users/hamiltonchua/.openclaw/workspace/voice-chat/frontend/dist/assets/
```
Expected: `index.html`, and `assets/` containing `.js` and `.css` files.

**Step 3: Check TypeScript has no errors**

```bash
cd /Users/hamiltonchua/.openclaw/workspace/voice-chat/frontend
npx tsc --noEmit 2>&1
```
Expected: No errors (or only minor type warnings).

**Step 4: Verify server.py syntax one more time**

```bash
python -c "import ast; ast.parse(open('/Users/hamiltonchua/.openclaw/workspace/voice-chat/server.py').read()); print('OK')"
```

**Step 5: Final commit with summary**

```bash
cd /Users/hamiltonchua/.openclaw/workspace/voice-chat
git add -A
git commit -m "feat: Phase 9 complete — Vite+React+TS+Tailwind+shadcn/ui frontend

- Scaffold: Vite 6, React 19, TypeScript, Tailwind CSS v4, shadcn/ui (new-york dark)
- Hooks: useWebSocket, useAudio, useWakeWordStream, useVAD, useManualRecording, useAudioLevel
- Components: Header, ChatDisplay, Controls, EnrollModal, MeetingBanner, ReconnectBanner, StatusDisplay
- All original functionality preserved: wake word, VAD, manual PTT, meeting mode, speaker verify, reconnect
- server.py updated to serve frontend/dist/ with fallback to index.html"
```

---

## Notes for Implementer

### VAD library types
`@ricky0123/vad-web` may need type augmentation. If TypeScript complains about missing types, add:
```typescript
// frontend/src/vad.d.ts
declare module '@ricky0123/vad-web' {
  export class MicVAD {
    static new(options: Record<string, unknown>): Promise<MicVAD>
    start(): void
    pause(): void
    listening: boolean
  }
}
```

### shadcn/ui CSS conflict
After `npx shadcn@latest init`, it may modify `index.css`. Ensure the original app CSS variables (`--bg`, `--surface`, etc.) are preserved alongside shadcn's `--background`, `--foreground` etc. The shadcn components (Button, Dialog) are available but the UI mostly uses vanilla CSS classes for consistency with the existing design.

### Vite WebSocket proxy
The proxy config (`/ws` → `ws://localhost:8765`) is for dev mode only. In production, FastAPI handles `/ws` directly.

### MeetingBanner re-render timing
`meetingCommandCapture` is a ref (not state) to avoid re-renders during audio processing. MeetingBanner reads the ref at render time — it may show stale text briefly but is functionally correct. This matches the original behavior.
