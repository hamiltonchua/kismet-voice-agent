# Phase 9 UI Overhaul — Two-Panel Voice Interface

> **For Claude:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task.

**Goal:** Redesign Kismet's visual layout into a two-panel interface (left Control Center + right Conversation) without breaking any existing WebSocket/audio/VAD/wake-word functionality.

**Architecture:** CSS grid two-panel layout with a fixed 320px left panel (Control Center with mic, status, settings drawer) and a full-height scrollable right panel (conversation transcript). All hook logic, types, and server communication remain untouched.

**Tech Stack:** React 19, TypeScript, Tailwind CSS v4, shadcn/ui (Sheet component needed), lucide-react icons, sonner toasts.

---

## Pre-flight Check

Before starting, confirm you're in the right worktree branch:
```bash
git branch  # should be phase9-ui-overhaul
cd /Users/hamiltonchua/.openclaw/workspace/voice-chat/frontend
```

---

### Task 1: Add shadcn Sheet component

**Files:**
- Create: `frontend/src/components/ui/sheet.tsx`

The Sheet component is needed for the SettingsDrawer. No sheet.tsx exists yet — add it.

**Step 1: Add the Sheet component**

Run from `frontend/`:
```bash
npx shadcn@latest add sheet --yes
```

If that fails (network/interactive issues), create manually:

`frontend/src/components/ui/sheet.tsx`:
```tsx
import * as React from "react"
import * as SheetPrimitive from "@radix-ui/react-dialog"
import { cn } from "@/lib/utils"

const Sheet = SheetPrimitive.Root
const SheetTrigger = SheetPrimitive.Trigger
const SheetClose = SheetPrimitive.Close
const SheetPortal = SheetPrimitive.Portal

const SheetOverlay = React.forwardRef<
  React.ElementRef<typeof SheetPrimitive.Overlay>,
  React.ComponentPropsWithoutRef<typeof SheetPrimitive.Overlay>
>(({ className, ...props }, ref) => (
  <SheetPrimitive.Overlay
    className={cn(
      "fixed inset-0 z-50 bg-black/80 data-[state=open]:animate-in data-[state=closed]:animate-out data-[state=closed]:fade-out-0 data-[state=open]:fade-in-0",
      className
    )}
    {...props}
    ref={ref}
  />
))
SheetOverlay.displayName = SheetPrimitive.Overlay.displayName

interface SheetContentProps
  extends React.ComponentPropsWithoutRef<typeof SheetPrimitive.Content> {
  side?: "top" | "right" | "bottom" | "left"
}

const SheetContent = React.forwardRef<
  React.ElementRef<typeof SheetPrimitive.Content>,
  SheetContentProps
>(({ side = "right", className, children, ...props }, ref) => (
  <SheetPortal>
    <SheetOverlay />
    <SheetPrimitive.Content
      ref={ref}
      className={cn(
        "fixed z-50 gap-4 bg-[var(--surface)] p-6 shadow-lg transition ease-in-out data-[state=open]:animate-in data-[state=closed]:animate-out data-[state=closed]:duration-300 data-[state=open]:duration-500",
        side === "left" && "inset-y-0 left-0 h-full w-3/4 data-[state=closed]:slide-out-to-left data-[state=open]:slide-in-from-left sm:max-w-sm",
        side === "right" && "inset-y-0 right-0 h-full w-3/4 data-[state=closed]:slide-out-to-right data-[state=open]:slide-in-from-right sm:max-w-sm",
        className
      )}
      {...props}
    >
      {children}
    </SheetPrimitive.Content>
  </SheetPortal>
))
SheetContent.displayName = SheetPrimitive.Content.displayName

const SheetHeader = ({ className, ...props }: React.HTMLAttributes<HTMLDivElement>) => (
  <div className={cn("flex flex-col space-y-2", className)} {...props} />
)
SheetHeader.displayName = "SheetHeader"

const SheetTitle = React.forwardRef<
  React.ElementRef<typeof SheetPrimitive.Title>,
  React.ComponentPropsWithoutRef<typeof SheetPrimitive.Title>
>(({ className, ...props }, ref) => (
  <SheetPrimitive.Title
    ref={ref}
    className={cn("text-lg font-semibold text-[var(--text)]", className)}
    {...props}
  />
))
SheetTitle.displayName = SheetPrimitive.Title.displayName

export {
  Sheet,
  SheetPortal,
  SheetOverlay,
  SheetTrigger,
  SheetClose,
  SheetContent,
  SheetHeader,
  SheetTitle,
}
```

**Step 2: Check if @radix-ui/react-dialog is available (Sheet uses it)**

The `dialog.tsx` shadcn component already uses radix-ui, so it's available through the `radix-ui` package. Verify:
```bash
cat frontend/src/components/ui/dialog.tsx | head -5
```

If the import path is `@radix-ui/react-dialog`, use that. If it's `radix-ui/react-dialog`, adjust sheet.tsx accordingly.

**Step 3: Verify sheet.tsx compiles**
```bash
cd frontend && npx tsc --noEmit 2>&1 | head -30
```

**Step 4: Commit**
```bash
git add frontend/src/components/ui/sheet.tsx
git commit -m "feat: add Sheet UI component for settings drawer"
```

---

### Task 2: Update index.css — animations + new layout variables

**Files:**
- Modify: `frontend/src/index.css`

The CSS needs new keyframe animations compatible with the new layout, plus cleanup of old layout-specific IDs that will be replaced by Tailwind classes.

**Step 1: Add new keyframes and keep existing ones**

At the end of `frontend/src/index.css`, append:

```css
/* =====================================================
   Phase 9 — Two-panel layout additions
   ===================================================== */

/* Streaming cursor blink */
@keyframes cursor-blink {
  0%, 50% { opacity: 1; }
  51%, 100% { opacity: 0; }
}

/* Meeting mode left accent bar pulse */
@keyframes meeting-pulse {
  0%, 100% { opacity: 0.6; }
  50% { opacity: 1; }
}

/* Audio level ring for new panel layout */
.audio-ring {
  position: absolute;
  inset: -6px;
  border-radius: 50%;
  border: 2px solid transparent;
  pointer-events: none;
  transition: border-color 0.05s, box-shadow 0.05s;
}
.audio-ring.active {
  border-color: rgba(0, 217, 126, 0.6);
}

/* Mic button states — new class names for left panel */
.mic-btn {
  width: 80px;
  height: 80px;
  border-radius: 50%;
  border: none;
  cursor: pointer;
  display: flex;
  align-items: center;
  justify-content: center;
  background: var(--accent);
  box-shadow: 0 0 0 0 rgba(233, 69, 96, 0.4);
  position: relative;
  transition: transform 0.15s, box-shadow 0.15s, background 0.3s;
}
.mic-btn:hover { transform: scale(1.05); }
.mic-btn:active { transform: scale(0.95); }

.mic-btn.recording {
  animation: pulse 1.5s infinite;
  background: #ff2244;
}
.mic-btn.vad-active {
  background: var(--green);
}
.mic-btn.vad-active.recording {
  animation: pulse-green 1.5s infinite;
  background: #00ff99;
}
.mic-btn.sleeping {
  background: var(--purple);
  animation: pulse-sleep 3s infinite;
}

/* Secondary control buttons */
.ctrl-btn {
  width: 48px;
  height: 48px;
  border-radius: 50%;
  border: none;
  cursor: pointer;
  display: flex;
  align-items: center;
  justify-content: center;
  background: var(--surface2);
  color: var(--text2);
  transition: transform 0.15s, background 0.2s;
}
.ctrl-btn:hover { transform: scale(1.05); background: var(--accent2); color: var(--text); }
.ctrl-btn:active { transform: scale(0.95); }
.ctrl-btn.wake-active { background: var(--purple); color: white; }
.ctrl-btn.meeting-active { background: var(--accent); color: white; }

/* Message row style (new, replaces bubble style) */
.msg-row {
  display: flex;
  flex-direction: column;
  padding: 10px 16px;
  border-left: 3px solid transparent;
  line-height: 1.5;
  font-size: 0.95rem;
  transition: border-color 0.2s;
}
.msg-row.assistant { border-left-color: var(--accent); }
.msg-row.user { border-left-color: #0f3460; }
.msg-row.system { border-left-color: rgba(255,255,255,0.1); }
.msg-row .role-label {
  font-size: 0.7rem;
  color: var(--text2);
  text-transform: uppercase;
  letter-spacing: 0.05em;
  margin-bottom: 3px;
}
.msg-row .msg-text { color: var(--text); }
.msg-row.streaming { border-left-color: var(--accent); }
.msg-row.streaming .msg-text::after {
  content: '▋';
  animation: cursor-blink 1s infinite;
  color: var(--accent);
}
.msg-row.interrupted { opacity: 0.5; font-style: italic; }
.msg-row.rejected { opacity: 0.4; border-left-color: var(--accent); }
.msg-row .msg-meta {
  font-size: 0.7rem;
  color: var(--text2);
  margin-top: 4px;
}

/* Meeting entry row */
.meeting-row {
  padding: 4px 16px;
  font-size: 0.85rem;
  line-height: 1.4;
  border-bottom: 1px solid rgba(255,255,255,0.04);
  display: flex;
  gap: 6px;
  align-items: baseline;
}
.meeting-row .meeting-time { color: var(--text2); font-size: 0.7rem; }
.meeting-row .meeting-speaker { font-weight: 600; }
.meeting-row .meeting-speaker.ham { color: var(--green); }
.meeting-row .meeting-speaker.other { color: #0f3460; }
.meeting-row .meeting-speaker.kismet { color: var(--accent); }

/* Connection dot */
.conn-dot {
  width: 8px;
  height: 8px;
  border-radius: 50%;
  display: inline-block;
  transition: background 0.3s;
}
.conn-dot.connected { background: var(--green); }
.conn-dot.disconnected { background: var(--accent); }
.conn-dot.sleeping { background: var(--purple); }
.conn-dot.connecting { background: var(--text2); animation: pulse-sleep 2s infinite; }

/* Status text colors (reuse existing, now inline) */
.status-active { color: var(--green); }
.status-listening { color: var(--accent); }
.status-sleeping { color: var(--purple); }
.status-default { color: var(--text2); }

/* Mobile bottom dock */
@media (max-width: 767px) {
  .mobile-dock {
    position: fixed;
    bottom: 0;
    left: 0;
    right: 0;
    background: linear-gradient(transparent, var(--bg) 20%);
    padding: 16px;
    display: flex;
    justify-content: center;
    align-items: center;
    gap: 16px;
    z-index: 40;
  }
}
```

**Step 2: Verify CSS compiles with build**
```bash
cd /Users/hamiltonchua/.openclaw/workspace/voice-chat/frontend
npm run build 2>&1 | tail -20
```

**Step 3: Commit**
```bash
git add frontend/src/index.css
git commit -m "feat: add phase9 CSS classes and layout styles to index.css"
```

---

### Task 3: Create SettingsDrawer component

**Files:**
- Create: `frontend/src/components/SettingsDrawer.tsx`

This replaces the enrollment + verify buttons from Header, putting them in a slide-in drawer triggered by a gear icon in the left panel bottom.

**Step 1: Create SettingsDrawer.tsx**

```tsx
// frontend/src/components/SettingsDrawer.tsx
import { Settings } from 'lucide-react'
import { Sheet, SheetContent, SheetHeader, SheetTitle, SheetTrigger } from './ui/sheet'
import { ENROLL_REQUIRED } from '../constants'

interface SettingsDrawerProps {
  enrolled: boolean
  verifyEnabled: boolean
  onEnroll: () => void
  onVerifyToggle: () => void
}

export function SettingsDrawer({ enrolled, verifyEnabled, onEnroll, onVerifyToggle }: SettingsDrawerProps) {
  return (
    <Sheet>
      <SheetTrigger asChild>
        <button
          className="ctrl-btn"
          title="Settings"
        >
          <Settings size={20} />
        </button>
      </SheetTrigger>
      <SheetContent side="left">
        <SheetHeader>
          <SheetTitle>Settings</SheetTitle>
        </SheetHeader>

        <div style={{ marginTop: 24, display: 'flex', flexDirection: 'column', gap: 20 }}>
          {/* Speaker Verification */}
          <div>
            <p style={{ fontSize: '0.75rem', color: 'var(--text2)', textTransform: 'uppercase', letterSpacing: '0.05em', marginBottom: 12 }}>
              Speaker Verification
            </p>

            <div style={{ display: 'flex', flexDirection: 'column', gap: 10 }}>
              <button
                className="ctrl-btn"
                onClick={onEnroll}
                title="Enroll your voice for speaker verification"
                style={{ width: '100%', borderRadius: 8, height: 40, fontSize: '0.85rem', color: 'var(--text2)', background: 'var(--surface2)', border: 'none', cursor: 'pointer', padding: '0 12px', display: 'flex', alignItems: 'center', gap: 8 }}
              >
                <span>🎤</span>
                <span>{enrolled ? 'Re-enroll Voice' : 'Enroll Voice'}</span>
              </button>

              {enrolled && (
                <div style={{ display: 'flex', alignItems: 'center', justifyContent: 'space-between', padding: '8px 12px', background: 'var(--surface2)', borderRadius: 8 }}>
                  <span style={{ fontSize: '0.85rem', color: 'var(--text2)' }}>Verify Speaker</span>
                  <button
                    onClick={onVerifyToggle}
                    style={{
                      width: 40,
                      height: 22,
                      borderRadius: 11,
                      border: 'none',
                      cursor: 'pointer',
                      background: verifyEnabled ? 'var(--green)' : 'var(--surface)',
                      position: 'relative',
                      transition: 'background 0.2s',
                    }}
                    title="Toggle speaker verification"
                  >
                    <span style={{
                      position: 'absolute',
                      top: 2,
                      left: verifyEnabled ? 20 : 2,
                      width: 18,
                      height: 18,
                      borderRadius: '50%',
                      background: 'white',
                      transition: 'left 0.2s',
                    }} />
                  </button>
                </div>
              )}
            </div>

            <p style={{ fontSize: '0.7rem', color: 'var(--text2)', marginTop: 8 }}>
              {enrolled
                ? `Enrolled (${ENROLL_REQUIRED} samples). Verification is ${verifyEnabled ? 'on' : 'off'}.`
                : `Not enrolled. Record ${ENROLL_REQUIRED} voice samples to enable speaker verification.`}
            </p>
          </div>
        </div>
      </SheetContent>
    </Sheet>
  )
}
```

**Step 2: Verify types compile**
```bash
cd /Users/hamiltonchua/.openclaw/workspace/voice-chat/frontend
npx tsc --noEmit 2>&1 | head -30
```

**Step 3: Commit**
```bash
git add frontend/src/components/SettingsDrawer.tsx
git commit -m "feat: add SettingsDrawer component with enrollment and verify controls"
```

---

### Task 4: Rewrite Header.tsx — title + connection status only

**Files:**
- Modify: `frontend/src/components/Header.tsx`

The new Header is just the app title + a connection status dot + status text. Buttons move to the left panel. The Header is only shown on mobile (top bar). On desktop, it's hidden in favor of the left panel title.

**Step 1: Rewrite Header.tsx**

Full replacement:

```tsx
// frontend/src/components/Header.tsx
interface HeaderProps {
  statusText: string
  statusClass: 'active' | 'listening' | 'sleeping' | ''
  connectionDot: 'connected' | 'disconnected' | 'sleeping' | 'connecting'
  meetingMode: boolean
  enrolled: boolean
  verifyEnabled: boolean
  onEnroll: () => void
  onVerifyToggle: () => void
  onMeetingToggle: () => void
}

export function Header({ statusText, statusClass, connectionDot, meetingMode, enrolled, verifyEnabled, onEnroll, onVerifyToggle, onMeetingToggle }: HeaderProps) {
  // Mobile-only top bar — hidden on desktop (md+)
  return (
    <header
      className="md:hidden"
      style={{
        display: 'flex',
        alignItems: 'center',
        justifyContent: 'space-between',
        padding: '12px 16px',
        background: 'var(--surface)',
        borderBottom: '1px solid rgba(255,255,255,0.08)',
        position: 'sticky',
        top: 0,
        zIndex: 30,
      }}
    >
      <h1 style={{ fontSize: '1.1rem', fontWeight: 600 }}>Kismet</h1>
      <div style={{ display: 'flex', alignItems: 'center', gap: 8 }}>
        <span className={`conn-dot ${connectionDot}`} />
        <span
          className={`status-${statusClass || 'default'}`}
          style={{ fontSize: '0.8rem' }}
        >
          {statusText}
        </span>
      </div>
      <div style={{ display: 'flex', gap: 8 }}>
        <button
          style={{ background: 'none', border: 'none', cursor: 'pointer', color: 'var(--text2)', fontSize: '0.75rem', padding: '4px 8px', borderRadius: 6, background: 'var(--surface2)' } as React.CSSProperties}
          onClick={onEnroll}
        >
          {enrolled ? 'Re-enroll' : 'Enroll'}
        </button>
        {enrolled && (
          <button
            style={{ background: verifyEnabled ? 'var(--green)' : 'var(--surface2)', border: 'none', cursor: 'pointer', color: verifyEnabled ? 'var(--bg)' : 'var(--text2)', fontSize: '0.75rem', padding: '4px 8px', borderRadius: 6 } as React.CSSProperties}
            onClick={onVerifyToggle}
          >
            {verifyEnabled ? 'Verify: On' : 'Verify: Off'}
          </button>
        )}
        <button
          style={{ background: meetingMode ? 'var(--accent)' : 'var(--surface2)', border: 'none', cursor: 'pointer', color: meetingMode ? 'white' : 'var(--text2)', fontSize: '0.75rem', padding: '4px 8px', borderRadius: 6 } as React.CSSProperties}
          onClick={onMeetingToggle}
        >
          {meetingMode ? 'End Meeting' : 'Meeting'}
        </button>
      </div>
    </header>
  )
}
```

**Step 2: Run type check**
```bash
cd /Users/hamiltonchua/.openclaw/workspace/voice-chat/frontend
npx tsc --noEmit 2>&1 | head -30
```

**Step 3: Commit**
```bash
git add frontend/src/components/Header.tsx
git commit -m "feat: simplify Header to mobile-only top bar with status dot"
```

---

### Task 5: Rewrite ReconnectBanner.tsx — full-width top overlay

**Files:**
- Modify: `frontend/src/components/ReconnectBanner.tsx`

**Step 1: Rewrite ReconnectBanner.tsx**

```tsx
// frontend/src/components/ReconnectBanner.tsx
interface ReconnectBannerProps {
  show: boolean
  onReconnect: () => void
}

export function ReconnectBanner({ show, onReconnect }: ReconnectBannerProps) {
  if (!show) return null
  return (
    <div
      onClick={onReconnect}
      style={{
        position: 'fixed',
        top: 0,
        left: 0,
        right: 0,
        background: 'var(--accent)',
        color: 'white',
        textAlign: 'center',
        padding: '10px',
        fontSize: '0.85rem',
        zIndex: 300,
        cursor: 'pointer',
        fontWeight: 600,
      }}
    >
      Connection lost. Click to retry.
    </div>
  )
}
```

**Step 2: Commit**
```bash
git add frontend/src/components/ReconnectBanner.tsx
git commit -m "feat: update ReconnectBanner to full-width fixed overlay"
```

---

### Task 6: Rewrite MeetingBanner.tsx — positioned in right panel

**Files:**
- Modify: `frontend/src/components/MeetingBanner.tsx`

**Step 1: Rewrite MeetingBanner.tsx**

```tsx
// frontend/src/components/MeetingBanner.tsx
interface MeetingBannerProps {
  show: boolean
  meetingCommandCapture: React.RefObject<boolean>
}

export function MeetingBanner({ show, meetingCommandCapture }: MeetingBannerProps) {
  if (!show) return null

  const text = meetingCommandCapture.current
    ? 'LISTENING — Speak your command...'
    : 'MEETING MODE — Transcribing • Say "Hey Friday" to ask Kismet'

  return (
    <div
      style={{
        background: 'rgba(233,69,96,0.12)',
        borderBottom: '1px solid rgba(233,69,96,0.2)',
        color: 'var(--accent)',
        padding: '8px 16px',
        fontSize: '0.78rem',
        fontWeight: 600,
        letterSpacing: '0.04em',
        animation: 'meeting-pulse 2s infinite',
      }}
    >
      {meetingCommandCapture.current ? '🎤 ' : '🔴 '}{text}
    </div>
  )
}
```

**Step 2: Commit**
```bash
git add frontend/src/components/MeetingBanner.tsx
git commit -m "feat: update MeetingBanner style for right panel placement"
```

---

### Task 7: Rewrite ChatDisplay.tsx — message rows, no bubbles

**Files:**
- Modify: `frontend/src/components/ChatDisplay.tsx`

Key changes:
- Full-width rows with left accent border (not chat bubbles)
- Role label above message text
- Meeting entries use new `.meeting-row` class
- Streaming pulse on border, cursor at end of text
- Interrupted: italic + opacity, no "[interrupted]" text

**Step 1: Rewrite ChatDisplay.tsx**

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
    <div
      ref={chatRef}
      style={{
        flex: 1,
        overflowY: 'auto',
        paddingBottom: 32,
        scrollBehavior: 'smooth',
      }}
    >
      {messages.map(msg => (
        <div
          key={msg.id}
          className={[
            'msg-row',
            msg.role,
            msg.streaming ? 'streaming' : '',
            msg.interrupted ? 'interrupted' : '',
            msg.rejected ? 'rejected' : '',
          ].filter(Boolean).join(' ')}
        >
          <span className="role-label">{msg.role}</span>
          <span className="msg-text">{msg.text}</span>
          {msg.meta && <div className="msg-meta">{msg.meta}</div>}
        </div>
      ))}

      {meetingMode && meetingEntries.map(entry => (
        <div key={entry.id} className="meeting-row">
          <span className="meeting-time">{entry.time}</span>
          <span className={`meeting-speaker ${entry.speakerType}`}>{entry.speaker}:</span>
          <span style={{ color: 'var(--text)', flex: 1 }}>{entry.text}</span>
        </div>
      ))}
    </div>
  )
}
```

**Step 2: Commit**
```bash
git add frontend/src/components/ChatDisplay.tsx
git commit -m "feat: rewrite ChatDisplay with message row style instead of chat bubbles"
```

---

### Task 8: Rewrite Controls.tsx — left panel button cluster

**Files:**
- Modify: `frontend/src/components/Controls.tsx`

Controls now renders:
- Large mic button (80px) centered
- Secondary row: Clear, WakeToggle, MeetingToggle (48px each)
- Meeting command button (floating, right side of right panel on desktop, bottom-right on mobile)
- Wake word status text beneath

**Step 1: Rewrite Controls.tsx**

```tsx
// frontend/src/components/Controls.tsx
import { useRef, useCallback } from 'react'
import { Trash2, Ear, EarOff, Radio } from 'lucide-react'
import { float32ToBase64 } from '../hooks/useManualRecording'
import { SAMPLE_RATE } from '../constants'
import type { MicVAD } from '@ricky0123/vad-web'

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
  onMeetingToggle: () => void
  send: (msg: Record<string, unknown>) => void
  vadRef: React.RefObject<MicVAD | null>
  meetingModeRef: React.RefObject<boolean>
  wakeWordEnabledRef: React.RefObject<boolean>
  startWakeStream: () => Promise<void>
  setStatus: (text: string, cls?: 'active' | 'listening' | 'sleeping' | '') => void
  wakeWordName: string | null
}

export function Controls({
  micState, wakeWordActive, wakeWordEnabled, audioLevelRingRef,
  meetingMode, onMicDown, onMicUp, onMicLeave,
  onClear, onWakeToggle, onMeetingToggle,
  send, vadRef, meetingModeRef,
  wakeWordEnabledRef, startWakeStream, setStatus, wakeWordName,
}: ControlsProps) {
  void wakeWordEnabledRef
  void startWakeStream

  const commandRecordingRef = useRef(false)
  const commandChunksRef = useRef<Float32Array[]>([])
  const commandStreamRef = useRef<MediaStream | null>(null)
  const commandCtxRef = useRef<AudioContext | null>(null)
  const commandProcessorRef = useRef<ScriptProcessorNode | null>(null)

  const micBtnClass = [
    'mic-btn',
    micState === 'recording' ? 'recording' : '',
    micState === 'vad-active' ? 'vad-active' : '',
    micState === 'sleeping' ? 'sleeping' : '',
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

  const WakeIcon = wakeWordActive ? Ear : EarOff

  return (
    <>
      {/* Floating meeting command button — visible when in meeting mode */}
      {meetingMode && (
        <button
          style={{
            position: 'fixed',
            bottom: 80,
            right: 24,
            width: 52,
            height: 52,
            borderRadius: '50%',
            background: 'var(--accent)',
            border: 'none',
            color: 'white',
            fontSize: '1.3rem',
            cursor: 'pointer',
            zIndex: 50,
            boxShadow: '0 2px 12px rgba(0,0,0,0.4)',
            display: 'flex',
            alignItems: 'center',
            justifyContent: 'center',
          }}
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

      {/* Mic button */}
      <div style={{ display: 'flex', flexDirection: 'column', alignItems: 'center', gap: 16 }}>
        <button
          className={micBtnClass}
          title="Hold to talk, or say the wake word"
          onMouseDown={onMicDown}
          onMouseUp={onMicUp}
          onMouseLeave={onMicLeave}
          onTouchStart={onMicDown}
          onTouchEnd={onMicUp}
        >
          <div className="audio-ring" ref={audioLevelRingRef} />
          <svg width="36" height="36" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2" strokeLinecap="round" strokeLinejoin="round">
            <path d="M12 1a3 3 0 0 0-3 3v8a3 3 0 0 0 6 0V4a3 3 0 0 0-3-3z"/>
            <path d="M19 10v2a7 7 0 0 1-14 0v-2"/>
            <line x1="12" y1="19" x2="12" y2="23"/>
            <line x1="8" y1="23" x2="16" y2="23"/>
          </svg>
        </button>

        {/* Secondary buttons row */}
        <div style={{ display: 'flex', gap: 12, alignItems: 'center' }}>
          <button
            className="ctrl-btn"
            title="Clear conversation"
            onClick={onClear}
          >
            <Trash2 size={18} />
          </button>

          <button
            className={`ctrl-btn ${wakeWordActive ? 'wake-active' : ''}`}
            title={wakeWordEnabled ? 'Toggle wake word mode' : 'Toggle hands-free listening'}
            onClick={onWakeToggle}
          >
            <WakeIcon size={18} />
          </button>

          <button
            className={`ctrl-btn ${meetingMode ? 'meeting-active' : ''}`}
            title={meetingMode ? 'End meeting mode' : 'Start meeting mode'}
            onClick={onMeetingToggle}
          >
            <Radio size={18} />
          </button>
        </div>

        {/* Wake word hint text */}
        {wakeWordActive && micState === 'sleeping' && wakeWordName && (
          <p style={{ fontSize: '0.75rem', color: 'var(--text2)', textAlign: 'center', marginTop: 4 }}>
            Say &ldquo;{wakeWordName}&rdquo; to wake me
          </p>
        )}
      </div>
    </>
  )
}
```

**Important note:** Controls now accepts `onMeetingToggle` and `wakeWordName` props — these must be added to the App.tsx call site in Task 9.

**Step 2: Run type check**
```bash
cd /Users/hamiltonchua/.openclaw/workspace/voice-chat/frontend
npx tsc --noEmit 2>&1 | head -40
```

**Step 3: Commit**
```bash
git add frontend/src/components/Controls.tsx
git commit -m "feat: rewrite Controls with left panel layout, lucide icons, meeting toggle"
```

---

### Task 9: Rewrite App.tsx — two-panel grid layout

**Files:**
- Modify: `frontend/src/App.tsx`

This is the biggest change. The JSX return is replaced with the two-panel grid. Hook logic stays 100% intact — only the JSX and props change.

**Step 1: Determine connection dot state**

Add a derived value in App.tsx (before the return statement, after all hooks):

```tsx
// Derive connection dot state from existing state
const connectionDot = showReconnectBanner
  ? 'disconnected'
  : micState === 'sleeping'
  ? 'sleeping'
  : _connectionStatus === 'connecting' || _connectionStatus === 'reconnecting'
  ? 'connecting'
  : 'connected'
```

**Step 2: Update imports in App.tsx**

Add `SettingsDrawer` import. Remove `StatusDisplay` from imports (it's now inline).

```tsx
import { SettingsDrawer } from './components/SettingsDrawer'
```

Remove:
```tsx
import { StatusDisplay } from './components/StatusDisplay'
```

**Step 3: Replace the return JSX in App.tsx**

The new two-panel layout. Replace the entire `return (...)` block (lines 577–635):

```tsx
  return (
    <div style={{ display: 'flex', flexDirection: 'column', height: '100vh', background: 'var(--bg)', color: 'var(--text)', overflow: 'hidden' }}>
      <Toaster position="top-right" theme="dark" />
      <ReconnectBanner
        show={showReconnectBanner}
        onReconnect={() => {
          setShowReconnectBanner(false)
          setStatus('Reconnecting...', '')
          reconnect()
        }}
      />

      {/* Mobile top bar */}
      <Header
        statusText={statusText}
        statusClass={statusClass}
        connectionDot={connectionDot}
        meetingMode={meetingMode}
        enrolled={enrolled}
        verifyEnabled={verifyEnabled}
        onEnroll={handleEnrollStart}
        onVerifyToggle={handleVerifyToggle}
        onMeetingToggle={() => meetingMode ? exitMeetingMode() : enterMeetingMode()}
      />

      {/* Two-panel grid — desktop only */}
      <div
        className="hidden md:grid"
        style={{
          flex: 1,
          display: 'grid',
          gridTemplateColumns: '320px 1fr',
          overflow: 'hidden',
        }}
      >
        {/* Left panel — Control Center */}
        <div
          style={{
            background: 'var(--surface)',
            borderRight: meetingMode ? '3px solid var(--accent)' : '1px solid rgba(255,255,255,0.08)',
            display: 'flex',
            flexDirection: 'column',
            padding: '28px 20px 20px',
            gap: 20,
            overflow: 'hidden',
          }}
        >
          {/* Top: Title + status */}
          <div>
            <h1 style={{ fontSize: '1.4rem', fontWeight: 700, marginBottom: 10 }}>Kismet</h1>
            <div style={{ display: 'flex', alignItems: 'center', gap: 8 }}>
              <span className={`conn-dot ${connectionDot}`} />
              <span
                className={`status-${statusClass || 'default'}`}
                style={{ fontSize: '0.82rem', transition: 'color 0.2s' }}
              >
                {meetingMode ? 'Meeting Mode — Recording' : statusText}
              </span>
            </div>
          </div>

          {/* Middle: Button cluster */}
          <div style={{ flex: 1, display: 'flex', flexDirection: 'column', alignItems: 'center', justifyContent: 'center', gap: 0 }}>
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
              onMeetingToggle={() => meetingMode ? exitMeetingMode() : enterMeetingMode()}
              send={send}
              vadRef={vadRef}
              meetingModeRef={meetingModeRef}
              wakeWordEnabledRef={wakeWordEnabledRef}
              startWakeStream={startWakeStream}
              setStatus={setStatus}
              wakeWordName={wakeWordName}
            />
          </div>

          {/* Bottom: Settings gear */}
          <div style={{ display: 'flex', justifyContent: 'flex-end' }}>
            <SettingsDrawer
              enrolled={enrolled}
              verifyEnabled={verifyEnabled}
              onEnroll={handleEnrollStart}
              onVerifyToggle={handleVerifyToggle}
            />
          </div>
        </div>

        {/* Right panel — Conversation */}
        <div style={{ display: 'flex', flexDirection: 'column', overflow: 'hidden' }}>
          <MeetingBanner show={meetingMode} meetingCommandCapture={meetingCommandCaptureRef} />
          <ChatDisplay
            messages={messages}
            meetingEntries={meetingEntries}
            meetingMode={meetingMode}
          />
        </div>
      </div>

      {/* Mobile layout — single column */}
      <div
        className="flex flex-col md:hidden"
        style={{ flex: 1, overflow: 'hidden' }}
      >
        <MeetingBanner show={meetingMode} meetingCommandCapture={meetingCommandCaptureRef} />
        <ChatDisplay
          messages={messages}
          meetingEntries={meetingEntries}
          meetingMode={meetingMode}
        />
        {/* Mobile bottom dock */}
        <div className="mobile-dock">
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
            onMeetingToggle={() => meetingMode ? exitMeetingMode() : enterMeetingMode()}
            send={send}
            vadRef={vadRef}
            meetingModeRef={meetingModeRef}
            wakeWordEnabledRef={wakeWordEnabledRef}
            startWakeStream={startWakeStream}
            setStatus={setStatus}
            wakeWordName={wakeWordName}
          />
        </div>
      </div>

      <EnrollModal
        show={showEnrollModal}
        step={enrollStep}
        onCancel={handleEnrollCancel}
        send={send}
      />
    </div>
  )
```

**Step 4: Run type check**
```bash
cd /Users/hamiltonchua/.openclaw/workspace/voice-chat/frontend
npx tsc --noEmit 2>&1
```

Fix any errors. Common ones:
- `connectionDot` type mismatch — ensure the derived value type is `'connected' | 'disconnected' | 'sleeping' | 'connecting'`
- Missing `isRecording` in scope — it's already returned from `useManualRecording` hook (line ~162 in App.tsx)
- `stopManual` — already in scope from hook destructuring

**Step 5: Commit**
```bash
git add frontend/src/App.tsx
git commit -m "feat: rewrite App.tsx with two-panel grid layout and updated component props"
```

---

### Task 10: Final build and type check

**Step 1: Full TypeScript check**
```bash
cd /Users/hamiltonchua/.openclaw/workspace/voice-chat/frontend
npx tsc --noEmit 2>&1
```

Fix all type errors before proceeding.

**Step 2: Production build**
```bash
cd /Users/hamiltonchua/.openclaw/workspace/voice-chat/frontend
npm run build 2>&1
```

Fix any build errors.

**Step 3: Verify dist/ was updated**
```bash
ls -la /Users/hamiltonchua/.openclaw/workspace/voice-chat/frontend/dist/
```

**Step 4: Final commit**
```bash
git add -A
git commit -m "feat: phase9 UI overhaul complete — two-panel layout, left Control Center, right conversation"
```

---

## Known Gotchas

1. **Sheet component radix import path**: Check how existing `dialog.tsx` imports radix — match that pattern in `sheet.tsx`.

2. **`hidden md:grid` Tailwind classes**: Tailwind v4 uses `@media` queries differently. If `md:grid` doesn't work, use inline style with a `useMediaQuery` hook or just always show the grid and handle mobile with CSS only.

3. **Controls duplicate rendering** (desktop + mobile): This is intentional — same Controls component is rendered twice. The `audioLevelRingRef` is shared, which is fine since only one will be visible at a time. If issues arise, pass `null` ref to the mobile instance.

4. **`connectionDot` type**: TypeScript may infer it as `string`. Explicitly type it:
   ```tsx
   const connectionDot: 'connected' | 'disconnected' | 'sleeping' | 'connecting' = ...
   ```

5. **`isRecording` in scope**: `useManualRecording` returns `{ start, stop, isRecording }`. In App.tsx it's destructured as `{ start: startManual, stop: stopManual, isRecording }` — already in scope.

6. **Tailwind `md:hidden` / `hidden md:grid`**: These rely on Tailwind's responsive prefix. In Tailwind v4, verify these work. Alternative: use a CSS custom media query in index.css.

7. **EnrollModal stays unchanged**: Don't touch EnrollModal.tsx — it works as-is. The SettingsDrawer just triggers `onEnroll` which shows the modal.
