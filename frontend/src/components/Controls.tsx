// frontend/src/components/Controls.tsx
import { useRef, useCallback } from 'react'
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
  send: (msg: Record<string, unknown>) => void
  vadRef: React.RefObject<MicVAD | null>
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
  void wakeWordEnabledRef
  void startWakeStream
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
    <svg width="20" height="20" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2" strokeLinecap="round" strokeLinejoin="round">
      <path d="M12 2a3 3 0 0 0-3 3v4a3 3 0 0 0 6 0V5a3 3 0 0 0-3-3z"/>
      <path d="M19 10v1a7 7 0 0 1-14 0v-1"/>
      <path d="M12 19v2"/><path d="M8 21h8"/>
      {!wakeWordActive && <path d="M3 3l18 18" strokeWidth="2"/>}
    </svg>
  ) : (
    <svg width="20" height="20" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2" strokeLinecap="round" strokeLinejoin="round">
      <path d="M12 2a3 3 0 0 0-3 3v4a3 3 0 0 0 6 0V5a3 3 0 0 0-3-3z"/>
      <path d="M19 10v1a7 7 0 0 1-14 0v-1"/>
      <path d="M12 19v2"/><path d="M8 21h8"/>
    </svg>
  )

  return (
    <>
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
        <button className="btn" id="clearBtn" title="Clear conversation" onClick={onClear}>
          <svg width="20" height="20" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2" strokeLinecap="round" strokeLinejoin="round">
            <polyline points="3 6 5 6 21 6"/>
            <path d="M19 6l-1 14a2 2 0 0 1-2 2H8a2 2 0 0 1-2-2L5 6"/>
            <path d="M10 11v6"/><path d="M14 11v6"/>
            <path d="M9 6V4a1 1 0 0 1 1-1h4a1 1 0 0 1 1 1v2"/>
          </svg>
        </button>

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
