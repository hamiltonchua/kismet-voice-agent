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
