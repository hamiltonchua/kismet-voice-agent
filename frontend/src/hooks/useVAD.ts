// frontend/src/hooks/useVAD.ts
import { useRef, useCallback } from 'react'
import { MicVAD } from '@ricky0123/vad-web'

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

    vadRef.current = await MicVAD.new({
      positiveSpeechThreshold: 0.9,
      negativeSpeechThreshold: 0.35,
      minSpeechMs: 250,
      preSpeechPadMs: 300,
      redemptionMs: 600,
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
