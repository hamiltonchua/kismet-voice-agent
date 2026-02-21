// frontend/src/components/MeetingBanner.tsx
import * as React from 'react'

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
