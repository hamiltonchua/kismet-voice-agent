// frontend/src/components/MeetingBanner.tsx

interface MeetingBannerProps {
  show: boolean
  meetingCommandCapture: React.RefObject<boolean>
}

export function MeetingBanner({ show, meetingCommandCapture }: MeetingBannerProps) {
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
