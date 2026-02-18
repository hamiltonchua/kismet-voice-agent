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
      <div className="header-row">
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
