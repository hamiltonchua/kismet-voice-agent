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
