import { Loader2 } from 'lucide-react'

interface TaskBadgeProps {
  count: number
}

export function TaskBadge({ count }: TaskBadgeProps) {
  if (count === 0) return null

  return (
    <span
      style={{
        display: 'inline-flex',
        alignItems: 'center',
        gap: 4,
        background: 'var(--surface2)',
        color: 'var(--text2)',
        fontSize: '0.65rem',
        fontWeight: 600,
        padding: '2px 6px',
        borderRadius: 10,
        letterSpacing: '0.02em',
      }}
    >
      <Loader2 size={10} className="animate-spin" />
      {count}
    </span>
  )
}
