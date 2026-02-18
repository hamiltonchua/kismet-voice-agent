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
