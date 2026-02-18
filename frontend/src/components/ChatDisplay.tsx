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
    <div id="chat" ref={chatRef}>
      {messages.map(msg => (
        <div
          key={msg.id}
          className={[
            'msg',
            msg.role,
            msg.streaming ? 'streaming' : '',
            msg.interrupted ? 'interrupted' : '',
            msg.rejected ? 'rejected' : '',
          ].filter(Boolean).join(' ')}
        >
          {msg.text}
          {msg.meta && <div className="meta">{msg.meta}</div>}
        </div>
      ))}
      {meetingMode && meetingEntries.map(entry => (
        <div key={entry.id} className="meeting-entry">
          <span className="timestamp">{entry.time}</span>
          <span className={`speaker ${entry.speakerType}`}>{entry.speaker}:</span>
          {' '}{entry.text}
        </div>
      ))}
    </div>
  )
}
