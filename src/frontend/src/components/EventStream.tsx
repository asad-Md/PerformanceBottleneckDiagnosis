import type { RealtimeEventItem } from '../types/realtime';

interface EventStreamProps {
  events: RealtimeEventItem[];
}

export function EventStream({ events }: EventStreamProps) {
  return (
    <section className="panel eventPanel">
      <div className="panelHeader">
        <div>
          <p className="eyebrow">Event stream</p>
          <h2>Recent activity</h2>
        </div>
      </div>

      <div className="eventList">
        {events.map((event) => (
          <div key={`${event.time}-${event.text}`} className={`eventItem ${event.level}`}>
            <span className="eventTime">{event.time}</span>
            <span className="eventText">{event.text}</span>
          </div>
        ))}
      </div>
    </section>
  );
}
