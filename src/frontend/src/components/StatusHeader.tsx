import type { RealtimeSnapshot } from '../types/realtime';

interface StatusHeaderProps {
  snapshot: RealtimeSnapshot;
}

export function StatusHeader({ snapshot }: StatusHeaderProps) {
  const statusText = snapshot.statusLabel;
  const timeOptions: Intl.DateTimeFormatOptions = {
    hour: '2-digit',
    minute: '2-digit',
    second: '2-digit',
    timeZone: 'UTC',
  };

  return (
    <header className="statusHeader">
      <div>
        <p className="eyebrow">Realtime status</p>
        <h1>{statusText}</h1>
      </div>

      <div className="headerMeta">
        <div className="metaItem">
          <span className="metaLabel">Current time</span>
            <strong>{new Date(snapshot.currentTimestamp).toLocaleTimeString([], timeOptions)} UTC</strong>
        </div>
        <div className="metaItem">
          <span className="metaLabel">Last update</span>
            <strong>{new Date(snapshot.lastUpdated).toLocaleTimeString([], timeOptions)} UTC</strong>
        </div>
        <div className="metaItem">
          <span className="metaLabel">Detected</span>
          <strong>{snapshot.bottleneckCount}</strong>
        </div>
      </div>
    </header>
  );
}
