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
          <span className="metaLabel">Last polled</span>
          <strong>{new Date(snapshot.lastPolled).toLocaleTimeString([], timeOptions)} UTC</strong>
        </div>
        <div className="metaItem">
          <span className="metaLabel">Sample timestamp (CLOCK_MONOTONIC ns)</span>
          <strong>
            {snapshot.sampleTimestampStatus === 'available' && snapshot.sampleTimestampNs
              ? snapshot.sampleTimestampNs
              : snapshot.sampleTimestampStatus === 'invalid'
                ? 'Invalid map timestamp'
                : 'Unavailable'}
          </strong>
        </div>
        <div className="metaItem">
          <span className="metaLabel">Sample age</span>
          <strong>
            {snapshot.sampleAgeSeconds === null
              ? `Unavailable${snapshot.connectionStatus === 'stale' ? ' · stale' : ''}`
              : `${snapshot.sampleAgeSeconds.toFixed(1)} s${snapshot.connectionStatus === 'stale' ? ' · stale' : ''}`}
          </strong>
        </div>
        <div className="metaItem">
          <span className="metaLabel">Processes</span>
          <strong>{snapshot.processCount}</strong>
        </div>
      </div>
    </header>
  );
}
