import type { RealtimeSnapshot } from '../types/realtime';

interface StatusHeaderProps {
  snapshot: RealtimeSnapshot | null;
  theme: 'dark' | 'light';
  onThemeToggle: () => void;
}

export function StatusHeader({ snapshot, theme, onThemeToggle }: StatusHeaderProps) {
  const statusText = snapshot?.statusLabel ?? 'CONNECTING';
  const timeOptions: Intl.DateTimeFormatOptions = {
    hour: '2-digit',
    minute: '2-digit',
    second: '2-digit',
    timeZone: 'UTC',
  };

  return (
    <header className="statusHeader">
      <div className="brandBlock">
        <div className="brandMark" aria-hidden="true"><span /><span /><span /></div>
        <div>
          <p className="eyebrow">Linux observability</p>
          <h1>Performance Bottleneck Diagnosis</h1>
          <p className="headerSubtitle">eBPF powered process monitoring</p>
        </div>
      </div>

      <div className="headerStatusGroup">
        <div className="headerStatus">
          <span className={`connectionBadge ${snapshot?.connectionStatus ?? 'reconnecting'}`}>
            <span className="statusDot" aria-hidden="true" />
            {statusText}
          </span>
          <span className="pollInterval">Polling every 5 seconds</span>
        </div>
        <button
          className="themeToggle"
          type="button"
          onClick={onThemeToggle}
          aria-label={`Switch to ${theme === 'dark' ? 'light' : 'dark'} mode`}
          title={`Switch to ${theme === 'dark' ? 'light' : 'dark'} mode`}
        >
          {theme === 'dark' ? (
            <svg aria-hidden="true" viewBox="0 0 20 20" fill="none">
              <circle cx="10" cy="10" r="3.5" stroke="currentColor" strokeWidth="1.6" />
              <path d="M10 1.8v2M10 16.2v2M18.2 10h-2M3.8 10h-2m14-5.8-1.4 1.4M5.6 14.4l-1.4 1.4m11.6 0-1.4-1.4M5.6 5.6 4.2 4.2" stroke="currentColor" strokeWidth="1.6" strokeLinecap="round" />
            </svg>
          ) : (
            <svg aria-hidden="true" viewBox="0 0 20 20" fill="none">
              <path d="M16.8 12.3A7.2 7.2 0 0 1 7.7 3.2a7.2 7.2 0 1 0 9.1 9.1Z" stroke="currentColor" strokeWidth="1.6" strokeLinejoin="round" />
            </svg>
          )}
          <span>{theme === 'dark' ? 'Light mode' : 'Dark mode'}</span>
        </button>
      </div>

      <div className="headerMeta">
        <div className="metaItem">
          <span className="metaLabel">Last polled</span>
          <strong>{snapshot ? `${new Date(snapshot.lastPolled).toLocaleTimeString([], timeOptions)} UTC` : 'Awaiting first poll'}</strong>
        </div>
        <div className="metaItem">
          <span className="metaLabel">Sample timestamp (CLOCK_MONOTONIC ns)</span>
          <strong>
            {snapshot?.sampleTimestampStatus === 'available' && snapshot.sampleTimestampNs
              ? snapshot.sampleTimestampNs
              : snapshot?.sampleTimestampStatus === 'invalid'
                ? 'Invalid map timestamp'
                : snapshot ? 'Unavailable' : 'Awaiting sample'}
          </strong>
        </div>
        <div className="metaItem">
          <span className="metaLabel">Sample age</span>
          <strong>
            {!snapshot || snapshot.sampleAgeSeconds === null
              ? `Unavailable${snapshot?.connectionStatus === 'stale' ? ' · stale' : ''}`
              : `${snapshot.sampleAgeSeconds.toFixed(1)} s${snapshot.connectionStatus === 'stale' ? ' · stale' : ''}`}
          </strong>
        </div>
        <div className="metaItem">
          <span className="metaLabel">Sample clock</span>
          <strong>{snapshot?.sampleTimestampClock ?? 'CLOCK_MONOTONIC'}</strong>
        </div>
      </div>
    </header>
  );
}
