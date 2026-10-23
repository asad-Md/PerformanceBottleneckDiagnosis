import { useEffect, useState } from 'react';
import { DiagnosisPanel } from './components/DiagnosisPanel';
import { EventStream } from './components/EventStream';
import { HistoryTable } from './components/HistoryTable';
import { StatusHeader } from './components/StatusHeader';
import { createRealtimeAdapter, createRealtimeErrorSnapshot } from './services/realtimeAdapter';
import type { RealtimeSnapshot } from './types/realtime';

const adapter = createRealtimeAdapter();
type Theme = 'dark' | 'light';

function initialTheme(): Theme {
  try {
    const savedTheme = window.localStorage.getItem('pbd-theme');
    if (savedTheme === 'dark' || savedTheme === 'light') return savedTheme;
  } catch {
    // Fall back to the system preference when storage is unavailable.
  }
  return window.matchMedia('(prefers-color-scheme: light)').matches ? 'light' : 'dark';
}

export function App() {
  const [snapshot, setSnapshot] = useState<RealtimeSnapshot | null>(null);
  const [theme, setTheme] = useState<Theme>(initialTheme);

  useEffect(() => {
    document.documentElement.dataset.theme = theme;
    try {
      window.localStorage.setItem('pbd-theme', theme);
    } catch {
      // Theme still applies for this session when storage is unavailable.
    }
  }, [theme]);

  useEffect(() => {
    let isMounted = true;
    let requestPending = false;

    async function load() {
      if (requestPending) return;
      requestPending = true;
      try {
        const result = await adapter.loadSnapshot();
        if (isMounted) setSnapshot(result);
      } catch (error) {
        if (isMounted) setSnapshot(createRealtimeErrorSnapshot(error));
      } finally {
        requestPending = false;
      }
    }

    load();
    const interval = window.setInterval(load, 5000);
    return () => {
      isMounted = false;
      window.clearInterval(interval);
    };
  }, []);

  if (!snapshot) {
    return (
      <main className="appShell">
        <div className="pageFrame">
          <StatusHeader snapshot={null} theme={theme} onThemeToggle={() => setTheme((current) => current === 'dark' ? 'light' : 'dark')} />
          <div className="loadingState" role="status" aria-live="polite">
            <div className="loadingCard">
              <span className="loadingMark" aria-hidden="true" />
              <p className="eyebrow">Performance Bottleneck Diagnosis</p>
              <h1>Connecting to live monitor</h1>
              <p>Waiting for the first eBPF snapshot from the backend.</p>
            </div>
          </div>
        </div>
      </main>
    );
  }

  return (
    <main className="appShell">
      <div className="pageFrame">
        <StatusHeader snapshot={snapshot} theme={theme} onThemeToggle={() => setTheme((current) => current === 'dark' ? 'light' : 'dark')} />

        <section className="systemSummary" aria-label="Monitoring summary">
          <div className="summaryHeading">
            <div>
              <p className="eyebrow">System overview</p>
              <h2>Live monitoring</h2>
            </div>
            <span className={`connectionBadge ${snapshot.connectionStatus}`}>
              <span className="statusDot" aria-hidden="true" />
              {snapshot.connectionStatus === 'live' ? 'Connected' : snapshot.connectionStatus === 'stale' ? 'Stale data' : 'Disconnected'}
            </span>
          </div>
          <p className="summaryCopy">{snapshot.summary}</p>
          <div className="overviewGrid">
            <div className="overviewStat"><span>Fresh PID / CPU rows</span><strong>{snapshot.freshRowCount}</strong><small>{snapshot.staleRowCount} stale rows excluded</small></div>
            <div className="overviewStat"><span>Selected process</span><strong className="overviewProcess">{snapshot.prediction?.comm ?? 'None selected'}</strong><small>{snapshot.prediction ? `PID ${snapshot.prediction.pid} · CPU ${snapshot.prediction.cpu}` : 'No process data available'}</small></div>
            <div className="overviewStat"><span>Champion V7 severity</span><strong>{snapshot.prediction?.severity_prediction?.available ? snapshot.prediction.severity_prediction.prediction === 'Not-High' ? 'Not High' : snapshot.prediction.severity_prediction.prediction ?? 'Unavailable' : 'Unavailable'}</strong><small>{snapshot.models.champion_v7.available ? 'Model available' : 'Model unavailable'}</small></div>
            <div className="overviewStat"><span>Sample age</span><strong>{snapshot.sampleAgeSeconds === null ? 'Unavailable' : `${snapshot.sampleAgeSeconds.toFixed(1)} s`}</strong><small>{snapshot.sampleTimestampStatus === 'available' ? 'CLOCK_MONOTONIC sample' : snapshot.sampleTimestampStatus === 'invalid' ? 'Invalid sample timestamp' : 'Timestamp unavailable'}</small></div>
          </div>
        </section>

        <DiagnosisPanel prediction={snapshot.prediction} metrics={snapshot.metrics} models={snapshot.models} connectionStatus={snapshot.connectionStatus} />

        <EventStream events={snapshot.events} />

        <HistoryTable items={snapshot.observations} totalCount={snapshot.freshRowCount} />
      </div>
    </main>
  );
}
