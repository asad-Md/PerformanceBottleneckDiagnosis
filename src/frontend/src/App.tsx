import { useEffect, useState } from 'react';
import { DiagnosisPanel } from './components/DiagnosisPanel';
import { EventStream } from './components/EventStream';
import { HistoryTable } from './components/HistoryTable';
import { StatusHeader } from './components/StatusHeader';
import { createRealtimeAdapter, createRealtimeErrorSnapshot } from './services/realtimeAdapter';
import type { RealtimeSnapshot } from './types/realtime';

const adapter = createRealtimeAdapter();

export function App() {
  const [snapshot, setSnapshot] = useState<RealtimeSnapshot | null>(null);

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
    return <div className="loadingState">Loading realtime diagnosis…</div>;
  }

  return (
    <div className="appShell">
      <div className="pageFrame">
        <StatusHeader snapshot={snapshot} />

        <p className="systemSummary">{snapshot.summary}</p>

        <DiagnosisPanel prediction={snapshot.prediction} metrics={snapshot.metrics} models={snapshot.models} connectionStatus={snapshot.connectionStatus} />

        <EventStream events={snapshot.events} />

        <HistoryTable items={snapshot.history} />
      </div>
    </div>
  );
}
