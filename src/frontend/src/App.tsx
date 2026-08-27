import { useEffect, useState } from 'react';
import { DiagnosisPanel } from './components/DiagnosisPanel';
import { EventStream } from './components/EventStream';
import { HistoryTable } from './components/HistoryTable';
import { StatusHeader } from './components/StatusHeader';
import { createRealtimeAdapter } from './services/realtimeAdapter';
import type { RealtimeSnapshot } from './types/realtime';

const adapter = createRealtimeAdapter();

export function App() {
  const [snapshot, setSnapshot] = useState<RealtimeSnapshot | null>(null);

  useEffect(() => {
    let isMounted = true;

    async function load() {
      const result = await adapter.loadSnapshot();
      if (isMounted) {
        setSnapshot(result);
      }
    }

    load();
    return () => {
      isMounted = false;
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

        <DiagnosisPanel prediction={snapshot.prediction} metrics={snapshot.metrics} />

        <EventStream events={snapshot.events} />

        <HistoryTable items={snapshot.history} />
      </div>
    </div>
  );
}
