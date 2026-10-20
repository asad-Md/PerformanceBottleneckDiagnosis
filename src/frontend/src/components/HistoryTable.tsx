import type { RealtimeHistoryItem } from '../types/realtime';

interface HistoryTableProps {
  items: RealtimeHistoryItem[];
}

export function HistoryTable({ items }: HistoryTableProps) {
  return (
    <section className="panel historyPanel">
      <div className="panelHeader">
        <div>
          <p className="eyebrow">Diagnosis history</p>
          <h2>Recent results</h2>
        </div>
      </div>
      {items.length === 0 ? (
          <p>No process results are present in the current live snapshot.</p>
      ) : (
        <div className="tableWrap">
          <table>
            <thead>
              <tr><th>Sample timestamp (CLOCK_MONOTONIC ns)</th><th>Sample age</th><th>Process</th><th>PID</th><th>CPU</th><th>Champion V7 severity</th><th>MLP attribution</th></tr>
            </thead>
            <tbody>
              {items.map((item) => (
                <tr key={`${item.sample_timestamp_ns ?? item.sample_timestamp_status}-${item.pid}-${item.cpu}`}>
                  <td>{item.sample_timestamp_status === 'available' && item.sample_timestamp_ns
                    ? item.sample_timestamp_ns
                    : item.sample_timestamp_status === 'invalid'
                      ? 'Invalid map timestamp'
                      : 'Unavailable'}</td>
                  <td>{item.sample_age_seconds === null
                    ? `Unavailable${item.data_status === 'stale' ? ' · stale' : ''}`
                    : `${item.sample_age_seconds.toFixed(1)} s${item.data_status === 'stale' ? ' · stale' : ''}`}</td>
                  <td>{item.comm}</td>
                  <td>{item.pid}</td>
                  <td>{item.cpu}</td>
                  <td>{item.severity_prediction?.available ? item.severity_prediction.prediction : 'Unavailable'}</td>
                  <td>{item.attribution_status !== 'available'
                    ? `Unavailable (${item.attribution_status})`
                    : item.attribution_labels?.length
                      ? item.attribution_labels.join(', ')
                      : 'No labels above threshold'}</td>
                </tr>
              ))}
            </tbody>
          </table>
        </div>
      )}
    </section>
  );
}
