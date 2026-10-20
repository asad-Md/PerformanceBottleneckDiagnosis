import { formatUnavailablePrediction } from '../services/realtimeAdapter';
import type { RealtimeObservation } from '../types/realtime';

interface HistoryTableProps {
  items: RealtimeObservation[];
  totalCount: number;
}

export function HistoryTable({ items, totalCount }: HistoryTableProps) {
  return (
    <section className="panel historyPanel">
      <div className="panelHeader">
        <div>
          <p className="eyebrow">PID/CPU observations</p>
          <h2>Current map snapshot</h2>
          <p>
            Latest sample timestamps first · showing {items.length} of {totalCount} additional rows
          </p>
        </div>
      </div>
      {items.length === 0 ? (
          <p>No additional PID/CPU rows are present in the current map snapshot.</p>
      ) : (
        <div className="tableWrap">
          <table>
            <thead>
              <tr><th>Sample timestamp (CLOCK_MONOTONIC ns)</th><th>Sample age</th><th>Observation</th><th>Process</th><th>PID</th><th>CPU</th><th>Champion V7 severity</th><th>MLP attribution</th></tr>
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
                  <td>{item.data_status === 'stale' ? 'Stale observation' : 'Fresh observation'}</td>
                  <td>{item.comm}</td>
                  <td>{item.pid}</td>
                  <td>{item.cpu}</td>
                  <td>{item.severity_prediction?.available
                    ? item.severity_prediction.prediction
                    : formatUnavailablePrediction(item.severity_prediction, item.data_status)}</td>
                  <td>{item.attribution_prediction?.available
                    ? item.attribution_prediction.labels?.length
                      ? item.attribution_prediction.labels.join(', ')
                      : 'No labels above threshold'
                    : formatUnavailablePrediction(item.attribution_prediction, item.data_status)}
                  </td>
                </tr>
              ))}
            </tbody>
          </table>
        </div>
      )}
    </section>
  );
}
