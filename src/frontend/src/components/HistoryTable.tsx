import { useState } from 'react';
import { formatUnavailablePrediction } from '../services/realtimeAdapter';
import {
  filterAndSortProcesses,
  formatProbabilityPercent,
  highClassProbability,
  predictedClassConfidence,
  summarizeVisibleProcesses,
  type ProcessSort,
} from '../services/processTable';
import type { RealtimeObservation } from '../types/realtime';

interface HistoryTableProps {
  items: RealtimeObservation[];
  totalCount: number;
}

function formatMetric(value: number | null, unit = ''): string {
  return value === null
    ? 'N/A'
    : `${new Intl.NumberFormat(undefined, { maximumFractionDigits: 1 }).format(value)}${unit}`;
}

export function HistoryTable({ items, totalCount }: HistoryTableProps) {
  const [search, setSearch] = useState('');
  const [stressOnly, setStressOnly] = useState(false);
  const [sort, setSort] = useState<ProcessSort>('high-probability');
  const visible = filterAndSortProcesses(items, { search, stressOnly, sort });
  const summary = summarizeVisibleProcesses(visible);

  return (
    <section className="panel historyPanel" aria-labelledby="process-list-title">
      <div className="panelHeader processHeader">
        <div>
          <p className="eyebrow">Live process snapshot</p>
          <h2 id="process-list-title">Process diagnosis</h2>
          <p className="processCaveat">
            Model probabilities are per-process outputs, not guarantees of actual performance degradation.
          </p>
        </div>
      </div>

      <div className="processControls">
        <label className="searchField">
          <span>Search PID or process</span>
          <input
            type="search"
            value={search}
            onChange={(event) => setSearch(event.target.value)}
            placeholder="PID or process name"
          />
        </label>
        <label className="selectField">
          <span>Sort by</span>
          <select value={sort} onChange={(event) => setSort(event.target.value as ProcessSort)}>
            <option value="high-probability">High probability, highest first</option>
            <option value="process-name">Process name</option>
            <option value="pid">PID</option>
          </select>
        </label>
        <label className="toggleField">
          <input type="checkbox" checked={stressOnly} onChange={(event) => setStressOnly(event.target.checked)} />
          <span>Stress-test processes</span>
        </label>
      </div>

      <div className="processSummary" aria-live="polite">
        <div><span>Visible / total</span><strong>{summary.visible} / {totalCount}</strong></div>
        <div><span>High</span><strong>{summary.high}</strong></div>
        <div><span>Not High</span><strong>{summary.notHigh}</strong></div>
        <div><span>Confidence available</span><strong>{summary.confidenceAvailable}</strong></div>
        <p>Counts reflect the current search and filters. Unavailable classifications are not counted.</p>
      </div>

      {visible.length === 0 ? (
        <p className="emptyProcessState">
          {items.length === 0
            ? 'No fresh process rows are available in the current map snapshot.'
            : 'No process rows match the current search and filters.'}
        </p>
      ) : (
        <div className="tableWrap processTableWrap">
          <table className="processTable">
            <thead>
              <tr>
                <th>Process</th>
                <th>PID</th>
                <th>CPU</th>
                <th>Data</th>
                <th>Champion V7</th>
                <th>Confidence</th>
                <th>High probability</th>
                <th>Runtime</th>
                <th>Context switches</th>
                <th>CPU migrations</th>
                <th>Allocated bytes</th>
                <th>Page faults (minor / major)</th>
                <th>MLP attribution</th>
              </tr>
            </thead>
            <tbody>
              {visible.map((item) => {
                const prediction = item.severity_prediction;
                const confidence = predictedClassConfidence(prediction);
                const highProbability = highClassProbability(prediction);
                const classification = prediction?.available
                  ? prediction.prediction === 'Not-High' ? 'Not High' : prediction.prediction ?? 'Unavailable'
                  : formatUnavailablePrediction(prediction, item.data_status);
                return (
                  <tr key={`${item.pid}-${item.cpu}`}>
                    <td className="processNameCell">{item.comm || 'Unknown process'}</td>
                    <td>{item.pid}</td>
                    <td>{item.cpu}</td>
                    <td><span className={`dataState ${item.data_status}`}>{item.data_status === 'live' ? 'Fresh' : 'Stale'}</span></td>
                    <td>
                      <span className={`classification ${classification === 'High' ? 'high' : classification === 'Not High' ? 'notHigh' : 'unavailable'}`}>
                        {classification}
                      </span>
                    </td>
                    <td className="confidenceCell">
                      {confidence === null ? (
                        <span className="unavailableValue">N/A</span>
                      ) : (
                        <span className="confidenceValue">{formatProbabilityPercent(confidence)}</span>
                      )}
                    </td>
                    <td>{formatProbabilityPercent(highProbability)}</td>
                    <td>{formatMetric(item.metrics.runtimeMs, ' ms')}</td>
                    <td>{formatMetric(item.metrics.contextSwitches)}</td>
                    <td>{formatMetric(item.metrics.cpuMigrations)}</td>
                    <td>{formatMetric(item.metrics.allocationBytes, ' B')}</td>
                    <td>{formatMetric(item.metrics.minorFaults)} / {formatMetric(item.metrics.majorFaults)}</td>
                    <td>{item.attribution_prediction?.available
                      ? item.attribution_labels?.length ? item.attribution_labels.join(', ') : 'No labels selected'
                      : formatUnavailablePrediction(item.attribution_prediction, item.data_status)}</td>
                  </tr>
                );
              })}
            </tbody>
          </table>
        </div>
      )}
    </section>
  );
}
