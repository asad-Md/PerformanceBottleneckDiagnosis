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

      <div className="tableWrap">
        <table>
          <thead>
            <tr>
              <th>Time</th>
              <th>Process</th>
              <th>PID</th>
              <th>CPU</th>
              <th>Prediction</th>
              <th>Class</th>
              <th>Bottleneck</th>
              <th>Confidence</th>
            </tr>
          </thead>
          <tbody>
            {items.map((item) => (
              <tr key={`${item.timestamp}-${item.pid}`}>
                <td>{item.timestamp}</td>
                <td>{item.comm}</td>
                <td>{item.pid}</td>
                <td>{item.cpu}</td>
                <td>{item.prediction}</td>
                <td>{item.pred_class}</td>
                <td>{item.is_bottleneck ? 'Yes' : 'No'}</td>
                <td>{item.confidence}%</td>
              </tr>
            ))}
          </tbody>
        </table>
      </div>
    </section>
  );
}
