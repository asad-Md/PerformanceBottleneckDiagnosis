import type { RealtimeFeatureMetric, RealtimePredictionResult } from '../types/realtime';

interface DiagnosisPanelProps {
  prediction: RealtimePredictionResult | null;
  metrics: RealtimeFeatureMetric[];
}

export function DiagnosisPanel({ prediction, metrics }: DiagnosisPanelProps) {
  if (!prediction) {
    return (
      <section className="panel diagnosisPanel">
        <div className="panelHeader">
          <div>
            <p className="eyebrow">Current diagnosis</p>
            <h2>Waiting for realtime analysis</h2>
          </div>
          <span className="labelBadge">Waiting</span>
        </div>
      </section>
    );
  }

  const identityMetrics = [
    { label: 'Process', value: prediction.comm },
    { label: 'PID', value: String(prediction.pid) },
    { label: 'CPU', value: String(prediction.cpu) },
    { label: 'Prediction', value: prediction.prediction },
    { label: 'Class', value: String(prediction.pred_class) },
    { label: 'Bottleneck', value: prediction.is_bottleneck ? 'Yes' : 'No' },
  ];

  return (
    <section className="panel diagnosisPanel">
      <div className="panelHeader">
        <div>
          <p className="eyebrow">Current diagnosis</p>
          <h2>{prediction.is_bottleneck ? 'Bottleneck detected' : 'No bottleneck detected'}</h2>
        </div>
        <span className="labelBadge">{prediction.is_bottleneck ? 'Active' : 'Stable'}</span>
      </div>

      <div className="diagnosisIdentity">
        {identityMetrics.map((metric) => (
          <div key={metric.label} className="identityItem">
            <span>{metric.label}</span>
            <strong>{metric.value}</strong>
          </div>
        ))}
        <div className="identityItem confidenceItem">
          <span>Confidence</span>
          <strong>{prediction.confidence}%</strong>
        </div>
      </div>

      <div className="supportingSignals">
        <div className="supportingSignalsHeader">
          <p className="eyebrow">Supporting signals</p>
          <span>Observed by the realtime pipeline</span>
        </div>
        <div className="detailGrid">
        {metrics.map((metric) => (
          <div key={metric.label} className="detailItem">
            <span>{metric.label}</span>
            <strong>{metric.value} <small>{metric.unit}</small></strong>
          </div>
        ))}
        </div>
      </div>
    </section>
  );
}
