import type {
  ModelStatuses,
  RealtimeFeatureMetric,
  RealtimePredictionResult,
  RealtimeSnapshot,
} from '../types/realtime';
import {
  formatProbabilityPercent,
  highClassProbability,
  normalizeProbability,
  predictedClassConfidence,
} from '../services/processTable';

interface DiagnosisPanelProps {
  prediction: RealtimePredictionResult | null;
  metrics: RealtimeFeatureMetric[];
  models: ModelStatuses;
  connectionStatus: RealtimeSnapshot['connectionStatus'];
}

export function DiagnosisPanel({ prediction, metrics, models, connectionStatus }: DiagnosisPanelProps) {
  const attribution = prediction?.attribution_prediction;
  const severity = prediction?.severity_prediction;
  const isLive = connectionStatus === 'live';
  const confidence = predictedClassConfidence(severity);
  const highProbability = highClassProbability(severity);
  const threshold = normalizeProbability(severity?.threshold);

  return (
    <section className="panel diagnosisPanel">
      <div className="panelHeader">
        <div>
          <p className="eyebrow">Process diagnosis</p>
          <h2>{prediction ? `${prediction.comm} · PID ${prediction.pid}` : 'No process selected'}</h2>
        </div>
        <span className="labelBadge">
          {isLive ? 'LIVE eBPF' : connectionStatus === 'stale' ? 'STALE eBPF' : 'DISCONNECTED'}
        </span>
      </div>

      {prediction ? (
        <div className="diagnosisIdentity">
          <div className="identityItem"><span>Process</span><strong>{prediction.comm}</strong></div>
          <div className="identityItem"><span>PID</span><strong>{prediction.pid}</strong></div>
          <div className="identityItem"><span>CPU ID</span><strong>{prediction.cpu}</strong></div>
          <div className="identityItem"><span>Missing raw metrics</span><strong>{prediction.missing_metrics.length}</strong></div>
        </div>
      ) : null}

      <section className="supportingSignals">
        <div className="supportingSignalsHeader">
          <p className="eyebrow">Champion V7 · Severity</p>
          <span>Binary Random Forest · High vs Not-High</span>
        </div>
        {!models.champion_v7.available || !severity?.available ? (
          <p className="modelError">
            {severity?.error ?? models.champion_v7.error ?? 'Champion V7 severity is unavailable.'}
          </p>
        ) : (
          <>
            <div className="detailGrid">
              <div className="detailItem">
                <span>Classification</span><strong>{severity.prediction === 'Not-High' ? 'Not High' : severity.prediction ?? 'Unavailable'}</strong>
              </div>
              <div className="detailItem">
                <span>Prediction confidence</span>
                <strong className="selectedConfidence">{formatProbabilityPercent(confidence)}</strong>
              </div>
              <div className="detailItem">
                <span>High probability</span>
                <strong>{formatProbabilityPercent(highProbability)}</strong>
              </div>
              <div className="detailItem">
                <span>Decision threshold</span>
                <strong>{formatProbabilityPercent(threshold)}</strong>
              </div>
            </div>
            {severity.zero_filled_inputs?.length ? (
              <p className="modelNote">
                Zero-filled live inputs: {severity.zero_filled_inputs.join(', ')}.
              </p>
            ) : null}
          </>
        )}
      </section>

      <section className="supportingSignals">
        <div className="supportingSignalsHeader">
          <p className="eyebrow">Baseline MLP · Attribution</p>
          <span>Sigmoid output scores · 50% decision threshold</span>
        </div>
        {!models.attribution_mlp.available || !attribution?.available ? (
          <p className="modelError">
            {attribution?.error ?? models.attribution_mlp.error ?? 'Attribution is unavailable.'}
          </p>
        ) : (
          <>
            <p className="attributionLabels">
              {attribution.labels?.length
                ? `Selected labels: ${attribution.labels.join(', ')}`
                : 'Valid model result: no labels crossed the decision threshold.'}
            </p>
            <div className="detailGrid">
              {Object.entries(attribution.probabilities ?? {}).map(([label, probability]) => {
                const selected = attribution.labels?.includes(label) ?? false;
                const validProbability = normalizeProbability(probability);
                return (
                  <div key={label} className="detailItem">
                    <span>{label}{selected ? ' · Selected' : ''}</span>
                    <strong>{formatProbabilityPercent(validProbability)}<small> score</small></strong>
                    <div
                      className="scoreTrack"
                      role="progressbar"
                      aria-label={`${label} sigmoid output score`}
                      aria-valuenow={validProbability === null ? undefined : validProbability * 100}
                      aria-valuemin={0}
                      aria-valuemax={100}
                    >
                      <div style={{ width: validProbability === null ? '0%' : `${validProbability * 100}%`, height: '100%' }} />
                    </div>
                  </div>
                );
              })}
            </div>
          </>
        )}
      </section>

      <section className="supportingSignals">
        <div className="supportingSignalsHeader">
          <p className="eyebrow">Live process metrics</p>
          <span>Observed eBPF counters</span>
        </div>
        {metrics.length ? (
          <div className="detailGrid">
            {metrics.map((metric) => (
              <div key={metric.id} className="detailItem" title={metric.description}>
                <span>{metric.label}</span>
                <strong>
                  {metric.value === null
                    ? 'Not collected'
                    : <>{new Intl.NumberFormat(undefined, { maximumFractionDigits: 2 }).format(metric.value)} <small>{metric.unit}</small></>}
                </strong>
              </div>
            ))}
          </div>
        ) : (
          <p className="modelNote">No process metrics are available in this snapshot.</p>
        )}
      </section>
    </section>
  );
}
