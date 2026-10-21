import type { RealtimeObservation, SeverityLabel, SeverityPrediction } from '../types/realtime';

export type ProcessSort = 'high-probability' | 'process-name' | 'pid';

export function normalizeProbability(value: unknown): number | null {
  return typeof value === 'number' && Number.isFinite(value) && value >= 0 && value <= 1
    ? value
    : null;
}

export function formatProbabilityPercent(value: unknown): string {
  const probability = normalizeProbability(value);
  return probability === null ? 'N/A' : `${(probability * 100).toFixed(1)}%`;
}

function classProbability(
  prediction: SeverityPrediction | undefined,
  label: SeverityLabel,
): number | null {
  if (!prediction?.available) return null;
  const probabilities = prediction.class_probabilities;
  if (probabilities && Object.prototype.hasOwnProperty.call(probabilities, label)) {
    return normalizeProbability(probabilities[label]);
  }
  if (label === 'High') return normalizeProbability(prediction.high_probability);
  return null;
}

export function predictedClassConfidence(prediction: SeverityPrediction | undefined): number | null {
  if (!prediction?.available) return null;
  const label = prediction.prediction;
  if (label === 'High' || label === 'Not-High') {
    const classScore = classProbability(prediction, label);
    if (classScore !== null) return classScore;
    if (prediction.class_probabilities?.[label] !== undefined) return null;
  } else {
    return null;
  }

  const percent = prediction.confidence;
  return typeof percent === 'number' && Number.isFinite(percent) && percent >= 0 && percent <= 100
    ? percent / 100
    : null;
}

export function highClassProbability(prediction: SeverityPrediction | undefined): number | null {
  return classProbability(prediction, 'High');
}

export function isStressTestProcess(process: RealtimeObservation): boolean {
  return /\bstress-?ng\b|\bstress\b/i.test(process.comm || '');
}

function validClassification(process: RealtimeObservation): SeverityLabel | null {
  const prediction = process.severity_prediction;
  if (!prediction?.available) return null;
  return prediction.prediction === 'High' || prediction.prediction === 'Not-High'
    ? prediction.prediction
    : null;
}

export function filterAndSortProcesses(
  items: RealtimeObservation[],
  options: { search: string; stressOnly: boolean; includeStale: boolean; sort: ProcessSort },
): RealtimeObservation[] {
  const query = options.search.trim().toLocaleLowerCase();
  return items
    .filter((item) => options.includeStale || item.data_status !== 'stale')
    .filter((item) => !options.stressOnly || isStressTestProcess(item))
    .filter((item) => {
      if (!query) return true;
      return String(item.pid).includes(query) || (item.comm || '').toLocaleLowerCase().includes(query);
    })
    .sort((left, right) => {
      if (options.sort === 'process-name') {
        return (left.comm || '').localeCompare(right.comm || '') || left.pid - right.pid;
      }
      if (options.sort === 'pid') return left.pid - right.pid || left.cpu - right.cpu;

      const leftProbability = highClassProbability(left.severity_prediction);
      const rightProbability = highClassProbability(right.severity_prediction);
      if (leftProbability === null && rightProbability !== null) return 1;
      if (rightProbability === null && leftProbability !== null) return -1;
      if (leftProbability !== null && rightProbability !== null && leftProbability !== rightProbability) {
        return rightProbability - leftProbability;
      }
      return left.pid - right.pid || left.cpu - right.cpu;
    });
}

export function summarizeVisibleProcesses(items: RealtimeObservation[]) {
  let high = 0;
  let notHigh = 0;
  let confidenceAvailable = 0;
  for (const item of items) {
    const prediction = item.severity_prediction;
    if (prediction?.available && prediction.prediction === 'High') high += 1;
    if (prediction?.available && prediction.prediction === 'Not-High') notHigh += 1;
    if (predictedClassConfidence(prediction) !== null) confidenceAvailable += 1;
  }
  return { visible: items.length, high, notHigh, confidenceAvailable };
}