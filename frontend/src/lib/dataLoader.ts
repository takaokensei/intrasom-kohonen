import type {
  SeriesItem,
  SOMModelWithVariants,
  MetricRow,
  ParameterStudyEntry,
  TextModelWithVariants,
  NewsSample,
} from '../store/useDashboardStore';
import type { PCAParameters } from './pca';

export interface SyntheticDataPayload {
  series: SeriesItem[];
  somModels: Record<string, SOMModelWithVariants>;
  metrics: MetricRow[];
  paramStudyResults: ParameterStudyEntry[];
}

export interface TextDataPayload {
  textModels: Record<string, Record<string, TextModelWithVariants>>;
  textMetrics: Record<string, Record<string, { ARI: number; NMI: number }>>;
  newsSamples: Record<string, NewsSample[]>;
  pcaParams: Record<string, PCAParameters>;
}

/**
 * Loads synthetic control time series, SOM models, metrics, and parameter study data.
 */
export async function fetchSyntheticData(): Promise<SyntheticDataPayload> {
  const [seriesRes, modelsRes, metricsRes, studyRes] = await Promise.all([
    fetch('/data/series.json'),
    fetch('/data/som_models.json'),
    fetch('/data/metrics.json'),
    fetch('/data/parameter_study.json').catch(() => null),
  ]);

  if (!seriesRes.ok || !modelsRes.ok || !metricsRes.ok) {
    throw new Error('HTTP status error loading synthetic control files');
  }

  const series = await seriesRes.json();
  const somModels = await modelsRes.json();
  const metrics = await metricsRes.json();
  const paramStudyResults: ParameterStudyEntry[] =
    studyRes && studyRes.ok ? await studyRes.json() : [];

  return { series, somModels, metrics, paramStudyResults };
}

/**
 * Loads text clustering SOM models, metrics, news samples, and PCA parameters.
 */
export async function fetchTextData(): Promise<TextDataPayload> {
  const [modelsRes, metricsRes, samplesRes, pcaRes] = await Promise.all([
    fetch('/data/text_models.json'),
    fetch('/data/text_metrics.json'),
    fetch('/data/news_samples.json'),
    fetch('/data/pca_params.json'),
  ]);

  if (!modelsRes.ok || !metricsRes.ok || !samplesRes.ok || !pcaRes.ok) {
    throw new Error('HTTP status error loading text SOM files');
  }

  const textModels = await modelsRes.json();
  const textMetrics = await metricsRes.json();
  const newsSamples = await samplesRes.json();
  const pcaParams = await pcaRes.json();

  return { textModels, textMetrics, newsSamples, pcaParams };
}

/**
 * Checks if the local FastAPI backend is online.
 */
export async function checkBackendHealth(): Promise<boolean> {
  try {
    const response = await fetch('http://127.0.0.1:8000/health');
    return response.ok;
  } catch {
    return false;
  }
}
