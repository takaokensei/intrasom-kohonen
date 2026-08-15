import { projectAndFindBMU, type PCAParameters } from './pca';
import { classifyByKeywords } from './keywordClassifier';
import type {
  TextModel,
  NewsSample,
} from '../store/useDashboardStore';

export interface ClassificationResult {
  bmu: number;
  dominantClass: string;
  purity: number;
  score: number;
  source: 'local' | 'cloud' | 'fallback';
}

export interface ClassifyParams {
  text: string;
  representation: 'SBERT' | 'TF-IDF';
  dataset: '20news' | '6class';
  lattice: 'HEX' | 'RECT';
  model: TextModel | null;
  pcaParams: PCAParameters | null;
  newsSamples: NewsSample[];
  hfToken?: string;
  hfProxyUrl?: string;
  localBackendUrl?: string;
}

export interface ClassificationOutcome {
  classificationResult: ClassificationResult | null;
  backendOnline: boolean;
}

/**
 * Pure classification function covering the 3-tier fallback strategy:
 * 1. Local FastAPI backend (`/classify`)
 * 2. Cloud Hugging Face Inference API (`/api/hf-sbert` proxy for SBERT)
 * 3. Client-side heuristic keyword BMU matching
 */
export async function classifyTextPure(
  params: ClassifyParams
): Promise<ClassificationOutcome> {
  const {
    text,
    representation,
    dataset,
    lattice,
    model,
    pcaParams,
    newsSamples,
    hfToken,
    hfProxyUrl = '/api/hf-sbert',
    localBackendUrl = 'http://127.0.0.1:8000/classify',
  } = params;

  if (!text || !text.trim()) {
    return {
      classificationResult: null,
      backendOnline: false,
    };
  }

  // 1. Try local FastAPI backend
  try {
    const response = await fetch(localBackendUrl, {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({
        text,
        representation,
        dataset,
        lattice,
      }),
    });

    if (response.ok) {
      const result = await response.json();
      return {
        classificationResult: {
          bmu: result.bmu,
          dominantClass: result.dominantClass,
          purity: result.purity,
          score: result.score,
          source: 'local',
        },
        backendOnline: true,
      };
    }
  } catch {
    // Local backend unavailable, fall through to cloud / heuristics
  }

  // 2. Try Hugging Face Inference API for SBERT
  if (representation === 'SBERT') {
    try {
      const headers: Record<string, string> = {
        'Content-Type': 'application/json',
      };

      if (hfToken) {
        headers['Authorization'] = `Bearer ${hfToken}`;
      }

      const hfResponse = await fetch(hfProxyUrl, {
        method: 'POST',
        headers,
        body: JSON.stringify({ inputs: text }),
      });

      if (hfResponse.ok) {
        const emb = await hfResponse.json();
        if (Array.isArray(emb) && emb.length === 384 && pcaParams && model) {
          const bmuResult = projectAndFindBMU(
            emb,
            pcaParams,
            model,
            lattice === 'RECT'
          );
          return {
            classificationResult: {
              bmu: bmuResult.bestNeuronId,
              dominantClass: bmuResult.dominantClass,
              purity: bmuResult.purity,
              score: bmuResult.score,
              source: 'cloud',
            },
            backendOnline: false,
          };
        }
      }
    } catch (hfErr) {
      console.error('Hugging Face Inference API failed:', hfErr);
    }
  }

  // 3. Client-side heuristic keyword fallback matching
  if (!model) {
    return {
      classificationResult: null,
      backendOnline: false,
    };
  }

  const fallbackResult = classifyByKeywords(text, newsSamples, model);
  return {
    classificationResult: {
      bmu: fallbackResult.bmu,
      dominantClass: fallbackResult.dominantClass,
      purity: fallbackResult.purity,
      score: fallbackResult.score,
      source: 'fallback',
    },
    backendOnline: false,
  };
}
