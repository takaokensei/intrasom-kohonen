import { describe, it, expect, vi, beforeEach, afterEach } from 'vitest';
import { classifyTextPure, type ClassifyParams } from '../lib/classification';
import type { PCAParameters } from '../lib/pca';
import type { TextModel, NewsSample } from '../store/useDashboardStore';

const dummyModel: TextModel = {
  cols: 2,
  rows: 2,
  neurons: [
    {
      id: 1,
      x: 0,
      y: 0,
      row: 0,
      col: 0,
      umatrix_value: 0.1,
      dominant_class: 'Graphics',
      purity: 0.9,
      total_samples: 10,
      doc_indices: [0],
      codebook: new Array(20).fill(0.1),
    },
    {
      id: 2,
      x: 1,
      y: 0,
      row: 0,
      col: 1,
      umatrix_value: 0.2,
      dominant_class: 'Space',
      purity: 0.85,
      total_samples: 8,
      doc_indices: [1],
      codebook: new Array(20).fill(0.5),
    },
  ],
  umatrix_edges: [],
  umatrix_edge_min: 0,
  umatrix_edge_max: 1,
};

const dummyPCA: PCAParameters = {
  mean: new Array(384).fill(0.01),
  components: Array.from({ length: 20 }, () => new Array(384).fill(0.05)),
};

const dummySamples: NewsSample[] = [
  {
    id: 0,
    text: 'computer graphics 3d rendering opengl shader polygon',
    class: 'Graphics',
  },
  {
    id: 1,
    text: 'nasa space rocket orbit mars satellite astronomy',
    class: 'Space',
  },
];

describe('F5: 3-Tier Text Classification Strategy', () => {
  beforeEach(() => {
    vi.restoreAllMocks();
  });

  afterEach(() => {
    vi.unstubAllGlobals();
  });

  it('returns null classification for empty or whitespace query', async () => {
    const params: ClassifyParams = {
      text: '   ',
      representation: 'SBERT',
      dataset: '20news',
      lattice: 'HEX',
      model: dummyModel,
      pcaParams: dummyPCA,
      newsSamples: dummySamples,
    };

    const outcome = await classifyTextPure(params);
    expect(outcome.classificationResult).toBeNull();
    expect(outcome.backendOnline).toBe(false);
  });

  it('Branch 1: uses Local FastAPI Backend when available', async () => {
    const mockLocalResponse = {
      bmu: 1,
      dominantClass: 'Graphics',
      purity: 0.95,
      score: 0.88,
    };

    vi.stubGlobal('fetch', vi.fn().mockImplementation((url: string) => {
      if (url.includes(':8000/classify')) {
        return Promise.resolve({
          ok: true,
          json: () => Promise.resolve(mockLocalResponse),
        });
      }
      return Promise.reject(new Error('Network error'));
    }));

    const params: ClassifyParams = {
      text: 'rendering computer graphics pixels',
      representation: 'SBERT',
      dataset: '20news',
      lattice: 'HEX',
      model: dummyModel,
      pcaParams: dummyPCA,
      newsSamples: dummySamples,
    };

    const outcome = await classifyTextPure(params);

    expect(outcome.backendOnline).toBe(true);
    expect(outcome.classificationResult).toEqual({
      bmu: 1,
      dominantClass: 'Graphics',
      purity: 0.95,
      score: 0.88,
      source: 'local',
    });
  });

  it('Branch 2: falls back to Hugging Face Cloud Inference API for SBERT when local backend is offline', async () => {
    const mockEmbedding = new Array(384).fill(0.05);

    vi.stubGlobal('fetch', vi.fn().mockImplementation((url: string) => {
      if (url.includes(':8000/classify')) {
        return Promise.reject(new Error('Connection refused'));
      }
      if (url.includes('/api/hf-sbert')) {
        return Promise.resolve({
          ok: true,
          json: () => Promise.resolve(mockEmbedding),
        });
      }
      return Promise.reject(new Error('Unknown url'));
    }));

    const params: ClassifyParams = {
      text: 'astronomy planetary orbit exploration',
      representation: 'SBERT',
      dataset: '20news',
      lattice: 'HEX',
      model: dummyModel,
      pcaParams: dummyPCA,
      newsSamples: dummySamples,
    };

    const outcome = await classifyTextPure(params);

    expect(outcome.backendOnline).toBe(false);
    expect(outcome.classificationResult).not.toBeNull();
    expect(outcome.classificationResult?.source).toBe('cloud');
    expect([1, 2]).toContain(outcome.classificationResult?.bmu);
  });

  it('Branch 3: falls back to client-side keyword heuristics when both local and cloud fail', async () => {
    vi.stubGlobal('fetch', vi.fn().mockImplementation(() => {
      return Promise.reject(new Error('Network unreachable'));
    }));

    const params: ClassifyParams = {
      text: 'nasa mars satellite exploration astronomy',
      representation: 'SBERT',
      dataset: '20news',
      lattice: 'HEX',
      model: dummyModel,
      pcaParams: dummyPCA,
      newsSamples: dummySamples,
    };

    const outcome = await classifyTextPure(params);

    expect(outcome.backendOnline).toBe(false);
    expect(outcome.classificationResult).not.toBeNull();
    expect(outcome.classificationResult?.source).toBe('fallback');
    expect(outcome.classificationResult?.dominantClass).toBe('Space');
    expect(outcome.classificationResult?.bmu).toBe(2);
  });
});
