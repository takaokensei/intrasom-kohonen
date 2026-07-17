import { describe, expect, it } from 'vitest';
import { projectAndFindBMU } from '../lib/pca';
import type { TextModel } from '../store/useDashboardStore';

describe('projectAndFindBMU', () => {
  const dummyEmbedding = new Array(384).fill(0.1);
  const dummyPcaParams = {
    mean: new Array(384).fill(0.05),
    components: new Array(20).fill(null).map(() => new Array(384).fill(0.01))
  };
  
  const dummyModel: TextModel = {
    cols: 5,
    rows: 5,
    neurons: [
      {
        id: 1,
        row: 0,
        col: 0,
        x: 0,
        y: 0,
        umatrix_value: 0.1,
        total_samples: 1,
        dominant_class: "Politica",
        purity: 0.9,
        doc_indices: [],
        codebook: new Array(20).fill(0.1) // Closest
      },
      {
        id: 2,
        row: 0,
        col: 1,
        x: 1,
        y: 0,
        umatrix_value: 0.2,
        total_samples: 1,
        dominant_class: "Variedades",
        purity: 0.8,
        doc_indices: [],
        codebook: new Array(20).fill(5.0) // Far
      }
    ]
  };

  it('should project embedding and return closest BMU neuron details', () => {
    // Expected projection vector element calculation:
    // (0.1 - 0.05) * 384 * 0.01 = 0.05 * 3.84 = 0.192
    // Diff to neuron 1 codebook elements: 0.192 - 0.1 = 0.092
    // Dist to neuron 1: sqrt(20 * 0.092^2) = sqrt(20 * 0.008464) ≈ 0.41
    const res = projectAndFindBMU(dummyEmbedding, dummyPcaParams, dummyModel);
    expect(res.bestNeuronId).toBe(1);
    expect(res.dominantClass).toBe("Politica");
    expect(res.purity).toBe(0.9);
    expect(res.score).toBeGreaterThanOrEqual(95); // dist < 1.0 triggers floor score of 95
  });
});
