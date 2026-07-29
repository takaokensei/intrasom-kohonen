import { describe, expect, it } from 'vitest';
import type { UMatrixTorusNeuron } from '../components/UMatrixTorus';

describe('UMatrixTorus data structure & parametrization', () => {
  const dummyNeurons: UMatrixTorusNeuron[] = [
    {
      id: 1,
      row: 0,
      col: 0,
      umatrix_value: 0.1,
      total_samples: 5,
      dominant_class: 'Normal',
      purity: 0.8
    },
    {
      id: 2,
      row: 0,
      col: 1,
      umatrix_value: 0.5,
      total_samples: 3,
      dominant_class: 'Cyclic Shift',
      purity: 0.9
    },
    {
      id: 3,
      row: 1,
      col: 0,
      umatrix_value: 0.2,
      total_samples: 2,
      dominant_class: 'Normal',
      purity: 0.7
    },
    {
      id: 4,
      row: 1,
      col: 1,
      umatrix_value: 0.8,
      total_samples: 4,
      dominant_class: 'Upward Shift',
      purity: 1.0
    }
  ];

  it('validates mock neurons data structure for 3D Torus mesh parametrization', () => {
    expect(dummyNeurons).toHaveLength(4);
    const minVal = Math.min(...dummyNeurons.map(n => n.umatrix_value));
    const maxVal = Math.max(...dummyNeurons.map(n => n.umatrix_value));
    expect(minVal).toBe(0.1);
    expect(maxVal).toBe(0.8);
  });
});
