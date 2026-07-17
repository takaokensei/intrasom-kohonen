import { describe, expect, it } from 'vitest';
import { classifyByKeywords } from '../lib/keywordClassifier';
import type { NewsSample, TextModel } from '../store/useDashboardStore';

describe('classifyByKeywords', () => {
  const dummySamples: NewsSample[] = [
    { id: 101, text: "O jogador fez um gol sensacional no campeonato", class: "Esportes" },
    { id: 102, text: "A inflação subiu e a bolsa de valores caiu", class: "Economia" }
  ];

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
        dominant_class: "Esportes",
        purity: 1.0,
        doc_indices: [101]
      },
      {
        id: 2,
        row: 0,
        col: 1,
        x: 1,
        y: 0,
        umatrix_value: 0.15,
        total_samples: 1,
        dominant_class: "Economia",
        purity: 1.0,
        doc_indices: [102]
      }
    ]
  };

  it('should match a sample via token overlap', () => {
    const res = classifyByKeywords("campeonato jogador gol", dummySamples, dummyModel);
    expect(res.dominantClass).toBe("Esportes");
    expect(res.bmu).toBe(1);
    expect(res.score).toBeGreaterThan(80);
  });

  it('should fall back to keyword heuristic if no exact sample overlap is found', () => {
    const res = classifyByKeywords("Temos novidades sobre a inflação e a economia do país", [], dummyModel);
    expect(res.dominantClass).toBe("Economia");
    expect(res.score).toBeGreaterThan(50);
  });

  it('should return Desconhecido if no keywords or samples match', () => {
    const res = classifyByKeywords("banana maca abacaxi", [], dummyModel);
    expect(res.dominantClass).toBe("Desconhecido");
    expect(res.bmu).toBe(0);
    expect(res.score).toBe(0);
  });
});
