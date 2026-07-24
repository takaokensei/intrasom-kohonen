import type { TextModel } from '../store/useDashboardStore';

export interface BMUResult {
  bestNeuronId: number;
  dominantClass: string;
  purity: number;
  score: number;
}

/**
 * Projects a 384D SBERT embedding to 20D using PCA parameters and finds the best matching unit (BMU) on the SOM.
 * Works for both HEX_toroid (IntraSOM, normalization='var') and RECT_planar (MiniSom, StandardScaler Z-score)
 * models — the caller resolves the correct model via getActiveTextModel() in the store.
 *
 * NOTE on maxExpectedDist calibration:
 *   - HEX_toroid (IntraSOM, normalization='var'): calibrated at 6.0 (empirical baseline).
 *   - RECT_planar (MiniSom, StandardScaler): calibrated at 4.0 — lower because Z-score
 *     normalization compresses the feature space variance relative to IntraSOM's var norm.
 *
 * ESPELHADO: A lógica de confiança do api.py usa avg_dist como denominador (escala relativa,
 * auto-adaptável), enquanto este arquivo usa maxExpectedDist (escala absoluta). São abordagens
 * distintas — api.py não precisou de segundo valor, mas as constantes abaixo devem ser
 * mantidas em sincronia se a calibração for revisada após mais testes manuais.
 */
export function projectAndFindBMU(
  embedding: number[],
  pcaParams: { mean: number[]; components: number[][] },
  model: TextModel,
  isRectModel: boolean = false
): BMUResult {
  // Apply PCA projection: Q_20 = (emb - mean) * components
  const vec_20 = new Array(20).fill(0);
  for (let i = 0; i < 20; i++) {
    let sum = 0;
    for (let j = 0; j < 384; j++) {
      sum += (embedding[j] - pcaParams.mean[j]) * pcaParams.components[i][j];
    }
    vec_20[i] = sum;
  }
  
  // Find BMU in model neurons using Euclidean distance
  let bestNeuronId = 1;
  let minDistance = Infinity;
  let bestNeuron = model.neurons[0];
  
  model.neurons.forEach(neuron => {
    if (neuron.codebook && neuron.codebook.length === 20) {
      let sumSq = 0;
      for (let i = 0; i < 20; i++) {
        const diff = vec_20[i] - neuron.codebook[i];
        sumSq += diff * diff;
      }
      const dist = Math.sqrt(sumSq);
      if (dist < minDistance) {
        minDistance = dist;
        bestNeuronId = neuron.id;
        bestNeuron = neuron;
      }
    }
  });
  
  // Calculate score (confidence) using calibrated maxExpectedDist per model type.
  // RECT_planar uses StandardScaler Z-score; HEX_toroid uses IntraSOM normalization='var'.
  const maxExpectedDist = isRectModel ? 4.0 : 6.0;
  const confidence = Math.max(0, Math.min(100, Math.round((1.0 - (minDistance / maxExpectedDist)) * 100)));
  
  // ESPELHADO: Esta fórmula de confiança está espelhada no api.py do backend.
  // Qualquer alteração na lógica de confiança deve ser replicada manualmente nos dois arquivos.
  const adjustedConfidence = confidence > 0 ? Math.round(50 + (confidence / 2)) : 0;
  const finalScore = minDistance < 1.0 ? Math.max(adjustedConfidence, 95) : adjustedConfidence;
  
  return {
    bestNeuronId,
    dominantClass: bestNeuron.dominant_class,
    purity: bestNeuron.purity,
    score: finalScore
  };
}
