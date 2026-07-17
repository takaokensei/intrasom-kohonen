import type { NewsSample, TextModel } from '../store/useDashboardStore';

export interface KeywordClassifierResult {
  bmu: number;
  dominantClass: string;
  purity: number;
  score: number;
}

const TURISMO_KEYWORDS = ["turismo", "viagem", "hotel", "voo", "praia", "turista", "passagem", "destino", "roteiro", "ferias", "pousada", "resort", "passeio", "agencia", "monumento", "atracoes", "viajar"];
const ESPORTES_KEYWORDS = ["esporte", "jogo", "time", "jogador", "campeonato", "partida", "futebol", "atleta", "copa", "estadio", "quadra", "campo", "vitoria", "derrota", "gol", "treino", "olimpiada"];
const POLICIA_KEYWORDS = ["policia", "crime", "preso", "delegado", "prisao", "roubo", "furto", "assalto", "assassinato", "drogas", "investigacao", "acusado", "suspeito", "vitima", "violencia", "mandado", "policial"];
const ECONOMIA_KEYWORDS = ["economia", "inflacao", "pib", "juros", "mercado", "dolar", "bolsa", "empresa", "crescimento", "setor", "reforma", "imposto", "banco", "receita", "investimento", "financas"];
const POLITICA_KEYWORDS = ["politica", "governo", "presidente", "senado", "deputado", "eleicao", "voto", "partido", "ministro", "projeto", "lei", "brasilia", "congresso", "câmara", "prefeito", "candidato"];
const VARIEDADES_KEYWORDS = ["variedades", "famosos", "novela", "filme", "cinema", "show", "musica", "artista", "celebridade", "teatro", "moda", "cultura", "evento", "estreia", "audiencia", "festival"];

/**
 * Fallback classification method based on news sample overlap or keyword matching.
 */
export function classifyByKeywords(
  text: string,
  samples: NewsSample[],
  model: TextModel
): KeywordClassifierResult {
  const queryLower = text.toLowerCase();
  
  let bestDocIdx = 0;
  let maxOverlap = 0;
  
  samples.forEach(doc => {
    const docTokens = new Set(doc.text.toLowerCase().split(/\W+/));
    let overlap = 0;
    docTokens.forEach(token => {
      if (token.length > 3 && queryLower.includes(token)) {
        overlap++;
      }
    });
    
    if (overlap > maxOverlap) {
      maxOverlap = overlap;
      bestDocIdx = doc.id;
    }
  });

  let targetNeuron = null;
  let score = 0;
  let dominantClass = "Desconhecido";
  let purity = 0;
  let bmu = 0;

  let turismoHits = 0;
  let esportesHits = 0;
  let policiaHits = 0;
  let economiaHits = 0;
  let politicaHits = 0;
  let variedadesHits = 0;

  const words = queryLower.split(/\W+/);
  words.forEach(word => {
    if (word.length > 2) {
      if (TURISMO_KEYWORDS.some(kw => word === kw || word.includes(kw))) turismoHits++;
      if (ESPORTES_KEYWORDS.some(kw => word === kw || word.includes(kw))) esportesHits++;
      if (POLICIA_KEYWORDS.some(kw => word === kw || word.includes(kw))) policiaHits++;
      if (ECONOMIA_KEYWORDS.some(kw => word === kw || word.includes(kw))) economiaHits++;
      if (POLITICA_KEYWORDS.some(kw => word === kw || word.includes(kw))) politicaHits++;
      if (VARIEDADES_KEYWORDS.some(kw => word === kw || word.includes(kw))) variedadesHits++;
    }
  });

  const totalHits = turismoHits + esportesHits + policiaHits + economiaHits + politicaHits + variedadesHits;

  if (maxOverlap > 0) {
    targetNeuron = model.neurons.find(n => n.doc_indices.includes(bestDocIdx));
    if (targetNeuron) {
      bmu = targetNeuron.id;
      dominantClass = targetNeuron.dominant_class;
      purity = targetNeuron.purity;
      score = Math.min(85 + maxOverlap * 2.5, 99);
    }
  } else if (totalHits > 0) {
    const maxHits = Math.max(turismoHits, esportesHits, policiaHits, economiaHits, politicaHits, variedadesHits);
    let predictedCat = "Turismo";
    if (maxHits === turismoHits) predictedCat = "Turismo";
    else if (maxHits === esportesHits) predictedCat = "Esportes";
    else if (maxHits === policiaHits) predictedCat = "Policia";
    else if (maxHits === economiaHits) predictedCat = "Economia";
    else if (maxHits === politicaHits) predictedCat = "Politica";
    else predictedCat = "Variedades";

    const categoryNeurons = model.neurons.filter(n => n.dominant_class === predictedCat);
    targetNeuron = categoryNeurons[Math.floor(Math.random() * categoryNeurons.length)] || model.neurons[0];
    
    bmu = targetNeuron.id;
    dominantClass = targetNeuron.dominant_class;
    purity = targetNeuron.purity;
    score = Math.min(60 + totalHits * 10, 85);
  } else {
    bmu = 0;
    dominantClass = "Desconhecido";
    purity = 0;
    score = 0;
  }

  return {
    bmu,
    dominantClass,
    purity,
    score
  };
}
