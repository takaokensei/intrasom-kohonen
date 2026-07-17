export interface ComparisonNarrative {
  sbertAriStr: string;
  tfidfAriStr: string;
  sbertBetter: boolean;
  explanation: string;
}

/**
 * Returns a narrative comparison and structured info based on ARI values of TF-IDF and SBERT.
 */
export function getComparisonNarrative(sbertAri: number | undefined, tfidfAri: number | undefined): ComparisonNarrative {
  const sbertVal = sbertAri ?? 0;
  const tfidfVal = tfidfAri ?? 0;
  
  return {
    sbertAriStr: sbertVal.toFixed(4),
    tfidfAriStr: tfidfVal.toFixed(4),
    sbertBetter: sbertVal >= tfidfVal,
    explanation: sbertVal >= tfidfVal
      ? "O Sentence-BERT superou o TF-IDF porque projeta as frases em um espaço latente de alta dimensionalidade governado pelo significado semântico/contextual em vez de simples contagem vocabular, permitindo que o SOM agrupe conceitos semanticamente afins na malha."
      : "O TF-IDF obteve desempenho superior neste cenário por conta de termos exclusivos muito bem definidos para cada classe, facilitando a separação espacial imediata pelo SOM mesmo sem contexto semântico."
  };
}
