import { useEffect } from 'react';
import { useDashboardStore } from '../store/useDashboardStore';
import { TextHexGrid } from '../components/TextHexGrid';
import { ClassifierPanel } from '../components/ClassifierPanel';
import { CheckCircle } from 'lucide-react';
import { getComparisonNarrative } from '../lib/metrics';
import { ErrorState } from '../components/ErrorState';
import { SOMParamControls } from '../components/SOMParamControls';

export function TextScreen() {
  const {
    selectedTextDataset,
    textMetrics,
    textModels,
    loadingText,
    errorText,
    loadTextData
  } = useDashboardStore();

  useEffect(() => {
    if (!textModels[selectedTextDataset] && !loadingText && !errorText) {
      loadTextData().catch(err => {
        console.error("Mount loading of text models failed:", err);
      });
    }
  }, [selectedTextDataset, textModels, loadingText, errorText, loadTextData]);

  if (errorText) {
    return <ErrorState message={errorText} onRetry={loadTextData} />;
  }

  const datasetMetrics = textMetrics[selectedTextDataset] || {};
  const sbertAriVal = datasetMetrics["SBERT"]?.ARI;
  const tfidfAriVal = datasetMetrics["TF_IDF"]?.ARI;
  
  const { sbertAriStr, tfidfAriStr, sbertBetter } = getComparisonNarrative(sbertAriVal, tfidfAriVal);

  return (
    <main className="grow flex-shrink-0 p-6 pb-8 grid grid-cols-1 lg:grid-cols-12 gap-6 items-start relative z-10">
      {/* Left Side: Hex grid comparison */}
      <section className="lg:col-span-7 flex flex-col min-w-0">
        <TextHexGrid />
      </section>

      {/* Right Side: Interactive classifier & explanations */}
      <aside className="lg:col-span-5 flex flex-col space-y-6 min-w-0">
        <ClassifierPanel />
        <SOMParamControls />
        
        {/* Scientific Explanation Panel */}
        <div className="glass-panel rounded-2xl p-5 flex flex-col text-xs leading-relaxed text-tokyo-text">
          <h4 className="font-bold text-tokyo-magenta uppercase font-mono tracking-wider mb-3 flex items-center gap-1.5">
            <CheckCircle size={14} className="text-tokyo-green" />
            Como o Sentence-BERT se compara ao TF-IDF?
          </h4>
          
          <div className="space-y-2.5 text-tokyo-text text-opacity-90">
            <p>
              Nos nossos experimentos de clusterização, o SOM treinado com <strong>Sentence-BERT (SBERT)</strong> obteve um índice de concordância externa Rand Ajustado (ARI) de <strong>{sbertAriStr}</strong> contra <strong>{tfidfAriStr}</strong> do TF-IDF no dataset selecionado.
            </p>
            <p>
              {sbertBetter ? (
                <span>
                  O <strong>Sentence-BERT</strong> superou o TF-IDF porque projeta as frases em um espaço latente de alta dimensionalidade governado pelo significado semântico/contextual em vez de simples contagem vocabular, permitindo que o SOM agrupe conceitos semanticamente afins na malha.
                </span>
              ) : (
                <span>
                  O <strong>TF-IDF</strong> obteve desempenho superior neste cenário por conta de termos exclusivos muito bem definidos para cada classe, facilitando a separação espacial imediata pelo SOM mesmo sem contexto semântico.
                </span>
              )}
            </p>
          </div>
        </div>
      </aside>
    </main>
  );
}
