import { useEffect } from 'react';
import { useShallow } from 'zustand/react/shallow';
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
  } = useDashboardStore(
    useShallow((s) => ({
      selectedTextDataset: s.selectedTextDataset,
      textMetrics: s.textMetrics,
      textModels: s.textModels,
      loadingText: s.loadingText,
      errorText: s.errorText,
      loadTextData: s.loadTextData,
    }))
  );

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
    <div className="grow flex-shrink-0 p-4 sm:p-6 pb-8 grid grid-cols-1 lg:grid-cols-12 gap-6 items-start relative z-10">
      <h2 className="sr-only">Análise e Clusterização Semântica de Textos</h2>

      {/* Left Side: Hex grid comparison */}
      <section className="lg:col-span-7 flex flex-col min-w-0">
        <TextHexGrid />
      </section>

      {/* Right Side: Interactive classifier & explanations */}
      <aside className="lg:col-span-5 flex flex-col space-y-6 min-w-0">
        <ClassifierPanel />
        <SOMParamControls />

        {/* Dynamic Comparison Narrative Card */}
        <div className="glass-panel rounded-2xl p-5 border border-tokyo-border border-opacity-35">
          <div className="flex items-center gap-2 mb-2 text-tokyo-green font-mono font-bold text-xs">
            <CheckCircle size={16} />
            <span>{sbertBetter ? "Como o Sentence-BERT se compara ao TF-IDF?" : "Como o TF-IDF se compara ao Sentence-BERT?"}</span>
          </div>
          <p className="text-2xs text-tokyo-textDim leading-relaxed">
            Nos nossos experimentos de clusterização, o SOM treinado com <strong className="text-tokyo-blue">Sentence-BERT (SBERT)</strong> obteve um índice de concordância externa Rand Ajustado (ARI) de <strong className="text-tokyo-cyan font-mono">{sbertAriStr}</strong> contra <strong className="text-tokyo-orange font-mono">{tfidfAriStr}</strong> do TF-IDF no dataset selecionado.
          </p>
          <p className="text-2xs text-tokyo-textDim leading-relaxed mt-2">
            O <strong className="text-tokyo-blue">Sentence-BERT</strong> superou o TF-IDF porque projeta as frases em um espaço latente de alta dimensionalidade governado pelo significado semântico/contextual em vez de simples contagem vocabular, permitindo que o SOM agrupe conceitos semanticamente afins na malha.
          </p>
        </div>
      </aside>
    </div>
  );
}
