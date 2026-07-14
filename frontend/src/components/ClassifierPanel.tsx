import { useDashboardStore } from '../store/useDashboardStore';
import { Sparkles, BrainCircuit, RefreshCw } from 'lucide-react';

const NEWS_COLORS: Record<string, string> = {
  "Graphics": "text-[#3182bd]",
  "Space": "text-[#31a354]",
  "Baseball": "text-[#e6550d]",
  "Mideast": "text-[#756bb1]"
};

const NEWS_BG_COLORS: Record<string, string> = {
  "Graphics": "bg-[#3182bd] bg-opacity-10 border-[#3182bd] border-opacity-30",
  "Space": "bg-[#31a354] bg-opacity-10 border-[#31a354] border-opacity-30",
  "Baseball": "bg-[#e6550d] bg-opacity-10 border-[#e6550d] border-opacity-30",
  "Mideast": "bg-[#756bb1] bg-opacity-10 border-[#756bb1] border-opacity-30"
};

export function ClassifierPanel() {
  const {
    selectedTextRep,
    setSelectedTextRep,
    customTextQuery,
    setCustomTextQuery,
    classificationResult,
    classifyText,
    resetClassification,
    newsSamples,
    textMetrics,
    loadingText
  } = useDashboardStore();
  
  if (loadingText) {
    return (
      <div className="glass-panel rounded-2xl p-5 h-full flex flex-col justify-center items-center">
        <div className="animate-spin rounded-full h-8 w-8 border-b-2 border-tokyo-blue"></div>
        <span className="text-xs text-tokyo-muted mt-2 font-mono">Carregando classificador...</span>
      </div>
    );
  }

  const currentMetrics = textMetrics[selectedTextRep === 'SBERT' ? 'SBERT' : 'TF_IDF'] || { ARI: 0, NMI: 0 };

  return (
    <div className="glass-panel rounded-2xl p-5 flex flex-col h-full overflow-hidden">
      <div className="flex justify-between items-center mb-3">
        <h3 className="text-sm font-bold text-tokyo-text uppercase font-mono tracking-wider">
          Classificador de Textos em Tempo Real
        </h3>
        
        <select
          value={selectedTextRep}
          onChange={(e) => setSelectedTextRep(e.target.value as 'SBERT' | 'TF-IDF')}
          className="bg-tokyo-panel border border-tokyo-border rounded px-2 py-0.5 text-[10px] text-tokyo-text focus:outline-none focus:border-tokyo-blue font-mono"
        >
          <option value="SBERT">Sentence-BERT (Semântico)</option>
          <option value="TF-IDF">TF-IDF (Frequencial)</option>
        </select>
      </div>
      
      {/* Metrics Header */}
      <div className="grid grid-cols-2 gap-3 mb-4 p-2 bg-tokyo-dark bg-opacity-40 rounded-lg border border-tokyo-border border-opacity-20 text-[10px] font-mono text-tokyo-muted">
        <div>
          Desempenho Geral (ARI): <span className="text-tokyo-text font-bold">{currentMetrics.ARI ? (currentMetrics.ARI as number).toFixed(4) : '-'}</span>
        </div>
        <div>
          Informação Mútua (NMI): <span className="text-tokyo-text font-bold">{currentMetrics.NMI ? (currentMetrics.NMI as number).toFixed(4) : '-'}</span>
        </div>
      </div>
      
      {/* Text Area Input */}
      <div className="flex-1 flex flex-col space-y-3 min-h-0">
        <div className="relative flex-1 min-h-[90px]">
          <textarea
            value={customTextQuery}
            onChange={(e) => {
              setCustomTextQuery(e.target.value);
              classifyText(e.target.value);
            }}
            placeholder="Digite ou clique em uma das notícias de exemplo abaixo para ver em qual neurônio do Mapa de Kohonen ela se projeta..."
            className="w-full h-full bg-tokyo-dark bg-opacity-60 border border-tokyo-border rounded-xl p-3 text-xs text-tokyo-text placeholder-tokyo-muted focus:outline-none focus:border-tokyo-blue resize-none overflow-y-auto"
          />
          {customTextQuery && (
            <button
              onClick={resetClassification}
              className="absolute right-3.5 bottom-3 text-tokyo-muted hover:text-tokyo-red transition-colors"
              title="Limpar"
            >
              <RefreshCw size={14} />
            </button>
          )}
        </div>
        
        {/* News Samples Selectors */}
        <div className="h-[90px] overflow-y-auto border-t border-tokyo-border border-opacity-20 pt-2 flex flex-col space-y-1.5">
          <span className="text-[9px] text-tokyo-muted uppercase font-mono tracking-wider">Amostras do Dataset:</span>
          <div className="grid grid-cols-2 gap-2">
            {newsSamples.slice(0, 4).map((sample) => (
              <button
                key={sample.id}
                onClick={() => {
                  setCustomTextQuery(sample.text);
                  classifyText(sample.text);
                }}
                className="p-2 text-left bg-tokyo-panel bg-opacity-35 hover:bg-opacity-70 rounded-lg border border-tokyo-border border-opacity-30 transition flex flex-col text-[10px]"
              >
                <span className="font-bold text-tokyo-blue uppercase font-mono text-[8px]">
                  Categoria real: {sample.class}
                </span>
                <span className="text-tokyo-text truncate w-full mt-0.5">
                  {sample.text}
                </span>
              </button>
            ))}
          </div>
        </div>
        
        {/* Classification Result display */}
        {classificationResult ? (
          <div className={`p-3 rounded-xl border flex items-center justify-between transition-all ${NEWS_BG_COLORS[classificationResult.dominantClass]}`}>
            <div className="flex items-center space-x-3">
              <div className="p-2 bg-tokyo-dark bg-opacity-40 rounded-lg text-tokyo-blue">
                <BrainCircuit size={18} />
              </div>
              <div>
                <div className="text-[10px] text-tokyo-muted uppercase font-mono tracking-wider">SOM Categoria Predita</div>
                <div className={`text-sm font-bold flex items-center gap-1.5 ${NEWS_COLORS[classificationResult.dominantClass]}`}>
                  {classificationResult.dominantClass}
                  <span className="text-xs text-tokyo-text font-normal opacity-90 font-mono">
                    (Neurônio N{classificationResult.bmu})
                  </span>
                </div>
              </div>
            </div>
            
            <div className="text-right text-[10px] font-mono text-tokyo-muted">
              <div>Purity: <span className="text-tokyo-text font-bold">{(classificationResult.purity * 100).toFixed(0)}%</span></div>
              <div>Confiança: <span className="text-tokyo-text font-bold">{classificationResult.score}%</span></div>
            </div>
          </div>
        ) : (
          <div className="p-3 bg-tokyo-panel bg-opacity-20 border border-tokyo-border border-opacity-20 rounded-xl flex items-center justify-center text-[10px] text-tokyo-muted font-mono gap-1.5">
            <Sparkles size={14} className="text-tokyo-blue" />
            <span>Projete um texto para ver o neurônio vencedor (BMU) acender no mapa</span>
          </div>
        )}
      </div>
    </div>
  );
}
