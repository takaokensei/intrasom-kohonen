import { useRef, useEffect } from 'react';
import { useDashboardStore } from '../store/useDashboardStore';
import { Sparkles, BrainCircuit, RefreshCw } from 'lucide-react';

const NEWS_COLORS: Record<string, string> = {
  "Turismo": "text-[#3182bd]",
  "Esportes": "text-[#31a354]",
  "Policia": "text-[#e6550d]",
  "Economia": "text-[#756bb1]",
  "Politica": "text-[#e7ba52]",
  "Variedades": "text-[#d6616b]"
};

const NEWS_BG_COLORS: Record<string, string> = {
  "Turismo": "bg-[#3182bd] bg-opacity-10 border-[#3182bd] border-opacity-30",
  "Esportes": "bg-[#31a354] bg-opacity-10 border-[#31a354] border-opacity-30",
  "Policia": "bg-[#e6550d] bg-opacity-10 border-[#e6550d] border-opacity-30",
  "Economia": "bg-[#756bb1] bg-opacity-10 border-[#756bb1] border-opacity-30",
  "Politica": "bg-[#e7ba52] bg-opacity-10 border-[#e7ba52] border-opacity-30",
  "Variedades": "bg-[#d6616b] bg-opacity-10 border-[#d6616b] border-opacity-30"
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
    loadingText,
    backendOnline
  } = useDashboardStore();
  
  const debounceRef = useRef<ReturnType<typeof setTimeout> | null>(null);

  // Clear timeout on unmount
  useEffect(() => {
    return () => {
      if (debounceRef.current) {
        clearTimeout(debounceRef.current);
      }
    };
  }, []);

  const handleTextChange = (text: string) => {
    setCustomTextQuery(text);
    
    if (debounceRef.current) {
      clearTimeout(debounceRef.current);
    }
    
    debounceRef.current = setTimeout(() => {
      classifyText(text);
    }, 450); // 450ms debounce
  };

  const handleSampleClick = (text: string) => {
    if (debounceRef.current) {
      clearTimeout(debounceRef.current);
    }
    setCustomTextQuery(text);
    classifyText(text);
  };

  const handleReset = () => {
    if (debounceRef.current) {
      clearTimeout(debounceRef.current);
    }
    resetClassification();
  };

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
      {/* Header */}
      <div className="flex items-center justify-between mb-4">
        <h3 className="text-sm font-bold text-tokyo-text uppercase font-mono tracking-wider flex items-center gap-1.5">
          <Sparkles size={15} className="text-tokyo-magenta" />
          Classificador de Textos em Tempo Real
        </h3>
        
        <select
          value={selectedTextRep}
          onChange={(e) => setSelectedTextRep(e.target.value as 'SBERT' | 'TF-IDF')}
          className="bg-tokyo-dark border border-tokyo-border text-[10px] font-mono font-bold rounded p-1 text-tokyo-text focus:outline-none focus:border-tokyo-blue cursor-pointer"
        >
          <option value="SBERT">Sentence-BERT (Semântico)</option>
          <option value="TF-IDF">TF-IDF (Frequencial)</option>
        </select>
      </div>
      
      {/* Backend Status Indicator */}
      <div className="flex items-center justify-between mb-3 text-[9px] font-mono">
        <span className="text-tokyo-muted uppercase font-bold">Status da Inferência:</span>
        {backendOnline ? (
          <span className="flex items-center gap-1 text-tokyo-green font-bold bg-tokyo-green bg-opacity-10 px-1.5 py-0.5 rounded border border-tokyo-green border-opacity-35">
            <span className="w-1.5 h-1.5 rounded-full bg-tokyo-green animate-pulse" />
            Servidor Local: ATIVO (Real)
          </span>
        ) : selectedTextRep === 'SBERT' ? (
          <span className="flex items-center gap-1 text-tokyo-blue font-bold bg-tokyo-blue bg-opacity-10 px-1.5 py-0.5 rounded border border-tokyo-blue border-opacity-35" title="Utilizando API de inferência do Hugging Face com projeção PCA local. 100% fiel e sem backend necessário!">
            <span className="w-1.5 h-1.5 rounded-full bg-tokyo-blue animate-pulse" />
            Nuvem HF API: ATIVO (Real)
          </span>
        ) : (
          <span className="flex items-center gap-1 text-tokyo-orange font-bold bg-tokyo-orange bg-opacity-10 px-1.5 py-0.5 rounded border border-[#ff9e64] border-opacity-35" title="Rode 'python src/api.py' no terminal para ativar inferência real do TF-IDF. Usando busca por palavras-chave.">
            <span className="w-1.5 h-1.5 rounded-full bg-[#ff9e64]" />
            Inativo (Fallback Heurístico)
          </span>
        )}
      </div>

      {/* Metrics Header */}
      <div className="grid grid-cols-2 gap-3 mb-4 p-2 bg-tokyo-dark bg-opacity-40 rounded-lg border border-tokyo-border border-opacity-20 text-[10px] font-mono text-[#9aa5ce]">
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
            onChange={(e) => handleTextChange(e.target.value)}
            placeholder="Digite ou clique em uma das notícias de exemplo abaixo para ver em qual neurônio do Mapa de Kohonen ela se projeta..."
            className="w-full h-full bg-tokyo-dark bg-opacity-60 border border-tokyo-border rounded-xl p-3 text-xs text-tokyo-text placeholder-tokyo-text placeholder-opacity-50 focus:outline-none focus:border-tokyo-blue resize-none overflow-y-auto"
          />
          {customTextQuery && (
            <button
              onClick={handleReset}
              className="absolute right-3.5 bottom-3 text-[#9aa5ce] hover:text-tokyo-red transition-colors"
              title="Limpar"
            >
              <RefreshCw size={14} />
            </button>
          )}
        </div>
        
        {/* News Samples Dropdown Selector */}
        <div className="flex flex-col space-y-2 pt-2 border-t border-tokyo-border border-opacity-20">
          <div className="flex items-center justify-between">
            <span className="text-[9px] text-[#9aa5ce] uppercase font-mono tracking-wider font-semibold">Testar com Amostras do Dataset:</span>
            <button 
              onClick={() => {
                const randomIdx = Math.floor(Math.random() * newsSamples.length);
                handleSampleClick(newsSamples[randomIdx].text);
              }}
              className="text-[9px] text-tokyo-blue hover:text-tokyo-teal font-mono flex items-center gap-1 transition-colors"
              title="Testar uma notícia aleatória"
            >
              <RefreshCw size={10} />
              Aleatório
            </button>
          </div>
          
          <select
            onChange={(e) => {
              const val = e.target.value;
              if (val) handleSampleClick(val);
            }}
            className="w-full bg-tokyo-dark bg-opacity-50 border border-tokyo-border border-opacity-30 rounded-lg px-2.5 py-2 text-xs text-tokyo-text focus:outline-none focus:border-tokyo-blue cursor-pointer transition-colors"
            value={newsSamples.find(s => s.text === customTextQuery)?.text || ''}
          >
            <option value="" disabled>-- Selecione uma notícia para testar --</option>
            {newsSamples.map((sample) => (
              <option key={sample.id} value={sample.text} className="bg-tokyo-dark text-tokyo-text">
                [{sample.class}] {sample.text.substring(0, 80)}...
              </option>
            ))}
          </select>
        </div>
        
        {/* Classification Result display */}
        {classificationResult ? (
          <div className={`p-4 rounded-xl border flex flex-col space-y-3.5 transition-all ${
            classificationResult.dominantClass === "Desconhecido"
              ? "bg-tokyo-panel bg-opacity-35 border-tokyo-border border-opacity-30"
              : (NEWS_BG_COLORS[classificationResult.dominantClass] || "")
          }`}>
            {/* Top row: Label & Source Badge */}
            <div className="flex items-center justify-between">
              <div className="flex items-center space-x-2">
                <div className="p-1.5 bg-tokyo-dark bg-opacity-40 rounded-md text-tokyo-blue">
                  <BrainCircuit size={14} />
                </div>
                <span className="text-[10px] text-[#9aa5ce] uppercase font-mono tracking-wider font-semibold">
                  SOM Categoria Predita
                </span>
              </div>
              
              {classificationResult.source === 'local' && (
                <span className="text-[8px] bg-tokyo-green bg-opacity-20 text-tokyo-green px-1.5 py-0.5 rounded border border-tokyo-green border-opacity-30 font-bold uppercase tracking-normal">Local</span>
              )}
              {classificationResult.source === 'cloud' && (
                <span className="text-[8px] bg-tokyo-blue bg-opacity-20 text-tokyo-blue px-1.5 py-0.5 rounded border border-tokyo-blue border-opacity-30 font-bold uppercase tracking-normal">Nuvem HF</span>
              )}
              {classificationResult.source === 'fallback' && (
                <span className="text-[8px] bg-tokyo-panel text-[#9aa5ce] px-1.5 py-0.5 rounded border border-tokyo-border border-opacity-40 font-bold uppercase tracking-normal">Heurística</span>
              )}
            </div>

            {/* Middle Row: Big Class Name & Neuron ID */}
            <div className="flex items-baseline justify-between border-b border-tokyo-border border-opacity-15 pb-2">
              <span className={`text-base font-extrabold tracking-tight ${
                classificationResult.dominantClass === "Desconhecido"
                  ? "text-[#9aa5ce]"
                  : (NEWS_COLORS[classificationResult.dominantClass] || "")
              }`}>
                {classificationResult.dominantClass === "Desconhecido" ? "Não Identificado" : classificationResult.dominantClass}
              </span>
              
              <span className="text-[10px] bg-tokyo-dark bg-opacity-50 text-tokyo-text px-2 py-0.5 rounded font-mono font-semibold">
                {classificationResult.bmu > 0 
                  ? `Neurônio N${classificationResult.bmu}` 
                  : 'Sem Correspondência'}
              </span>
            </div>

            {/* Bottom Row: Metrics Grid with Progress Bars */}
            <div className="grid grid-cols-2 gap-4 text-[10px]">
              {classificationResult.dominantClass !== "Desconhecido" && (
                <div className="flex flex-col space-y-1">
                  <div className="flex justify-between text-[#9aa5ce] font-semibold">
                    <span>Pureza:</span>
                    <span className="text-tokyo-text font-mono font-bold">{(classificationResult.purity * 100).toFixed(0)}%</span>
                  </div>
                  <div className="w-full bg-tokyo-dark bg-opacity-40 rounded-full h-1">
                    <div 
                      className="bg-tokyo-blue h-1 rounded-full transition-all duration-500" 
                      style={{ width: `${classificationResult.purity * 100}%` }}
                    />
                  </div>
                </div>
              )}
              
              <div className="flex flex-col space-y-1">
                <div className="flex justify-between text-[#9aa5ce] font-semibold">
                  <span>Confiança:</span>
                  <span className="text-tokyo-text font-mono font-bold">{classificationResult.score}%</span>
                </div>
                <div className="w-full bg-tokyo-dark bg-opacity-40 rounded-full h-1">
                  <div 
                    className="bg-tokyo-teal h-1 rounded-full transition-all duration-500" 
                    style={{ width: `${classificationResult.score}%` }}
                  />
                </div>
              </div>
            </div>
          </div>
        ) : (
          <div className="p-3 bg-tokyo-panel bg-opacity-20 border border-tokyo-border border-opacity-20 rounded-xl flex items-center justify-center text-[10px] text-[#9aa5ce] font-mono gap-1.5">
            <Sparkles size={14} className="text-tokyo-blue" />
            <span>Projete um texto para ver o neurônio vencedor (BMU) acender no mapa</span>
          </div>
        )}
      </div>
    </div>
  );
}
