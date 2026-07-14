import { useEffect } from 'react';
import { useDashboardStore } from './store/useDashboardStore';
import { HexGrid } from './components/HexGrid';
import { TimeSeriesPlot } from './components/TimeSeriesPlot';
import { MetricTable } from './components/MetricTable';
import { RadarChart } from './components/RadarChart';
import { TextHexGrid } from './components/TextHexGrid';
import { ClassifierPanel } from './components/ClassifierPanel';
import { Brain, Settings, FileText, LineChart, Cpu, Info, CheckCircle } from 'lucide-react';

function App() {
  const {
    activeTab,
    setActiveTab,
    selectedMapSize,
    setSelectedMapSize,
    selectedNeuronId,
    setSelectedNeuronId,
    somModels,
    loadSyntheticData,
    loadingSynthetic
  } = useDashboardStore();

  // Load initial data on mount
  useEffect(() => {
    loadSyntheticData();
  }, [loadSyntheticData]);

  const model = somModels[selectedMapSize];
  const selectedNeuron = selectedNeuronId && model
    ? model.neurons.find(n => n.id === selectedNeuronId)
    : null;

  return (
    <div className="min-h-screen bg-[#1a1b26] text-[#a9b1d6] flex flex-col relative overflow-x-hidden overflow-y-auto font-sans">
      {/* Decorative neon ambient glows */}
      <div className="glow-spot-blue -top-20 -left-20" />
      <div className="glow-spot-purple bottom-10 right-10" />

      {/* Header bar */}
      <header className="px-6 py-4 border-b border-tokyo-border bg-tokyo-dark bg-opacity-75 backdrop-blur-md z-10 flex flex-col md:flex-row justify-between items-center gap-4">
        <div className="flex items-center space-x-3.5">
          <div className="p-2.5 bg-tokyo-blue bg-opacity-10 rounded-xl border border-tokyo-blue border-opacity-25 text-tokyo-blue">
            <Brain size={26} className="animate-float" />
          </div>
          <div>
            <h1 className="text-lg font-bold text-tokyo-text flex items-center gap-2">
              IntraSOM Kohonen Maps Analyzer
            </h1>
            <p className="text-[10px] text-[#9aa5ce] font-semibold font-mono tracking-wide uppercase">
              Projeto de NLP & Séries Temporais — Cauã Vitor (UFRN) — Prof. José Alfredo F. Costa
            </p>
          </div>
        </div>

        {/* Tab Selector */}
        <div className="flex bg-tokyo-dark bg-opacity-80 p-1 rounded-xl border border-tokyo-border z-10">
          <button
            onClick={() => setActiveTab('synthetic')}
            className={`flex items-center gap-2 px-4 py-2 rounded-lg text-xs font-semibold transition-all ${
              activeTab === 'synthetic'
                ? 'bg-tokyo-blue text-tokyo-bg shadow-lg'
                : 'text-[#9aa5ce] hover:text-tokyo-text'
            }`}
          >
            <LineChart size={14} />
            Synthetic Control
          </button>
          <button
            onClick={() => setActiveTab('text')}
            className={`flex items-center gap-2 px-4 py-2 rounded-lg text-xs font-semibold transition-all ${
              activeTab === 'text'
                ? 'bg-tokyo-blue text-tokyo-bg shadow-lg'
                : 'text-[#9aa5ce] hover:text-tokyo-text'
            }`}
          >
            <FileText size={14} />
            Clusterização de Textos
          </button>
        </div>
      </header>

      {/* Main content grid */}
      {activeTab === 'synthetic' ? (
        <main className="flex-1 p-6 grid grid-cols-1 lg:grid-cols-12 gap-6 relative z-10">
          {/* Left Area - HexGrid Map & Metrics */}
          <section className="lg:col-span-8 flex flex-col space-y-6 min-w-0">
            <div className="grid grid-cols-1 md:grid-cols-2 gap-6 flex-1 min-h-[350px]">
              <HexGrid />
              <TimeSeriesPlot />
            </div>
            
            <div className="grid grid-cols-1 md:grid-cols-12 gap-6 h-80">
              <div className="md:col-span-7 h-full">
                <MetricTable />
              </div>
              <div className="md:col-span-5 h-full">
                <RadarChart />
              </div>
            </div>
          </section>

          {/* Right Area - Config & Selection Details */}
          <aside className="lg:col-span-4 flex flex-col space-y-6 h-full min-w-0">
            {/* SOM Model Configurations */}
            <div className="glass-panel rounded-2xl p-5 flex flex-col">
              <h3 className="text-sm font-bold text-tokyo-text mb-4 uppercase font-mono tracking-wider flex items-center gap-1.5">
                <Settings size={15} className="text-tokyo-blue" />
                Parâmetros do Mapa SOM
              </h3>
              
              <div className="space-y-4">
                {/* Size Selector */}
                <div className="flex flex-col space-y-1.5">
                  <label className="text-[10px] text-[#9aa5ce] font-semibold uppercase font-mono tracking-wider">Dimensões da Grade (cols x rows)</label>
                  <div className="grid grid-cols-3 gap-2">
                    {(['10x10', '15x15', '20x20'] as const).map(size => (
                      <button
                        key={size}
                        onClick={() => setSelectedMapSize(size)}
                        className={`py-2 rounded-lg text-xs font-mono font-bold border transition ${
                          selectedMapSize === size
                            ? 'bg-tokyo-blue text-tokyo-bg border-tokyo-blue'
                            : 'bg-tokyo-panel text-tokyo-text border-tokyo-border hover:bg-opacity-80'
                        }`}
                      >
                        {size}
                      </button>
                    ))}
                  </div>
                </div>

                {/* Constant params display */}
                <div className="grid grid-cols-2 gap-2 text-[10px] bg-tokyo-dark bg-opacity-40 p-3 rounded-lg border border-tokyo-border border-opacity-20 font-mono text-[#9aa5ce] leading-relaxed">
                  <div>Topologia: <span className="text-tokyo-text">Toroide (Hexa)</span></div>
                  <div>Vizinhaça: <span className="text-tokyo-text">Gaussiana</span></div>
                  <div>Inicialização: <span className="text-tokyo-text">Aleatória</span></div>
                  <div>Normalização: <span className="text-tokyo-text">Z-Score (var)</span></div>
                </div>
              </div>
            </div>

            {/* Selection Details Panel */}
            <div className="glass-panel rounded-2xl p-5 flex-1 flex flex-col min-h-[220px]">
              <h3 className="text-sm font-bold text-tokyo-text mb-4 uppercase font-mono tracking-wider flex items-center gap-1.5">
                <Cpu size={15} className="text-tokyo-magenta" />
                Detalhes do Neurônio
              </h3>
              
              {loadingSynthetic ? (
                <div className="flex-1 flex flex-col items-center justify-center text-center p-6 text-[#9aa5ce]">
                  <div className="animate-spin rounded-full h-6 w-6 border-b-2 border-tokyo-blue mb-2"></div>
                  <span className="text-xs font-mono">Carregando dados...</span>
                </div>
              ) : selectedNeuron ? (
                <div className="flex-1 flex flex-col space-y-3.5 text-xs">
                  <div className="grid grid-cols-2 gap-3">
                    <div className="bg-tokyo-panel bg-opacity-40 p-2.5 rounded-lg border border-tokyo-border border-opacity-25">
                      <span className="text-[9px] text-[#9aa5ce] font-semibold uppercase font-mono block">ID do Neurônio</span>
                      <span className="text-sm font-bold text-tokyo-text">N{selectedNeuron.id}</span>
                      <span className="text-[10px] text-[#9aa5ce] font-mono block">Col: {selectedNeuron.col}, Row: {selectedNeuron.row}</span>
                    </div>
                    
                    <div className="bg-tokyo-panel bg-opacity-40 p-2.5 rounded-lg border border-tokyo-border border-opacity-25">
                      <span className="text-[9px] text-[#9aa5ce] font-semibold uppercase font-mono block">Distância U-Matrix</span>
                      <span className="text-sm font-bold text-tokyo-magenta font-mono">{selectedNeuron.umatrix_value.toFixed(4)}</span>
                    </div>
                  </div>

                  <div className="bg-tokyo-panel bg-opacity-40 p-3 rounded-lg border border-tokyo-border border-opacity-25">
                    <span className="text-[9px] text-[#9aa5ce] font-semibold uppercase font-mono block">Análise Pós-Treinamento</span>
                    <div className="flex justify-between items-center mt-1">
                      <div>
                        <span className="text-[10px] text-[#9aa5ce] block">Classe Dominante</span>
                        <span className="font-bold text-tokyo-blue">{selectedNeuron.dominant_class}</span>
                      </div>
                      <div className="text-right">
                        <span className="text-[10px] text-[#9aa5ce] block">Pureza Local</span>
                        <span className="font-bold text-tokyo-green">{(selectedNeuron.purity * 100).toFixed(0)}%</span>
                      </div>
                    </div>
                  </div>

                  <div className="flex-1 flex flex-col min-h-0 bg-tokyo-panel bg-opacity-20 border border-tokyo-border border-opacity-20 rounded-lg p-3">
                    <span className="text-[9px] text-[#9aa5ce] font-semibold uppercase font-mono block mb-1.5">
                      Amostras Mapeadas ({selectedNeuron.total_samples})
                    </span>
                    <div className="flex-1 overflow-y-auto pr-1 flex flex-wrap gap-1 content-start">
                      {selectedNeuron.sample_ids.length > 0 ? (
                        selectedNeuron.sample_ids.map(id => (
                          <span 
                            key={id} 
                            className="px-1.5 py-0.5 bg-tokyo-dark bg-opacity-65 rounded border border-tokyo-border text-[9px] font-mono hover:text-white transition-colors"
                          >
                            #{id}
                          </span>
                        ))
                      ) : (
                        <span className="text-[10px] text-[#9aa5ce] italic font-mono">Nenhuma amostra mapeada neste neurônio</span>
                      )}
                    </div>
                  </div>
                  
                  <button
                    onClick={() => setSelectedNeuronId(null)}
                    className="w-full py-2 bg-tokyo-panel hover:bg-opacity-80 border border-tokyo-border text-tokyo-text font-semibold rounded-lg text-xs transition"
                  >
                    Limpar Seleção
                  </button>
                </div>
              ) : (
                <div className="flex-1 flex flex-col items-center justify-center text-center p-6 text-[#9aa5ce]">
                  <Info size={32} className="mb-2 text-tokyo-border" />
                  <p className="text-xs leading-relaxed max-w-[200px]">
                    Clique em qualquer hexágono do mapa para visualizar o perfil detalhado, os sinais temporais correspondentes e os pesos sinápticos do neurônio.
                  </p>
                </div>
              )}
            </div>
          </aside>
        </main>
      ) : (
        <main className="flex-1 p-6 grid grid-cols-1 lg:grid-cols-12 gap-6 relative z-10">
          {/* Left Side: Hex grid comparison */}
          <section className="lg:col-span-7 flex flex-col min-w-0">
            <TextHexGrid />
          </section>

          {/* Right Side: Interactive classifier & explanations */}
          <aside className="lg:col-span-5 flex flex-col space-y-6 h-full min-w-0">
            <ClassifierPanel />
            
            {/* Scientific Explanation Panel */}
            <div className="glass-panel rounded-2xl p-5 flex flex-col text-xs leading-relaxed text-tokyo-text">
              <h4 className="font-bold text-tokyo-magenta uppercase font-mono tracking-wider mb-3 flex items-center gap-1.5">
                <CheckCircle size={14} className="text-tokyo-green" />
                Por que o Sentence-BERT superou o TF-IDF?
              </h4>
              
              <div className="space-y-2.5 text-tokyo-text text-opacity-90">
                <p>
                  Nos nossos experimentos de clusterização, o SOM treinado com <strong>Sentence-BERT (SBERT)</strong> obteve um ganho considerável nas métricas quantitativas (ARI de <strong>0.2738</strong> contra <strong>0.1973</strong> do TF-IDF).
                </p>
                <p>
                  O <strong>TF-IDF</strong> representa documentos baseando-se apenas na contagem exata de termos (Bag-of-Words). Se duas notícias falam do mesmo assunto (ex: Astronomia) mas usam sinônimos diferentes (ex: <em>"space"</em> e <em>"cosmos"</em>), o TF-IDF não detecta a similaridade física.
                </p>
                <p>
                  Já o <strong>Sentence-BERT</strong> projeta as frases em um espaço latente de alta dimensionalidade onde a proximidade é governada pelo <strong>significado contextual (semântica)</strong>. O SOM consegue organizar esse espaço de forma muito mais coesa, mantendo as categorias bem agrupadas e separadas na malha hexagonal.
                </p>
              </div>
            </div>
          </aside>
        </main>
      )}

      {/* Footer bar */}
      <footer className="px-6 py-3 bg-tokyo-dark bg-opacity-90 border-t border-tokyo-border text-[9.5px] text-[#9aa5ce] font-semibold flex justify-between items-center z-10">
        <span>Base de Dados: 600 séries temporais (Synthetic Control) | 400 notícias (20 Newsgroups)</span>
        <div className="flex space-x-4">
          <span>USP IntraSOM Library Integration</span>
          <span>Tokyo Night Design System</span>
          <span>Vite + React + TS v19.2</span>
        </div>
      </footer>
    </div>
  );
}

export default App;
