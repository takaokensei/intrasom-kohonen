import { useEffect } from 'react';
import { useDashboardStore } from '../store/useDashboardStore';
import { HexGrid } from '../components/HexGrid';
import { TimeSeriesPlot } from '../components/TimeSeriesPlot';
import { MetricTable } from '../components/MetricTable';
import { RadarChart } from '../components/RadarChart';
import { Cpu, Info } from 'lucide-react';
import { ErrorState } from '../components/ErrorState';
import { SOMParamControls } from '../components/SOMParamControls';

export function SyntheticScreen() {
  const {
    selectedMapSize,
    selectedNeuronId,
    setSelectedNeuronId,
    somModels,
    loadingSynthetic,
    errorSynthetic,
    loadSyntheticData,
  } = useDashboardStore();

  useEffect(() => {
    if (Object.keys(somModels).length === 0 && !loadingSynthetic && !errorSynthetic) {
      loadSyntheticData().catch(err => {
        console.error("Mount loading of synthetic data failed:", err);
      });
    }
  }, [somModels, loadingSynthetic, errorSynthetic, loadSyntheticData]);

  if (errorSynthetic) {
    return <ErrorState message={errorSynthetic} onRetry={loadSyntheticData} />;
  }

  const model = somModels[selectedMapSize];
  const selectedNeuron = selectedNeuronId && model
    ? model.neurons.find(n => n.id === selectedNeuronId)
    : null;

  return (
    <main className="grow flex-shrink-0 p-6 pb-8 grid grid-cols-1 lg:grid-cols-12 gap-6 items-start relative z-10">
      {/* Left Area - HexGrid Map & Metrics */}
      <section className="lg:col-span-8 flex flex-col space-y-6 min-w-0">
        <div className="grid grid-cols-1 md:grid-cols-2 gap-6 items-start">
          <HexGrid />
          <TimeSeriesPlot />
        </div>
        
        <div className="grid grid-cols-1 md:grid-cols-12 gap-6 items-start">
          <div className="md:col-span-7">
            <MetricTable />
          </div>
          <div className="md:col-span-5">
            <RadarChart />
          </div>
        </div>
      </section>

      {/* Right Area - Config & Selection Details */}
      <aside className="lg:col-span-4 flex flex-col space-y-6 min-w-0 self-start">
        {/* SOM Model Configurations */}
        <SOMParamControls />

        {/* Selection Details Panel */}
        <div className="glass-panel rounded-2xl p-5 flex flex-col">
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

              <div className="bg-tokyo-panel bg-opacity-20 border border-tokyo-border border-opacity-20 rounded-lg p-3">
                <span className="text-[9px] text-[#9aa5ce] font-semibold uppercase font-mono block mb-1.5">
                  Amostras Mapeadas ({selectedNeuron.total_samples})
                </span>
                <div className="flex flex-wrap gap-1">
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
  );
}
