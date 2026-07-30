import { useDashboardStore } from '../store/useDashboardStore';
import { matchParamStudyEntry } from '../lib/paramStudyMatch';
import { Settings, Sliders, Layers, RefreshCw, Filter, ArrowDownCircle } from 'lucide-react';

export function SOMParamControls() {
  const {
    lattice,
    topology,
    initialRadius,
    finalRadius,
    epochs,
    setLattice,
    setTopology,
    setInitialRadius,
    setFinalRadius,
    setEpochs,
    activeTab,
    selectedMapSize,
    setSelectedMapSize,
    getActiveSOMModel,
    paramStudyResults
  } = useDashboardStore();

  const isTextTab = activeTab === 'text';
  const activeSOMModel = getActiveSOMModel();

  // Match live selected parameters against pre-computed parameter study table
  const matchedStudyEntry = matchParamStudyEntry(paramStudyResults, initialRadius, finalRadius, epochs);

  return (
    <div className="glass-panel rounded-2xl p-5 flex flex-col space-y-5 text-tokyo-text">
      {/* Main Header */}
      <div className="flex items-center justify-between border-b border-tokyo-border border-opacity-30 pb-3">
        <h2 className="text-sm font-bold uppercase font-mono tracking-wider flex items-center gap-2 text-tokyo-blue">
          <Settings size={16} />
          Parâmetros do Algoritmo de Kohonen
        </h2>
        <span className="text-[10px] font-mono bg-tokyo-dark px-2 py-0.5 rounded border border-tokyo-border text-tokyo-magenta font-semibold">
          IntraSOM Config
        </span>
      </div>

      {/* SECTION 1: Active Model & Grid Routing */}
      <div className="flex flex-col space-y-3 bg-tokyo-dark bg-opacity-30 p-3.5 rounded-xl border border-tokyo-border border-opacity-30">
        <div className="flex items-center justify-between border-b border-tokyo-border border-opacity-20 pb-2">
          <span className="text-[11px] font-bold text-tokyo-cyan uppercase font-mono tracking-wider flex items-center gap-1.5">
            <Layers size={13} />
            1. Seleção do Modelo Ativo (Roteamento de Mapa)
          </span>
          <span className="text-[9px] font-mono bg-tokyo-blue bg-opacity-20 text-tokyo-blue px-1.5 py-0.5 rounded border border-tokyo-blue border-opacity-30 font-semibold">
            Carrega Mapa Pré-Treinado
          </span>
        </div>

        {/* Map Size Selector */}
        <div className="flex flex-col space-y-1.5">
          <label className="text-[10px] text-tokyo-muted font-semibold uppercase font-mono tracking-wider">
            Dimensões da Grade (Solicitado)
          </label>
          <div className="grid grid-cols-3 gap-2">
            {(['5x5', '7x7', '10x10', '12x12', '15x15', '20x20'] as const).map((size) => {
              const isDisabled = isTextTab && size !== '10x10';
              return (
                <button
                  key={size}
                  disabled={isDisabled}
                  onClick={() => setSelectedMapSize(size)}
                  title={isDisabled ? 'Modelos de texto são treinados exclusivamente no tamanho 10x10' : undefined}
                  className={`py-1.5 rounded-lg text-xs font-mono font-bold border transition ${
                    isDisabled
                      ? 'opacity-40 cursor-not-allowed bg-tokyo-dark text-tokyo-muted border-tokyo-border'
                      : selectedMapSize === size
                      ? 'bg-tokyo-blue text-tokyo-bg border-tokyo-blue shadow-[0_0_12px_rgba(59,130,246,0.3)] active-press-scale'
                      : 'bg-tokyo-dark text-tokyo-text border-tokyo-border hover:border-tokyo-blue hover:bg-opacity-80 active-press-scale'
                  }`}
                >
                  {size}
                </button>
              );
            })}
          </div>
          {activeSOMModel && (
            <div className="flex justify-between items-center text-[10px] font-mono text-tokyo-muted bg-tokyo-dark bg-opacity-60 px-2 py-1 rounded border border-tokyo-border border-opacity-30">
              <span>Dimensão Efetiva (Motor): <strong className="text-tokyo-cyan">{activeSOMModel.cols}×{activeSOMModel.rows}</strong></span>
              <span className="font-bold text-tokyo-text">{activeSOMModel.cols * activeSOMModel.rows} neurônios</span>
            </div>
          )}
          {isTextTab && (
            <span className="text-[10px] font-mono text-tokyo-muted italic">
              ℹ️ Modelos textuais são treinados exclusivamente na dimensão 10x10.
            </span>
          )}
        </div>

        {/* Geometry / Lattice Selector */}
        <div className="flex flex-col space-y-1.5 pt-1">
          <label className="text-[10px] text-tokyo-muted font-semibold uppercase font-mono tracking-wider flex items-center gap-1">
            <Sliders size={12} className="text-tokyo-green" />
            Geometria da Grade (Lattice)
          </label>
          <div className="grid grid-cols-2 gap-2">
            <button
              onClick={() => setLattice('HEX')}
              className={`py-1.5 px-2 rounded-lg text-xs font-mono font-bold border transition flex items-center justify-center gap-1.5 active-press-scale ${
                lattice === 'HEX'
                  ? 'bg-tokyo-green text-tokyo-bg border-tokyo-green shadow-[0_0_10px_rgba(16,185,129,0.3)]'
                  : 'bg-tokyo-dark text-tokyo-text border-tokyo-border hover:border-tokyo-green'
              }`}
            >
              <span className="w-2.5 h-2.5 rounded-sm rotate-45 border border-current" />
              Hexagonal (HEX)
            </button>

            <button
              onClick={() => setLattice('RECT')}
              title="Geometria Retangular: Modelo treinado com o motor IntraSOM 1.1.1 (_rect_dist_tor corrigida)."
              className={`py-1.5 px-2 rounded-lg text-xs font-mono font-bold border transition flex items-center justify-center gap-1.5 active-press-scale ${
                lattice === 'RECT'
                  ? 'bg-tokyo-orange text-tokyo-bg border-tokyo-orange shadow-[0_0_10px_rgba(249,115,22,0.3)]'
                  : 'bg-tokyo-dark text-tokyo-text border-tokyo-border hover:border-tokyo-orange'
              }`}
            >
              <span className="w-2.5 h-2.5 rounded-sm border border-current" />
              Retangular (RECT)
            </button>
          </div>
        </div>

        {/* Topology Selector */}
        <div className="flex flex-col space-y-1.5 pt-1">
          <label className="text-[10px] text-tokyo-muted font-semibold uppercase font-mono tracking-wider flex items-center gap-1">
            <RefreshCw size={12} className="text-tokyo-magenta" />
            Topologia do Mapa (Mapshape)
          </label>
          <div className="grid grid-cols-2 gap-2">
            <button
              onClick={() => setTopology('toroid')}
              title="Topologia toroidal (Rosca - bordas conectadas sem efeitos de borda)"
              className={`py-1.5 px-2 rounded-lg text-xs font-mono font-bold border transition flex items-center justify-center gap-1.5 active-press-scale ${
                topology === 'toroid'
                  ? 'bg-tokyo-magenta text-tokyo-bg border-tokyo-magenta shadow-[0_0_10px_rgba(217,70,239,0.3)]'
                  : 'bg-tokyo-dark text-tokyo-text border-tokyo-border hover:border-tokyo-magenta'
              }`}
            >
              Toroide (Rosca)
            </button>

            <button
              onClick={() => setTopology('planar')}
              title="Topologia plana (Bordas desconectadas)"
              className={`py-1.5 px-2 rounded-lg text-xs font-mono font-bold border transition flex items-center justify-center gap-1.5 active-press-scale ${
                topology === 'planar'
                  ? 'bg-tokyo-yellow text-tokyo-bg border-tokyo-yellow shadow-[0_0_10px_rgba(234,179,8,0.3)]'
                  : 'bg-tokyo-dark text-tokyo-text border-tokyo-border hover:border-tokyo-yellow'
              }`}
            >
              Plana (Sem Karnaugh)
            </button>
          </div>
        </div>
      </div>

      {/* SECTION 2: Hyperparameter Sensitivity Study (Highlighting Filter) */}
      <div className="flex flex-col space-y-3 bg-tokyo-dark bg-opacity-30 p-3.5 rounded-xl border border-tokyo-border border-opacity-30">
        <div className="flex items-center justify-between border-b border-tokyo-border border-opacity-20 pb-2">
          <span className="text-[11px] font-bold text-tokyo-purple uppercase font-mono tracking-wider flex items-center gap-1.5">
            <Filter size={13} />
            2. Estudo de Sensibilidade de Hiperparâmetros
          </span>
          <span className="text-[9px] font-mono bg-tokyo-purple bg-opacity-20 text-tokyo-purple px-1.5 py-0.5 rounded border border-tokyo-purple border-opacity-30 font-semibold">
            Destaque de Tabela
          </span>
        </div>

        <p className="text-[9.5px] font-mono text-tokyo-muted leading-relaxed">
          ℹ️ Os seletores abaixo destacam a combinação correspondente na tabela <strong>Estudo de Parâmetros</strong> (não alteram o mapa exibido acima).
        </p>

        {/* Neighborhood & Epochs Grid */}
        <div className="grid grid-cols-3 gap-2 text-[11px] font-mono">
          {/* Initial Radius */}
          <div className="flex flex-col space-y-1">
            <span className="text-[9px] text-tokyo-muted uppercase font-semibold">Raio Inicial</span>
            <select
              value={initialRadius}
              onChange={(e) => setInitialRadius(e.target.value as '80%' | '50%' | '100%')}
              className="bg-tokyo-dark border border-tokyo-border text-tokyo-text rounded p-1 text-[11px] font-bold focus:outline-none focus:border-tokyo-purple cursor-pointer"
            >
              <option value="80%">80% (Rec.)</option>
              <option value="50%">50%</option>
              <option value="100%">100%</option>
            </select>
          </div>

          {/* Final Radius */}
          <div className="flex flex-col space-y-1">
            <span className="text-[9px] text-tokyo-muted uppercase font-semibold">Raio Final</span>
            <select
              value={finalRadius}
              onChange={(e) => setFinalRadius(e.target.value as '1' | '2')}
              className="bg-tokyo-dark border border-tokyo-border text-tokyo-text rounded p-1 text-[11px] font-bold focus:outline-none focus:border-tokyo-purple cursor-pointer"
            >
              <option value="1">1 (Fino)</option>
              <option value="2">2 neurônios</option>
            </select>
          </div>

          {/* Epochs */}
          <div className="flex flex-col space-y-1">
            <span className="text-[9px] text-tokyo-muted uppercase font-semibold">Épocas</span>
            <select
              value={epochs}
              onChange={(e) => setEpochs(Number(e.target.value) as 500 | 200 | 100)}
              className="bg-tokyo-dark border border-tokyo-border text-tokyo-text rounded p-1 text-[11px] font-bold focus:outline-none focus:border-tokyo-purple cursor-pointer"
            >
              <option value={500}>500 (Rec.)</option>
              <option value={200}>200 épocas</option>
              <option value={100}>100 épocas</option>
            </select>
          </div>
        </div>

        {/* Immediate Live Summary Line (Item 1) */}
        {matchedStudyEntry ? (
          <div className="bg-tokyo-purple bg-opacity-15 border border-tokyo-purple border-opacity-40 p-2 rounded-lg text-[10px] font-mono flex items-center justify-between text-tokyo-purple">
            <span>Configuração Selecionada:</span>
            <span className="font-bold">
              QE: <strong className="text-tokyo-text">{matchedStudyEntry.quantization_error.toFixed(4)}</strong> · TE: <strong className="text-tokyo-text">{matchedStudyEntry.topographic_error.toFixed(4)}</strong>
            </span>
          </div>
        ) : (
          <div className="bg-tokyo-dark bg-opacity-50 border border-tokyo-border border-opacity-30 p-2 rounded-lg text-[10px] font-mono text-tokyo-muted italic">
            Nenhuma combinação pré-computada para estes valores.
          </div>
        )}

        {/* Functional Scroll-to-Link with Highlight Trigger (Item 2) */}
        <button
          type="button"
          onClick={() => {
            const el = document.getElementById('parameter-study-panel');
            if (el) {
              el.scrollIntoView({ behavior: 'smooth', block: 'center' });
              window.dispatchEvent(new CustomEvent('highlight-parameter-study'));
            }
          }}
          className="w-full flex items-center justify-center gap-2 p-2 rounded-lg bg-tokyo-purple bg-opacity-20 border border-tokyo-purple border-opacity-40 text-tokyo-purple hover:bg-opacity-30 hover:border-opacity-60 text-[10px] font-mono font-bold transition active-press-scale cursor-pointer"
        >
          <ArrowDownCircle size={14} className="animate-bounce" />
          <span>Rolar até a tabela "Estudo de Parâmetros" & destacar linha</span>
        </button>
      </div>

      {/* Active Settings Summary Footer */}
      <div className="bg-tokyo-dark bg-opacity-60 p-2.5 rounded-xl border border-tokyo-border border-opacity-30 text-[10px] font-mono leading-relaxed space-y-1 text-tokyo-muted">
        <div className="flex justify-between">
          <span>Motor / Malha:</span>
          <span className={`font-bold ${lattice === 'RECT' ? 'text-tokyo-orange' : 'text-tokyo-text'}`}>
            {lattice === 'HEX' ? 'IntraSOM 1.1.1 (Hexagonal)' : 'IntraSOM 1.1.1 (Retangular)'}
          </span>
        </div>
        <div className="flex justify-between">
          <span>Topologia Ativa:</span>
          <span className={`font-bold ${
            topology === 'toroid'
              ? 'text-tokyo-magenta'
              : 'text-tokyo-yellow'
          }`}>
            {topology === 'toroid'
              ? `Toroide (Rosca ${lattice})`
              : `Plana (${lattice})`}
          </span>
        </div>
        <div className="flex justify-between border-t border-tokyo-border border-opacity-20 pt-1 mt-1">
          <span>Padronização do Treino:</span>
          <span className="text-tokyo-green font-bold">Batch Síncrono (Kohonen 2013) · Inicialização Linear PCA</span>
        </div>
      </div>
    </div>
  );
}
