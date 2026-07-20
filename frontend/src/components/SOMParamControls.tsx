import { useDashboardStore } from '../store/useDashboardStore';
import { Settings, Sliders, Layers, RefreshCw } from 'lucide-react';

export function SOMParamControls() {
  const {
    lattice,
    topology,
    initialRadius,
    finalRadius,
    epochs,
    trainingMode,
    initialization,
    setLattice,
    setTopology,
    setInitialRadius,
    setFinalRadius,
    setEpochs,
    setTrainingMode,
    setInitialization,
    selectedMapSize,
    setSelectedMapSize
  } = useDashboardStore();

  return (
    <div className="glass-panel rounded-2xl p-5 flex flex-col space-y-4 text-tokyo-text">
      <div className="flex items-center justify-between border-b border-tokyo-border border-opacity-30 pb-3">
        <h3 className="text-sm font-bold uppercase font-mono tracking-wider flex items-center gap-2 text-tokyo-blue">
          <Settings size={16} />
          Parâmetros do Algoritmo de Kohonen
        </h3>
        <span className="text-[10px] font-mono bg-tokyo-dark px-2 py-0.5 rounded border border-tokyo-border text-tokyo-magenta font-semibold">
          IntraSOM Config
        </span>
      </div>

      {/* Map Size Selector */}
      <div className="flex flex-col space-y-1.5">
        <label className="text-[10px] text-tokyo-muted font-semibold uppercase font-mono tracking-wider flex items-center gap-1">
          <Layers size={12} className="text-tokyo-cyan" />
          Dimensões da Grade (cols x rows)
        </label>
        <div className="grid grid-cols-3 gap-2">
          {(['10x10', '15x15', '20x20'] as const).map((size) => (
            <button
              key={size}
              onClick={() => setSelectedMapSize(size)}
              className={`py-1.5 rounded-lg text-xs font-mono font-bold border transition active-press-scale ${
                selectedMapSize === size
                  ? 'bg-tokyo-blue text-tokyo-bg border-tokyo-blue shadow-[0_0_12px_rgba(59,130,246,0.3)]'
                  : 'bg-tokyo-dark text-tokyo-text border-tokyo-border hover:border-tokyo-blue hover:bg-opacity-80'
              }`}
            >
              {size}
            </button>
          ))}
        </div>
      </div>

      {/* Geometry / Lattice Selector (HEX vs RECT) */}
      <div className="flex flex-col space-y-1.5">
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

          {/* RECT button — limitation documented: intrasom build_umatrix requires hexa */}
          <button
            onClick={() => setLattice('RECT')}
            title="Limitação da biblioteca intrasom: build_umatrix suporta apenas lattice hexagonal (linha 1928 de intrasom.py). O seletor muda a representação visual, mas os dados exibidos permanecem do modelo hexagonal."
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
        {/* Limitation notice shown when RECT is selected */}
        {lattice === 'RECT' && (
          <div className="mt-1.5 px-2.5 py-1.5 rounded-lg bg-orange-900/30 border border-orange-500/40 text-[10px] text-orange-300 leading-tight">
            ⚠️ <strong>Limitação da lib intrasom:</strong> U-Matrix requer grade hexagonal.
            Os dados exibidos são do modelo HEX treinado — apenas a geometria SVG muda.
            Para treino retangular real, seria necessária uma versão futura da biblioteca.
          </div>
        )}
      </div>

      {/* Topology Selector (Toroid vs Planar) */}
      <div className="flex flex-col space-y-1.5">
        <label className="text-[10px] text-tokyo-muted font-semibold uppercase font-mono tracking-wider flex items-center gap-1">
          <RefreshCw size={12} className="text-tokyo-magenta" />
          Topologia do Mapa (Mapshape)
        </label>
        <div className="grid grid-cols-2 gap-2">
          <button
            onClick={() => setTopology('toroid')}
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
            title={selectedMapSize === '10x10'
              ? 'Topologia planar: bordas desconectadas (sem efeito toroidal). Modelo 10x10 treinado com HEX+planar real.'
              : 'Topologia planar disponível apenas para o mapa 10x10 (dados pré-treinados). Para outros tamanhos, a seleção é informativa.'}
            className={`py-1.5 px-2 rounded-lg text-xs font-mono font-bold border transition flex items-center justify-center gap-1.5 active-press-scale ${
              topology === 'planar'
                ? 'bg-tokyo-yellow text-tokyo-bg border-tokyo-yellow shadow-[0_0_10px_rgba(234,179,8,0.3)]'
                : 'bg-tokyo-dark text-tokyo-text border-tokyo-border hover:border-tokyo-yellow'
            }`}
          >
            Plana (Sem Karnaugh)
          </button>
        </div>
        {/* Show info when planar selected on non-10x10 map */}
        {topology === 'planar' && selectedMapSize !== '10x10' && (
          <div className="mt-1.5 px-2.5 py-1.5 rounded-lg bg-yellow-900/30 border border-yellow-500/40 text-[10px] text-yellow-300 leading-tight">
            ℹ️ Variante planar pré-treinada disponível apenas para <strong>10×10</strong>.
            Para {selectedMapSize}, exibindo dados do modelo toroidal como fallback.
          </div>
        )}
      </div>

      {/* Neighborhood & Training Parameters Grid */}
      <div className="grid grid-cols-2 gap-3 pt-1 border-t border-tokyo-border border-opacity-20 text-[11px] font-mono">
        {/* Initial Radius */}
        <div className="flex flex-col space-y-1">
          <span className="text-[10px] text-tokyo-muted uppercase font-semibold">Vizinhança Inicial</span>
          <select
            value={initialRadius}
            onChange={(e) => setInitialRadius(e.target.value as '80%' | '50%' | '100%')}
            className="bg-tokyo-dark border border-tokyo-border text-tokyo-text rounded p-1 text-xs font-bold focus:outline-none focus:border-tokyo-blue cursor-pointer"
          >
            <option value="80%">80% do mapa (Recomendado)</option>
            <option value="50%">50% do mapa</option>
            <option value="100%">100% do mapa (Global)</option>
          </select>
        </div>

        {/* Final Radius */}
        <div className="flex flex-col space-y-1">
          <span className="text-[10px] text-tokyo-muted uppercase font-semibold">Vizinhança Final</span>
          <select
            value={finalRadius}
            onChange={(e) => setFinalRadius(e.target.value as '1' | '2')}
            className="bg-tokyo-dark border border-tokyo-border text-tokyo-text rounded p-1 text-xs font-bold focus:outline-none focus:border-tokyo-blue cursor-pointer"
          >
            <option value="1">1 neurônio (Ajuste Fino)</option>
            <option value="2">2 neurônios</option>
          </select>
        </div>

        {/* Epochs */}
        <div className="flex flex-col space-y-1">
          <span className="text-[10px] text-tokyo-muted uppercase font-semibold">Épocas de Treino</span>
          <select
            value={epochs}
            onChange={(e) => setEpochs(Number(e.target.value) as 500 | 200 | 100)}
            className="bg-tokyo-dark border border-tokyo-border text-tokyo-text rounded p-1 text-xs font-bold focus:outline-none focus:border-tokyo-blue cursor-pointer"
          >
            <option value={500}>500 épocas (Recomendado)</option>
            <option value={200}>200 épocas</option>
            <option value={100}>100 épocas</option>
          </select>
        </div>

        {/* Training Mode */}
        <div className="flex flex-col space-y-1">
          <span className="text-[10px] text-tokyo-muted uppercase font-semibold">Modo de Treino</span>
          <select
            value={trainingMode}
            onChange={(e) => setTrainingMode(e.target.value as 'batch' | 'online')}
            className="bg-tokyo-dark border border-tokyo-border text-tokyo-text rounded p-1 text-xs font-bold focus:outline-none focus:border-tokyo-blue cursor-pointer"
          >
            <option value="batch">Batch (Lote Síncrono)</option>
            <option value="online">Online (Estocástico)</option>
          </select>
        </div>
      </div>

      {/* Initialization */}
      <div className="flex flex-col space-y-1 pt-1 border-t border-tokyo-border border-opacity-20">
        <span className="text-[10px] text-tokyo-muted uppercase font-semibold font-mono">Inicialização de Pesos</span>
        <select
          value={initialization}
          onChange={(e) => setInitialization(e.target.value as 'linear' | 'random')}
          className="bg-tokyo-dark border border-tokyo-border text-tokyo-text rounded p-1 text-xs font-bold font-mono focus:outline-none focus:border-tokyo-blue cursor-pointer"
        >
          <option value="linear">Linear (PCA - Autovalores)</option>
          <option value="random">Aleatória</option>
        </select>
      </div>

      {/* Active Settings Summary Footer */}
      <div className="bg-tokyo-dark bg-opacity-60 p-2.5 rounded-xl border border-tokyo-border border-opacity-30 text-[10px] font-mono leading-relaxed space-y-1 text-tokyo-muted">
        <div className="flex justify-between">
          <span>Malha Ativa:</span>
          <span className={`font-bold ${lattice === 'RECT' ? 'text-tokyo-orange' : 'text-tokyo-text'}`}>
            {lattice === 'HEX' ? 'Hexagonal (HEX)' : 'Retangular (RECT) ⚠️ visual only'}
          </span>
        </div>
        <div className="flex justify-between">
          <span>Topologia Ativa:</span>
          <span className={`font-bold ${
            topology === 'planar' && selectedMapSize === '10x10'
              ? 'text-tokyo-yellow'
              : topology === 'planar'
              ? 'text-yellow-500'
              : 'text-tokyo-text'
          }`}>
            {topology === 'toroid'
              ? 'Toroide (Rosca)'
              : selectedMapSize === '10x10'
              ? 'Plana ✓ dados reais 10×10'
              : 'Plana (fallback → dados toroid)'}
          </span>
        </div>
        <div className="flex justify-between">
          <span>Configuração Recomendada:</span>
          <span className="text-tokyo-green font-bold">80% → 1n, 500 épocas, Batch, PCA</span>
        </div>
      </div>
    </div>
  );
}
