import { useState } from 'react';
import { useDashboardStore, type NeuronItem } from '../store/useDashboardStore';
import { useFullscreen } from '../hooks/useFullscreen';
import { Maximize2, Minimize2, Download, X } from 'lucide-react';

const CLASS_COLORS: Record<string, string> = {
  "Normal": "#7aa2f7",
  "Cyclic": "#7dcfff",
  "Increasing Trend": "#ff9e64",
  "Decreasing Trend": "#e0af68",
  "Upward Shift": "#9ece6a",
  "Downward Shift": "#f7768e"
};

const getUMatrixColor = (val: number, max: number) => {
  const norm = val / (max || 1);
  if (norm < 0.5) {
    const r = Math.floor(26 + (187 - 26) * (norm * 2));
    const g = Math.floor(27 + (154 - 27) * (norm * 2));
    const b = Math.floor(38 + (247 - 38) * (norm * 2));
    return `rgb(${r}, ${g}, ${b})`;
  } else {
    const factor = (norm - 0.5) * 2;
    const r = Math.floor(187 + (125 - 187) * factor);
    const g = Math.floor(154 + (207 - 154) * factor);
    const b = Math.floor(247 + (255 - 247) * factor);
    return `rgb(${r}, ${g}, ${b})`;
  }
};

// ──────────────────────────────────────────────────────────
// Inline NeuronDetailPanel (used only inside FullScreen)
// ──────────────────────────────────────────────────────────
function NeuronDetailPanel({ neuron, series, onClose }: {
  neuron: NeuronItem;
  series: { id: number; values: number[]; class: string }[];
  onClose: () => void;
}) {
  const neuronSeries = series.filter(s => neuron.sample_ids.includes(s.id));
  const allValues = neuronSeries.flatMap(s => s.values);
  const minVal = allValues.length ? Math.min(...allValues) - 2 : -10;
  const maxVal = allValues.length ? Math.max(...allValues) + 2 : 10;

  const W = 560;
  const H = 320;
  const PAD = 28;

  const getSvgPath = (values: number[]) =>
    values.map((val, step) => {
      const x = PAD + (step / (values.length - 1)) * (W - 2 * PAD);
      const y = H - PAD - ((val - minVal) / (maxVal - minVal || 1)) * (H - 2 * PAD);
      return `${step === 0 ? 'M' : 'L'} ${x.toFixed(1)} ${y.toFixed(1)}`;
    }).join(' ');

  const codebookPath = getSvgPath(neuron.codebook);
  const classColor = CLASS_COLORS[neuron.dominant_class] || '#7aa2f7';

  return (
    <div className="flex flex-col h-full w-full">
      {/* Panel header */}
      <div className="flex items-start justify-between mb-5">
        <div>
          <h3 className="text-base font-bold text-white uppercase font-mono tracking-wider">
            Padrões no Neurônio N{neuron.id}
          </h3>
          <p className="text-[11px] text-[#9aa5ce] font-mono mt-0.5">
            {neuron.total_samples} amostras mapeadas
          </p>
        </div>
        <div className="flex items-center gap-2">
          <span
            className="px-2 py-1 rounded text-[10px] font-bold font-mono uppercase border"
            style={{ color: classColor, borderColor: classColor + '55', background: classColor + '15' }}
          >
            {neuron.dominant_class}
          </span>
          <span className="px-2 py-1 rounded text-[10px] font-bold font-mono border border-tokyo-blue text-tokyo-blue bg-tokyo-blue bg-opacity-10">
            Pureza: {(neuron.purity * 100).toFixed(0)}%
          </span>
          <button
            onClick={onClose}
            className="ml-1 p-1.5 rounded-lg hover:bg-[#1f2335] transition-colors text-[#565f89] hover:text-white"
            title="Fechar painel"
          >
            <X size={16} />
          </button>
        </div>
      </div>

      {/* Stats row */}
      <div className="grid grid-cols-3 gap-3 mb-5">
        {[
          { label: 'Amostras', value: neuron.total_samples, color: '#7aa2f7' },
          { label: 'U-Dist', value: neuron.umatrix_value.toFixed(4), color: '#bb9af7' },
          { label: 'Posição', value: `(${neuron.col}, ${neuron.row})`, color: '#7dcfff' },
        ].map(item => (
          <div key={item.label} className="bg-[#1f2335] rounded-xl p-3 border border-[#3b4261] border-opacity-60 flex flex-col gap-0.5">
            <span className="text-[9px] text-[#565f89] uppercase font-mono">{item.label}</span>
            <span className="text-sm font-bold font-mono" style={{ color: item.color }}>{item.value}</span>
          </div>
        ))}
      </div>

      {/* Chart area */}
      <div className="flex-1 bg-[#13131a] rounded-xl border border-[#3b4261] border-opacity-40 relative overflow-hidden flex items-center justify-center">
        <svg viewBox={`0 0 ${W} ${H}`} className="w-full h-full">
          {/* Grid lines */}
          {[0, 1, 2, 3, 4].map(idx => {
            const y = PAD + (idx / 4) * (H - 2 * PAD);
            return (
              <line key={idx} x1={PAD} y1={y} x2={W - PAD} y2={y}
                stroke="rgba(122,162,247,0.06)" strokeWidth="1" />
            );
          })}

          {/* Sample series */}
          {neuronSeries.map(s => (
            <path
              key={s.id}
              d={getSvgPath(s.values)}
              fill="none"
              stroke={classColor}
              strokeWidth="1.5"
              strokeOpacity="0.5"
            />
          ))}

          {/* Codebook glow */}
          <path d={codebookPath} fill="none"
            stroke={classColor} strokeWidth="7" strokeOpacity="0.25"
            strokeLinecap="round" strokeLinejoin="round" />
          {/* Codebook line */}
          <path d={codebookPath} fill="none"
            stroke="#ffffff" strokeWidth="2.5"
            strokeLinecap="round" strokeLinejoin="round" />
        </svg>
      </div>

      {/* Footer legend */}
      <div className="flex justify-between items-center mt-3 text-[10px] text-[#565f89] font-mono">
        <span>Linhas coloridas: séries temporais reais associadas a este neurônio</span>
        <span className="flex items-center gap-1.5">
          <span className="w-5 h-0.5 bg-white rounded-full inline-block" />
          <span className="text-[#9aa5ce] font-semibold">Vetor de Pesos (Codebook)</span>
        </span>
      </div>
    </div>
  );
}

// ──────────────────────────────────────────────────────────
// Main HexGrid component
// ──────────────────────────────────────────────────────────
export function HexGrid() {
  const { selectedMapSize, selectedNeuronId, setSelectedNeuronId, somModels, loadingSynthetic, series } = useDashboardStore();
  const { isFullscreen, toggleFullscreen } = useFullscreen();
  const [colorMode, setColorMode] = useState<'class' | 'umatrix'>('class');

  // Tooltip local state
  const [hoveredNeuron, setHoveredNeuron] = useState<NeuronItem | null>(null);
  const [tooltipPos, setTooltipPos] = useState<{ x: number; y: number }>({ x: 0, y: 0 });

  if (loadingSynthetic) {
    return (
      <div className="glass-panel rounded-2xl p-5 h-full flex flex-col justify-center items-center">
        <div className="animate-spin rounded-full h-8 w-8 border-b-2 border-tokyo-blue animate-pulse"></div>
        <span className="text-xs text-tokyo-muted mt-2 font-mono">Carregando mapa...</span>
      </div>
    );
  }

  const model = somModels[selectedMapSize];
  if (!model) return null;

  const { cols, rows, neurons } = model;

  const xCoords = neurons.map(n => n.x);
  const yCoords = neurons.map(n => n.y);
  const minX = Math.min(...xCoords);
  const maxX = Math.max(...xCoords);
  const minY = Math.min(...yCoords);
  const maxY = Math.max(...yCoords);

  const padding = 30;
  const svgWidth = isFullscreen ? 800 : 540;
  const svgHeight = isFullscreen ? 550 : 360;

  const scaleX = (x: number) => padding + ((x - minX) / (maxX - minX || 1)) * (svgWidth - 2 * padding);
  const scaleY = (y: number) => padding + ((y - minY) / (maxY - minY || 1)) * (svgHeight - 2 * padding);

  const r = Math.min(
    (svgWidth - 2 * padding) / (cols * 1.6),
    (svgHeight - 2 * padding) / (rows * 1.45)
  ) * 0.95;

  const getHexPoints = (cx: number, cy: number, radius: number) => {
    const points = [];
    for (let i = 0; i < 6; i++) {
      const angle = (i * Math.PI) / 3 - Math.PI / 6;
      points.push(`${cx + radius * Math.cos(angle)},${cy + radius * Math.sin(angle)}`);
    }
    return points.join(' ');
  };

  const maxUMatrixVal = Math.max(...neurons.map(n => n.umatrix_value));

  const downloadSVG = () => {
    const svgEl = document.getElementById('som-hex-grid-svg');
    if (!svgEl) return;
    const svgString = new XMLSerializer().serializeToString(svgEl);
    const svgBlob = new Blob([svgString], { type: 'image/svg+xml;charset=utf-8' });
    const svgUrl = URL.createObjectURL(svgBlob);
    const downloadLink = document.createElement('a');
    downloadLink.href = svgUrl;
    downloadLink.download = `som_${selectedMapSize}_${colorMode}.svg`;
    document.body.appendChild(downloadLink);
    downloadLink.click();
    document.body.removeChild(downloadLink);
  };

  // Sparkline path generator
  const getSparklinePath = (weights: number[]) => {
    const w = 120;
    const h = 40;
    const minW = Math.min(...weights);
    const maxW = Math.max(...weights);
    return weights.map((val, step) => {
      const x = (step / 59) * w;
      const y = h - 2 - ((val - minW) / (maxW - minW || 1)) * (h - 4);
      return `${step === 0 ? 'M' : 'L'} ${x} ${y}`;
    }).join(' ');
  };

  // Derive selected neuron object for the side panel
  const selectedNeuron = selectedNeuronId !== null
    ? neurons.find(n => n.id === selectedNeuronId) ?? null
    : null;

  // Panel is open when: fullscreen + a neuron selected + it has samples
  const sidePanelOpen = isFullscreen && selectedNeuron !== null && selectedNeuron.total_samples > 0;

  return (
    <div
      className={
        isFullscreen
          ? "fixed inset-0 bg-[#16161e] z-50 p-8 flex flex-col"
          : "glass-panel rounded-2xl p-5 flex flex-col"
      }
    >
      <style>{`
        @keyframes popIn {
          0% { transform: scale(0); opacity: 0; }
          75% { transform: scale(1.08); opacity: 0.9; }
          100% { transform: scale(1); opacity: 1; }
        }
        .hex-polygon {
          animation: popIn 0.4s cubic-bezier(0.34, 1.56, 0.64, 1) forwards;
        }
        @keyframes slideInRight {
          from { transform: translateX(100%); opacity: 0; }
          to   { transform: translateX(0);    opacity: 1; }
        }
        @keyframes slideOutRight {
          from { transform: translateX(0);    opacity: 1; }
          to   { transform: translateX(100%); opacity: 0; }
        }
        .side-panel-enter {
          animation: slideInRight 0.42s cubic-bezier(0.22, 1, 0.36, 1) forwards;
        }
      `}</style>

      {/* ── Header ── */}
      <div className="flex justify-between items-center mb-4">
        <h3 className="text-sm font-bold text-tokyo-text uppercase font-mono tracking-wider">
          Malha Hexagonal Kohonen ({selectedMapSize})
        </h3>

        <div className="flex items-center space-x-2">
          {/* Toggle Mode */}
          <div className="flex rounded border border-tokyo-border overflow-hidden">
            <button
              onClick={() => setColorMode('class')}
              className={`px-3 py-1 text-xs transition ${colorMode === 'class' ? 'bg-tokyo-blue text-tokyo-bg font-semibold' : 'bg-tokyo-panel text-tokyo-text hover:bg-opacity-80'}`}
            >
              Classes
            </button>
            <button
              onClick={() => setColorMode('umatrix')}
              className={`px-3 py-1 text-xs transition ${colorMode === 'umatrix' ? 'bg-tokyo-blue text-tokyo-bg font-semibold' : 'bg-tokyo-panel text-tokyo-text hover:bg-opacity-80'}`}
            >
              U-Matrix
            </button>
          </div>

          {/* Export SVG */}
          <button
            onClick={downloadSVG}
            className="p-1.5 hover:bg-tokyo-panel rounded-lg transition-colors text-tokyo-muted hover:text-tokyo-text"
            title="Exportar SVG"
          >
            <Download size={16} />
          </button>

          {/* Fullscreen */}
          <button
            onClick={toggleFullscreen}
            className="p-1.5 hover:bg-tokyo-panel rounded-lg transition-colors text-tokyo-muted hover:text-tokyo-text"
            title={isFullscreen ? "Sair da Tela Cheia" : "Tela Cheia"}
          >
            {isFullscreen ? <Minimize2 size={16} /> : <Maximize2 size={16} />}
          </button>
        </div>
      </div>

      {/* ── Content: hex map + optional sliding side panel ── */}
      <div className={`${isFullscreen ? 'flex-1' : ''} flex overflow-hidden gap-5 min-h-0`}>

        {/* Hex Map column — shrinks when panel is open */}
        <div
          className="flex flex-col min-h-0 transition-all duration-500 ease-[cubic-bezier(0.22,1,0.36,1)]"
          style={{ flex: sidePanelOpen ? '0 0 48%' : '1 1 100%' }}
        >
          {/* SVG Hex Map */}
          <div className={`${isFullscreen ? 'flex-1' : 'min-h-[320px]'} flex justify-center items-center relative overflow-hidden bg-tokyo-dark bg-opacity-40 rounded-xl border border-tokyo-border border-opacity-30`}>
            <svg
              id="som-hex-grid-svg"
              viewBox={`0 0 ${svgWidth} ${svgHeight}`}
              className="w-full h-full max-h-[500px]"
            >
              <g>
                {neurons.map((neuron) => {
                  const cx = scaleX(neuron.x);
                  const cy = scaleY(neuron.y);

                  const isSelected = selectedNeuronId === neuron.id;

                  let fill = '#1f2335';
                  let stroke = 'rgba(122, 162, 247, 0.15)';
                  let strokeWidth = '1';

                  if (colorMode === 'class') {
                    if (neuron.total_samples > 0) {
                      fill = CLASS_COLORS[neuron.dominant_class] || '#1f2335';
                    }
                  } else {
                    fill = getUMatrixColor(neuron.umatrix_value, maxUMatrixVal);
                  }

                  if (isSelected) {
                    stroke = '#ffffff';
                    strokeWidth = '2.5';
                  }

                  const pointsStr = getHexPoints(cx, cy, r);

                  return (
                    <g
                      key={`${selectedMapSize}-${neuron.id}`}
                      role="button"
                      tabIndex={0}
                      aria-label={`Neurônio N${neuron.id}, Classe Dominante: ${neuron.total_samples > 0 ? neuron.dominant_class : 'Vazio'}, Amostras: ${neuron.total_samples}, Valor U-Matrix: ${neuron.umatrix_value.toFixed(3)}`}
                      onClick={() => setSelectedNeuronId(isSelected ? null : neuron.id)}
                      onKeyDown={(e) => {
                        if (e.key === 'Enter' || e.key === ' ') {
                          e.preventDefault();
                          setSelectedNeuronId(isSelected ? null : neuron.id);
                        }
                      }}
                      onMouseEnter={(e) => {
                        setHoveredNeuron(neuron);
                        const rect = e.currentTarget.getBoundingClientRect();
                        setTooltipPos({
                          x: rect.left + rect.width / 2,
                          y: rect.top - 10
                        });
                      }}
                      onMouseLeave={() => setHoveredNeuron(null)}
                      className="cursor-pointer group focus:outline-none"
                    >
                      <polygon
                        points={pointsStr}
                        fill={fill}
                        fillOpacity={neuron.total_samples === 0 && colorMode === 'class' ? 0.2 : 0.8}
                        stroke={stroke}
                        strokeWidth={strokeWidth}
                        className="hex-polygon transition-all duration-200 group-hover:fill-opacity-100 group-hover:stroke-tokyo-blue group-hover:stroke-opacity-80 group-focus:stroke-white group-focus:stroke-opacity-100"
                        style={{
                          transformOrigin: `${cx}px ${cy}px`,
                          animationDelay: `${(neuron.row * cols + neuron.col) * 5}ms`
                        }}
                      />

                      <text
                        x={cx}
                        y={cy + 3}
                        textAnchor="middle"
                        fill={colorMode === 'class' ? (neuron.total_samples > 0 ? '#16161e' : '#565f89') : '#ffffff'}
                        fontSize={selectedMapSize === '20x20' ? '6px' : '8px'}
                        fontWeight="bold"
                        className="select-none pointer-events-none group-hover:fill-white transition-colors"
                      >
                        {neuron.id}
                      </text>

                      {isSelected && (
                        <circle
                          cx={cx}
                          cy={cy}
                          r={r * 0.4}
                          fill="none"
                          stroke="#ffffff"
                          strokeWidth="1.5"
                          className="animate-ping"
                        />
                      )}
                    </g>
                  );
                })}
              </g>
            </svg>
          </div>

          {/* Legend */}
          {colorMode === 'class' ? (
            <div className="grid grid-cols-3 gap-2 mt-4 text-[10px] bg-tokyo-dark bg-opacity-30 p-2.5 rounded-lg border border-tokyo-border border-opacity-35">
              {Object.entries(CLASS_COLORS).map(([name, color]) => (
                <div key={name} className="flex items-center space-x-1.5 text-tokyo-text">
                  <span className="w-2.5 h-2.5 rounded-full flex-shrink-0" style={{ backgroundColor: color }} />
                  <span className="truncate">{name}</span>
                </div>
              ))}
              <div className="flex items-center space-x-1.5 text-[#9aa5ce]">
                <span className="w-2.5 h-2.5 rounded-full flex-shrink-0 bg-tokyo-panel border border-dashed border-tokyo-text border-opacity-45" />
                <span>Vazio</span>
              </div>
            </div>
          ) : (
            <div className="flex justify-between items-center mt-4 text-[10px] bg-tokyo-dark bg-opacity-30 p-2.5 rounded-lg border border-tokyo-border border-opacity-35">
              <span className="text-[#9aa5ce] font-semibold uppercase font-mono">Mais Similar (Valores baixos)</span>
              <div className="w-24 h-2 rounded bg-gradient-to-r from-[#1a1b26] via-[#bb9af7] to-[#7dcfff] border border-tokyo-border" />
              <span className="text-[#9aa5ce] font-semibold uppercase font-mono">Menos Similar (Fronteiras)</span>
            </div>
          )}
        </div>

        {/* ── Sliding side panel (fullscreen-only) ── */}
        {isFullscreen && (
          <div
            className={`flex flex-col min-h-0 overflow-hidden rounded-xl border border-[#3b4261] border-opacity-50 bg-[#1a1b2e] p-5 transition-all duration-500 ease-[cubic-bezier(0.22,1,0.36,1)] ${
              sidePanelOpen ? 'side-panel-enter opacity-100' : 'opacity-0 pointer-events-none'
            }`}
            style={{
              flex: sidePanelOpen ? '0 0 49%' : '0 0 0%',
              width: sidePanelOpen ? undefined : 0,
              padding: sidePanelOpen ? undefined : 0,
              border: sidePanelOpen ? undefined : 'none',
            }}
          >
            {sidePanelOpen && selectedNeuron && (
              <NeuronDetailPanel
                neuron={selectedNeuron}
                series={series}
                onClose={() => setSelectedNeuronId(null)}
              />
            )}
          </div>
        )}
      </div>

      {/* ── Floating Sparkline Tooltip ── */}
      {hoveredNeuron && (
        <div
          className="fixed z-50 bg-[#16161e] border border-tokyo-border p-3 rounded-xl shadow-2xl flex flex-col space-y-2 pointer-events-none transform -translate-x-1/2 -translate-y-full"
          style={{
            left: tooltipPos.x,
            top: tooltipPos.y,
            boxShadow: '0 10px 30px rgba(0,0,0,0.5), 0 0 10px rgba(122,162,247,0.1)'
          }}
        >
          <div className="flex justify-between items-center text-[10px] font-mono border-b border-tokyo-border border-opacity-30 pb-1.5">
            <span className="text-tokyo-text font-bold">Neurônio N{hoveredNeuron.id}</span>
            <span className="text-[#9aa5ce]">({hoveredNeuron.col}, {hoveredNeuron.row})</span>
          </div>

          <div className="text-[10px] space-y-0.5">
            <div className="flex justify-between space-x-4">
              <span className="text-[#9aa5ce]">Dominante:</span>
              <span className="font-bold text-tokyo-blue">{hoveredNeuron.dominant_class}</span>
            </div>
            <div className="flex justify-between">
              <span className="text-[#9aa5ce]">Amostras:</span>
              <span className="font-mono text-tokyo-text">{hoveredNeuron.total_samples}</span>
            </div>
            <div className="flex justify-between">
              <span className="text-[#9aa5ce]">Pureza:</span>
              <span className="font-mono text-tokyo-green">{(hoveredNeuron.purity * 100).toFixed(0)}%</span>
            </div>
            <div className="flex justify-between">
              <span className="text-[#9aa5ce]">U-Dist:</span>
              <span className="font-mono text-tokyo-magenta">{hoveredNeuron.umatrix_value.toFixed(4)}</span>
            </div>
          </div>

          {hoveredNeuron.total_samples > 0 && (
            <div className="flex flex-col border-t border-tokyo-border border-opacity-30 pt-1.5">
              <span className="text-[8px] text-[#9aa5ce] uppercase font-mono mb-1">Perfil Sináptico (Codebook):</span>
              <svg width="120" height="40" className="bg-[#1f2335] bg-opacity-40 rounded">
                <path
                  d={getSparklinePath(hoveredNeuron.codebook)}
                  fill="none"
                  stroke="#7aa2f7"
                  strokeWidth="1.5"
                />
              </svg>
            </div>
          )}
        </div>
      )}
    </div>
  );
}
