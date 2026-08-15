import { useState, useEffect, useMemo, memo } from 'react';
import { useShallow } from 'zustand/react/shallow';
import { useDashboardStore, type NeuronItem } from '../store/useDashboardStore';
import { useFullscreen } from '../hooks/useFullscreen';
import { Maximize2, Minimize2, Download } from 'lucide-react';
import { SYNTHETIC_CLASS_COLORS as CLASS_COLORS, getUMatrixColor } from '../lib/colors';
import { NeuronDetailPanel } from './NeuronDetailPanel';
import { getHexPoints, computeContiguousHexRadius, getHexCenter } from '../lib/geometry';
import { FullscreenPanel } from './FullscreenPanel';
import { UMatrix3D } from './UMatrix3D';
import { UMatrixTorus } from './UMatrixTorus';

// ──────────────────────────────────────────────────────────
// Main HexGrid component
// ──────────────────────────────────────────────────────────
export const HexGrid = memo(function HexGrid() {
  const { selectedMapSize, selectedNeuronId, setSelectedNeuronId, loadingSynthetic, lattice, getServedTopology, series, getActiveSOMModel } = useDashboardStore(
    useShallow((s) => ({
      selectedMapSize: s.selectedMapSize,
      selectedNeuronId: s.selectedNeuronId,
      setSelectedNeuronId: s.setSelectedNeuronId,
      loadingSynthetic: s.loadingSynthetic,
      lattice: s.lattice,
      getServedTopology: s.getServedTopology,
      series: s.series,
      getActiveSOMModel: s.getActiveSOMModel,
    }))
  );
  const { isFullscreen, toggleFullscreen } = useFullscreen();
  const servedTopology = getServedTopology();
  const [colorMode, setColorMode] = useState<'class' | 'umatrix'>('class');
  const [viewDimension, setViewDimension] = useState<'2D' | '3D'>('2D');
  const [threeModeOverride, setThreeModeOverride] = useState<'terrain' | 'torus' | null>(null);

  const threeMode = threeModeOverride ?? (servedTopology === 'toroid' ? 'torus' : 'terrain');

  // Ao mudar a topologia servida, limpa o override manual para que o padrao da topologia vença
  useEffect(() => {
    setThreeModeOverride(null);
  }, [servedTopology]);

  // Tooltip local state
  const [hoveredNeuron, setHoveredNeuron] = useState<NeuronItem | null>(null);
  const [tooltipPos, setTooltipPos] = useState<{ x: number; y: number }>({ x: 0, y: 0 });

  // Route to the correct pre-trained variant based on topology/mapSize
  const model = getActiveSOMModel();
  const neurons = model?.neurons;
  const cols = model?.cols || 1;
  const rows = model?.rows || 1;

  const padding = 25;
  const svgWidth = isFullscreen ? 800 : 540;
  const svgHeight = isFullscreen ? 550 : 380;

  const { r, minUMatrixVal, maxUMatrixVal, neuronLayouts, interstitialCells, offsetX, offsetY } = useMemo(() => {
    if (!neurons || neurons.length === 0) {
      return { r: 0, minUMatrixVal: 0, maxUMatrixVal: 0, neuronLayouts: [], interstitialCells: [], offsetX: 0, offsetY: 0 };
    }

    const isUMatrix = colorMode === 'umatrix';
    const colsEff = isUMatrix ? 2 * cols - 1 : cols;
    const rowsEff = isUMatrix ? 2 * rows - 1 : rows;

    const { radius, offsetX, offsetY } = computeContiguousHexRadius(colsEff, rowsEff, svgWidth, svgHeight, padding, lattice);

    const uMatrixVals = neurons.map(n => n.umatrix_value);
    const minUVal = Math.min(...uMatrixVals);
    const maxUVal = Math.max(...uMatrixVals);

    const layouts = neurons.map(neuron => {
      const rIndex = isUMatrix ? 2 * neuron.row : neuron.row;
      const cIndex = isUMatrix ? 2 * neuron.col : neuron.col;
      const { cx, cy } = getHexCenter(rIndex, cIndex, radius, padding, lattice);
      const pointsStr = getHexPoints(cx, cy, radius);
      return {
        ...neuron,
        cx,
        cy,
        pointsStr
      };
    });

    const eMin = model?.umatrix_edge_min ?? minUVal;
    const eMax = model?.umatrix_edge_max ?? maxUVal;
    const edges = model?.umatrix_edges || [];
    const neuronRowCol = new Map(neurons.map(n => [n.id, { row: n.row, col: n.col }]));

    const interstitials = isUMatrix ? edges.map((edge, idx) => {
      const n1 = neuronRowCol.get(edge.from);
      const n2 = neuronRowCol.get(edge.to);
      if (!n1 || !n2) return null;
      const rIndex = n1.row + n2.row;
      const cIndex = n1.col + n2.col;
      const { cx, cy } = getHexCenter(rIndex, cIndex, radius, padding, lattice);
      const pointsStr = getHexPoints(cx, cy, radius);
      const fill = getUMatrixColor(edge.distance, eMin, eMax);
      return {
        key: `edge-${edge.from}-${edge.to}-${idx}`,
        cx,
        cy,
        pointsStr,
        fill,
        distance: edge.distance,
        from: edge.from,
        to: edge.to
      };
    }).filter(Boolean) as Array<{ key: string; cx: number; cy: number; pointsStr: string; fill: string; distance: number; from: number; to: number }> : [];

    return {
      r: radius,
      minUMatrixVal: minUVal,
      maxUMatrixVal: maxUVal,
      neuronLayouts: layouts,
      interstitialCells: interstitials,
      edgeMin: eMin,
      edgeMax: eMax,
      offsetX,
      offsetY
    };
  }, [neurons, cols, rows, svgWidth, svgHeight, model, lattice, colorMode]);

  if (loadingSynthetic) {
    return (
      <div className="glass-panel rounded-2xl p-5 flex flex-col h-full min-h-[380px] animate-pulse">
        <div className="flex justify-between items-center mb-4">
          <div className="h-4 bg-[#2e3440] rounded w-1/3" />
          <div className="h-6 bg-[#2e3440] rounded w-24" />
        </div>
        <div className="flex-1 bg-tokyo-dark bg-opacity-40 rounded-xl border border-tokyo-border border-opacity-30 flex justify-center items-center p-4">
          <div className="grid grid-cols-10 gap-2.5 w-full max-w-[450px]">
            {Array.from({ length: 60 }).map((_, i) => (
              <div
                key={i}
                className="aspect-square bg-[#1f2335] rounded-lg border border-tokyo-border border-opacity-20 animate-pulse"
                style={{ animationDelay: `${(i % 10) * 35}ms` }}
              />
            ))}
          </div>
        </div>
      </div>
    );
  }

  if (!model) return null;

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
    ? (neurons ?? []).find(n => n.id === selectedNeuronId) ?? null
    : null;

  // Panel is open when: fullscreen + a neuron selected + it has samples
  const sidePanelOpen = isFullscreen && selectedNeuron !== null && selectedNeuron.total_samples > 0;

  return (
    <FullscreenPanel
      isFullscreen={isFullscreen}
      className="glass-panel rounded-2xl p-5 flex flex-col"
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
        <div className="flex items-center gap-2">
          <h2 className="text-sm font-bold text-tokyo-text uppercase font-mono tracking-wider">
            Malha {lattice === 'HEX' ? 'Hexagonal (HEX)' : 'Retangular (RECT)'} Kohonen ({selectedMapSize})
          </h2>
          <span className={`text-[9px] font-mono px-2 py-0.5 rounded border font-semibold ${
            servedTopology === 'toroid'
              ? 'bg-tokyo-magenta bg-opacity-10 text-tokyo-magenta border-tokyo-magenta border-opacity-30'
              : 'bg-tokyo-yellow bg-opacity-10 text-tokyo-yellow border-tokyo-yellow border-opacity-30'
          }`}>
            {servedTopology === 'toroid' ? 'Toroide ON' : 'Plana (Sem Karnaugh)'}
          </span>
        </div>

        <div className="flex items-center space-x-2">
          {/* Toggle Mode */}
          <div className="flex rounded border border-tokyo-border overflow-hidden">
            <button
              onClick={() => setColorMode('class')}
              className={`px-3 py-1 text-xs transition active-press-scale ${colorMode === 'class' ? 'bg-tokyo-blue text-tokyo-bg font-semibold' : 'bg-tokyo-panel text-tokyo-text hover:bg-opacity-80'}`}
            >
              Classes
            </button>
            <button
              onClick={() => setColorMode('umatrix')}
              className={`px-3 py-1 text-xs transition active-press-scale ${colorMode === 'umatrix' ? 'bg-tokyo-blue text-tokyo-bg font-semibold' : 'bg-tokyo-panel text-tokyo-text hover:bg-opacity-80'}`}
            >
              U-Matrix
            </button>
          </div>

          {/* Toggle 2D / 3D (visible only when colorMode === 'umatrix') */}
          {colorMode === 'umatrix' && (
            <div className="flex rounded border border-tokyo-border overflow-hidden">
              <button
                onClick={() => setViewDimension('2D')}
                className={`px-2.5 py-1 text-xs font-mono transition active-press-scale ${viewDimension === '2D' ? 'bg-tokyo-purple text-tokyo-bg font-bold' : 'bg-tokyo-panel text-tokyo-text hover:bg-opacity-80'}`}
              >
                2D
              </button>
              <button
                onClick={() => setViewDimension('3D')}
                className={`px-2.5 py-1 text-xs font-mono transition active-press-scale ${viewDimension === '3D' ? 'bg-tokyo-purple text-tokyo-bg font-bold' : 'bg-tokyo-panel text-tokyo-text hover:bg-opacity-80'}`}
              >
                3D
              </button>
            </div>
          )}

          {/* Sub-toggle Terreno 3D vs Toroide 3D */}
          {colorMode === 'umatrix' && viewDimension === '3D' && (
            <div className="flex rounded border border-tokyo-border overflow-hidden">
              <button
                onClick={() => setThreeModeOverride('terrain')}
                className={`px-2 py-1 text-[11px] font-mono transition active-press-scale ${threeMode === 'terrain' ? 'bg-tokyo-cyan text-tokyo-bg font-bold' : 'bg-tokyo-panel text-tokyo-text hover:bg-opacity-80'}`}
              >
                Terreno
              </button>
              <button
                onClick={() => setThreeModeOverride('torus')}
                className={`px-2 py-1 text-[11px] font-mono transition active-press-scale ${threeMode === 'torus' ? 'bg-tokyo-magenta text-tokyo-bg font-bold' : 'bg-tokyo-panel text-tokyo-text hover:bg-opacity-80'}`}
              >
                Toroide 🍩
              </button>
            </div>
          )}

          {/* Export SVG */}
          <button
            onClick={downloadSVG}
            className="p-1.5 hover:bg-tokyo-panel rounded-lg transition-colors text-tokyo-muted hover:text-tokyo-text active-press-scale"
            title="Exportar SVG"
          >
            <Download size={16} />
          </button>

          {/* Fullscreen */}
          <button
            onClick={toggleFullscreen}
            className="p-1.5 hover:bg-tokyo-panel rounded-lg transition-colors text-tokyo-muted hover:text-tokyo-text active-press-scale"
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
          className="flex flex-col min-h-0 flex-1 transition-all duration-500 ease-[cubic-bezier(0.22,1,0.36,1)]"
          style={{ flex: sidePanelOpen ? '0 0 48%' : '1 1 100%' }}
        >
          {/* SVG Hex Map, 3D Terrain, or 3D Torus */}
          <div className={`${isFullscreen ? 'flex-1' : 'min-h-[380px]'} flex justify-center items-center relative overflow-hidden bg-tokyo-dark bg-opacity-40 rounded-xl border border-tokyo-border border-opacity-30`}>
          {colorMode === 'umatrix' && viewDimension === '3D' ? (
            threeMode === 'torus' ? (
              <UMatrixTorus
                neurons={neurons || []}
                cols={cols}
                rows={rows}
                umatrix_edges={model.umatrix_edges}
                edgeMin={model.umatrix_edge_min}
                edgeMax={model.umatrix_edge_max}
                lattice={lattice}
              />
            ) : (
              <UMatrix3D
                neurons={neurons || []}
                cols={cols}
                rows={rows}
                umatrix_edges={model.umatrix_edges}
                edgeMin={model.umatrix_edge_min}
                edgeMax={model.umatrix_edge_max}
                lattice={lattice}
              />
            )
          ) : (
            <svg
              id="som-hex-grid-svg"
              viewBox={`0 0 ${svgWidth} ${svgHeight}`}
              className="w-full h-full max-h-[500px]"
            >
              <g transform={`translate(${offsetX}, ${offsetY})`}>
                {colorMode === 'umatrix' && interstitialCells.map(cell => (
                  lattice === 'RECT' ? (
                    <rect
                      key={cell.key}
                      x={cell.cx - r * Math.sqrt(3) / 2}
                      y={cell.cy - r * Math.sqrt(3) / 2}
                      width={r * Math.sqrt(3)}
                      height={r * Math.sqrt(3)}
                      fill={cell.fill}
                      fillOpacity={0.95}
                      stroke="rgba(0,0,0,0.15)"
                      strokeWidth="0.3"
                      className="transition-all duration-200"
                    >
                      <title>{`Distância U-Matrix (N${cell.from} ↔ N${cell.to}): ${cell.distance.toFixed(3)}`}</title>
                    </rect>
                  ) : (
                    <polygon
                      key={cell.key}
                      points={cell.pointsStr}
                      fill={cell.fill}
                      fillOpacity={0.95}
                      stroke="rgba(0,0,0,0.15)"
                      strokeWidth="0.3"
                      className="transition-all duration-200"
                    >
                      <title>{`Distância U-Matrix (N${cell.from} ↔ N${cell.to}): ${cell.distance.toFixed(3)}`}</title>
                    </polygon>
                  )
                ))}
                {neuronLayouts.map((neuron, index) => {
                  const { cx, cy, pointsStr } = neuron;

                  const isSelected = selectedNeuronId === neuron.id;

                  let fill = '#1f2335';
                  let stroke = 'rgba(122, 162, 247, 0.15)';
                  let strokeWidth = '1';

                  if (colorMode === 'class') {
                    if (neuron.total_samples > 0) {
                      fill = CLASS_COLORS[neuron.dominant_class as keyof typeof CLASS_COLORS] || '#1f2335';
                    }
                  } else {
                    fill = getUMatrixColor(neuron.umatrix_value, minUMatrixVal, maxUMatrixVal);
                  }

                  if (isSelected) {
                    stroke = '#ffffff';
                    strokeWidth = '2.5';
                  }

                  const delay = index * (500 / neuronLayouts.length);

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
                      className="cursor-pointer group focus:outline-none som-hex-group animate-hex-entrance"
                      style={{
                        transformOrigin: `${cx}px ${cy}px`,
                        animationDelay: `${delay}ms`
                      }}
                    >
                      {lattice === 'RECT' ? (
                        <rect
                          x={cx - r * 0.85}
                          y={cy - r * 0.85}
                          width={r * 1.7}
                          height={r * 1.7}
                          rx={4}
                          fill={fill}
                          fillOpacity={neuron.total_samples === 0 && colorMode === 'class' ? 0.2 : 0.8}
                          stroke={stroke}
                          strokeWidth={strokeWidth}
                          className="hex-polygon transition-all duration-200 group-hover:fill-opacity-100 group-hover:stroke-tokyo-blue group-hover:stroke-opacity-80 group-focus:stroke-white group-focus:stroke-opacity-100"
                          style={{
                            transformOrigin: `${cx}px ${cy}px`,
                            animationDelay: `${delay}ms`
                          }}
                        />
                      ) : (
                        <polygon
                          points={pointsStr}
                          fill={fill}
                          fillOpacity={neuron.total_samples === 0 && colorMode === 'class' ? 0.2 : 0.8}
                          stroke={stroke}
                          strokeWidth={strokeWidth}
                          className="hex-polygon transition-all duration-200 group-hover:fill-opacity-100 group-hover:stroke-tokyo-blue group-hover:stroke-opacity-80 group-focus:stroke-white group-focus:stroke-opacity-100"
                          style={{
                            transformOrigin: `${cx}px ${cy}px`,
                            animationDelay: `${delay}ms`
                          }}
                        />
                      )}

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
          )}
        </div>

        {/* Legend */}
        {colorMode === 'class' ? (
          <div className="grid grid-cols-3 gap-2 mt-4 text-2xs bg-tokyo-dark bg-opacity-30 p-2.5 rounded-lg border border-tokyo-border border-opacity-35">
            {Object.entries(CLASS_COLORS).map(([name, color]) => (
              <div key={name} className="flex items-center space-x-1.5 text-tokyo-text">
                <span className="w-2.5 h-2.5 rounded-full flex-shrink-0" style={{ backgroundColor: color as string }} />
                <span className="truncate">{name}</span>
              </div>
            ))}
            <div className="flex items-center space-x-1.5 text-tokyo-textDim">
              <span className="w-2.5 h-2.5 rounded-full flex-shrink-0 bg-tokyo-panel border border-dashed border-tokyo-text border-opacity-45" />
              <span>Vazio</span>
            </div>
          </div>
        ) : (
          <div className="flex justify-between items-center mt-4 text-2xs bg-tokyo-dark bg-opacity-30 p-2.5 rounded-lg border border-tokyo-border border-opacity-35">
            <span className="text-tokyo-textDim font-semibold uppercase font-mono">Mais Similar (Valores baixos)</span>
            <div className="flex flex-col items-center space-y-1">
              <div className="w-24 h-2 rounded bg-gradient-to-r from-[#1a1b26] via-[#bb9af7] to-[#7dcfff] border border-tokyo-border" />
              {viewDimension === '3D' && (
                <span className="text-[9px] text-tokyo-cyan font-mono">Altura Y = Descontinuidade U-Matrix</span>
              )}
            </div>
            <span className="text-tokyo-textDim font-semibold uppercase font-mono">Menos Similar (Fronteiras)</span>
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
          <div className="flex justify-between items-center text-2xs font-mono border-b border-tokyo-border border-opacity-30 pb-1.5">
            <span className="text-tokyo-text font-bold">Neurônio N{hoveredNeuron.id}</span>
            <span className="text-tokyo-textDim">({hoveredNeuron.col}, {hoveredNeuron.row})</span>
          </div>

          <div className="text-2xs space-y-0.5">
            <div className="flex justify-between space-x-4">
              <span className="text-tokyo-textDim">Dominante:</span>
              <span className="font-bold text-tokyo-blue">{hoveredNeuron.dominant_class}</span>
            </div>
            <div className="flex justify-between">
              <span className="text-tokyo-textDim">Amostras:</span>
              <span className="font-mono text-tokyo-text">{hoveredNeuron.total_samples}</span>
            </div>
            <div className="flex justify-between">
              <span className="text-tokyo-textDim">Pureza:</span>
              <span className="font-mono text-tokyo-green">{(hoveredNeuron.purity * 100).toFixed(0)}%</span>
            </div>
            <div className="flex justify-between">
              <span className="text-tokyo-textDim">U-Dist:</span>
              <span className="font-mono text-tokyo-magenta">{hoveredNeuron.umatrix_value.toFixed(4)}</span>
            </div>
          </div>

          {hoveredNeuron.total_samples > 0 && (
            <div className="flex flex-col border-t border-tokyo-border border-opacity-30 pt-1.5">
              <span className="text-[8px] text-tokyo-textDim uppercase font-mono mb-1">Perfil Sináptico (Codebook):</span>
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
    </FullscreenPanel>
  );
});
