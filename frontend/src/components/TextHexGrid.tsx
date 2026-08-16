import { useState, useEffect, useMemo, memo } from 'react';
import { useShallow } from 'zustand/react/shallow';
import { useDashboardStore } from '../store/useDashboardStore';
import { useFullscreen } from '../hooks/useFullscreen';
import { Maximize2, Minimize2 } from 'lucide-react';
import { getClassColor, TEXT_CLASS_COLORS, getUMatrixColor } from '../lib/colors';
import { getHexPoints, computeContiguousHexRadius, getHexCenter } from '../lib/geometry';
import { FullscreenPanel } from './FullscreenPanel';
import { UMatrix3D } from './UMatrix3D';
import { UMatrixTorus } from './UMatrixTorus';

export const TextHexGrid = memo(function TextHexGrid() {
  const { 
    selectedTextDataset, 
    selectedTextRep,
    selectedDocId, 
    setSelectedDocId, 
    loadingText, 
    classificationResult,
    lattice,
    topology,
    getServedTopology,
    getActiveTextModel
  } = useDashboardStore(
    useShallow((s) => ({
      selectedTextDataset: s.selectedTextDataset,
      selectedTextRep: s.selectedTextRep,
      selectedDocId: s.selectedDocId,
      setSelectedDocId: s.setSelectedDocId,
      loadingText: s.loadingText,
      classificationResult: s.classificationResult,
      lattice: s.lattice,
      topology: s.topology,
      getServedTopology: s.getServedTopology,
      getActiveTextModel: s.getActiveTextModel,
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
  
  const model = getActiveTextModel();
  const neurons = model?.neurons;
  const cols = model?.cols || 1;
  const rows = model?.rows || 1;

  const padding = 20;
  const svgWidth = isFullscreen ? 800 : 500;
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
      offsetX,
      offsetY
    };
  }, [neurons, cols, rows, svgWidth, svgHeight, model, lattice, colorMode]);

  if (loadingText) {
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

  // Always use the correct color palette for the active dataset
  const activeColors = TEXT_CLASS_COLORS[selectedTextDataset] ?? {};

  return (
    <FullscreenPanel
      isFullscreen={isFullscreen}
      className="glass-panel rounded-2xl p-5 flex flex-col h-full overflow-visible"
    >
      <style>{`
        @keyframes ripple {
          0% { transform: scale(1); opacity: 0.8; }
          100% { transform: scale(3.5); opacity: 0; }
        }
        .ripple-circle {
          animation: ripple 1.2s cubic-bezier(0.1, 0.8, 0.3, 1) infinite;
        }
      `}</style>

      <div className="flex justify-between items-center mb-4">
        <div>
          <div className="flex flex-wrap items-center gap-2">
            <h2 className="text-sm font-bold text-tokyo-text uppercase font-mono tracking-wider">
              Malha {lattice === 'HEX' ? 'Hexagonal (HEX)' : 'Retangular (RECT)'} - Notícias (10x10)
            </h2>
            <span className={`text-[9px] font-mono px-2 py-0.5 rounded border font-semibold ${
              topology === 'toroid' || servedTopology === 'toroid'
                ? 'bg-tokyo-magenta bg-opacity-10 text-tokyo-magenta border-tokyo-magenta border-opacity-30'
                : 'bg-tokyo-yellow bg-opacity-10 text-tokyo-yellow border-tokyo-yellow border-opacity-30'
            }`}>
              {servedTopology === 'toroid' ? 'Toroide ON' : 'Plana (Sem Karnaugh)'}
            </span>
            <span className="text-[9px] font-mono px-2 py-0.5 rounded border border-tokyo-blue border-opacity-30 bg-tokyo-blue bg-opacity-10 text-tokyo-blue font-bold uppercase">
              {selectedTextRep}
            </span>
          </div>
          <p className="text-2xs text-tokyo-muted font-mono mt-0.5">
            {selectedTextDataset === '20news' 
              ? '4 categorias do dataset 20 Newsgroups (400 documentos)'
              : '6 categorias do dataset 6News com Texto Expandido (317 documentos)'}
          </p>
        </div>
        
        <div className="flex items-center space-x-2">
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

          <button 
            onClick={toggleFullscreen}
            className="p-1.5 hover:bg-tokyo-panel rounded-lg transition-colors text-tokyo-muted hover:text-tokyo-text active-press-scale"
            title={isFullscreen ? "Sair da Tela Cheia" : "Tela Cheia"}
          >
            {isFullscreen ? <Minimize2 size={16} /> : <Maximize2 size={16} />}
          </button>
        </div>
      </div>
      
      {/* Hex Grid SVG, 3D Terrain, or 3D Torus */}
      <div className="flex-1 flex justify-center items-center relative overflow-hidden bg-tokyo-dark bg-opacity-40 rounded-xl border border-tokyo-border border-opacity-30 min-h-[380px]">
        {colorMode === 'umatrix' && viewDimension === '3D' && model ? (
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
            viewBox={`0 0 ${svgWidth} ${svgHeight}`} 
            className="w-full h-full max-h-[460px]"
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
              {neuronLayouts.map((neuron) => {
                const { cx, cy, pointsStr } = neuron;
                
                const isClassifiedBMU = classificationResult?.bmu === neuron.id;
                
                let isSelectedDocBMU = false;
                if (selectedDocId !== null) {
                  isSelectedDocBMU = neuron.doc_indices.includes(selectedDocId);
                }
                
                const isHighlighted = isClassifiedBMU || isSelectedDocBMU;
                
                let fill = '#1f2335';
                let stroke = 'rgba(122, 162, 247, 0.15)';
                let strokeWidth = '1';
                
                if (colorMode === 'class') {
                  if (neuron.total_samples > 0) {
                    fill = getClassColor(selectedTextDataset, neuron.dominant_class);
                  }
                } else {
                  fill = getUMatrixColor(neuron.umatrix_value, minUMatrixVal, maxUMatrixVal);
                }
                
                if (isHighlighted) {
                  stroke = isClassifiedBMU ? '#bb9af7' : '#7aa2f7';
                  strokeWidth = '2.5';
                }

                return (
                  <g
                    key={neuron.id}
                    onClick={() => {
                      if (neuron.doc_indices.length > 0) {
                        setSelectedDocId(neuron.doc_indices[0]);
                      }
                    }}
                    className="cursor-pointer group font-mono"
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
                        className="hex-polygon transition-all duration-200 group-hover:fill-opacity-100 group-hover:stroke-tokyo-blue group-hover:stroke-opacity-80"
                      />
                    ) : (
                      <polygon
                        points={pointsStr}
                        fill={fill}
                        fillOpacity={neuron.total_samples === 0 && colorMode === 'class' ? 0.2 : 0.8}
                        stroke={stroke}
                        strokeWidth={strokeWidth}
                        className="hex-polygon transition-all duration-200 group-hover:fill-opacity-100 group-hover:stroke-tokyo-blue group-hover:stroke-opacity-80"
                      />
                    )}

                    <text
                      x={cx}
                      y={cy + 3}
                      textAnchor="middle"
                      fill={colorMode === 'class' ? (neuron.total_samples > 0 ? '#16161e' : '#565f89') : '#ffffff'}
                      fontSize={cols > 10 ? '7px' : '9px'}
                      fontWeight="bold"
                      className="select-none pointer-events-none group-hover:fill-white transition-colors"
                    >
                      {neuron.id}
                    </text>

                    {isHighlighted && (
                      <circle
                        cx={cx}
                        cy={cy}
                        r={r * 0.4}
                        fill="none"
                        stroke={isClassifiedBMU ? "#bb9af7" : "#7aa2f7"}
                        strokeWidth="2"
                        className="animate-ping"
                      />
                    )}

                    <title>
                      {`Neurônio N${neuron.id} (${neuron.col}, ${neuron.row})\n` +
                       `Classe Dominante: ${neuron.dominant_class}\n` +
                       `Amostras: ${neuron.total_samples}\n` +
                       `Pureza: ${(neuron.purity * 100).toFixed(0)}%`}
                    </title>
                  </g>
                );
              })}
            </g>
          </svg>
        )}
      </div>
      
      {/* Legend */}
      {colorMode === 'class' ? (
        <div className="grid grid-cols-5 gap-1.5 mt-4 text-2xs bg-tokyo-dark bg-opacity-30 p-2.5 rounded-lg border border-tokyo-border border-opacity-35">
          {Object.entries(activeColors).map(([name, color]) => (
            <div key={name} className="flex items-center space-x-1 text-tokyo-text">
              <span className="w-2.5 h-2.5 rounded-full flex-shrink-0" style={{ backgroundColor: color }} />
              <span className="truncate">{name}</span>
            </div>
          ))}
          <div className="flex items-center space-x-1 text-tokyo-textDim">
            <span className="w-2.5 h-2.5 rounded-full flex-shrink-0 bg-tokyo-panel border border-dashed border-tokyo-text border-opacity-40" />
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
    </FullscreenPanel>
  );
});
