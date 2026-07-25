import { useState, useMemo, memo } from 'react';
import { useDashboardStore } from '../store/useDashboardStore';
import { useFullscreen } from '../hooks/useFullscreen';
import { Maximize2, Minimize2 } from 'lucide-react';
import { getClassColor, TEXT_CLASS_COLORS, getUMatrixColor } from '../lib/colors';
import { getHexPoints } from '../lib/geometry';
import { FullscreenPanel } from './FullscreenPanel';

export const TextHexGrid = memo(function TextHexGrid() {
  const { 
    selectedTextDataset, 
    selectedDocId, 
    setSelectedDocId, 
    loadingText, 
    classificationResult,
    lattice,
    topology,
    getActiveTextModel
  } = useDashboardStore();
  const { isFullscreen, toggleFullscreen } = useFullscreen();
  const [colorMode, setColorMode] = useState<'class' | 'umatrix'>('class');
  
  const model = getActiveTextModel();
  const neurons = model?.neurons;
  const cols = model?.cols || 1;
  const rows = model?.rows || 1;

  const padding = 20;
  const svgWidth = isFullscreen ? 800 : 500;
  const svgHeight = isFullscreen ? 550 : 360;

  const { r, minUMatrixVal, maxUMatrixVal, neuronLayouts, interstitialCells } = useMemo(() => {
    if (!neurons || neurons.length === 0) {
      return { r: 0, minUMatrixVal: 0, maxUMatrixVal: 0, neuronLayouts: [], interstitialCells: [] };
    }

    const xCoords = neurons.map(n => n.x);
    const yCoords = neurons.map(n => n.y);
    const minX = Math.min(...xCoords);
    const maxX = Math.max(...xCoords);
    const minY = Math.min(...yCoords);
    const maxY = Math.max(...yCoords);

    const scaleX = (x: number) => padding + ((x - minX) / (maxX - minX || 1)) * (svgWidth - 2 * padding);
    const scaleY = (y: number) => padding + ((y - minY) / (maxY - minY || 1)) * (svgHeight - 2 * padding);

    const radius = Math.min(
      (svgWidth - 2 * padding) / (cols * 1.6),
      (svgHeight - 2 * padding) / (rows * 1.45)
    ) * 0.95;

    const uMatrixVals = neurons.map(n => n.umatrix_value);
    const minUVal = Math.min(...uMatrixVals);
    const maxUVal = Math.max(...uMatrixVals);

    const layouts = neurons.map(neuron => {
      const cx = scaleX(neuron.x);
      const cy = scaleY(neuron.y);
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
    const neuronMap = new Map(layouts.map(n => [n.id, n]));

    const interstitials = edges.map((edge, idx) => {
      const n1 = neuronMap.get(edge.from);
      const n2 = neuronMap.get(edge.to);
      if (!n1 || !n2) return null;
      const cx = (n1.cx + n2.cx) / 2;
      const cy = (n1.cy + n2.cy) / 2;
      const fill = getUMatrixColor(edge.distance, eMin, eMax);
      return {
        key: `edge-${edge.from}-${edge.to}-${idx}`,
        cx,
        cy,
        fill,
        distance: edge.distance,
        from: edge.from,
        to: edge.to
      };
    }).filter(Boolean) as Array<{ key: string; cx: number; cy: number; fill: string; distance: number; from: number; to: number }>;

    return {
      r: radius,
      minUMatrixVal: minUVal,
      maxUMatrixVal: maxUVal,
      neuronLayouts: layouts,
      interstitialCells: interstitials
    };
  }, [neurons, cols, rows, svgWidth, svgHeight, model]);

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
          <div className="flex items-center gap-2">
            <h3 className="text-sm font-bold text-tokyo-text uppercase font-mono tracking-wider">
              Malha {lattice === 'HEX' ? 'Hexagonal (HEX)' : 'Retangular (RECT)'} - Notícias (10x10)
            </h3>
            <span className={`text-[9px] font-mono px-2 py-0.5 rounded border font-semibold ${
              topology === 'toroid'
                ? 'bg-tokyo-magenta bg-opacity-10 text-tokyo-magenta border-tokyo-magenta border-opacity-30'
                : 'bg-tokyo-yellow bg-opacity-10 text-tokyo-yellow border-tokyo-yellow border-opacity-30'
            }`}>
              {topology === 'toroid' ? 'Toroide ON' : 'Plana (Sem Karnaugh)'}
            </span>
          </div>
          <p className="text-[10px] text-tokyo-muted font-mono mt-0.5">
            {selectedTextDataset === '20news' 
              ? '4 categorias do dataset 20 Newsgroups (400 documentos)'
              : '6 categorias do dataset 6News com Texto Expandido (317 documentos)'}
          </p>
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

          <button 
            onClick={toggleFullscreen}
            className="p-1.5 hover:bg-tokyo-panel rounded-lg transition-colors text-tokyo-muted hover:text-tokyo-text active-press-scale"
            title={isFullscreen ? "Sair da Tela Cheia" : "Tela Cheia"}
          >
            {isFullscreen ? <Minimize2 size={16} /> : <Maximize2 size={16} />}
          </button>
        </div>
      </div>
      
      {/* Hex Grid SVG */}
      <div className="flex-1 flex justify-center items-center relative overflow-hidden bg-tokyo-dark bg-opacity-40 rounded-xl border border-tokyo-border border-opacity-30 min-h-[220px]">
        <svg 
          viewBox={`0 0 ${svgWidth} ${svgHeight}`} 
          className="w-full h-full max-h-[460px]"
        >
          <g>
            {colorMode === 'umatrix' && interstitialCells.map(cell => (
              <circle
                key={cell.key}
                cx={cell.cx}
                cy={cell.cy}
                r={r * 0.45}
                fill={cell.fill}
                fillOpacity={0.9}
                stroke="rgba(0,0,0,0.3)"
                strokeWidth="0.5"
                className="transition-all duration-200"
              >
                <title>{`Distância U-Matrix (N${cell.from} ↔ N${cell.to}): ${cell.distance.toFixed(3)}`}</title>
              </circle>
            ))}
            {neuronLayouts.map((neuron, index) => {
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
                stroke = '#ffffff';
                strokeWidth = '2.5';
              }
              
              const delay = index * (500 / neuronLayouts.length);
              
              return (
                <g 
                  key={neuron.id}
                  role="button"
                  tabIndex={0}
                  aria-label={`Neurônio N${neuron.id}, Classe Dominante: ${neuron.total_samples > 0 ? neuron.dominant_class : 'Vazio'}, Amostras: ${neuron.total_samples}, Pureza: ${(neuron.purity * 100).toFixed(0)}%`}
                  className="cursor-pointer group focus:outline-none som-hex-group animate-hex-entrance"
                  onClick={() => {
                    if (neuron.doc_indices.length > 0) {
                      setSelectedDocId(neuron.doc_indices[0]);
                    }
                  }}
                  onKeyDown={(e) => {
                    if (e.key === 'Enter' || e.key === ' ') {
                      e.preventDefault();
                      if (neuron.doc_indices.length > 0) {
                        setSelectedDocId(neuron.doc_indices[0]);
                      }
                    }
                  }}
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
                    fontSize="7px"
                    fontWeight="bold"
                    className="select-none pointer-events-none group-hover:fill-white"
                  >
                    {neuron.id}
                  </text>
                  
                  {isClassifiedBMU && (
                    <circle
                      cx={cx}
                      cy={cy}
                      r={r * 0.8}
                      fill="none"
                      stroke="#ffffff"
                      strokeWidth="2"
                      className="ripple-circle pointer-events-none"
                      style={{
                        transformOrigin: `${cx}px ${cy}px`
                      }}
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
      </div>
      
      {/* Legend */}
      {colorMode === 'class' ? (
        <div className="grid grid-cols-5 gap-1.5 mt-4 text-[10px] bg-tokyo-dark bg-opacity-30 p-2.5 rounded-lg border border-tokyo-border border-opacity-35">
          {Object.entries(activeColors).map(([name, color]) => (
            <div key={name} className="flex items-center space-x-1 text-tokyo-text">
              <span className="w-2.5 h-2.5 rounded-full flex-shrink-0" style={{ backgroundColor: color }} />
              <span className="truncate">{name}</span>
            </div>
          ))}
          <div className="flex items-center space-x-1 text-[#9aa5ce]">
            <span className="w-2.5 h-2.5 rounded-full flex-shrink-0 bg-tokyo-panel border border-dashed border-tokyo-text border-opacity-40" />
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
    </FullscreenPanel>
  );
});
