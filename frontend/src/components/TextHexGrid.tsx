import { useMemo, memo } from 'react';
import { useDashboardStore } from '../store/useDashboardStore';
import { useFullscreen } from '../hooks/useFullscreen';
import { Maximize2, Minimize2 } from 'lucide-react';
import { getClassColor, TEXT_CLASS_COLORS } from '../lib/colors';
import { getHexPoints } from '../lib/geometry';
import { FullscreenPanel } from './FullscreenPanel';

export const TextHexGrid = memo(function TextHexGrid() {
  const { 
    selectedTextDataset, 
    selectedTextRep, 
    selectedDocId, 
    setSelectedDocId, 
    textModels, 
    loadingText, 
    classificationResult,
    lattice,
    topology 
  } = useDashboardStore();
  const { isFullscreen, toggleFullscreen } = useFullscreen();
  
  const model = textModels[selectedTextDataset]?.[selectedTextRep];
  const neurons = model?.neurons;
  const cols = model?.cols || 1;
  const rows = model?.rows || 1;

  const padding = 20;
  const svgWidth = isFullscreen ? 800 : 500;
  const svgHeight = isFullscreen ? 550 : 360;

  const { r, neuronLayouts } = useMemo(() => {
    if (!neurons || neurons.length === 0) {
      return { r: 0, neuronLayouts: [] };
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

    return { r: radius, neuronLayouts: layouts };
  }, [neurons, cols, rows, svgWidth, svgHeight]);

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
        
        <button 
          onClick={toggleFullscreen}
          className="p-1.5 hover:bg-tokyo-panel rounded-lg transition-colors text-tokyo-muted hover:text-tokyo-text active-press-scale"
          title={isFullscreen ? "Sair da Tela Cheia" : "Tela Cheia"}
        >
          {isFullscreen ? <Minimize2 size={16} /> : <Maximize2 size={16} />}
        </button>
      </div>
      
      {/* Hex Grid SVG */}
      <div className="flex-1 flex justify-center items-center relative overflow-hidden bg-tokyo-dark bg-opacity-40 rounded-xl border border-tokyo-border border-opacity-30 min-h-[220px]">
        <svg 
          viewBox={`0 0 ${svgWidth} ${svgHeight}`} 
          className="w-full h-full max-h-[460px]"
        >
          <g>
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
              
              if (neuron.total_samples > 0) {
                // Use the color from the active dataset palette
                fill = getClassColor(selectedTextDataset, neuron.dominant_class);
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
                      fillOpacity={neuron.total_samples === 0 ? 0.2 : 0.8}
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
                      fillOpacity={neuron.total_samples === 0 ? 0.2 : 0.8}
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
                    fill={neuron.total_samples > 0 ? '#16161e' : '#565f89'}
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
      
      {/* Legend — always reflects the active dataset's palette */}
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
    </FullscreenPanel>
  );
});
