import { useDashboardStore } from '../store/useDashboardStore';
import { useFullscreen } from '../hooks/useFullscreen';
import { Maximize2, Minimize2 } from 'lucide-react';

const NEWS_COLORS: Record<string, string> = {
  "Graphics": "#3182bd",
  "Space": "#31a354",
  "Baseball": "#e6550d",
  "Mideast": "#756bb1"
};

export function TextHexGrid() {
  const { selectedTextRep, selectedDocId, setSelectedDocId, textModels, loadingText, classificationResult } = useDashboardStore();
  const { isFullscreen, toggleFullscreen } = useFullscreen();
  
  if (loadingText) {
    return (
      <div className="glass-panel rounded-2xl p-5 h-full flex flex-col justify-center items-center">
        <div className="animate-spin rounded-full h-8 w-8 border-b-2 border-tokyo-blue"></div>
        <span className="text-xs text-tokyo-muted mt-2 font-mono">Carregando mapa de textos...</span>
      </div>
    );
  }

  const model = textModels[selectedTextRep];
  if (!model) return null;
  
  const { cols, rows, neurons } = model;
  
  const xCoords = neurons.map(n => n.x);
  const yCoords = neurons.map(n => n.y);
  const minX = Math.min(...xCoords);
  const maxX = Math.max(...xCoords);
  const minY = Math.min(...yCoords);
  const maxY = Math.max(...yCoords);

  const padding = 20;
  const svgWidth = isFullscreen ? 800 : 500;
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

  return (
    <div 
      className={
        isFullscreen 
          ? "fixed inset-0 bg-[#16161e] bg-opacity-98 z-50 p-8 flex flex-col" 
          : "glass-panel rounded-2xl p-5 flex flex-col h-full overflow-hidden"
      }
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
          <h3 className="text-sm font-bold text-tokyo-text uppercase font-mono tracking-wider">
            Organização Semântica de Notícias (10x10)
          </h3>
          <p className="text-[10px] text-tokyo-muted font-mono mt-0.5">
            4 categorias do dataset 20 Newsgroups (400 documentos)
          </p>
        </div>
        
        <button 
          onClick={toggleFullscreen}
          className="p-1.5 hover:bg-tokyo-panel rounded-lg transition-colors text-tokyo-muted hover:text-tokyo-text"
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
            {neurons.map((neuron) => {
              const cx = scaleX(neuron.x);
              const cy = scaleY(neuron.y);
              
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
                fill = NEWS_COLORS[neuron.dominant_class] || '#1f2335';
              }
              
              if (isHighlighted) {
                stroke = '#ffffff';
                strokeWidth = '2.5';
              }
              
              const pointsStr = getHexPoints(cx, cy, r);
              
              return (
                <g 
                  key={neuron.id}
                  className="cursor-pointer group"
                  onClick={() => {
                    if (neuron.doc_indices.length > 0) {
                      setSelectedDocId(neuron.doc_indices[0]);
                    }
                  }}
                >
                  <polygon
                    points={pointsStr}
                    fill={fill}
                    fillOpacity={neuron.total_samples === 0 ? 0.2 : 0.8}
                    stroke={stroke}
                    strokeWidth={strokeWidth}
                    className="transition-all duration-200 group-hover:fill-opacity-100 group-hover:stroke-tokyo-blue group-hover:stroke-opacity-80"
                  />
                  
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
                      style={{ transformOrigin: `${cx}px ${cy}px` }}
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
      <div className="grid grid-cols-5 gap-1.5 mt-4 text-[10px] bg-tokyo-dark bg-opacity-30 p-2.5 rounded-lg border border-tokyo-border border-opacity-35">
        {Object.entries(NEWS_COLORS).map(([name, color]) => (
          <div key={name} className="flex items-center space-x-1 text-tokyo-text">
            <span className="w-2.5 h-2.5 rounded-full flex-shrink-0" style={{ backgroundColor: color }} />
            <span className="truncate">{name}</span>
          </div>
        ))}
        <div className="flex items-center space-x-1 text-tokyo-muted">
          <span className="w-2.5 h-2.5 rounded-full flex-shrink-0 bg-tokyo-panel border border-dashed border-tokyo-muted" />
          <span>Vazio</span>
        </div>
      </div>
    </div>
  );
}
