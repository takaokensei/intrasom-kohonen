import { useState } from 'react';
import { useDashboardStore } from '../store/useDashboardStore';

const MODEL_COLORS: Record<string, string> = {
  "SOM 5x5": "#7aa2f7",     // Blue
  "SOM 7x7": "#7dcfff",     // Cyan
  "SOM 10x10": "#bb9af7",   // Magenta
  "SOM 12x12": "#ff9e64",   // Orange
  "SOM 15x15": "#9ece6a",   // Green
  "SOM 20x20": "#f7768e"    // Red
};

export function RadarChart() {
  const { metrics, loadingSynthetic } = useDashboardStore();
  const [hoveredModel, setHoveredModel] = useState<string | null>(null);

  if (loadingSynthetic || metrics.length === 0) {
    return (
      <div className="glass-panel rounded-2xl p-5 h-full flex items-center justify-center">
        <span className="text-xs text-tokyo-muted font-mono">Carregando Radar...</span>
      </div>
    );
  }

  // Filter only SOM models for radar comparison
  const somMetrics = metrics.filter(m => m.Modelo.startsWith("SOM"));
  
  const axes = [
    { name: "ARI", key: "ARI", max: 0.7 },
    { name: "NMI", key: "NMI", max: 0.85 },
    { name: "Silhueta", key: "Silhouette", max: 0.3 },
    { name: "Pureza", key: "Pureza Neurônios", max: 1.0 },
    { name: "Erro Topo (Inv)", key: "Erro Topográfico", max: 0.3, invert: true }
  ];

  const cx = 150;
  const cy = 130;
  const r = 90; // max radius

  // Helper to calculate coordinates of value on an axis
  const getCoords = (axisIdx: number, val: number, max: number, invert = false) => {
    const angle = (axisIdx * 2 * Math.PI) / axes.length - Math.PI / 2;
    // Normalize value
    let norm = val / max;
    if (invert) norm = 1.0 - norm; // lower is better
    norm = Math.max(0.1, Math.min(1.0, norm)); // clip
    
    const x = cx + r * norm * Math.cos(angle);
    const y = cy + r * norm * Math.sin(angle);
    return { x, y };
  };

  return (
    <div className="glass-panel rounded-2xl p-5 flex flex-col h-full overflow-hidden">
      <h3 className="text-sm font-bold text-tokyo-text mb-2 uppercase font-mono tracking-wider">
        Trade-Off de Tamanhos do SOM
      </h3>
      
      <div className="flex-1 flex flex-col md:flex-row items-center justify-between min-h-0">
        {/* SVG Radar */}
        <div className="w-[300px] h-[250px] flex-shrink-0 flex items-center justify-center">
          <svg viewBox="0 0 300 260" className="w-full h-full">
            {/* Grid circles (spiderweb) */}
            {[0.25, 0.5, 0.75, 1.0].map((scale, sIdx) => {
              const points = [];
              for (let i = 0; i <= axes.length; i++) {
                const angle = (i * 2 * Math.PI) / axes.length - Math.PI / 2;
                points.push(`${cx + r * scale * Math.cos(angle)},${cy + r * scale * Math.sin(angle)}`);
              }
              return (
                <polygon
                  key={sIdx}
                  points={points.join(' ')}
                  fill="none"
                  stroke="rgba(122, 162, 247, 0.15)"
                  strokeWidth="1"
                  strokeDasharray="3 3"
                />
              );
            })}

            {/* Axes lines */}
            {axes.map((axis, aIdx) => {
              const angle = (aIdx * 2 * Math.PI) / axes.length - Math.PI / 2;
              const x2 = cx + r * Math.cos(angle);
              const y2 = cy + r * Math.sin(angle);
              return (
                <g key={aIdx}>
                  <line
                    x1={cx}
                    y1={cy}
                    x2={x2}
                    y2={y2}
                    stroke="rgba(122, 162, 247, 0.25)"
                    strokeWidth="1.5"
                  />
                  {/* Axis Label */}
                  <text
                    x={cx + (r + 15) * Math.cos(angle)}
                    y={cy + (r + 12) * Math.sin(angle) + 4}
                    textAnchor="middle"
                    fill="#565f89"
                    fontSize="9px"
                    fontWeight="bold"
                    className="font-mono uppercase"
                  >
                    {axis.name}
                  </text>
                </g>
              );
            })}

            {/* Model polygons */}
            {somMetrics.map((row) => {
              const isHovered = hoveredModel === row.Modelo;
              const points = axes.map((axis, aIdx) => {
                const val = (row[axis.key as keyof typeof row] as number) || 0;
                const { x, y } = getCoords(aIdx, val, axis.max, axis.invert);
                return `${x},${y}`;
              }).join(' ');

              const color = MODEL_COLORS[row.Modelo] || '#7aa2f7';

              return (
                <polygon
                  key={row.Modelo}
                  points={points}
                  fill={color}
                  fillOpacity={hoveredModel ? (isHovered ? 0.35 : 0.03) : 0.15}
                  stroke={color}
                  strokeWidth={isHovered ? 3 : 1.5}
                  strokeOpacity={hoveredModel ? (isHovered ? 1.0 : 0.1) : 0.8}
                  className="transition-all duration-200"
                />
              );
            })}
          </svg>
        </div>

        {/* Legend Panel & Description */}
        <div className="flex-1 flex flex-col justify-center space-y-2.5 ml-4 self-stretch text-[10px]">
          <span className="text-tokyo-muted uppercase font-mono tracking-wider">Selecione o Modelo:</span>
          
          <div className="grid grid-cols-2 gap-1.5">
            {somMetrics.map(row => (
              <div 
                key={row.Modelo}
                onMouseEnter={() => setHoveredModel(row.Modelo)}
                onMouseLeave={() => setHoveredModel(null)}
                className={`flex items-center space-x-2 p-1.5 rounded border transition-all cursor-pointer ${
                  hoveredModel === row.Modelo 
                    ? 'bg-tokyo-panel border-tokyo-blue' 
                    : 'bg-tokyo-panel bg-opacity-20 border-transparent'
                }`}
              >
                <span 
                  className="w-2.5 h-2.5 rounded-full flex-shrink-0" 
                  style={{ backgroundColor: MODEL_COLORS[row.Modelo] || '#7aa2f7' }} 
                />
                <span className="font-semibold text-tokyo-text">{row.Modelo}</span>
              </div>
            ))}
          </div>

          <p className="text-[9px] text-tokyo-muted leading-relaxed pt-2 border-t border-tokyo-border border-opacity-20">
            Passe o mouse por cima do modelo para comparar o tradeoff: mapas maiores aumentam **Pureza** e **Erro Topográfico**, enquanto os intermediários (7x7, 10x10) otimizam o **ARI** e **NMI**.
          </p>
        </div>
      </div>
    </div>
  );
}
