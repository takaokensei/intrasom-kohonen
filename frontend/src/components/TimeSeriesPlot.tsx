import { useDashboardStore } from '../store/useDashboardStore';
import { useFullscreen } from '../hooks/useFullscreen';
import { Maximize2, Minimize2, Download } from 'lucide-react';

const CLASS_COLORS: Record<string, string> = {
  "Normal": "#7aa2f7",
  "Cyclic": "#7dcfff",
  "Increasing Trend": "#ff9e64",
  "Decreasing Trend": "#e0af68",
  "Upward Shift": "#9ece6a",
  "Downward Shift": "#f7768e"
};

export function TimeSeriesPlot() {
  const { selectedMapSize, selectedNeuronId, highlightedClass, setHighlightedClass, series, somModels, loadingSynthetic } = useDashboardStore();
  const { isFullscreen, toggleFullscreen } = useFullscreen();

  if (loadingSynthetic) {
    return (
      <div className="glass-panel rounded-2xl p-5 h-full flex flex-col justify-center items-center">
        <div className="animate-spin rounded-full h-8 w-8 border-b-2 border-tokyo-blue"></div>
        <span className="text-xs text-tokyo-muted mt-2 font-mono">Carregando séries...</span>
      </div>
    );
  }

  if (series.length === 0) return null;

  // Find min and max values to scale Y axis dynamically
  const allValues = series.flatMap(s => s.values);
  const minVal = Math.min(...allValues) - 2;
  const maxVal = Math.max(...allValues) + 2;
  
  const width = isFullscreen ? 800 : 500;
  const height = isFullscreen ? 450 : 240;
  const padding = 20;
  
  const getSvgPath = (values: number[]) => {
    return values.map((val, step) => {
      const x = padding + (step / 59) * (width - 2 * padding);
      const y = height - padding - ((val - minVal) / (maxVal - minVal || 1)) * (height - 2 * padding);
      return `${step === 0 ? 'M' : 'L'} ${x.toFixed(1)} ${y.toFixed(1)}`;
    }).join(' ');
  };
  
  let displayedSeries = series;
  let codebookPath = '';
  let selectedNeuronInfo = null;
  
  if (selectedNeuronId !== null) {
    const model = somModels[selectedMapSize];
    const neuron = model?.neurons.find(n => n.id === selectedNeuronId);
    
    if (neuron) {
      selectedNeuronInfo = neuron;
      codebookPath = getSvgPath(neuron.codebook);
      displayedSeries = series.filter(s => neuron.sample_ids.includes(s.id));
    }
  } else if (highlightedClass !== null) {
    displayedSeries = series.filter(s => s.class === highlightedClass);
  } else {
    // Show clean subset of 18 series (3 from each class)
    const subset: typeof series = [];
    const classes = Object.keys(CLASS_COLORS);
    classes.forEach(cat => {
      const catSeries = series.filter(s => s.class === cat).slice(0, 3);
      subset.push(...catSeries);
    });
    displayedSeries = subset;
  }

  const downloadSVG = () => {
    const svgEl = document.getElementById('som-time-series-svg');
    if (!svgEl) return;
    const svgString = new XMLSerializer().serializeToString(svgEl);
    const svgBlob = new Blob([svgString], { type: 'image/svg+xml;charset=utf-8' });
    const svgUrl = URL.createObjectURL(svgBlob);
    const downloadLink = document.createElement('a');
    downloadLink.href = svgUrl;
    downloadLink.download = `time_series_${selectedNeuronId !== null ? `neuron_N${selectedNeuronId}` : 'overview'}.svg`;
    document.body.appendChild(downloadLink);
    downloadLink.click();
    document.body.removeChild(downloadLink);
  };

  return (
    <div 
      className={
        isFullscreen 
          ? "fixed inset-0 bg-[#16161e] bg-opacity-98 z-50 p-8 flex flex-col" 
          : "glass-panel rounded-2xl p-5 flex flex-col h-full overflow-hidden"
      }
    >
      <div className="flex justify-between items-start mb-4">
        <div>
          <h3 className="text-sm font-bold text-tokyo-text uppercase font-mono tracking-wider">
            {selectedNeuronId !== null ? `Padrões no Neurônio N${selectedNeuronId}` : 'Visualizador de Sinais Temporais'}
          </h3>
          <p className="text-[10px] text-tokyo-muted font-mono mt-0.5">
            {selectedNeuronId !== null 
              ? `${selectedNeuronInfo?.total_samples} amostras mapeadas` 
              : 'Séries temporais sintéticas do Synthetic Control (UCI)'}
          </p>
        </div>
        
        <div className="flex items-center space-x-2 text-[10px]">
          {selectedNeuronId !== null && selectedNeuronInfo && (
            <div className="flex space-x-2 mr-2">
              <span className="px-2 py-0.5 bg-tokyo-panel rounded border border-tokyo-border font-bold text-tokyo-text uppercase font-mono">
                Classe: {selectedNeuronInfo.dominant_class}
              </span>
              <span className="px-2 py-0.5 bg-tokyo-panel rounded border border-tokyo-border font-bold text-tokyo-blue font-mono">
                Pureza: {(selectedNeuronInfo.purity * 100).toFixed(0)}%
              </span>
            </div>
          )}

          {/* Export SVG */}
          <button 
            onClick={downloadSVG}
            className="p-1.5 hover:bg-tokyo-panel rounded-lg transition-colors text-[#9aa5ce] hover:text-tokyo-text"
            title="Exportar SVG"
          >
            <Download size={16} />
          </button>
          
          {/* Fullscreen */}
          <button 
            onClick={toggleFullscreen}
            className="p-1.5 hover:bg-tokyo-panel rounded-lg transition-colors text-[#9aa5ce] hover:text-tokyo-text"
            title={isFullscreen ? "Sair da Tela Cheia" : "Tela Cheia"}
          >
            {isFullscreen ? <Minimize2 size={16} /> : <Maximize2 size={16} />}
          </button>
        </div>
      </div>
      
      {/* Plot Area */}
      <div className="flex-1 bg-tokyo-dark bg-opacity-40 rounded-xl border border-tokyo-border border-opacity-30 relative overflow-hidden flex items-center justify-center min-h-[160px]">
        <svg id="som-time-series-svg" viewBox={`0 0 ${width} ${height}`} className="w-full h-full">
          {/* Grid lines */}
          {[0, 1, 2, 3, 4].map(idx => {
            const y = padding + (idx / 4) * (height - 2 * padding);
            return (
              <line 
                key={idx} 
                x1={padding} 
                y1={y} 
                x2={width - padding} 
                y2={y} 
                stroke="rgba(122, 162, 247, 0.05)" 
                strokeWidth="1" 
              />
            );
          })}
          
          {/* Mapped series paths */}
          {displayedSeries.map(s => {
            const isHighlighted = highlightedClass === null || s.class === highlightedClass;
            const path = getSvgPath(s.values);
            const color = CLASS_COLORS[s.class];
            
            return (
              <path
                key={s.id}
                d={path}
                fill="none"
                stroke={color}
                strokeWidth={selectedNeuronId !== null ? "1.5" : "1"}
                strokeOpacity={selectedNeuronId !== null ? 0.65 : isHighlighted ? 0.35 : 0.05}
                className="transition-all duration-300"
              />
            );
          })}
          
          {/* Codebook vector path */}
          {selectedNeuronId !== null && codebookPath && (
            <>
              <path
                d={codebookPath}
                fill="none"
                stroke="#7aa2f7"
                strokeWidth="5"
                strokeOpacity="0.4"
                strokeLinecap="round"
                strokeLinejoin="round"
              />
              <path
                d={codebookPath}
                fill="none"
                stroke="#ffffff"
                strokeWidth="2.5"
                strokeLinecap="round"
                strokeLinejoin="round"
              />
            </>
          )}
        </svg>
      </div>
      
      {/* Footer controls */}
      {selectedNeuronId === null ? (
        <div className="flex flex-wrap gap-1.5 mt-3">
          <span className="text-[10px] text-[#9aa5ce] uppercase font-mono self-center mr-1 font-semibold">Filtrar por Classe:</span>
          {Object.keys(CLASS_COLORS).map(cname => {
            const isActive = highlightedClass === cname;
            return (
              <button
                key={cname}
                onClick={() => setHighlightedClass(isActive ? null : cname)}
                className={`px-2 py-0.5 rounded text-[9px] font-mono border transition ${
                  isActive 
                    ? 'bg-tokyo-blue text-tokyo-bg font-bold border-tokyo-blue' 
                    : 'bg-tokyo-panel text-tokyo-text border-tokyo-border hover:bg-opacity-80'
                }`}
              >
                {cname}
              </button>
            );
          })}
        </div>
      ) : (
        <div className="flex justify-between items-center mt-3 text-[10px] text-[#9aa5ce] font-mono">
          <span>Linhas coloridas: séries temporais reais associadas a este neurônio</span>
          <span className="flex items-center gap-1">
            <span className="w-4 h-0.5 bg-white shadow-lg inline-block" />
            <span className="text-tokyo-text font-bold">Vetor de Pesos (Codebook)</span>
          </span>
        </div>
      )}
    </div>
  );
}
