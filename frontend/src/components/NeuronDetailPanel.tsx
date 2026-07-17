import { X } from 'lucide-react';
import type { NeuronItem } from '../store/useDashboardStore';
import { SYNTHETIC_CLASS_COLORS as CLASS_COLORS } from '../lib/colors';

interface NeuronDetailPanelProps {
  neuron: NeuronItem;
  series: { id: number; values: number[]; class: string }[];
  onClose: () => void;
}

export function NeuronDetailPanel({ neuron, series, onClose }: NeuronDetailPanelProps) {
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

  const codebookPath = getSvgPath(neuron.codebook || []);
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
          {neuron.codebook && (
            <>
              <path d={codebookPath} fill="none"
                stroke={classColor} strokeWidth="7" strokeOpacity="0.25"
                strokeLinecap="round" strokeLinejoin="round" />
              {/* Codebook line */}
              <path d={codebookPath} fill="none"
                stroke="#ffffff" strokeWidth="2.5"
                strokeLinecap="round" strokeLinejoin="round" />
            </>
          )}
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
