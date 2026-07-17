import { memo } from 'react';
import { useDashboardStore } from '../store/useDashboardStore';
import { useFullscreen } from '../hooks/useFullscreen';
import { Maximize2, Minimize2 } from 'lucide-react';
import { FullscreenPanel } from './FullscreenPanel';

export const MetricTable = memo(function MetricTable() {
  const { metrics, loadingSynthetic } = useDashboardStore();
  const { isFullscreen, toggleFullscreen } = useFullscreen();

  if (loadingSynthetic) {
    return (
      <div className="glass-panel rounded-2xl p-6 flex flex-col min-h-[220px] animate-pulse">
        <div className="flex justify-between items-center mb-4">
          <div className="h-4 bg-[#2e3440] rounded w-1/2" />
        </div>
        <div className="border border-tokyo-border border-opacity-40 rounded-lg overflow-hidden space-y-3 p-4 bg-tokyo-dark bg-opacity-30">
          {/* Header row */}
          <div className="grid grid-cols-5 gap-4 border-b border-tokyo-border border-opacity-20 pb-2">
            {Array.from({ length: 5 }).map((_, i) => (
              <div key={i} className="h-3 bg-[#1f2335] rounded" />
            ))}
          </div>
          {/* Data rows */}
          {Array.from({ length: 4 }).map((_, r) => (
            <div key={r} className="grid grid-cols-5 gap-4 py-1">
              {Array.from({ length: 5 }).map((_, c) => (
                <div
                  key={c}
                  className="h-2.5 bg-[#1f2335] rounded animate-pulse"
                  style={{ animationDelay: `${(r * 5 + c) * 30}ms` }}
                />
              ))}
            </div>
          ))}
        </div>
      </div>
    );
  }

  if (metrics.length === 0) return null;

  return (
    <FullscreenPanel
      isFullscreen={isFullscreen}
      className="glass-panel rounded-2xl p-6 flex flex-col"
    >
      <div className="flex justify-between items-center mb-4">
        <h3 className="text-sm font-bold text-tokyo-text uppercase font-mono tracking-wider">
          Comparativo Quantitativo de Modelos (Séries Sintéticas)
        </h3>
        
        <button 
          onClick={toggleFullscreen}
          className="p-1.5 hover:bg-tokyo-panel rounded-lg transition-colors text-tokyo-muted hover:text-tokyo-text active-press-scale"
          title={isFullscreen ? "Sair da Tela Cheia" : "Tela Cheia"}
        >
          {isFullscreen ? <Minimize2 size={16} /> : <Maximize2 size={16} />}
        </button>
      </div>
      
      <div className="rounded-lg border border-tokyo-border border-opacity-40">
        <table className="w-full text-left border-collapse min-w-[700px]">
          <thead>
            <tr className="bg-tokyo-dark bg-opacity-70 text-[10px] text-[#9aa5ce] font-semibold uppercase font-mono border-b border-tokyo-border">
              <th className="px-4 py-3 font-semibold sticky top-0 bg-[#16161e]">Modelo</th>
              <th className="px-4 py-3 font-semibold text-right sticky top-0 bg-[#16161e]">ARI</th>
              <th className="px-4 py-3 font-semibold text-right sticky top-0 bg-[#16161e]">NMI</th>
              <th className="px-4 py-3 font-semibold text-right sticky top-0 bg-[#16161e]">Silhueta</th>
              <th className="px-4 py-3 font-semibold text-right sticky top-0 bg-[#16161e]">Davies-Bouldin</th>
              <th className="px-4 py-3 font-semibold text-right sticky top-0 bg-[#16161e]">Calinski-Harabasz</th>
              <th className="px-4 py-3 font-semibold text-right text-tokyo-blue sticky top-0 bg-[#16161e]">Erro Quant.</th>
              <th className="px-4 py-3 font-semibold text-right text-tokyo-magenta sticky top-0 bg-[#16161e]">Erro Topo.</th>
            </tr>
          </thead>
          <tbody className="divide-y divide-tokyo-border divide-opacity-30 text-xs">
            {metrics.map((row, idx) => {
              const isSOM = row.Modelo.startsWith('SOM');
              return (
                <tr 
                  key={idx} 
                  className={`hover:bg-tokyo-panel hover:bg-opacity-40 transition-colors ${
                    isSOM ? 'text-tokyo-text' : 'text-[#9aa5ce] text-opacity-80 bg-tokyo-dark bg-opacity-20'
                  }`}
                >
                  <td className="px-4 py-3 font-semibold">{row.Modelo}</td>
                  <td className="px-4 py-3 text-right font-mono">{(row.ARI as number).toFixed(4)}</td>
                  <td className="px-4 py-3 text-right font-mono">{(row.NMI as number).toFixed(4)}</td>
                  <td className="px-4 py-3 text-right font-mono">{(row.Silhouette as number).toFixed(4)}</td>
                  <td className="px-4 py-3 text-right font-mono">{(row["Davies-Bouldin"] as number).toFixed(4)}</td>
                  <td className="px-4 py-3 text-right font-mono">{(row["Calinski-Harabasz"] as number).toFixed(2)}</td>
                  <td className="px-4 py-3 text-right font-mono text-tokyo-blue">
                    {row["Erro Quantização"] ? (row["Erro Quantização"] as number).toFixed(4) : '-'}
                  </td>
                  <td className="px-4 py-3 text-right font-mono text-tokyo-magenta">
                    {row["Erro Topográfico"] !== undefined && row["Erro Topográfico"] !== null ? (row["Erro Topográfico"] as number).toFixed(4) : '-'}
                  </td>
                </tr>
              );
            })}
          </tbody>
        </table>
      </div>
      
      <div className="mt-4 p-3 bg-tokyo-dark bg-opacity-40 rounded-lg border border-tokyo-border border-opacity-35 text-[10px] text-[#9aa5ce] leading-relaxed flex justify-between items-center">
        <span>
          <strong>Análise dos Baselines:</strong> Agglomerative e DBSCAN foram calculados no espaço de Z-Score. O SOM 5x5 e 10x10 superam todos os algoritmos tradicionais em ARI/NMI.
        </span>
      </div>
    </FullscreenPanel>
  );
});
