import { useDashboardStore } from '../store/useDashboardStore';
import { useFullscreen } from '../hooks/useFullscreen';
import { Maximize2, Minimize2 } from 'lucide-react';

export function MetricTable() {
  const { metrics, loadingSynthetic } = useDashboardStore();
  const { isFullscreen, toggleFullscreen } = useFullscreen();

  if (loadingSynthetic) {
    return (
      <div className="glass-panel rounded-2xl p-6 flex flex-col justify-center items-center min-h-[200px]">
        <div className="animate-spin rounded-full h-8 w-8 border-b-2 border-tokyo-blue"></div>
        <span className="text-xs text-tokyo-muted mt-2 font-mono">Carregando métricas...</span>
      </div>
    );
  }

  if (metrics.length === 0) return null;

  return (
    <div 
      className={
        isFullscreen 
          ? "fixed inset-0 bg-[#16161e] bg-opacity-98 z-50 p-8 flex flex-col" 
          : "glass-panel rounded-2xl p-6 flex flex-col"
      }
    >
      <div className="flex justify-between items-center mb-4">
        <h3 className="text-sm font-bold text-tokyo-text uppercase font-mono tracking-wider">
          Comparativo Quantitativo de Modelos (Séries Sintéticas)
        </h3>
        
        <button 
          onClick={toggleFullscreen}
          className="p-1.5 hover:bg-tokyo-panel rounded-lg transition-colors text-tokyo-muted hover:text-tokyo-text"
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
    </div>
  );
}
