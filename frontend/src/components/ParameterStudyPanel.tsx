import React from 'react';
import { useDashboardStore } from '../store/useDashboardStore';
import type { ParameterStudyEntry } from '../store/useDashboardStore';

const COLUMNS = [
  { key: 'label',               label: 'Configuração',         align: 'left'  },
  { key: 'total_epochs',        label: 'Épocas',               align: 'right' },
  { key: 'radius_initial_pct',  label: 'Raio Inicial (%)',      align: 'right' },
  { key: 'radius_final',        label: 'Raio Final',            align: 'right' },
  { key: 'quantization_error',  label: 'QE ↓',                 align: 'right' },
  { key: 'topographic_error',   label: 'TE ↓',                 align: 'right' },
] as const;

function badge(value: number, min: number, max: number, lowerIsBetter = true): string {
  const norm = (value - min) / (max - min + 1e-9);
  const score = lowerIsBetter ? 1 - norm : norm;
  if (score > 0.66) return 'bg-green-900/60 text-green-300';
  if (score > 0.33) return 'bg-yellow-900/60 text-yellow-300';
  return 'bg-red-900/60 text-red-300';
}

export const ParameterStudyPanel: React.FC = () => {
  const { paramStudyResults, initialRadius, finalRadius, epochs } = useDashboardStore();

  if (!paramStudyResults || paramStudyResults.length === 0) return null;

  const qeValues = paramStudyResults.map(r => r.quantization_error);
  const teValues = paramStudyResults.map(r => r.topographic_error);
  const minQE = Math.min(...qeValues), maxQE = Math.max(...qeValues);
  const minTE = Math.min(...teValues), maxTE = Math.max(...teValues);

  // Determine active row from current param selectors
  const activeEpochs = epochs;
  const activeRin   = initialRadius === '50%' ? 50 : initialRadius === '100%' ? 100 : 80;
  const activeRfin  = finalRadius === '2' ? 2 : 1;

  function isActive(entry: ParameterStudyEntry): boolean {
    return (
      entry.total_epochs === activeEpochs &&
      entry.radius_initial_pct === activeRin &&
      entry.radius_final === activeRfin
    );
  }

  return (
    <div className="mt-6 rounded-xl border border-white/10 bg-white/5 backdrop-blur-sm overflow-hidden">
      <div className="px-4 py-3 border-b border-white/10 flex items-center gap-2">
        <span className="text-xs font-semibold uppercase tracking-widest text-purple-300">
          📊 Estudo de Parâmetros
        </span>
        <span className="text-xs text-white/40 ml-auto">
          Mapa 10×10 · HEX · Toroid · PCA · Batch
        </span>
      </div>
      <div className="overflow-x-auto">
        <table className="w-full text-xs">
          <thead>
            <tr className="border-b border-white/10">
              {COLUMNS.map(col => (
                <th
                  key={col.key}
                  className={`px-3 py-2 text-white/50 font-semibold uppercase tracking-wide text-${col.align}`}
                >
                  {col.label}
                </th>
              ))}
            </tr>
          </thead>
          <tbody>
            {paramStudyResults.map((entry, i) => {
              const active = isActive(entry);
              return (
                <tr
                  key={entry.key}
                  className={`border-b border-white/5 transition-colors ${
                    active
                      ? 'bg-purple-900/30 ring-1 ring-inset ring-purple-500/40'
                      : i % 2 === 0
                      ? 'bg-white/0'
                      : 'bg-white/[0.02]'
                  }`}
                >
                  <td className="px-3 py-2 text-left font-medium text-white/80">
                    {entry.label}
                    {active && (
                      <span className="ml-2 px-1.5 py-0.5 rounded text-[10px] bg-purple-600/60 text-purple-200">
                        ativo
                      </span>
                    )}
                  </td>
                  <td className="px-3 py-2 text-right text-white/60">{entry.total_epochs}</td>
                  <td className="px-3 py-2 text-right text-white/60">{entry.radius_initial_pct}%</td>
                  <td className="px-3 py-2 text-right text-white/60">{entry.radius_final}</td>
                  <td className="px-3 py-2 text-right">
                    <span className={`px-2 py-0.5 rounded ${badge(entry.quantization_error, minQE, maxQE)}`}>
                      {entry.quantization_error.toFixed(4)}
                    </span>
                  </td>
                  <td className="px-3 py-2 text-right">
                    <span className={`px-2 py-0.5 rounded ${badge(entry.topographic_error, minTE, maxTE)}`}>
                      {entry.topographic_error.toFixed(4)}
                    </span>
                  </td>
                </tr>
              );
            })}
          </tbody>
        </table>
      </div>
      <div className="px-4 py-2 border-t border-white/5 text-[10px] text-white/30">
        QE (Erro de Quantização) e TE (Erro Topográfico): valores menores indicam melhor qualidade.
        Linha destacada em roxo = configuração atualmente selecionada nos controles acima.
        Parâmetro variado por experimento; demais fixos na configuração recomendada.
      </div>
    </div>
  );
};

export default ParameterStudyPanel;
