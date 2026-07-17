import { AlertTriangle, RefreshCw } from 'lucide-react';

interface ErrorStateProps {
  message: string;
  onRetry: () => void;
}

export function ErrorState({ message, onRetry }: ErrorStateProps) {
  return (
    <div className="grow flex-shrink-0 p-6 flex flex-col items-center justify-center relative z-10 my-12">
      <div className="glass-panel rounded-2xl p-8 max-w-md w-full text-center space-y-4 border border-tokyo-red border-opacity-35">
        <div className="w-12 h-12 bg-tokyo-red bg-opacity-10 text-tokyo-red rounded-full flex items-center justify-center mx-auto">
          <AlertTriangle size={24} />
        </div>
        <h3 className="text-sm font-bold text-tokyo-text uppercase font-mono tracking-wider">Erro ao Carregar Dados</h3>
        <p className="text-xs text-[#9aa5ce] leading-relaxed">{message}</p>
        <button 
          onClick={onRetry} 
          className="w-full py-2.5 bg-tokyo-blue bg-opacity-20 hover:bg-opacity-30 text-tokyo-blue border border-tokyo-blue border-opacity-40 hover:border-opacity-65 rounded-lg text-xs font-bold font-mono uppercase tracking-wider flex items-center justify-center gap-1.5 transition-all"
        >
          <RefreshCw size={12} />
          Tentar Novamente
        </button>
      </div>
    </div>
  );
}
