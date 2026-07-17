import { useEffect } from 'react';
import { useDashboardStore } from './store/useDashboardStore';
import { ErrorState } from './components/ErrorState';
import { SyntheticScreen } from './screens/SyntheticScreen';
import { TextScreen } from './screens/TextScreen';
import { Brain, FileText, LineChart } from 'lucide-react';

function App() {
  const {
    activeTab,
    setActiveTab,
    loadSyntheticData,
    loadTextData,
    errorSynthetic,
    errorText,
    selectedTextDataset
  } = useDashboardStore();

  // Load initial data on mount
  useEffect(() => {
    loadSyntheticData();
  }, [loadSyntheticData]);

  // Handle tab routing & data loading
  const handleTabChange = (tab: 'synthetic' | 'text') => {
    setActiveTab(tab);
    if (tab === 'synthetic') {
      loadSyntheticData();
    } else {
      loadTextData();
    }
  };

  return (
    <div className="min-h-screen bg-[#1a1b26] text-[#a9b1d6] flex flex-col relative overflow-x-hidden font-sans">
      {/* Decorative neon ambient glows */}
      <div className="glow-spot-blue -top-20 -left-20" />
      <div className="glow-spot-purple bottom-10 right-10" />

      {/* Header bar */}
      <header className="px-6 py-4 border-b border-tokyo-border bg-tokyo-dark bg-opacity-75 backdrop-blur-md z-10 flex flex-col md:flex-row justify-between items-center gap-4">
        <div className="flex items-center space-x-3.5">
          <div className="p-2.5 bg-tokyo-blue bg-opacity-10 rounded-xl border border-tokyo-blue border-opacity-25 text-tokyo-blue">
            <Brain size={26} className="animate-float" />
          </div>
          <div>
            <h1 className="text-lg font-bold text-tokyo-text flex items-center gap-2">
              IntraSOM Kohonen Maps Analyzer
            </h1>
            <p className="text-[10px] text-[#9aa5ce] font-semibold font-mono tracking-wide uppercase">
              Projeto de NLP & Séries Temporais — Cauã Vitor (UFRN) — Prof. José Alfredo F. Costa
            </p>
          </div>
        </div>

        {/* Tab Selector */}
        <div className="flex bg-tokyo-dark bg-opacity-80 p-1 rounded-xl border border-tokyo-border z-10">
          <button
            onClick={() => handleTabChange('synthetic')}
            className={`flex items-center gap-2 px-4 py-2 rounded-lg text-xs font-semibold transition-all active-press-scale ${
              activeTab === 'synthetic'
                ? 'bg-tokyo-blue text-tokyo-bg shadow-lg'
                : 'text-[#9aa5ce] hover:text-tokyo-text'
            }`}
          >
            <LineChart size={14} />
            Synthetic Control
          </button>
          <button
            onClick={() => handleTabChange('text')}
            className={`flex items-center gap-2 px-4 py-2 rounded-lg text-xs font-semibold transition-all active-press-scale ${
              activeTab === 'text'
                ? 'bg-tokyo-blue text-tokyo-bg shadow-lg'
                : 'text-[#9aa5ce] hover:text-tokyo-text'
            }`}
          >
            <FileText size={14} />
            Clusterização de Textos
          </button>
        </div>
      </header>

      {/* Main content grid or Error handling */}
      {activeTab === 'synthetic' ? (
        errorSynthetic ? (
          <ErrorState message={errorSynthetic} onRetry={loadSyntheticData} />
        ) : (
          <div key="synthetic" className="grow flex flex-col animate-tab-change">
            <SyntheticScreen />
          </div>
        )
      ) : (
        errorText ? (
          <ErrorState message={errorText} onRetry={loadTextData} />
        ) : (
          <div key="text" className="grow flex flex-col animate-tab-change">
            <TextScreen />
          </div>
        )
      )}

      {/* Footer bar */}
      <footer className="px-6 py-3 bg-tokyo-dark bg-opacity-90 border-t border-tokyo-border text-[9.5px] text-[#9aa5ce] font-semibold flex flex-col md:flex-row justify-between items-center gap-2 md:gap-0 text-center md:text-left z-10">
        <span>
          Base de Dados: 600 séries temporais (Synthetic Control) | {selectedTextDataset === '20news' ? '400 notícias (20 Newsgroups)' : '317 notícias (Base Acadêmica 6 Classes)'}
        </span>
        <div className="flex flex-wrap justify-center gap-x-4 gap-y-1">
          <span>USP IntraSOM Library Integration</span>
          <span>Tokyo Night Design System</span>
          <span>Vite + React + TS v19.2</span>
        </div>
      </footer>
    </div>
  );
}

export default App;
