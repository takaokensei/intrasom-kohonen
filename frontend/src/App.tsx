import { useEffect, lazy, Suspense } from 'react';
import { useDashboardStore } from './store/useDashboardStore';
import { ErrorState } from './components/ErrorState';
import { Brain, FileText, LineChart } from 'lucide-react';

// F1: lazy-load both screens so Three.js/R3F (pulled in by the SOM
// visualizations) is only fetched/parsed once a screen actually mounts,
// instead of shipping in the initial bundle regardless of active tab.
const SyntheticScreen = lazy(() =>
  import('./screens/SyntheticScreen').then((m) => ({ default: m.SyntheticScreen }))
);
const TextScreen = lazy(() =>
  import('./screens/TextScreen').then((m) => ({ default: m.TextScreen }))
);

function ScreenFallback() {
  return (
    <div className="flex flex-1 items-center justify-center py-24 text-tokyo-muted text-sm">
      Carregando...
    </div>
  );
}

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
    <div className="min-h-screen bg-tokyo-bg text-tokyo-text flex flex-col relative overflow-x-hidden font-sans">
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
            <p className="text-2xs text-tokyo-textDim font-semibold font-mono tracking-wide uppercase">
              Projeto de NLP & Séries Temporais — Cauã Vitor (UFRN) — Prof. José Alfredo F. Costa
            </p>
          </div>
        </div>

        {/* Tab Selector Navigation Landmark */}
        <nav aria-label="Navegação principal">
          <div role="tablist" aria-label="Seções do analisador" className="flex bg-tokyo-dark bg-opacity-80 p-1 rounded-xl border border-tokyo-border z-10">
            <button
              role="tab"
              id="tab-synthetic"
              aria-selected={activeTab === 'synthetic'}
              aria-controls="panel-synthetic"
              onClick={() => handleTabChange('synthetic')}
              className={`flex items-center gap-2 px-4 py-2 rounded-lg text-xs font-semibold transition-all active-press-scale ${
                activeTab === 'synthetic'
                  ? 'bg-tokyo-blue text-tokyo-bg shadow-lg'
                  : 'text-tokyo-textDim hover:text-tokyo-text'
              }`}
            >
              <LineChart size={14} />
              Synthetic Control
            </button>
            <button
              role="tab"
              id="tab-text"
              aria-selected={activeTab === 'text'}
              aria-controls="panel-text"
              onClick={() => handleTabChange('text')}
              className={`flex items-center gap-2 px-4 py-2 rounded-lg text-xs font-semibold transition-all active-press-scale ${
                activeTab === 'text'
                  ? 'bg-tokyo-blue text-tokyo-bg shadow-lg'
                  : 'text-tokyo-textDim hover:text-tokyo-text'
              }`}
            >
              <FileText size={14} />
              Clusterização de Textos
            </button>
          </div>
        </nav>
      </header>

      {/* Main content grid or Error handling.
          Only the active tab's screen is mounted (F1) — the inactive one is
          neither rendered nor lazy-imported, so its Three.js dependency is
          not fetched until the user actually switches to it. Both tabpanel
          elements stay in the DOM for correct ARIA structure; the inactive
          one is simply empty rather than hidden-but-mounted. */}
      {/* Main content landmark (WCAG AA) */}
      <main role="main" className="grow flex flex-col">
        <div
          role="tabpanel"
          id="panel-synthetic"
          aria-labelledby="tab-synthetic"
          className={`grow flex flex-col ${activeTab === 'synthetic' ? 'animate-tab-change' : 'hidden'}`}
        >
          {activeTab === 'synthetic' && (
            errorSynthetic ? (
              <ErrorState message={errorSynthetic} onRetry={loadSyntheticData} />
            ) : (
              <Suspense fallback={<ScreenFallback />}>
                <SyntheticScreen />
              </Suspense>
            )
          )}
        </div>

        <div
          role="tabpanel"
          id="panel-text"
          aria-labelledby="tab-text"
          className={`grow flex flex-col ${activeTab === 'text' ? 'animate-tab-change' : 'hidden'}`}
        >
          {activeTab === 'text' && (
            errorText ? (
              <ErrorState message={errorText} onRetry={loadTextData} />
            ) : (
              <Suspense fallback={<ScreenFallback />}>
                <TextScreen />
              </Suspense>
            )
          )}
        </div>
      </main>

      {/* Footer bar */}
      <footer className="px-6 py-3 bg-tokyo-dark bg-opacity-90 border-t border-tokyo-border text-3xs text-tokyo-textDim font-semibold flex flex-col md:flex-row justify-between items-center gap-2 md:gap-0 text-center md:text-left z-10">
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
