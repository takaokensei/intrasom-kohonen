import { create } from 'zustand';
import { fetchSyntheticData, fetchTextData, checkBackendHealth } from '../lib/dataLoader';
import { classifyTextPure } from '../lib/classification';
import type { PCAParameters } from '../lib/pca';

export type TabType = 'synthetic' | 'text';

export interface SeriesItem {
  id: number;
  values: number[];
  class: string;
}

export interface NeuronItem {
  id: number;
  x: number;
  y: number;
  row: number;
  col: number;
  umatrix_value: number;
  dominant_class: string;
  purity: number;
  total_samples: number;
  sample_ids: number[];
  codebook: number[];
}

export interface UMatrixEdge {
  from: number;
  to: number;
  distance: number;
}

export interface SOMModel {
  cols: number;
  rows: number;
  neurons: NeuronItem[];
  umatrix_edges?: UMatrixEdge[];
  umatrix_edge_min?: number;
  umatrix_edge_max?: number;
}

// All 6 map sizes ship 4 real variants: HEX_toroid, HEX_planar, RECT_planar, RECT_toroid
export interface SOMModelWithVariants {
  has_variants: true;
  HEX_toroid?: SOMModel;
  HEX_planar?: SOMModel;
  RECT_planar?: SOMModel;
  RECT_toroid?: SOMModel;  // IntraSOM 1.1.1 — _rect_dist_tor corrigida
}

// Text models ship 4 variants: HEX_toroid, HEX_planar, RECT_planar, RECT_toroid
export interface TextModelWithVariants {
  has_variants: true;
  HEX_toroid?: TextModel;
  HEX_planar?: TextModel;
  RECT_planar?: TextModel;
  RECT_toroid?: TextModel;  // IntraSOM 1.1.1 — mesmo motor que HEX
}


export interface ParameterStudyEntry {
  key: string;
  label: string;
  total_epochs: number;
  rough_epochs: number;
  finetune_epochs: number;
  radius_initial: number;
  radius_final: number;
  radius_initial_pct: number;
  quantization_error: number;
  topographic_error: number;
}

export interface MetricRow {
  Modelo: string;
  lattice?: string;
  topology?: string;
  variant?: string;
  ARI: number;
  NMI: number;
  Silhouette: number;
  "Davies-Bouldin": number;
  "Calinski-Harabasz": number;
  "Pureza Neurônios"?: number;
  "Erro Quantização"?: number;
  "Erro Topográfico"?: number;
}

export interface TextNeuronItem {
  id: number;
  x: number;
  y: number;
  row: number;
  col: number;
  umatrix_value: number;
  dominant_class: string;
  purity: number;
  total_samples: number;
  doc_indices: number[];
  codebook?: number[];
}

export interface TextModel {
  cols: number;
  rows: number;
  neurons: TextNeuronItem[];
  umatrix_edges?: UMatrixEdge[];
  umatrix_edge_min?: number;
  umatrix_edge_max?: number;
}

export interface NewsSample {
  id: number;
  text: string;
  class: string;
}

interface DashboardState {
  activeTab: TabType;

  // Dynamic Data
  series: SeriesItem[];
  somModels: Record<string, SOMModel | SOMModelWithVariants>;
  metrics: MetricRow[];
  textModels: Record<string, Record<string, TextModel | TextModelWithVariants>>;
  textMetrics: Record<string, Record<string, { ARI: number; NMI: number }>>;
  newsSamples: Record<string, NewsSample[]>;
  paramStudyResults: ParameterStudyEntry[];
  selectedTextDataset: '20news' | '6class';
  setSelectedTextDataset: (dataset: '20news' | '6class') => void;

  // Derived selectors: return the correct model for the current lattice/topology
  getActiveSOMModel: () => SOMModel | null;
  // Routes to HEX_toroid or RECT_planar based on global lattice state
  getActiveTextModel: () => TextModel | null;
  
  // Loading states
  loadingSynthetic: boolean;
  loadingText: boolean;
  
  // Synthetic Control
  selectedMapSize: '5x5' | '7x7' | '10x10' | '12x12' | '15x15' | '20x20';
  selectedNeuronId: number | null;
  highlightedClass: string | null;
  
  // Text SOM
  selectedTextRep: 'SBERT' | 'TF-IDF' | 'BGE-M3' | 'Gemma-300M';
  selectedDocId: number | null;
  customTextQuery: string;
  classificationResult: {
    bmu: number;
    dominantClass: string;
    purity: number;
    score: number;
    source: 'local' | 'cloud' | 'fallback';
  } | null;
  backendOnline: boolean | null;
  pcaParams: Record<string, PCAParameters | Record<string, PCAParameters>> | null;
  errorSynthetic: string | null;
  errorText: string | null;
  
  // SOM Algorithm Parameters (Interactive Controls requested by Professor)
  lattice: 'HEX' | 'RECT';
  topology: 'toroid' | 'planar';
  initialRadius: '80%' | '50%' | '100%';
  finalRadius: '1' | '2';
  epochs: 500 | 200 | 100;
  
  setLattice: (lattice: 'HEX' | 'RECT') => void;
  setTopology: (topology: 'toroid' | 'planar') => void;
  setInitialRadius: (radius: '80%' | '50%' | '100%') => void;
  setFinalRadius: (radius: '1' | '2') => void;
  setEpochs: (epochs: 500 | 200 | 100) => void;

  getServedTopology: () => 'toroid' | 'planar';
  
  // Actions
  setActiveTab: (tab: TabType) => void;
  loadSyntheticData: () => Promise<void>;
  loadTextData: () => Promise<void>;
  setSelectedMapSize: (size: '5x5' | '7x7' | '10x10' | '12x12' | '15x15' | '20x20') => void;
  setSelectedNeuronId: (id: number | null) => void;
  setHighlightedClass: (className: string | null) => void;
  setSelectedTextRep: (rep: 'SBERT' | 'TF-IDF' | 'BGE-M3' | 'Gemma-300M') => void;
  setSelectedDocId: (id: number | null) => void;
  setCustomTextQuery: (query: string) => void;
  classifyText: (text: string) => void;
  resetClassification: () => void;
  checkBackend: () => Promise<void>;
}

export const useDashboardStore = create<DashboardState>((set, get) => ({
  activeTab: 'synthetic',
  
  // Data initial states
  series: [],
  somModels: {},
  metrics: [],
  textModels: {},
  textMetrics: {},
  newsSamples: {},
  paramStudyResults: [],
  
  // Loading initial states
  loadingSynthetic: false,
  loadingText: false,
  
  // Synthetic Control Defaults
  selectedMapSize: '10x10',
  selectedNeuronId: null,
  highlightedClass: null,
  
  // Text SOM Defaults
  selectedTextDataset: '20news',
  selectedTextRep: 'SBERT',
  selectedDocId: null,
  customTextQuery: '',
  classificationResult: null,
  backendOnline: null,
  pcaParams: null,
  errorSynthetic: null,
  errorText: null,
  // SOM Algorithm Parameter Defaults (Requested by Professor)
  lattice: 'HEX',
  topology: 'toroid',
  initialRadius: '80%',
  finalRadius: '1',
  epochs: 500,

  setLattice: (lattice) => set({ lattice }),
  // setTopology now selects the real trained variant for 10x10;
  // for other map sizes, only topology state is updated (no variant data yet)
  setTopology: (topology) => set({ topology }),
  setInitialRadius: (initialRadius) => set({ initialRadius }),
  setFinalRadius: (finalRadius) => set({ finalRadius }),
  setEpochs: (epochs) => set({ epochs }),

  getServedTopology: () => {
    // RECT_toroid now exists — return the real topology for all 4 variants
    const { topology } = get();
    return topology;
  },

  // Actions
  setSelectedTextDataset: (selectedTextDataset) => set({ 
    selectedTextDataset, 
    selectedDocId: null, 
    classificationResult: null, 
    customTextQuery: '' 
  }),
  
  // Actions
  setActiveTab: (activeTab) => {
    set({ activeTab });
    if (activeTab === 'synthetic') {
      get().loadSyntheticData();
    } else {
      get().loadTextData();
    }
  },
  
  loadSyntheticData: async () => {
    if (get().series.length > 0) return; // Already loaded

    set({ loadingSynthetic: true, errorSynthetic: null });
    try {
      const { series, somModels, metrics, paramStudyResults } = await fetchSyntheticData();
      set({ series, somModels, metrics, paramStudyResults, loadingSynthetic: false });
    } catch (err) {
      console.error('Error loading synthetic control data:', err);
      set({
        loadingSynthetic: false,
        errorSynthetic:
          'Falha ao carregar dados sintéticos do SOM. Verifique a existência dos arquivos JSON na pasta public/data/.',
      });
    }
  },

  // Returns the active SOMModel based on selected map size, lattice and topology.
  // Routes to real pre-trained IntraSOM variants across all 6 map sizes:
  // - RECT lattice + toroid topology -> RECT_toroid (IntraSOM 1.1.1)
  // - RECT lattice + planar topology -> RECT_planar (IntraSOM 1.1.1)
  // - HEX lattice + planar topology  -> HEX_planar  (IntraSOM)
  // - HEX lattice + toroid topology  -> HEX_toroid  (IntraSOM)
  getActiveSOMModel: () => {
    const { somModels, selectedMapSize, lattice, topology } = get();
    const entry = somModels[selectedMapSize];
    if (!entry) return null;
    if ('has_variants' in entry) {
      const variantKey = lattice === 'RECT'
        ? (topology === 'toroid' ? 'RECT_toroid' : 'RECT_planar')
        : (topology === 'planar' ? 'HEX_planar' : 'HEX_toroid');
      return (entry as SOMModelWithVariants)[variantKey] ?? null;
    }
    return entry as SOMModel;
  },

  // Returns the active TextModel based on global lattice & topology state.
  // Routes to real pre-trained IntraSOM variants:
  // - RECT lattice + toroid topology -> RECT_toroid (IntraSOM 1.1.1)
  // - RECT lattice + planar topology -> RECT_planar (IntraSOM 1.1.1)
  // - HEX lattice + planar topology  -> HEX_planar  (IntraSOM, normalization='var')
  // - HEX lattice + toroid topology  -> HEX_toroid  (IntraSOM, normalization='var')
  getActiveTextModel: () => {
    const { textModels, selectedTextDataset, selectedTextRep, lattice, topology } = get();
    const entry = textModels[selectedTextDataset]?.[selectedTextRep];
    if (!entry) return null;
    if ('has_variants' in entry) {
      const variantKey = lattice === 'RECT'
        ? (topology === 'toroid' ? 'RECT_toroid' : 'RECT_planar')
        : (topology === 'planar' ? 'HEX_planar' : 'HEX_toroid');
      return (entry as TextModelWithVariants)[variantKey] ?? null;
    }
    return entry as TextModel;
  },

  loadTextData: async () => {
    if (Object.keys(get().newsSamples).length > 0) {
      get().checkBackend();
      return; // Already loaded
    }
    
    set({ loadingText: true, errorText: null });
    try {
      const { textModels, textMetrics, newsSamples, pcaParams } = await fetchTextData();
      set({ textModels, textMetrics, newsSamples, pcaParams, loadingText: false });
      get().checkBackend();
    } catch (err) {
      console.error("Error loading text news data:", err);
      set({ 
        loadingText: false, 
        errorText: "Falha ao carregar os dados textuais do SOM. Verifique a existência dos arquivos JSON na pasta public/data/." 
      });
    }
  },
  
  setSelectedMapSize: (selectedMapSize) => set({ selectedMapSize, selectedNeuronId: null }),
  setSelectedNeuronId: (selectedNeuronId) => set({ selectedNeuronId }),
  setHighlightedClass: (highlightedClass) => set({ highlightedClass }),
  setSelectedTextRep: (selectedTextRep) => set({ selectedTextRep }),
  setSelectedDocId: (selectedDocId) => set({ selectedDocId }),
  setCustomTextQuery: (customTextQuery) => set({ customTextQuery }),
  
  classifyText: async (text) => {
    const { selectedTextRep, selectedTextDataset, lattice, pcaParams, newsSamples } = get();
    const model = get().getActiveTextModel();
    const hfToken = import.meta.env.VITE_HF_TOKEN;

    const dsPca = pcaParams?.[selectedTextDataset];
    const activePca: PCAParameters | null = dsPca
      ? ('mean' in dsPca 
          ? (dsPca as PCAParameters) 
          : ((dsPca as Record<string, PCAParameters>)[selectedTextRep] ?? (dsPca as Record<string, PCAParameters>)['SBERT'] ?? null))
      : null;

    const outcome = await classifyTextPure({
      text,
      representation: selectedTextRep,
      dataset: selectedTextDataset,
      lattice,
      model,
      pcaParams: activePca,
      newsSamples: newsSamples[selectedTextDataset] || [],
      hfToken,
    });

    set({
      classificationResult: outcome.classificationResult,
      backendOnline: outcome.backendOnline,
    });
  },
  
  resetClassification: () => set({ classificationResult: null, customTextQuery: '' }),
  
  checkBackend: async () => {
    const isOnline = await checkBackendHealth();
    set({ backendOnline: isOnline });
  }
}));
