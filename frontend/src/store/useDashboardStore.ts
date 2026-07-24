import { create } from 'zustand';
import { projectAndFindBMU } from '../lib/pca';
import { classifyByKeywords } from '../lib/keywordClassifier';

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

export interface SOMModel {
  cols: number;
  rows: number;
  neurons: NeuronItem[];
}

// All 6 map sizes ship 3 real variants: HEX_toroid, HEX_planar, RECT_planar
export interface SOMModelWithVariants {
  has_variants: true;
  HEX_toroid?: SOMModel;
  HEX_planar?: SOMModel;
  RECT_planar?: SOMModel;
}

// Text models ship 2 variants: HEX_toroid and RECT_planar (no planar variant for text)
export interface TextModelWithVariants {
  has_variants: true;
  HEX_toroid?: TextModel;
  RECT_planar?: TextModel;
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
  selectedMapSize: '10x10' | '15x15' | '20x20';
  selectedNeuronId: number | null;
  highlightedClass: string | null;
  
  // Text SOM
  selectedTextRep: 'SBERT' | 'TF-IDF';
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
  pcaParams: Record<string, { mean: number[]; components: number[][] }> | null;
  errorSynthetic: string | null;
  errorText: string | null;
  
  // SOM Algorithm Parameters (Interactive Controls requested by Professor)
  lattice: 'HEX' | 'RECT';
  topology: 'toroid' | 'planar';
  initialRadius: '80%' | '50%' | '100%';
  finalRadius: '1' | '2';
  epochs: 500 | 200 | 100;
  trainingMode: 'batch' | 'online';
  initialization: 'linear' | 'random';
  
  setLattice: (lattice: 'HEX' | 'RECT') => void;
  setTopology: (topology: 'toroid' | 'planar') => void;
  setInitialRadius: (radius: '80%' | '50%' | '100%') => void;
  setFinalRadius: (radius: '1' | '2') => void;
  setEpochs: (epochs: 500 | 200 | 100) => void;
  setTrainingMode: (mode: 'batch' | 'online') => void;
  setInitialization: (init: 'linear' | 'random') => void;

  // Actions
  setActiveTab: (tab: TabType) => void;
  loadSyntheticData: () => Promise<void>;
  loadTextData: () => Promise<void>;
  setSelectedMapSize: (size: '10x10' | '15x15' | '20x20') => void;
  setSelectedNeuronId: (id: number | null) => void;
  setHighlightedClass: (className: string | null) => void;
  setSelectedTextRep: (rep: 'SBERT' | 'TF-IDF') => void;
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
  trainingMode: 'batch',
  initialization: 'linear',

  setLattice: (lattice) => set({ lattice }),
  // setTopology now selects the real trained variant for 10x10;
  // for other map sizes, only topology state is updated (no variant data yet)
  setTopology: (topology) => set({ topology }),
  setInitialRadius: (initialRadius) => set({ initialRadius }),
  setFinalRadius: (finalRadius) => set({ finalRadius }),
  setEpochs: (epochs) => set({ epochs }),
  setTrainingMode: (trainingMode) => set({ trainingMode }),
  setInitialization: (initialization) => set({ initialization }),

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
      const [seriesRes, modelsRes, metricsRes, studyRes] = await Promise.all([
        fetch('/data/series.json'),
        fetch('/data/som_models.json'),
        fetch('/data/metrics.json'),
        fetch('/data/parameter_study.json').catch(() => null),
      ]);

      if (!seriesRes.ok || !modelsRes.ok || !metricsRes.ok) {
        throw new Error('HTTP status error loading synthetic control files');
      }

      const series = await seriesRes.json();
      const somModels = await modelsRes.json();
      const metrics = await metricsRes.json();
      const paramStudyResults: ParameterStudyEntry[] =
        studyRes && studyRes.ok ? await studyRes.json() : [];

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
  // Routes to real pre-trained variants across all 6 map sizes:
  // - RECT lattice -> RECT_planar (MiniSom)
  // - HEX lattice + planar topology -> HEX_planar (IntraSOM)
  // - HEX lattice + toroid topology -> HEX_toroid (IntraSOM)
  getActiveSOMModel: () => {
    const { somModels, selectedMapSize, lattice, topology } = get();
    const entry = somModels[selectedMapSize];
    if (!entry) return null;
    if ('has_variants' in entry) {
      const variantKey = lattice === 'RECT'
        ? 'RECT_planar'
        : (topology === 'planar' ? 'HEX_planar' : 'HEX_toroid');
      return (entry as SOMModelWithVariants)[variantKey] ?? null;
    }
    return entry as SOMModel;
  },

  // Returns the active TextModel based on the global lattice state.
  // Routes to real pre-trained variants:
  // - RECT lattice -> RECT_planar (MiniSom, Z-score normalized, pure rect grid coords)
  // - HEX lattice  -> HEX_toroid  (IntraSOM, normalization='var')
  // Mirrors getActiveSOMModel() pattern. No HEX_planar variant exists for text models.
  getActiveTextModel: () => {
    const { textModels, selectedTextDataset, selectedTextRep, lattice } = get();
    const entry = textModels[selectedTextDataset]?.[selectedTextRep];
    if (!entry) return null;
    if ('has_variants' in entry) {
      const variantKey = lattice === 'RECT' ? 'RECT_planar' : 'HEX_toroid';
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
      const [modelsRes, metricsRes, samplesRes, pcaRes] = await Promise.all([
        fetch('/data/text_models.json'),
        fetch('/data/text_metrics.json'),
        fetch('/data/news_samples.json'),
        fetch('/data/pca_params.json')
      ]);
      
      if (!modelsRes.ok || !metricsRes.ok || !samplesRes.ok || !pcaRes.ok) {
        throw new Error("HTTP status error loading text SOM files");
      }
      
      const textModels = await modelsRes.json();
      const textMetrics = await metricsRes.json();
      const newsSamples = await samplesRes.json();
      const pcaParams = await pcaRes.json();
      
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
    if (!text.trim()) {
      set({ classificationResult: null });
      return;
    }
    
    const rep = get().selectedTextRep;
    const dataset = get().selectedTextDataset;
    
    // 1. Try to fetch from FastAPI local backend
    try {
      const response = await fetch('http://127.0.0.1:8000/classify', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ text, representation: rep, dataset, lattice: get().lattice })
      });
      
      if (response.ok) {
        const result = await response.json();
        set({
          classificationResult: {
            bmu: result.bmu,
            dominantClass: result.dominantClass,
            purity: result.purity,
            score: result.score,
            source: 'local'
          },
          backendOnline: true
        });
        return;
      }
    } catch {
      // Backend is offline, try cloud HF Inference API for SBERT
    }
    
    // 2. Try Hugging Face Inference API for SBERT (Cloud-based real embedding projection)
    if (rep === 'SBERT') {
      try {
        const headers: Record<string, string> = {
          'Content-Type': 'application/json'
        };
        
        const token = import.meta.env.VITE_HF_TOKEN;
        if (token) {
          headers['Authorization'] = `Bearer ${token}`;
        }
        
        // Unified proxy URL: handled by Vite middleware in dev and Vercel Serverless Function in production
        const url = '/api/hf-sbert';
          
        const hfResponse = await fetch(url, {
          method: 'POST',
          headers,
          body: JSON.stringify({ inputs: text })
        });
        
        if (hfResponse.ok) {
          const emb = await hfResponse.json(); // Array of 384 floats
          const pcaParamsObj = get().pcaParams;
          const pca = pcaParamsObj ? pcaParamsObj[dataset] : null;
          const model = get().getActiveTextModel();
          
          if (Array.isArray(emb) && emb.length === 384 && pca && model) {
            const bmuResult = projectAndFindBMU(emb, pca, model, get().lattice === 'RECT');
            set({
              classificationResult: {
                bmu: bmuResult.bestNeuronId,
                dominantClass: bmuResult.dominantClass,
                purity: bmuResult.purity,
                score: bmuResult.score,
                source: 'cloud'
              },
              backendOnline: false
            });
            return;
          }
        }
      } catch (hfErr) {
        console.error("Hugging Face Inference API failed:", hfErr);
      }
    }
    
    // 3. Client-side fallback matching (Heuristics)
    const model = get().getActiveTextModel();
    if (!model) return;
    
    const samples = get().newsSamples[dataset] || [];
    const fallbackResult = classifyByKeywords(text, samples, model);
    
    set({
      classificationResult: {
        bmu: fallbackResult.bmu,
        dominantClass: fallbackResult.dominantClass,
        purity: fallbackResult.purity,
        score: fallbackResult.score,
        source: 'fallback'
      },
      backendOnline: false
    });
  },
  
  resetClassification: () => set({ classificationResult: null, customTextQuery: '' }),
  
  checkBackend: async () => {
    try {
      const response = await fetch('http://127.0.0.1:8000/health');
      if (response.ok) {
        set({ backendOnline: true });
        return;
      }
    } catch {
      // Offline
    }
    set({ backendOnline: false });
  }
}));
