import { create } from 'zustand';

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
  somModels: Record<string, SOMModel>;
  metrics: MetricRow[];
  textModels: Record<string, TextModel>;
  textMetrics: Record<string, { ARI: number; NMI: number }>;
  newsSamples: NewsSample[];
  
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
  } | null;
  
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
}

export const useDashboardStore = create<DashboardState>((set, get) => ({
  activeTab: 'synthetic',
  
  // Data initial states
  series: [],
  somModels: {},
  metrics: [],
  textModels: {},
  textMetrics: {},
  newsSamples: [],
  
  // Loading initial states
  loadingSynthetic: false,
  loadingText: false,
  
  // Synthetic Control Defaults
  selectedMapSize: '10x10',
  selectedNeuronId: null,
  highlightedClass: null,
  
  // Text SOM Defaults
  selectedTextRep: 'SBERT',
  selectedDocId: null,
  customTextQuery: '',
  classificationResult: null,
  
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
    
    set({ loadingSynthetic: true });
    try {
      const [seriesRes, modelsRes, metricsRes] = await Promise.all([
        fetch('/data/series.json'),
        fetch('/data/som_models.json'),
        fetch('/data/metrics.json')
      ]);
      
      const series = await seriesRes.json();
      const somModels = await modelsRes.json();
      const metrics = await metricsRes.json();
      
      set({ series, somModels, metrics, loadingSynthetic: false });
    } catch (err) {
      console.error("Error loading synthetic control data:", err);
      set({ loadingSynthetic: false });
    }
  },
  
  loadTextData: async () => {
    if (get().newsSamples.length > 0) return; // Already loaded
    
    set({ loadingText: true });
    try {
      const [modelsRes, metricsRes, samplesRes] = await Promise.all([
        fetch('/data/text_models.json'),
        fetch('/data/text_metrics.json'),
        fetch('/data/news_samples.json')
      ]);
      
      const textModels = await modelsRes.json();
      const textMetrics = await metricsRes.json();
      const newsSamples = await samplesRes.json();
      
      set({ textModels, textMetrics, newsSamples, loadingText: false });
    } catch (err) {
      console.error("Error loading text news data:", err);
      set({ loadingText: false });
    }
  },
  
  setSelectedMapSize: (selectedMapSize) => set({ selectedMapSize, selectedNeuronId: null }),
  setSelectedNeuronId: (selectedNeuronId) => set({ selectedNeuronId }),
  setHighlightedClass: (highlightedClass) => set({ highlightedClass }),
  setSelectedTextRep: (selectedTextRep) => set({ selectedTextRep }),
  setSelectedDocId: (selectedDocId) => set({ selectedDocId }),
  setCustomTextQuery: (customTextQuery) => set({ customTextQuery }),
  
  classifyText: (text) => {
    if (!text.trim()) {
      set({ classificationResult: null });
      return;
    }
    
    const queryLower = text.toLowerCase();
    const samples = get().newsSamples;
    
    let bestDocIdx = 0;
    let maxOverlap = -1;
    
    samples.forEach(doc => {
      const docTokens = new Set(doc.text.toLowerCase().split(/\W+/));
      let overlap = 0;
      docTokens.forEach(token => {
        if (token.length > 3 && queryLower.includes(token)) {
          overlap++;
        }
      });
      
      if (overlap > maxOverlap) {
        maxOverlap = overlap;
        bestDocIdx = doc.id;
      }
    });
    
    const rep = get().selectedTextRep;
    const model = get().textModels[rep];
    if (!model) return;
    
    let targetNeuron = model.neurons.find(n => n.doc_indices.includes(bestDocIdx));
    
    if (!targetNeuron) {
      let predictedCat = "Space";
      if (queryLower.includes("baseball") || queryLower.includes("pitcher") || queryLower.includes("game")) predictedCat = "Baseball";
      else if (queryLower.includes("graphic") || queryLower.includes("render") || queryLower.includes("image")) predictedCat = "Graphics";
      else if (queryLower.includes("israel") || queryLower.includes("turkish") || queryLower.includes("arab")) predictedCat = "Mideast";
      else if (queryLower.includes("orbit") || queryLower.includes("space") || queryLower.includes("nasa")) predictedCat = "Space";
      
      const categoryNeurons = model.neurons.filter(n => n.dominant_class === predictedCat);
      targetNeuron = categoryNeurons[Math.floor(Math.random() * categoryNeurons.length)] || model.neurons[0];
    }
    
    const score = maxOverlap > 0 ? Math.min(85 + maxOverlap * 2, 98) : 60 + Math.floor(Math.random() * 20);
    
    set({
      classificationResult: {
        bmu: targetNeuron.id,
        dominantClass: targetNeuron.dominant_class,
        purity: targetNeuron.purity,
        score: score
      }
    });
  },
  
  resetClassification: () => set({ classificationResult: null, customTextQuery: '' })
}));
