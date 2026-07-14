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
    isBackend?: boolean;
  } | null;
  backendOnline: boolean | null;
  
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
  backendOnline: null,
  
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
  },  loadTextData: async () => {
    if (get().newsSamples.length > 0) {
      get().checkBackend();
      return; // Already loaded
    }
    
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
      get().checkBackend();
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
  
  classifyText: async (text) => {
    if (!text.trim()) {
      set({ classificationResult: null });
      return;
    }
    
    const rep = get().selectedTextRep;
    
    // 1. Try to fetch from FastAPI local backend
    try {
      const response = await fetch('http://127.0.0.1:8000/classify', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ text, representation: rep })
      });
      
      if (response.ok) {
        const result = await response.json();
        set({
          classificationResult: {
            bmu: result.bmu,
            dominantClass: result.dominantClass,
            purity: result.purity,
            score: result.score,
            isBackend: true
          },
          backendOnline: true
        });
        return;
      }
    } catch (err) {
      // Backend is offline, fall back to local client matching
    }
    
    // 2. Client-side fallback matching
    const queryLower = text.toLowerCase();
    const samples = get().newsSamples;
    
    let bestDocIdx = 0;
    let maxOverlap = 0;
    
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
    
    const model = get().textModels[rep];
    if (!model) return;
    
    let targetNeuron = null;
    let score = 0;
    let dominantClass = "Desconhecido";
    let purity = 0;
    let bmu = 0;

    const spaceKeywords = ["space", "nasa", "moon", "orbit", "rocket", "launch", "sky", "star", "planet", "astronomy", "shuttle", "satellite", "mission", "mars", "earth", "solar", "telescope", "astronaut"];
    const baseballKeywords = ["baseball", "pitcher", "game", "run", "hit", "team", "player", "stadium", "bat", "glove", "inning", "league", "sox", "cubs", "red", "yankee", "series", "ball", "season", "batter"];
    const mideastKeywords = ["israel", "turkish", "arab", "mideast", "peace", "war", "jew", "muslim", "christian", "palestinian", "syria", "turkey", "government", "military", "soldier", "weapons", "kill", "attack", "armenia"];
    const graphicsKeywords = ["graphic", "render", "image", "polygon", "3d", "draw", "animation", "pixel", "texture", "vector", "format", "tiff", "jpeg", "raytracing", "computer", "screen", "color", "paint", "code", "library", "packages"];

    let spaceHits = 0;
    let baseballHits = 0;
    let mideastHits = 0;
    let graphicsHits = 0;

    const words = queryLower.split(/\W+/);
    words.forEach(word => {
      if (word.length > 2) {
        if (spaceKeywords.some(kw => word === kw || word.includes(kw))) spaceHits++;
        if (baseballKeywords.some(kw => word === kw || word.includes(kw))) baseballHits++;
        if (mideastKeywords.some(kw => word === kw || word.includes(kw))) mideastHits++;
        if (graphicsKeywords.some(kw => word === kw || word.includes(kw))) graphicsHits++;
      }
    });

    const totalHits = spaceHits + baseballHits + mideastHits + graphicsHits;

    if (maxOverlap > 0) {
      targetNeuron = model.neurons.find(n => n.doc_indices.includes(bestDocIdx));
      if (targetNeuron) {
        bmu = targetNeuron.id;
        dominantClass = targetNeuron.dominant_class;
        purity = targetNeuron.purity;
        score = Math.min(85 + maxOverlap * 2.5, 99);
      }
    } else if (totalHits > 0) {
      const maxHits = Math.max(spaceHits, baseballHits, mideastHits, graphicsHits);
      let predictedCat = "Space";
      if (maxHits === spaceHits) predictedCat = "Space";
      else if (maxHits === baseballHits) predictedCat = "Baseball";
      else if (maxHits === mideastHits) predictedCat = "Mideast";
      else predictedCat = "Graphics";

      const categoryNeurons = model.neurons.filter(n => n.dominant_class === predictedCat);
      targetNeuron = categoryNeurons[Math.floor(Math.random() * categoryNeurons.length)] || model.neurons[0];
      
      bmu = targetNeuron.id;
      dominantClass = targetNeuron.dominant_class;
      purity = targetNeuron.purity;
      score = Math.min(60 + totalHits * 10, 85);
    } else {
      bmu = 0;
      dominantClass = "Desconhecido";
      purity = 0;
      score = 0;
    }

    set({
      classificationResult: {
        bmu,
        dominantClass,
        purity,
        score,
        isBackend: false
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
    } catch (e) {
      // Offline
    }
    set({ backendOnline: false });
  }
}));
