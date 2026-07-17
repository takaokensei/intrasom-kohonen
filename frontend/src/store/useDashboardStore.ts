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
  somModels: Record<string, SOMModel>;
  metrics: MetricRow[];
  textModels: Record<string, Record<string, TextModel>>;
  textMetrics: Record<string, Record<string, { ARI: number; NMI: number }>>;
  newsSamples: Record<string, NewsSample[]>;
  selectedTextDataset: '20news' | '6class';
  setSelectedTextDataset: (dataset: '20news' | '6class') => void;
  
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
      const [seriesRes, modelsRes, metricsRes] = await Promise.all([
        fetch('/data/series.json'),
        fetch('/data/som_models.json'),
        fetch('/data/metrics.json')
      ]);
      
      if (!seriesRes.ok || !modelsRes.ok || !metricsRes.ok) {
        throw new Error("HTTP status error loading synthetic control files");
      }
      
      const series = await seriesRes.json();
      const somModels = await modelsRes.json();
      const metrics = await metricsRes.json();
      
      set({ series, somModels, metrics, loadingSynthetic: false });
    } catch (err) {
      console.error("Error loading synthetic control data:", err);
      set({ 
        loadingSynthetic: false, 
        errorSynthetic: "Falha ao carregar dados sintéticos do SOM. Verifique a existência dos arquivos JSON na pasta public/data/." 
      });
    }
  },  loadTextData: async () => {
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
        body: JSON.stringify({ text, representation: rep, dataset })
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
          const model = get().textModels[dataset] ? get().textModels[dataset][rep] : null;
          
          if (Array.isArray(emb) && emb.length === 384 && pca && model) {
            const bmuResult = projectAndFindBMU(emb, pca, model);
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
    const model = get().textModels[dataset] ? get().textModels[dataset][rep] : null;
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
