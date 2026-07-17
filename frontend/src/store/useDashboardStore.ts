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
    } catch (err) {
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
            // Apply PCA projection: Q_20 = (emb - mean) * components
            const vec_20 = new Array(20).fill(0);
            for (let i = 0; i < 20; i++) {
              let sum = 0;
              for (let j = 0; j < 384; j++) {
                sum += (emb[j] - pca.mean[j]) * pca.components[i][j];
              }
              vec_20[i] = sum;
            }
            
            // Find BMU in model neurons using Euclidean distance
            let bestNeuronId = 1;
            let minDistance = Infinity;
            let bestNeuron = model.neurons[0];
            
            model.neurons.forEach(neuron => {
              if (neuron.codebook && neuron.codebook.length === 20) {
                let sumSq = 0;
                for (let i = 0; i < 20; i++) {
                  const diff = vec_20[i] - neuron.codebook[i];
                  sumSq += diff * diff;
                }
                const dist = Math.sqrt(sumSq);
                if (dist < minDistance) {
                  minDistance = dist;
                  bestNeuronId = neuron.id;
                  bestNeuron = neuron;
                }
              }
            });
            
            // Calculate score (confidence)
            const maxExpectedDist = 6.0;
            const confidence = Math.max(0, Math.min(100, Math.round((1.0 - (minDistance / maxExpectedDist)) * 100)));
            const adjustedConfidence = confidence > 0 ? Math.round(50 + (confidence / 2)) : 0;
            const finalScore = minDistance < 1.0 ? Math.max(adjustedConfidence, 95) : adjustedConfidence;
            
            set({
              classificationResult: {
                bmu: bestNeuronId,
                dominantClass: bestNeuron.dominant_class,
                purity: bestNeuron.purity,
                score: finalScore,
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
    const queryLower = text.toLowerCase();
    const samples = get().newsSamples[dataset] || [];
    
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
    
    const model = get().textModels[dataset] ? get().textModels[dataset][rep] : null;
    if (!model) return;
    
    let targetNeuron = null;
    let score = 0;
    let dominantClass = "Desconhecido";
    let purity = 0;
    let bmu = 0;

    const turismoKeywords = ["turismo", "viagem", "hotel", "voo", "praia", "turista", "passagem", "destino", "roteiro", "ferias", "pousada", "resort", "passeio", "agencia", "monumento", "atracoes", "viajar"];
    const esportesKeywords = ["esporte", "jogo", "time", "jogador", "campeonato", "partida", "futebol", "atleta", "copa", "estadio", "quadra", "campo", "vitoria", "derrota", "gol", "treino", "olimpiada"];
    const policiaKeywords = ["policia", "crime", "preso", "delegado", "prisao", "roubo", "furto", "assalto", "assassinato", "drogas", "investigacao", "acusado", "suspeito", "vitima", "violencia", "mandado", "policial"];
    const economiaKeywords = ["economia", "inflacao", "pib", "juros", "mercado", "dolar", "bolsa", "empresa", "crescimento", "setor", "reforma", "imposto", "banco", "receita", "investimento", "financas"];
    const politicaKeywords = ["politica", "governo", "presidente", "senado", "deputado", "eleicao", "voto", "partido", "ministro", "projeto", "lei", "brasilia", "congresso", "câmara", "prefeito", "candidato"];
    const variedadesKeywords = ["variedades", "famosos", "novela", "filme", "cinema", "show", "musica", "artista", "celebridade", "teatro", "moda", "cultura", "evento", "estreia", "audiencia", "festival"];

    let turismoHits = 0;
    let esportesHits = 0;
    let policiaHits = 0;
    let economiaHits = 0;
    let politicaHits = 0;
    let variedadesHits = 0;

    const words = queryLower.split(/\W+/);
    words.forEach(word => {
      if (word.length > 2) {
        if (turismoKeywords.some(kw => word === kw || word.includes(kw))) turismoHits++;
        if (esportesKeywords.some(kw => word === kw || word.includes(kw))) esportesHits++;
        if (policiaKeywords.some(kw => word === kw || word.includes(kw))) policiaHits++;
        if (economiaKeywords.some(kw => word === kw || word.includes(kw))) economiaHits++;
        if (politicaKeywords.some(kw => word === kw || word.includes(kw))) politicaHits++;
        if (variedadesKeywords.some(kw => word === kw || word.includes(kw))) variedadesHits++;
      }
    });

    const totalHits = turismoHits + esportesHits + policiaHits + economiaHits + politicaHits + variedadesHits;

    if (maxOverlap > 0) {
      targetNeuron = model.neurons.find(n => n.doc_indices.includes(bestDocIdx));
      if (targetNeuron) {
        bmu = targetNeuron.id;
        dominantClass = targetNeuron.dominant_class;
        purity = targetNeuron.purity;
        score = Math.min(85 + maxOverlap * 2.5, 99);
      }
    } else if (totalHits > 0) {
      const maxHits = Math.max(turismoHits, esportesHits, policiaHits, economiaHits, politicaHits, variedadesHits);
      let predictedCat = "Turismo";
      if (maxHits === turismoHits) predictedCat = "Turismo";
      else if (maxHits === esportesHits) predictedCat = "Esportes";
      else if (maxHits === policiaHits) predictedCat = "Policia";
      else if (maxHits === economiaHits) predictedCat = "Economia";
      else if (maxHits === politicaHits) predictedCat = "Politica";
      else predictedCat = "Variedades";

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
    } catch (e) {
      // Offline
    }
    set({ backendOnline: false });
  }
}));
