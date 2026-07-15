import { defineConfig } from 'vite'
import react from '@vitejs/plugin-react'

// https://vite.dev/config/
export default defineConfig({
  plugins: [react()],
  server: {
    proxy: {
      // Proxy /api/hf-sbert in dev → HF Inference API (avoids DNS issues and CORS)
      '/api/hf-sbert': {
        target: 'https://api-inference.huggingface.co',
        changeOrigin: true,
        secure: true,
        rewrite: () => '/models/sentence-transformers/all-MiniLM-L6-v2',
      },
    },
  },
})
