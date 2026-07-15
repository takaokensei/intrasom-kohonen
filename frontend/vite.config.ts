import { defineConfig, loadEnv } from 'vite'
import react from '@vitejs/plugin-react'
import https from 'node:https'

// https://vite.dev/config/
export default defineConfig(({ mode }) => {
  // Load env variables (including .env.local) securely in Node.js context
  const env = loadEnv(mode, process.cwd(), '')

  return {
    plugins: [
      react(),
      {
        name: 'hf-sbert-middleware',
        configureServer(server) {
          // Intercept /api/hf-sbert in dev with a native Node.js HTTPS request
          server.middlewares.use('/api/hf-sbert', (req, res) => {
            if (req.method !== 'POST') {
              res.writeHead(405, { 'Content-Type': 'application/json' });
              res.end(JSON.stringify({ error: 'Method not allowed' }));
              return;
            }

            let body = '';
            req.on('data', chunk => { body += chunk.toString(); });
            req.on('end', () => {
              // Retrieve the token loaded from .env.local
              const token = env.VITE_HF_TOKEN || '';

              const postData = body;
              const headers: Record<string, string> = {
                'Content-Type': 'application/json',
                'Content-Length': Buffer.byteLength(postData).toString(),
              };

              if (token) {
                headers['Authorization'] = `Bearer ${token}`;
              }

              const options = {
                hostname: 'api-inference.huggingface.co',
                path: '/models/sentence-transformers/all-MiniLM-L6-v2',
                method: 'POST',
                headers,
              };

              const hfReq = https.request(options, (hfRes) => {
                let responseData = '';
                hfRes.on('data', chunk => { responseData += chunk; });
                hfRes.on('end', () => {
                  res.writeHead(hfRes.statusCode || 502, {
                    'Content-Type': 'application/json',
                    'Access-Control-Allow-Origin': '*',
                  });
                  res.end(responseData);
                });
              });

              hfReq.on('error', (err) => {
                res.writeHead(502, { 'Content-Type': 'application/json' });
                res.end(JSON.stringify({ error: err.message }));
              });

              hfReq.write(postData);
              hfReq.end();
            });
          });
        },
      },
    ],
  }
})
