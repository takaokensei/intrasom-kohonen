import { defineConfig } from 'vite'
import react from '@vitejs/plugin-react'
import https from 'node:https'

// https://vite.dev/config/
export default defineConfig({
  plugins: [
    react(),
    {
      name: 'hf-sbert-middleware',
      configureServer(server) {
        // Intercept /api/hf-sbert in dev with a native Node.js HTTPS request
        // This avoids CORS, DNS resolution failures, and TLS proxy issues
        server.middlewares.use('/api/hf-sbert', (req, res) => {
          if (req.method !== 'POST') {
            res.writeHead(405, { 'Content-Type': 'application/json' });
            res.end(JSON.stringify({ error: 'Method not allowed' }));
            return;
          }

          let body = '';
          req.on('data', chunk => { body += chunk.toString(); });
          req.on('end', () => {
            const token = process.env.VITE_HF_TOKEN;

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
                res.writeHead(hfRes.statusCode, {
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
})
