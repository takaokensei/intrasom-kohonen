export default async function handler(req, res) {
  // CORS Headers
  res.setHeader('Access-Control-Allow-Credentials', 'true');
  res.setHeader('Access-Control-Allow-Origin', '*');
  res.setHeader('Access-Control-Allow-Methods', 'GET,OPTIONS,PATCH,DELETE,POST,PUT');
  res.setHeader(
    'Access-Control-Allow-Headers',
    'X-CSRF-Token, X-Requested-With, Accept, Accept-Version, Content-Length, Content-MD5, Content-Type, Date, X-Api-Version, Authorization'
  );

  // Handle preflight OPTIONS request
  if (req.method === 'OPTIONS') {
    res.status(200).end();
    return;
  }

  if (req.method !== 'POST') {
    res.status(405).json({ error: 'Method not allowed' });
    return;
  }

  const { inputs } = req.body;
  if (!inputs) {
    res.status(400).json({ error: 'Missing inputs parameter' });
    return;
  }

  // Hugging Face authorization token
  const token = process.env.HF_TOKEN || process.env.VITE_HF_TOKEN || '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';

  const headers = {
    'Content-Type': 'application/json'
  };

  if (token) {
    headers['Authorization'] = `Bearer ${token}`;
  }

  try {
    const hfResponse = await fetch(
      'https://api-inference.huggingface.co/models/sentence-transformers/all-MiniLM-L6-v2',
      {
        method: 'POST',
        headers,
        body: JSON.stringify({ inputs })
      }
    );

    if (!hfResponse.ok) {
      const errorText = await hfResponse.text();
      res.status(hfResponse.status).json({
        error: `Hugging Face API status ${hfResponse.status}`,
        details: errorText
      });
      return;
    }

    const data = await hfResponse.json();
    res.status(200).json(data);
  } catch (err) {
    res.status(500).json({ error: err.message });
  }
}
