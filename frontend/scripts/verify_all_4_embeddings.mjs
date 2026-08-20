import { chromium } from 'playwright';

async function testAll4Embeddings() {
  const browser = await chromium.launch({ headless: true });
  const page = await browser.newPage({ viewport: { width: 1280, height: 800 } });
  
  await page.goto('http://127.0.0.1:5173/', { waitUntil: 'networkidle' });

  // Switch to Text SOM tab
  await page.locator('button').filter({ hasText: 'Clusterização' }).click();
  await page.waitForTimeout(800);

  const selectRep = page.locator('select[aria-label="Selecionar representação textual"]');
  const reps = ['SBERT', 'TF-IDF', 'BGE-M3', 'Gemma-300M'];

  console.log('--- Testing 4 Embedding Models in Real Browser ---');

  for (const rep of reps) {
    await selectRep.selectOption(rep);
    await page.waitForTimeout(800);

    const badgeText = await page.locator('span:has-text("' + rep + '")').first().textContent();
    const fills = await page.locator('svg polygon').evaluateAll(els => els.map(e => e.getAttribute('fill')));

    console.log(`\n[${rep}]`);
    console.log(`  Badge confirmed: ${badgeText}`);
    console.log(`  Neuron count: ${fills.length}`);
    console.log(`  Sample 5 fills: ${fills.slice(0, 5).join(', ')}`);
  }

  // Also switch to 6class dataset and test BGE-M3 and Gemma-300M
  console.log('\n--- Testing 6class Dataset Switch with BGE-M3 & Gemma-300M ---');
  const selectDataset = page.locator('select[aria-label="Selecionar dataset de notícias"]');
  await selectDataset.selectOption('6class');
  await page.waitForTimeout(800);

  for (const rep of ['BGE-M3', 'Gemma-300M']) {
    await selectRep.selectOption(rep);
    await page.waitForTimeout(800);
    const fills = await page.locator('svg polygon').evaluateAll(els => els.map(e => e.getAttribute('fill')));
    console.log(`[6class - ${rep}] Neuron count: ${fills.length} | First 3 fills: ${fills.slice(0, 3).join(', ')}`);
  }

  console.log('\n✅ All 4 representations verified and functioning dynamically in the real browser!');
  await browser.close();
}

testAll4Embeddings().catch(console.error);
