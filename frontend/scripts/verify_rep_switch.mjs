import { chromium } from 'playwright';

async function verifyRepSwitch() {
  const browser = await chromium.launch({ headless: true });
  const page = await browser.newPage({ viewport: { width: 1280, height: 800 } });
  await page.goto('http://127.0.0.1:5173/', { waitUntil: 'networkidle' });

  // Switch to Text tab
  await page.locator('button').filter({ hasText: 'Clusterização' }).click();
  await page.waitForTimeout(1000);

  // Read all polygon fills for SBERT
  const sbertFills = await page.locator('svg polygon').evaluateAll(els => els.map(e => e.getAttribute('fill')));
  console.log('SBERT polygon count:', sbertFills.length);
  console.log('SBERT first 5 fills:', sbertFills.slice(0, 5));

  // Change representation to TF-IDF
  const selectRep = page.locator('select[aria-label="Selecionar representação textual"]');
  await selectRep.selectOption('TF-IDF');
  await page.waitForTimeout(1000);

  // Read all polygon fills for TF-IDF
  const tfidfFills = await page.locator('svg polygon').evaluateAll(els => els.map(e => e.getAttribute('fill')));
  console.log('TF-IDF polygon count:', tfidfFills.length);
  console.log('TF-IDF first 5 fills:', tfidfFills.slice(0, 5));

  // Compare fills
  let diffCount = 0;
  for (let i = 0; i < sbertFills.length; i++) {
    if (sbertFills[i] !== tfidfFills[i]) diffCount++;
  }
  console.log(`\nResults: ${diffCount}/${sbertFills.length} neurons changed color/class when switching from SBERT to TF-IDF.`);
  if (diffCount > 0) {
    console.log('✅ Representation switch successfully updates the Kohonen Map visualization in real-time!');
  } else {
    console.log('❌ Map did not update.');
  }

  await browser.close();
}

verifyRepSwitch().catch(console.error);
