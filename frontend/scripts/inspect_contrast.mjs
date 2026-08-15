import { chromium } from 'playwright';
import fs from 'fs';
import path from 'path';

async function inspectContrast() {
  const browser = await chromium.launch({ headless: true });
  const page = await browser.newPage({ viewport: { width: 1280, height: 800 } });

  await page.goto('http://127.0.0.1:5173/', { waitUntil: 'networkidle' });
  await page.waitForTimeout(1000);

  const axeScript = fs.readFileSync(path.resolve('node_modules/axe-core/axe.min.js'), 'utf8');
  await page.evaluate(axeScript);
  const results = await page.evaluate(async () => {
    // @ts-ignore
    return await axe.run({ runOnly: ['color-contrast'] });
  });

  const violation = results.violations.find(v => v.id === 'color-contrast');
  if (!violation) {
    console.log('Zero color-contrast violations found!');
    await browser.close();
    return;
  }

  console.log(`Found ${violation.nodes.length} nodes with color-contrast violations on SyntheticScreen:`);
  violation.nodes.forEach((node, idx) => {
    const data = node.any[0]?.data;
    console.log(`\n[Node ${idx + 1}]`);
    console.log('Target:', node.target.join(' > '));
    console.log('HTML:', node.html);
    console.log('Data:', JSON.stringify(data, null, 2));
  });

  // Switch to Text Tab and check there too
  await page.locator('button:has-text("Clusterização de Textos"), button:has-text("Text")').first().click();
  await page.waitForTimeout(800);
  const resultsText = await page.evaluate(async () => {
    // @ts-ignore
    return await axe.run({ runOnly: ['color-contrast'] });
  });
  const violationText = resultsText.violations.find(v => v.id === 'color-contrast');
  if (violationText) {
    console.log(`\n\nFound ${violationText.nodes.length} nodes with color-contrast violations on TextScreen:`);
    violationText.nodes.forEach((node, idx) => {
      const data = node.any[0]?.data;
      console.log(`\n[Text Node ${idx + 1}]`);
      console.log('Target:', node.target.join(' > '));
      console.log('HTML:', node.html);
      console.log('Data:', JSON.stringify(data, null, 2));
    });
  }

  await browser.close();
}

inspectContrast().catch(console.error);
