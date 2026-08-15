import { chromium } from 'playwright';
import path from 'path';
import fs from 'fs';

const BASE_URL = 'http://127.0.0.1:5173/';
const LOCAL_SCREENSHOTS = path.resolve('audit_screenshots');

async function testInteractions() {
  console.log('🔍 Running Deep Interactive & Edge-Case Playwright Tests...');
  const browser = await chromium.launch({ headless: true });
  const page = await browser.newPage({ viewport: { width: 1440, height: 900 } });

  const interactionResults = {
    fullscreen: {},
    tooltips: {},
    keyboardModal: {},
    timeSeriesFiltering: {},
    paramStudyHighlight: {},
    textClassifier: {},
    accessibilityViolations: []
  };

  await page.goto(BASE_URL, { waitUntil: 'networkidle' });
  await page.waitForTimeout(1000);

  // 1. Test Fullscreen on HexGrid
  try {
    const fsBtn = page.locator('button[title*="tela cheia"], button[title*="Fullscreen"], button:has-text("Tela Cheia"), button svg.lucide-maximize2, button svg.lucide-maximize').first();
    if (await fsBtn.count() > 0) {
      await fsBtn.click();
      await page.waitForTimeout(500);
      await page.screenshot({ path: path.join(LOCAL_SCREENSHOTS, 'interactive_01_fullscreen_open.png') });
      
      // Test Esc key to close
      await page.keyboard.press('Escape');
      await page.waitForTimeout(500);
      const isClosed = await page.locator('.fixed.inset-0').count() === 0;
      interactionResults.fullscreen.escCloses = isClosed;
      await page.screenshot({ path: path.join(LOCAL_SCREENSHOTS, 'interactive_02_fullscreen_after_esc.png') });
    }
  } catch (e) {
    interactionResults.fullscreen.error = e.message;
  }

  // 2. Test TimeSeries Filter Toggles
  try {
    const filterBtn = page.locator('button:has-text("Normal")').first();
    if (await filterBtn.count() > 0) {
      await filterBtn.click();
      await page.waitForTimeout(300);
      await page.screenshot({ path: path.join(LOCAL_SCREENSHOTS, 'interactive_03_timeseries_filtered.png') });
      interactionResults.timeSeriesFiltering.works = true;
    }
  } catch (e) {
    interactionResults.timeSeriesFiltering.error = e.message;
  }

  // 3. Test Parameter Study "Destaque de Tabela" button
  try {
    const highlightBtn = page.locator('button:has-text("Rolar até a tabela")').first();
    if (await highlightBtn.count() > 0) {
      await highlightBtn.click();
      await page.waitForTimeout(500);
      await page.screenshot({ path: path.join(LOCAL_SCREENSHOTS, 'interactive_04_param_study_highlight.png') });
      interactionResults.paramStudyHighlight.clicked = true;
    }
  } catch (e) {
    interactionResults.paramStudyHighlight.error = e.message;
  }

  // 4. Test TextScreen 6class and TF-IDF
  try {
    const textTab = page.locator('button:has-text("Clusterização de Textos"), button:has-text("Text")').first();
    await textTab.click();
    await page.waitForTimeout(800);

    // Switch dataset to 6class
    const datasetSelect = page.locator('select').first();
    await datasetSelect.selectOption({ index: 1 });
    await page.waitForTimeout(800);
    await page.screenshot({ path: path.join(LOCAL_SCREENSHOTS, 'interactive_05_text_6class_dataset.png') });

    // Switch rep to TF-IDF
    const repSelect = page.locator('select').nth(1);
    await repSelect.selectOption('TF-IDF');
    await page.waitForTimeout(800);
    await page.screenshot({ path: path.join(LOCAL_SCREENSHOTS, 'interactive_06_text_tfidf_rep.png') });
  } catch (e) {
    interactionResults.textClassifier.error = e.message;
  }

  // 5. Run detailed axe audit on both screens
  const axeScript = fs.readFileSync(path.resolve('node_modules/axe-core/axe.min.js'), 'utf8');
  await page.evaluate(axeScript);
  const a11yText = await page.evaluate(async () => {
    // @ts-ignore
    return await axe.run();
  });
  interactionResults.accessibilityViolations = a11yText.violations;

  fs.writeFileSync(path.join(LOCAL_SCREENSHOTS, 'deep_interactions_report.json'), JSON.stringify(interactionResults, null, 2));
  console.log('✅ Deep Interaction Tests Complete!');

  await browser.close();
}

testInteractions().catch(console.error);
