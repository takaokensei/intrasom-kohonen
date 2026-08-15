import { chromium } from 'playwright';
import fs from 'fs';
import path from 'path';

const VIEWPORTS = [
  { name: '320px_mobile_small', width: 320, height: 640 },
  { name: '375px_mobile_standard', width: 375, height: 812 },
  { name: '768px_tablet_portrait', width: 768, height: 1024 },
  { name: '1024px_tablet_landscape', width: 1024, height: 768 },
  { name: '1280px_laptop', width: 1280, height: 800 },
  { name: '1440px_desktop', width: 1440, height: 900 },
  { name: '1920px_large_desktop', width: 1920, height: 1080 },
];

const BASE_URL = 'http://127.0.0.1:5173/';
const OUTPUT_DIR = path.resolve('..', 'screenshots', 'phase1_audit');
const LOCAL_SCREENSHOTS = path.resolve('audit_screenshots');

fs.mkdirSync(OUTPUT_DIR, { recursive: true });
fs.mkdirSync(LOCAL_SCREENSHOTS, { recursive: true });

async function runAudit() {
  console.log('🚀 Starting Comprehensive Browser Audit across 7 Viewports...');
  const browser = await chromium.launch({ headless: true });
  const report = {
    viewports: {},
    consoleMessages: [],
    networkErrors: [],
    a11yIssues: {},
    performanceTimings: {},
    interactions: {}
  };

  for (const vp of VIEWPORTS) {
    console.log(`\n📱 Auditing Viewport: ${vp.name} (${vp.width}x${vp.height})`);
    const context = await browser.newContext({
      viewport: { width: vp.width, height: vp.height },
      deviceScaleFactor: 1,
    });

    const page = await context.newPage();
    const vpLogs = [];
    const vpErrors = [];

    page.on('console', msg => {
      const entry = { type: msg.type(), text: msg.text() };
      vpLogs.push(entry);
      if (msg.type() === 'error') vpErrors.push(entry);
    });

    page.on('pageerror', err => {
      vpErrors.push({ type: 'pageerror', text: err.toString() });
    });

    page.on('requestfailed', req => {
      report.networkErrors.push({ url: req.url(), failure: req.failure()?.errorText });
    });

    // 1. Initial Load & Synthetic Screen (2D)
    await page.goto(BASE_URL, { waitUntil: 'networkidle' });
    await page.waitForTimeout(1000); // Allow render stabilization

    const saveShot = async (filename) => {
      const out1 = path.join(OUTPUT_DIR, `${vp.name}_${filename}.png`);
      const out2 = path.join(LOCAL_SCREENSHOTS, `${vp.name}_${filename}.png`);
      await page.screenshot({ path: out1, fullPage: true });
      await page.screenshot({ path: out2, fullPage: true });
    };

    await saveShot('01_synthetic_2d_class');

    // Check Horizontal Overflow
    const scrollWidth = await page.evaluate(() => document.documentElement.scrollWidth);
    const clientWidth = await page.evaluate(() => document.documentElement.clientWidth);
    const hasHorizontalOverflow = scrollWidth > clientWidth;

    // 2. Click neuron 42 (or any neuron) to open NeuronDetailPanel
    try {
      const neuron = page.locator('svg polygon, svg path, [data-neuron-id]').first();
      if (await neuron.count() > 0) {
        await neuron.click({ force: true });
        await page.waitForTimeout(500);
        await saveShot('02_synthetic_neuron_selected');
      }
    } catch (e) {
      console.warn(`Neuron click failed at ${vp.name}:`, e.message);
    }

    // 3. Switch to 3D View mode
    try {
      const view3DButton = page.locator('button:has-text("3D"), button:has-text("Visualização 3D")').first();
      if (await view3DButton.count() > 0) {
        await view3DButton.click();
        await page.waitForTimeout(1200);
        await saveShot('03_synthetic_3d_mode');
      }
    } catch (e) {
      console.warn(`3D switch failed at ${vp.name}:`, e.message);
    }

    // 4. Switch to Text Screen Tab
    try {
      const textTab = page.locator('button:has-text("Text"), button:has-text("Texto"), button:has-text("Notícias")').first();
      if (await textTab.count() > 0) {
        await textTab.click();
        await page.waitForTimeout(1200);
        await saveShot('04_text_screen_initial');

        // Test Classifier interaction
        const sampleBtn = page.locator('button:has-text("Aleatório"), button:has-text("Random")').first();
        if (await sampleBtn.count() > 0) {
          await sampleBtn.click();
          await page.waitForTimeout(800);
          await saveShot('05_text_classifier_result');
        }
      }
    } catch (e) {
      console.warn(`Text tab switch failed at ${vp.name}:`, e.message);
    }

    // 5. Run axe-core accessibility audit
    try {
      const axeScript = fs.readFileSync(path.resolve('node_modules/axe-core/axe.min.js'), 'utf8');
      await page.evaluate(axeScript);
      const a11yResults = await page.evaluate(async () => {
        // @ts-ignore
        return await axe.run();
      });
      if (a11yResults.violations.length > 0) {
        console.log(`[a11y] ${vp.name} violations:`, JSON.stringify(a11yResults.violations.map(v => ({ id: v.id, nodes: v.nodes.map(n => ({ target: n.target, html: n.html, data: n.any?.[0]?.data })) })), null, 2));
      }
      report.a11yIssues[vp.name] = {
        violationsCount: a11yResults.violations.length,
        violations: a11yResults.violations.map(v => ({
          id: v.id,
          impact: v.impact,
          description: v.description,
          nodes: v.nodes.length,
          nodesDetail: v.nodes.map(n => ({ target: n.target, html: n.html, data: n.any?.[0]?.data })),
          helpUrl: v.helpUrl
        }))
      };
    } catch (e) {
      console.warn(`Axe-core run failed at ${vp.name}:`, e.message);
    }

    report.viewports[vp.name] = {
      scrollWidth,
      clientWidth,
      hasHorizontalOverflow,
      errors: vpErrors,
      logsCount: vpLogs.length
    };

    await context.close();
  }

  // 6. Test Error State in 1280px
  console.log('\n⚠️ Testing Error State Handling...');
  const errorContext = await browser.newContext({ viewport: { width: 1280, height: 800 } });
  const errorPage = await errorContext.newPage();
  await errorPage.route('**/data/*.json', route => route.abort());
  await errorPage.goto(BASE_URL, { waitUntil: 'networkidle' });
  await errorPage.waitForTimeout(1000);
  await errorPage.screenshot({ path: path.join(OUTPUT_DIR, 'error_state_1280px.png'), fullPage: true });
  await errorPage.screenshot({ path: path.join(LOCAL_SCREENSHOTS, 'error_state_1280px.png'), fullPage: true });
  await errorContext.close();

  // 7. Keyboard Navigation Audit
  console.log('\n⌨️ Testing Keyboard Navigation...');
  const kbdContext = await browser.newContext({ viewport: { width: 1280, height: 800 } });
  const kbdPage = await kbdContext.newPage();
  await kbdPage.goto(BASE_URL, { waitUntil: 'networkidle' });

  const focusLog = [];
  for (let i = 0; i < 20; i++) {
    await kbdPage.keyboard.press('Tab');
    const focusedEl = await kbdPage.evaluate(() => {
      const el = document.activeElement;
      if (!el || el === document.body) return null;
      return {
        tagName: el.tagName,
        text: el.textContent?.substring(0, 40).trim(),
        className: el.className,
        outline: window.getComputedStyle(el).outline,
        boxShadow: window.getComputedStyle(el).boxShadow,
      };
    });
    if (focusedEl) focusLog.push(focusedEl);
  }
  report.interactions.keyboardFocusSequence = focusLog;
  await kbdPage.screenshot({ path: path.join(OUTPUT_DIR, 'keyboard_focus_tab.png') });
  await kbdPage.screenshot({ path: path.join(LOCAL_SCREENSHOTS, 'keyboard_focus_tab.png') });
  await kbdContext.close();

  await browser.close();

  fs.writeFileSync(path.join(OUTPUT_DIR, 'audit_report.json'), JSON.stringify(report, null, 2));
  fs.writeFileSync(path.join(LOCAL_SCREENSHOTS, 'audit_report.json'), JSON.stringify(report, null, 2));
  console.log('\n✅ Real Browser Audit Complete! Results saved to screenshots/phase1_audit/ and audit_screenshots/');
}

runAudit().catch(err => {
  console.error('Fatal audit error:', err);
  process.exit(1);
});
