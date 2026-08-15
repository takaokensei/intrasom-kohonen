import { chromium } from 'playwright';

async function measurePerf() {
  const browser = await chromium.launch({ headless: true });
  const context = await browser.newContext();
  const page = await context.newPage();

  const start = Date.now();
  await page.goto('http://127.0.0.1:5173/', { waitUntil: 'networkidle' });
  const loadTime = Date.now() - start;

  const perfMetrics = await page.evaluate(() => {
    const nav = performance.getEntriesByType('navigation')[0];
    const paint = performance.getEntriesByType('paint');
    const fcp = paint.find(p => p.name === 'first-contentful-paint')?.startTime;
    return {
      domContentLoaded: nav ? nav.domContentLoadedEventEnd - nav.startTime : null,
      loadEventEnd: nav ? nav.loadEventEnd - nav.startTime : null,
      fcp: fcp || null,
      resourceCount: performance.getEntriesByType('resource').length
    };
  });

  // Measure tab switch chunk fetch time
  const t0 = Date.now();
  await page.locator('button:has-text("Clusterização de Textos"), button:has-text("Text")').first().click();
  await page.waitForTimeout(300);
  const tabSwitchTime = Date.now() - t0;

  console.log('--- REAL PERFORMANCE MEASUREMENTS ---');
  console.log('Total Initial NetworkIdle Load:', loadTime, 'ms');
  console.log('DOM Content Loaded:', perfMetrics.domContentLoaded?.toFixed(1), 'ms');
  console.log('First Contentful Paint (FCP):', perfMetrics.fcp?.toFixed(1), 'ms');
  console.log('Resources Loaded:', perfMetrics.resourceCount);
  console.log('First Tab Switch (Lazy-load TextScreen chunk):', tabSwitchTime, 'ms');
  console.log('------------------------------------');

  await browser.close();
}

measurePerf().catch(console.error);
