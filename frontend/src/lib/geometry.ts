/**
 * Calculates the SVG point coordinates for a regular flat-topped hexagon.
 */
export function getHexPoints(cx: number, cy: number, radius: number): string {
  const points = [];
  for (let i = 0; i < 6; i++) {
    const angle = (i * Math.PI) / 3 - Math.PI / 6;
    points.push(`${cx + radius * Math.cos(angle)},${cy + radius * Math.sin(angle)}`);
  }
  return points.join(' ');
}

/** Real neurons map to EVEN grid coordinates in doubled grid (2n-1) x (2n-1). */
export function getExpandedGridPosition(
  neuronRow: number, neuronCol: number
): { gridRow: number; gridCol: number } {
  return { gridRow: neuronRow * 2, gridCol: neuronCol * 2 };
}

/** Interstitial edge cells map to midpoint between adjacent neurons in doubled grid. */
export function getInterstitialGridPosition(
  fromRow: number, fromCol: number,
  toRow: number, toCol: number
): { gridRow: number; gridCol: number } {
  return {
    gridRow: fromRow + toRow,   // sum equals midpoint in doubled-grid coords
    gridCol: fromCol + toCol,
  };
}

