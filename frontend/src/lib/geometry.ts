/**
 * Geometry utilities for regular hexagonal (pointy-topped) and rectangular SOM lattices.
 */

/**
 * Calculates the SVG point coordinates for a regular pointy-topped hexagon.
 */
export function getHexPoints(cx: number, cy: number, radius: number): string {
  const points = [];
  for (let i = 0; i < 6; i++) {
    const angle = (i * Math.PI) / 3 - Math.PI / 6;
    points.push(`${cx + radius * Math.cos(angle)},${cy + radius * Math.sin(angle)}`);
  }
  return points.join(' ');
}

/**
 * Calculates the exact radius needed for hexagons to touch edge-to-edge seamlessly.
 */
export function computeContiguousHexRadius(
  cols: number,
  rows: number,
  width: number,
  height: number,
  padding: number = 25
): number {
  if (cols <= 0 || rows <= 0) return 10;
  const maxRx = (width - 2 * padding) / ((cols + 0.5) * Math.sqrt(3));
  const maxRy = (height - 2 * padding) / (1.5 * rows + 0.5);
  return Math.max(5, Math.min(maxRx, maxRy));
}

/**
 * Calculates the exact center (cx, cy) of a cell in a contiguous lattice.
 */
export function getHexCenter(
  row: number,
  col: number,
  radius: number,
  padding: number = 25,
  lattice: string = 'HEX'
): { cx: number; cy: number } {
  if (lattice === 'RECT') {
    const side = radius * Math.sqrt(3);
    return {
      cx: padding + col * side + side / 2,
      cy: padding + row * side + side / 2,
    };
  }
  const hexWidth = radius * Math.sqrt(3);
  const cx = padding + (col + (row % 2 === 1 ? 0.5 : 0)) * hexWidth + hexWidth / 2;
  const cy = padding + row * (1.5 * radius) + radius;
  return { cx, cy };
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
    gridRow: fromRow + toRow,
    gridCol: fromCol + toCol,
  };
}
