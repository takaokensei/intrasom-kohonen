import { describe, it, expect } from 'vitest';

/**
 * Strict modular boundary crossing test for toroidal wraparound edge classification.
 * Matches the implementation in UMatrix3D.tsx.
 */
export function isWraparoundEdge(
  r1: number, c1: number,
  r2: number, c2: number,
  rows: number, cols: number
): boolean {
  const dr = Math.min(Math.abs(r1 - r2), rows - Math.abs(r1 - r2));
  const dc = Math.min(Math.abs(c1 - c2), cols - Math.abs(c1 - c2));
  const rawDr = Math.abs(r1 - r2);
  const rawDc = Math.abs(c1 - c2);
  const modularlyAdjacent = dr <= 1 && dc <= 1 && (dr + dc) >= 1;
  const crossesBoundary = rawDr > 1 || rawDc > 1;
  return modularlyAdjacent && crossesBoundary;
}

describe('Wraparound Edge Classifier (Strict Modular Boundary Test)', () => {
  it('correctly classifies internal vs wraparound edges on a 10x10 grid', () => {
    const rows = 10;
    const cols = 10;

    // Internal adjacent edges (should NOT be wraparound)
    expect(isWraparoundEdge(0, 0, 0, 1, rows, cols)).toBe(false);
    expect(isWraparoundEdge(0, 0, 1, 0, rows, cols)).toBe(false);
    expect(isWraparoundEdge(4, 5, 5, 5, rows, cols)).toBe(false);

    // Wraparound edges connecting row 0 and row 9 (opposite vertical boundaries)
    expect(isWraparoundEdge(0, 0, 9, 0, rows, cols)).toBe(true);
    expect(isWraparoundEdge(0, 4, 9, 4, rows, cols)).toBe(true);

    // Wraparound edges connecting col 0 and col 9 (opposite horizontal boundaries)
    expect(isWraparoundEdge(3, 0, 3, 9, rows, cols)).toBe(true);

    // Modularly non-adjacent distant nodes (should NOT be marked as single-step wraparound edges)
    expect(isWraparoundEdge(0, 0, 5, 5, rows, cols)).toBe(false);
  });
});
