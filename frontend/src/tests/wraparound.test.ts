import { describe, it, expect } from 'vitest';

/**
 * Wraparound detection logic for periodic boundary edges on a toroidal grid.
 */
function isWraparoundEdge(
  r1: number,
  c1: number,
  r2: number,
  c2: number
): boolean {
  const dr = Math.abs(r1 - r2);
  const dc = Math.abs(c1 - c2);

  // If distance along either axis is > 1 grid unit, the edge wraps across opposite boundaries
  return dr > 1 || dc > 1;
}

describe('Wraparound Edge Classifier', () => {
  it('correctly classifies internal vs wraparound edges on a 10x10 grid', () => {
    // Internal adjacent edges (should NOT be wraparound)
    expect(isWraparoundEdge(0, 0, 0, 1)).toBe(false);
    expect(isWraparoundEdge(0, 0, 1, 0)).toBe(false);
    expect(isWraparoundEdge(4, 5, 5, 5)).toBe(false);

    // Wraparound edges connecting row 0 and row 9 (opposite vertical boundaries)
    expect(isWraparoundEdge(0, 0, 9, 0)).toBe(true);
    expect(isWraparoundEdge(0, 4, 9, 4)).toBe(true);

    // Wraparound edges connecting col 0 and col 9 (opposite horizontal boundaries)
    expect(isWraparoundEdge(3, 0, 3, 9)).toBe(true);

    // Diagonal wraparound edges (e.g. top-left corner wrapping to bottom-right)
    expect(isWraparoundEdge(0, 0, 9, 9)).toBe(true);
  });
});
