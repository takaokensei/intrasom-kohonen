import { describe, expect, it } from 'vitest';
import { getHexPoints } from '../lib/geometry';

describe('getHexPoints', () => {
  it('should return a string containing 6 coordinate pairs', () => {
    const pointsStr = getHexPoints(100, 100, 10);
    const points = pointsStr.split(' ');
    expect(points).toHaveLength(6);
    points.forEach(point => {
      const coords = point.split(',');
      expect(coords).toHaveLength(2);
      const x = parseFloat(coords[0]);
      const y = parseFloat(coords[1]);
      expect(isNaN(x)).toBe(false);
      expect(isNaN(y)).toBe(false);
    });
  });
});
