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
