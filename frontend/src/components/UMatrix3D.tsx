// NOTA — Trabalho Futuro:
// Como os SOMs hexagonais são treinados com mapshape='toroid', a representação 3D topologicamente
// mais correta não é uma superfície plana — é um torus real (donut), onde a U-Matrix modula a
// cor/deslocamento radial ao longo da superfície do anel. Isso eliminaria de forma natural
// qualquer problema de "borda" na visualização (um torus não tem costura).

import { useMemo, useState, useRef, useCallback } from 'react';
import { Canvas } from '@react-three/fiber';
import { OrbitControls } from '@react-three/drei';
import * as THREE from 'three';
import { getUMatrixColor } from '../lib/colors';

export interface UMatrix3DEdge {
  from: number;
  to: number;
  distance: number;
}

export interface UMatrix3DNeuron {
  id: number;
  row: number;
  col: number;
  umatrix_value: number;
  dominant_class?: string;
  total_samples?: number;
  purity?: number;
}

export interface UMatrix3DProps {
  neurons: UMatrix3DNeuron[];
  cols: number;
  rows: number;
  umatrix_edges?: UMatrix3DEdge[];
  edgeMin?: number;
  edgeMax?: number;
  lattice?: 'HEX' | 'RECT';
}

interface HoveredPointInfo {
  neuronId: number;
  val: number;
  row: number;
  col: number;
  screenX: number;
  screenY: number;
  isWraparound?: boolean;
  targetId?: number;
}

/**
 * Converts "rgb(r, g, b)" string into a THREE.Color instance.
 */
function parseRgbToThreeColor(rgbStr: string): THREE.Color {
  const match = rgbStr.match(/rgb\(\s*(\d+)\s*,\s*(\d+)\s*,\s*(\d+)\s*\)/);
  if (match) {
    return new THREE.Color(
      parseInt(match[1], 10) / 255,
      parseInt(match[2], 10) / 255,
      parseInt(match[3], 10) / 255
    );
  }
  return new THREE.Color(0x1a1b26);
}

function UMatrixMesh({
  neurons,
  cols,
  rows,
  umatrix_edges = [],
  edgeMin,
  edgeMax,
  lattice = 'HEX',
  onHover
}: UMatrix3DProps & { onHover: (info: HoveredPointInfo | null) => void }) {
  const meshRef = useRef<THREE.Mesh>(null);

  // Compute min/max for normalization
  const { minVal, maxVal } = useMemo(() => {
    const vals = neurons.map(n => n.umatrix_value);
    const minV = edgeMin ?? (vals.length ? Math.min(...vals) : 0);
    const maxV = edgeMax ?? (vals.length ? Math.max(...vals) : 1);
    return { minVal: minV, maxVal: maxV };
  }, [neurons, edgeMin, edgeMax]);

  // Map of real neurons by (row, col)
  const neuronMap = useMemo(() => {
    const map = new Map<string, UMatrix3DNeuron>();
    neurons.forEach(n => {
      map.set(`${n.row},${n.col}`, n);
    });
    return map;
  }, [neurons]);

  // Map of edges by pair of neuron IDs ("p1-p2" -> distance)
  const edgeMap = useMemo(() => {
    const map = new Map<string, number>();
    umatrix_edges.forEach(e => {
      const p1 = Math.min(e.from, e.to);
      const p2 = Math.max(e.from, e.to);
      map.set(`${p1}-${p2}`, e.distance);
    });
    return map;
  }, [umatrix_edges]);

  // Separate edges into internal vs wraparound (toroidal)
  const { wraparoundEdges } = useMemo(() => {
    const nPos = new Map<number, { row: number; col: number }>();
    neurons.forEach(n => nPos.set(n.id, { row: n.row, col: n.col }));

    const wrap: Array<{ from: number; to: number; distance: number; n1: UMatrix3DNeuron; n2: UMatrix3DNeuron }> = [];

    umatrix_edges.forEach(e => {
      const p1 = nPos.get(e.from);
      const p2 = nPos.get(e.to);
      if (!p1 || !p2) return;
      const dr = Math.abs(p1.row - p2.row);
      const dc = Math.abs(p1.col - p2.col);
      const isWrap = dr > 1 || dc > 1;
      if (isWrap) {
        const n1 = neurons.find(n => n.id === e.from);
        const n2 = neurons.find(n => n.id === e.to);
        if (n1 && n2) {
          wrap.push({ from: e.from, to: e.to, distance: e.distance, n1, n2 });
        }
      }
    });

    return { wraparoundEdges: wrap };
  }, [neurons, umatrix_edges]);

  // Determine whether to use expanded grid (2N-1) x (2M-1)
  const isExpanded = umatrix_edges.length > 0;
  const gridCols = isExpanded ? 2 * cols - 1 : cols;
  const gridRows = isExpanded ? 2 * rows - 1 : rows;

  const widthScale = 8 / Math.max(gridCols, 1);
  const heightScale = 2.5;

  // Build 3D main mesh geometry with height map & vertex colors
  const { geometry, vertexNeuronMap } = useMemo(() => {
    const geo = new THREE.BufferGeometry();
    const positions: number[] = [];
    const colors: number[] = [];
    const indices: number[] = [];
    const vNeuronMap: Array<{ neuronId: number; val: number; row: number; col: number }> = [];

    // Helper to get exact value & neuron ID at grid position (r, c)
    const getValueAt = (r: number, c: number): { val: number; id: number } => {
      if (!isExpanded) {
        const n = neuronMap.get(`${r},${c}`);
        return { val: n?.umatrix_value ?? minVal, id: n?.id ?? 0 };
      }

      const rIsEven = r % 2 === 0;
      const cIsEven = c % 2 === 0;

      if (rIsEven && cIsEven) {
        // Real neuron
        const realR = r / 2;
        const realC = c / 2;
        const n = neuronMap.get(`${realR},${realC}`);
        return { val: n?.umatrix_value ?? minVal, id: n?.id ?? 0 };
      } else if (rIsEven && !cIsEven) {
        // Horizontal edge: between (r/2, (c-1)/2) and (r/2, (c+1)/2)
        const realR = r / 2;
        const c1 = (c - 1) / 2;
        const c2 = (c + 1) / 2;
        const n1 = neuronMap.get(`${realR},${c1}`);
        const n2 = neuronMap.get(`${realR},${c2}`);
        if (n1 && n2) {
          const pair = `${Math.min(n1.id, n2.id)}-${Math.max(n1.id, n2.id)}`;
          const edgeDist = edgeMap.get(pair);
          if (edgeDist !== undefined) {
            return { val: edgeDist, id: n1.id };
          }
        }
        const v1 = n1?.umatrix_value ?? minVal;
        const v2 = n2?.umatrix_value ?? minVal;
        return { val: (v1 + v2) / 2, id: n1?.id ?? n2?.id ?? 0 };
      } else if (!rIsEven && cIsEven) {
        // Vertical edge: between ((r-1)/2, c/2) and ((r+1)/2, c/2)
        const r1 = (r - 1) / 2;
        const r2 = (r + 1) / 2;
        const realC = c / 2;
        const n1 = neuronMap.get(`${r1},${realC}`);
        const n2 = neuronMap.get(`${r2},${realC}`);
        if (n1 && n2) {
          const pair = `${Math.min(n1.id, n2.id)}-${Math.max(n1.id, n2.id)}`;
          const edgeDist = edgeMap.get(pair);
          if (edgeDist !== undefined) {
            return { val: edgeDist, id: n1.id };
          }
        }
        const v1 = n1?.umatrix_value ?? minVal;
        const v2 = n2?.umatrix_value ?? minVal;
        return { val: (v1 + v2) / 2, id: n1?.id ?? n2?.id ?? 0 };
      } else {
        // TAREFA A: Both ODD (diagonal edge in hex odd-r lattice)
        // In pointy-topped odd-r hex lattice:
        // For upper row r1 = (r-1)/2 and lower row r2 = (r+1)/2:
        // - If r1 is EVEN (r1 % 2 === 0): diagonal edge is (r1, c2) <-> (r2, c1) where c1 = (c-1)/2, c2 = (c+1)/2
        // - If r1 is ODD  (r1 % 2 === 1): diagonal edge is (r1, c1) <-> (r2, c2)
        const r1 = (r - 1) / 2;
        const r2 = (r + 1) / 2;
        const c1 = (c - 1) / 2;
        const c2 = (c + 1) / 2;

        const isR1Even = r1 % 2 === 0;
        const diagN1 = isR1Even ? neuronMap.get(`${r1},${c2}`) : neuronMap.get(`${r1},${c1}`);
        const diagN2 = isR1Even ? neuronMap.get(`${r2},${c1}`) : neuronMap.get(`${r2},${c2}`);

        if (diagN1 && diagN2) {
          const pair = `${Math.min(diagN1.id, diagN2.id)}-${Math.max(diagN1.id, diagN2.id)}`;
          const edgeDist = edgeMap.get(pair);
          if (edgeDist !== undefined) {
            return { val: edgeDist, id: diagN1.id };
          }
        }

        // Alternative diagonal check fallback
        const altN1 = isR1Even ? neuronMap.get(`${r1},${c1}`) : neuronMap.get(`${r1},${c2}`);
        const altN2 = isR1Even ? neuronMap.get(`${r2},${c2}`) : neuronMap.get(`${r2},${c1}`);
        if (altN1 && altN2) {
          const altPair = `${Math.min(altN1.id, altN2.id)}-${Math.max(altN1.id, altN2.id)}`;
          const altEdgeDist = edgeMap.get(altPair);
          if (altEdgeDist !== undefined) {
            return { val: altEdgeDist, id: altN1.id };
          }
        }

        // Fallback: average 4 surrounding corner neurons
        const n1 = neuronMap.get(`${r1},${c1}`);
        const n2 = neuronMap.get(`${r1},${c2}`);
        const n3 = neuronMap.get(`${r2},${c1}`);
        const n4 = neuronMap.get(`${r2},${c2}`);
        const vals = [n1, n2, n3, n4].filter(Boolean).map(n => n!.umatrix_value);
        const avgVal = vals.length ? vals.reduce((a, b) => a + b, 0) / vals.length : minVal;
        const nearestId = diagN1?.id || n1?.id || n2?.id || 0;
        return { val: avgVal, id: nearestId };
      }
    };

    // Generate grid vertices
    for (let r = 0; r < gridRows; r++) {
      for (let c = 0; c < gridCols; c++) {
        const { val, id } = getValueAt(r, c);
        const normH = Math.max(0, Math.min(1, (val - minVal) / (maxVal - minVal || 1)));

        // X, Z coordinates with hex staggering based on REAL row parity
        const realRow = Math.floor(r / 2);
        const hexStag = (lattice === 'HEX' && realRow % 2 === 1) ? 0.5 : 0;
        const x = (c + hexStag - (gridCols - 1) / 2) * widthScale;
        const z = (r - (gridRows - 1) / 2) * widthScale * (lattice === 'HEX' ? Math.sqrt(3) / 2 : 1);
        const y = normH * heightScale;

        positions.push(x, y, z);

        // Vertex color based on U-Matrix value
        const cColor = parseRgbToThreeColor(getUMatrixColor(val, minVal, maxVal));
        colors.push(cColor.r, cColor.g, cColor.b);

        vNeuronMap.push({
          neuronId: id,
          val,
          row: isExpanded ? Math.floor(r / 2) : r,
          col: isExpanded ? Math.floor(c / 2) : c
        });
      }
    }

    // Triangulate grid cells into 2 triangles per quad
    for (let r = 0; r < gridRows - 1; r++) {
      for (let c = 0; c < gridCols - 1; c++) {
        const topLeft = r * gridCols + c;
        const topRight = topLeft + 1;
        const bottomLeft = (r + 1) * gridCols + c;
        const bottomRight = bottomLeft + 1;

        indices.push(topLeft, bottomLeft, topRight);
        indices.push(topRight, bottomLeft, bottomRight);
      }
    }

    geo.setAttribute('position', new THREE.Float32BufferAttribute(positions, 3));
    geo.setAttribute('color', new THREE.Float32BufferAttribute(colors, 3));
    geo.setIndex(indices);
    geo.computeVertexNormals();

    return { geometry: geo, vertexNeuronMap: vNeuronMap };
  }, [gridCols, gridRows, isExpanded, neuronMap, edgeMap, minVal, maxVal, lattice, widthScale, heightScale]);

  // TAREFA B: Render border strips for toroidal wraparound edges
  const borderStrips = useMemo(() => {
    return wraparoundEdges.map(w => {
      const normH = Math.max(0, Math.min(1, (w.distance - minVal) / (maxVal - minVal || 1)));
      const y = normH * heightScale;
      const rgbColor = parseRgbToThreeColor(getUMatrixColor(w.distance, minVal, maxVal));

      // Calculate position of border indicator near n1 pointing outward
      const expR1 = 2 * w.n1.row;
      const expC1 = 2 * w.n1.col;
      const hexStag1 = (lattice === 'HEX' && w.n1.row % 2 === 1) ? 0.5 : 0;

      let borderDx = 0;
      let borderDz = 0;

      if (w.n1.col === 0 && w.n2.col === cols - 1) {
        borderDx = -1.2;
      } else if (w.n1.col === cols - 1 && w.n2.col === 0) {
        borderDx = 1.2;
      }

      if (w.n1.row === 0 && w.n2.row === rows - 1) {
        borderDz = -1.2;
      } else if (w.n1.row === rows - 1 && w.n2.row === 0) {
        borderDz = 1.2;
      }

      const x = (expC1 + hexStag1 - (gridCols - 1) / 2 + borderDx) * widthScale;
      const z = (expR1 - (gridRows - 1) / 2 + borderDz) * widthScale * (lattice === 'HEX' ? Math.sqrt(3) / 2 : 1);

      return {
        key: `wrap-${w.from}-${w.to}`,
        from: w.from,
        to: w.to,
        distance: w.distance,
        position: [x, y, z] as [number, number, number],
        color: rgbColor,
        n1: w.n1,
        n2: w.n2
      };
    });
  }, [wraparoundEdges, minVal, maxVal, gridCols, gridRows, cols, rows, lattice, widthScale, heightScale]);

  const handlePointerMove = useCallback((e: any) => {
    e.stopPropagation();
    if (!e.point || e.index === undefined) return;
    const vertexIdx = e.index; // Index of face vertex
    const info = vertexNeuronMap[vertexIdx];
    if (info && e.nativeEvent) {
      onHover({
        neuronId: info.neuronId,
        val: info.val,
        row: info.row,
        col: info.col,
        screenX: e.nativeEvent.clientX,
        screenY: e.nativeEvent.clientY
      });
    }
  }, [vertexNeuronMap, onHover]);

  const handlePointerOut = useCallback(() => {
    onHover(null);
  }, [onHover]);

  return (
    <group>
      {/* Main Terrain Mesh */}
      <mesh
        ref={meshRef}
        geometry={geometry}
        onPointerMove={handlePointerMove}
        onPointerOut={handlePointerOut}
      >
        <meshStandardMaterial
          vertexColors
          side={THREE.DoubleSide}
          roughness={0.35}
          metalness={0.15}
          flatShading={false}
        />
      </mesh>

      {/* Wraparound Border Strips (Toroidal Edge Indicators) */}
      {borderStrips.map(strip => (
        <mesh
          key={strip.key}
          position={strip.position}
          onPointerMove={(e) => {
            e.stopPropagation();
            if (e.nativeEvent) {
              onHover({
                neuronId: strip.from,
                val: strip.distance,
                row: strip.n1.row,
                col: strip.n1.col,
                screenX: e.nativeEvent.clientX,
                screenY: e.nativeEvent.clientY,
                isWraparound: true,
                targetId: strip.to
              });
            }
          }}
          onPointerOut={() => onHover(null)}
        >
          <boxGeometry args={[0.35, 0.35, 0.35]} />
          <meshStandardMaterial
            color={strip.color}
            roughness={0.2}
            metalness={0.3}
            emissive={strip.color}
            emissiveIntensity={0.2}
          />
        </mesh>
      ))}
    </group>
  );
}

export function UMatrix3D(props: UMatrix3DProps) {
  const [hoverInfo, setHoverInfo] = useState<HoveredPointInfo | null>(null);

  return (
    <div className="relative w-full h-full min-h-[320px] rounded-xl overflow-hidden bg-[#1a1b26]">
      <Canvas
        camera={{ position: [9, 8, 10], fov: 42 }}
        className="w-full h-full"
      >
        <color attach="background" args={['#1a1b26']} />
        <ambientLight intensity={0.7} />
        <directionalLight position={[10, 15, 10]} intensity={0.8} />
        <directionalLight position={[-10, -5, -10]} intensity={0.25} />
        
        <UMatrixMesh {...props} onHover={setHoverInfo} />

        <OrbitControls
          enableDamping
          dampingFactor={0.05}
          maxPolarAngle={Math.PI / 2.05}
          minDistance={3}
          maxDistance={25}
        />
      </Canvas>

      {/* On-hover Tooltip overlay */}
      {hoverInfo && (
        <div
          className="absolute z-20 pointer-events-none glass-panel px-3 py-2 rounded-lg text-xs font-mono border border-tokyo-border shadow-xl transform -translate-x-1/2 -translate-y-full mb-2 bg-[#16161e] bg-opacity-95 text-tokyo-text"
          style={{
            left: `${hoverInfo.screenX}px`,
            top: `${hoverInfo.screenY}px`,
          }}
        >
          {hoverInfo.isWraparound ? (
            <>
              <div className="text-[10px] text-tokyo-magenta font-bold uppercase tracking-wider mb-0.5">
                🌀 Wraparound Toroidal (N{hoverInfo.neuronId} ↔ N{hoverInfo.targetId})
              </div>
              <div className="text-tokyo-cyan font-bold">
                Distância Aresta: <span className="text-tokyo-text">{hoverInfo.val.toFixed(4)}</span>
              </div>
              <div className="text-[9px] text-tokyo-muted mt-0.5 italic">
                Conecta com N{hoverInfo.targetId} via wraparound toroidal
              </div>
            </>
          ) : (
            <>
              <div className="text-[10px] text-tokyo-muted font-bold uppercase tracking-wider mb-0.5">
                Neurônio #{hoverInfo.neuronId} ({hoverInfo.row}, {hoverInfo.col})
              </div>
              <div className="text-tokyo-cyan font-bold">
                U-Matrix: <span className="text-tokyo-text">{hoverInfo.val.toFixed(4)}</span>
              </div>
            </>
          )}
        </div>
      )}

      {/* Floating 3D Navigation Hint */}
      <div className="absolute bottom-2 left-3 pointer-events-none text-[10px] font-mono text-tokyo-muted bg-[#16161e] bg-opacity-80 px-2 py-1 rounded border border-tokyo-border border-opacity-30 flex items-center gap-2">
        <span>🖱️ Arraste para girar | Scroll para zoom</span>
        <span className="text-[9px] text-tokyo-magenta bg-tokyo-magenta bg-opacity-10 px-1.5 py-0.5 rounded border border-tokyo-magenta border-opacity-20 font-semibold">
          📦 Cubos na borda = Wraparound Toroidal
        </span>
      </div>
    </div>
  );
}

