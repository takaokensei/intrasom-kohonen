// NOTA — Topologia Toroidal Real:
// Como os SOMs hexagonais são treinados com mapshape='toroid', a superfície 3D topologicamente
// exata é um torus real (donut) sem costuras. Cada um dos neurônios e arestas (incluindo as 39 de wraparound)
// conecta-se naturalmente na superfície contínua do tubo sem necessidade de artefatos de borda.

import { useMemo, useState, useRef, useCallback } from 'react';
import { Canvas } from '@react-three/fiber';
import { OrbitControls } from '@react-three/drei';
import * as THREE from 'three';
import { getUMatrixColor } from '../lib/colors';

export interface UMatrixTorusEdge {
  from: number;
  to: number;
  distance: number;
}

export interface UMatrixTorusNeuron {
  id: number;
  row: number;
  col: number;
  umatrix_value: number;
  dominant_class?: string;
  total_samples?: number;
  purity?: number;
}

export interface UMatrixTorusProps {
  neurons: UMatrixTorusNeuron[];
  cols: number;
  rows: number;
  umatrix_edges?: UMatrixTorusEdge[];
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

function UMatrixTorusMesh({
  neurons,
  cols,
  rows,
  umatrix_edges = [],
  edgeMin,
  edgeMax,
  lattice = 'HEX',
  onHover
}: UMatrixTorusProps & { onHover: (info: HoveredPointInfo | null) => void }) {
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
    const map = new Map<string, UMatrixTorusNeuron>();
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

  // Expanded grid dimensions (2N) x (2M) for seamless closed torus loops
  const isExpanded = umatrix_edges.length > 0;
  const gridCols = isExpanded ? 2 * cols : cols;
  const gridRows = isExpanded ? 2 * rows : rows;

  // Build 3D Torus Surface geometry with U-Matrix bump mapping & vertex colors
  const { geometry, vertexNeuronMap } = useMemo(() => {
    const geo = new THREE.BufferGeometry();
    const positions: number[] = [];
    const colors: number[] = [];
    const indices: number[] = [];
    const vNeuronMap: Array<{ neuronId: number; val: number; row: number; col: number }> = [];

    const R = 4.0;            // Major radius (center to tube center)
    const tubeR0 = 1.2;       // Base tube radius
    const bumpStrength = 0.5;  // U-Matrix bulge height modulation

    // Helper to get exact value & neuron ID at grid position (r, c) with toroidal modulo
    const getValueAt = (r: number, c: number): { val: number; id: number } => {
      const rMod = (r % (2 * rows) + 2 * rows) % (2 * rows);
      const cMod = (c % (2 * cols) + 2 * cols) % (2 * cols);

      if (!isExpanded) {
        const n = neuronMap.get(`${rMod},${cMod}`);
        return { val: n?.umatrix_value ?? minVal, id: n?.id ?? 0 };
      }

      const rIsEven = rMod % 2 === 0;
      const cIsEven = cMod % 2 === 0;

      if (rIsEven && cIsEven) {
        // Real neuron
        const realR = rMod / 2;
        const realC = cMod / 2;
        const n = neuronMap.get(`${realR},${realC}`);
        return { val: n?.umatrix_value ?? minVal, id: n?.id ?? 0 };
      } else if (rIsEven && !cIsEven) {
        // Horizontal edge (including toroidal wraparound column)
        const realR = rMod / 2;
        const c1 = Math.floor(cMod / 2);
        const c2 = (c1 + 1) % cols;
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
        // Vertical edge (including toroidal wraparound row)
        const r1 = Math.floor(rMod / 2);
        const r2 = (r1 + 1) % rows;
        const realC = cMod / 2;
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
        // Diagonal edge (odd-r hex parity with toroidal modulo)
        const r1 = Math.floor(rMod / 2);
        const r2 = (r1 + 1) % rows;
        const c1 = Math.floor(cMod / 2);
        const c2 = (c1 + 1) % cols;

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

        const altN1 = isR1Even ? neuronMap.get(`${r1},${c1}`) : neuronMap.get(`${r1},${c2}`);
        const altN2 = isR1Even ? neuronMap.get(`${r2},${c2}`) : neuronMap.get(`${r2},${c1}`);
        if (altN1 && altN2) {
          const altPair = `${Math.min(altN1.id, altN2.id)}-${Math.max(altN1.id, altN2.id)}`;
          const altEdgeDist = edgeMap.get(altPair);
          if (altEdgeDist !== undefined) {
            return { val: altEdgeDist, id: altN1.id };
          }
        }

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

    // Generate grid vertices parametrized on 3D Torus
    for (let r = 0; r <= gridRows; r++) {
      const u = (r / gridRows) * Math.PI * 2;
      const realRow = Math.floor((r % (2 * rows)) / 2);
      const hexStagAngle = (lattice === 'HEX' && realRow % 2 === 1) ? (Math.PI / gridCols) : 0;

      for (let c = 0; c <= gridCols; c++) {
        const v = (c / gridCols) * Math.PI * 2 + hexStagAngle;

        const { val, id } = getValueAt(r, c);
        const normH = Math.max(0, Math.min(1, (val - minVal) / (maxVal - minVal || 1)));
        const tubeR = tubeR0 * (1 + bumpStrength * normH);

        const x = (R + tubeR * Math.cos(v)) * Math.cos(u);
        const y = (R + tubeR * Math.cos(v)) * Math.sin(u);
        const z = tubeR * Math.sin(v);

        positions.push(x, y, z);

        const cColor = parseRgbToThreeColor(getUMatrixColor(val, minVal, maxVal));
        colors.push(cColor.r, cColor.g, cColor.b);

        vNeuronMap.push({
          neuronId: id,
          val,
          row: isExpanded ? Math.floor((r % (2 * rows)) / 2) : r % rows,
          col: isExpanded ? Math.floor((c % (2 * cols)) / 2) : c % cols
        });
      }
    }

    // Triangulate grid into torus surface quads
    const vertsPerRow = gridCols + 1;
    for (let r = 0; r < gridRows; r++) {
      for (let c = 0; c < gridCols; c++) {
        const topLeft = r * vertsPerRow + c;
        const topRight = topLeft + 1;
        const bottomLeft = (r + 1) * vertsPerRow + c;
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
  }, [gridCols, gridRows, isExpanded, neuronMap, edgeMap, minVal, maxVal, lattice, rows, cols]);

  const handlePointerMove = useCallback((e: any) => {
    e.stopPropagation();
    if (!e.point || e.index === undefined) return;
    const vertexIdx = e.index;
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
  );
}

export function UMatrixTorus(props: UMatrixTorusProps) {
  const [hoverInfo, setHoverInfo] = useState<HoveredPointInfo | null>(null);

  return (
    <div className="relative w-full h-full min-h-[320px] rounded-xl overflow-hidden bg-tokyo-bg">
      <Canvas
        camera={{ position: [10, 8, 10], fov: 42 }}
        className="w-full h-full"
      >
        <color attach="background" args={['#1a1b26']} />
        <ambientLight intensity={0.7} />
        <directionalLight position={[10, 15, 10]} intensity={0.8} />
        <directionalLight position={[-10, -5, -10]} intensity={0.25} />
        
        <UMatrixTorusMesh {...props} onHover={setHoverInfo} />

        <OrbitControls
          enableDamping
          dampingFactor={0.05}
          minDistance={4}
          maxDistance={30}
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
          <div className="text-2xs text-tokyo-magenta font-bold uppercase tracking-wider mb-0.5">
            🌀 Superfície Toroidal — Neurônio #{hoverInfo.neuronId} ({hoverInfo.row}, {hoverInfo.col})
          </div>
          <div className="text-tokyo-cyan font-bold">
            U-Matrix: <span className="text-tokyo-text">{hoverInfo.val.toFixed(4)}</span>
          </div>
        </div>
      )}

      {/* Floating 3D Navigation & Methodology Hint */}
      <div className="absolute bottom-2 left-3 right-3 pointer-events-none text-[9px] font-mono text-tokyo-muted bg-[#16161e] bg-opacity-85 px-2.5 py-1.5 rounded border border-tokyo-border border-opacity-30 flex items-center justify-between">
        <span>🖱️ Arraste para girar | Scroll para zoom</span>
        <span className="text-tokyo-magenta font-semibold truncate ml-2">
          🍩 SOM Toroidal na topologia real — sem costuras ou artefatos de borda.
        </span>
      </div>
    </div>
  );
}
