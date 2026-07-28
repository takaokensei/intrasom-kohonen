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

  // Map of edges by pair of neuron IDs
  const edgeMap = useMemo(() => {
    const map = new Map<string, number>();
    umatrix_edges.forEach(e => {
      const p1 = Math.min(e.from, e.to);
      const p2 = Math.max(e.from, e.to);
      map.set(`${p1}-${p2}`, e.distance);
    });
    return map;
  }, [umatrix_edges]);

  // Determine whether to use expanded grid (2N-1) x (2M-1)
  const isExpanded = umatrix_edges.length > 0;
  const gridCols = isExpanded ? 2 * cols - 1 : cols;
  const gridRows = isExpanded ? 2 * rows - 1 : rows;

  // Build 3D mesh geometry with height map & vertex colors
  const { geometry, vertexNeuronMap } = useMemo(() => {
    const geo = new THREE.BufferGeometry();
    const positions: number[] = [];
    const colors: number[] = [];
    const indices: number[] = [];
    const vNeuronMap: Array<{ neuronId: number; val: number; row: number; col: number }> = [];

    const widthScale = 8 / Math.max(gridCols, 1);
    const heightScale = 2.5;

    // Helper to get value at grid position (r, c)
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
      } else if (!rIsEven && !cIsEven) {
        // Center vertex between 4 neurons — average surrounding neurons
        const r1 = Math.floor(r / 2);
        const r2 = Math.ceil(r / 2);
        const c1 = Math.floor(c / 2);
        const c2 = Math.ceil(c / 2);
        const n1 = neuronMap.get(`${r1},${c1}`);
        const n2 = neuronMap.get(`${r1},${c2}`);
        const n3 = neuronMap.get(`${r2},${c1}`);
        const n4 = neuronMap.get(`${r2},${c2}`);
        const vals = [n1, n2, n3, n4].filter(Boolean).map(n => n!.umatrix_value);
        const avgVal = vals.length ? vals.reduce((a, b) => a + b, 0) / vals.length : minVal;
        const nearestId = n1?.id || n2?.id || n3?.id || n4?.id || 0;
        return { val: avgVal, id: nearestId };
      } else {
        // Interstitial edge cell
        const r1 = Math.floor(r / 2);
        const r2 = Math.ceil(r / 2);
        const c1 = Math.floor(c / 2);
        const c2 = Math.ceil(c / 2);
        const n1 = neuronMap.get(`${r1},${c1}`);
        const n2 = neuronMap.get(`${r2},${c2}`);
        if (n1 && n2) {
          const p1 = Math.min(n1.id, n2.id);
          const p2 = Math.max(n1.id, n2.id);
          const edgeDist = edgeMap.get(`${p1}-${p2}`);
          if (edgeDist !== undefined) {
            return { val: edgeDist, id: n1.id };
          }
        }
        const v1 = n1?.umatrix_value ?? minVal;
        const v2 = n2?.umatrix_value ?? minVal;
        return { val: (v1 + v2) / 2, id: n1?.id ?? n2?.id ?? 0 };
      }
    };

    // Generate grid vertices
    for (let r = 0; r < gridRows; r++) {
      for (let c = 0; c < gridCols; c++) {
        const { val, id } = getValueAt(r, c);
        const normH = Math.max(0, Math.min(1, (val - minVal) / (maxVal - minVal || 1)));

        // X, Z coordinates with hex staggering
        const hexStag = (lattice === 'HEX' && r % 2 === 1) ? 0.5 : 0;
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
  }, [gridCols, gridRows, isExpanded, neuronMap, edgeMap, minVal, maxVal, lattice]);

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
          <div className="text-[10px] text-tokyo-muted font-bold uppercase tracking-wider mb-0.5">
            Neurônio #{hoverInfo.neuronId} ({hoverInfo.row}, {hoverInfo.col})
          </div>
          <div className="text-tokyo-cyan font-bold">
            U-Matrix: <span className="text-tokyo-text">{hoverInfo.val.toFixed(4)}</span>
          </div>
        </div>
      )}

      {/* Floating 3D Navigation Hint */}
      <div className="absolute bottom-2 left-3 pointer-events-none text-[10px] font-mono text-tokyo-muted bg-[#16161e] bg-opacity-80 px-2 py-1 rounded border border-tokyo-border border-opacity-30">
        🖱️ Arraste para girar | Scroll para zoom
      </div>
    </div>
  );
}
