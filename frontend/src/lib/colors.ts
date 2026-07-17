/**
 * Paleta de cores centralizada para todos os datasets.
 *
 * Por que indexar por dataset:
 * - Cada dataset tem seu próprio vocabulário de classes.
 * - Indexar separadamente evita colisão de nomes (ex: "Esportes" e "Baseball"
 *   ambos mapeados para a mesma chave plana quebraria a legenda ao alternar).
 * - Adicionar um terceiro dataset exige apenas uma nova entrada neste objeto,
 *   sem tocar em nenhum componente.
 */

export const SYNTHETIC_CLASS_COLORS: Record<string, string> = {
  "Normal": "#7aa2f7",
  "Cyclic": "#7dcfff",
  "Increasing Trend": "#ff9e64",
  "Decreasing Trend": "#e0af68",
  "Upward Shift": "#9ece6a",
  "Downward Shift": "#f7768e",
};

export const TEXT_CLASS_COLORS: Record<string, Record<string, string>> = {
  "20news": {
    "Graphics": "#3182bd",
    "Space":    "#31a354",
    "Baseball": "#e6550d",
    "Mideast":  "#756bb1",
  },
  "6class": {
    "Turismo":    "#3182bd",
    "Esportes":   "#31a354",
    "Policia":    "#e6550d",
    "Economia":   "#756bb1",
    "Politica":   "#e7ba52",
    "Variedades": "#d6616b",
  },
};

/** Returns the hex color for a given dataset + class, or a neutral fallback. */
export function getClassColor(dataset: string, className: string): string {
  return TEXT_CLASS_COLORS[dataset]?.[className] ?? "#555577";
}

/**
 * Returns a Tailwind-safe inline style object `{ color: hex }` for JSX.
 * Using style props instead of dynamic class names avoids Tailwind JIT purge issues.
 */
export function getTextStyle(dataset: string, className: string): React.CSSProperties {
  return { color: getClassColor(dataset, className) };
}

export function getBgStyle(dataset: string, className: string): React.CSSProperties {
  const hex = getClassColor(dataset, className);
  return {
    backgroundColor: `${hex}1a`, // ~10% opacity
    borderColor: `${hex}4d`,     // ~30% opacity
  };
}
