import type { ParameterStudyEntry } from '../store/useDashboardStore';

export function matchParamStudyEntry(
  entries: ParameterStudyEntry[] | undefined | null,
  initialRadius: string,
  finalRadius: string,
  epochs: number
): ParameterStudyEntry | undefined {
  if (!entries || entries.length === 0) return undefined;
  const activeRin = initialRadius === '50%' ? 50 : initialRadius === '100%' ? 100 : 80;
  const activeRfin = finalRadius === '2' ? 2 : 1;
  return entries.find(
    (entry) =>
      entry.total_epochs === epochs &&
      entry.radius_initial_pct === activeRin &&
      entry.radius_final === activeRfin
  );
}
