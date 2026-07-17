import type { ReactNode } from 'react';

interface FullscreenPanelProps {
  isFullscreen: boolean;
  className?: string; // class name for non-fullscreen state
  children: ReactNode;
}

/**
 * Reusable layout wrapper for panel components that can toggle to fullscreen mode.
 * Standardizes styling (bg-opacity-98) and behavior.
 */
export function FullscreenPanel({ isFullscreen, className = '', children }: FullscreenPanelProps) {
  if (isFullscreen) {
    return (
      <div className="fixed inset-0 bg-[#16161e] bg-opacity-98 z-50 p-8 flex flex-col overflow-y-auto">
        {children}
      </div>
    );
  }

  return (
    <div className={className}>
      {children}
    </div>
  );
}
