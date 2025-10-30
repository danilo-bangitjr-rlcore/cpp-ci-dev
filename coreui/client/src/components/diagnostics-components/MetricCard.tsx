import type { ReactNode } from 'react';

interface MetricCardProps {
  children: ReactNode;
}

export function MetricCard({ children }: MetricCardProps) {
  return (
    <div className="p-3 bg-gray-50 rounded border border-gray-200">
      {children}
    </div>
  );
}
