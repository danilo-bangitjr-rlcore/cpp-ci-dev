import type { ReactNode } from 'react';

interface MetricHeaderProps {
  title: string;
  badge?: ReactNode;
}

export function MetricHeader({ title, badge }: MetricHeaderProps) {
  return (
    <div className="flex items-center justify-between mb-2">
      <span className="text-sm font-medium text-gray-700">{title}</span>
      {badge}
    </div>
  );
}
