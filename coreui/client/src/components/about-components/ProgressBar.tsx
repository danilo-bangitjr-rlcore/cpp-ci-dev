interface ProgressBarProps {
  percent: number;
  color?: 'blue' | 'purple' | 'green';
  height?: 'sm' | 'md';
}

export function ProgressBar({
  percent,
  color = 'blue',
  height = 'md',
}: ProgressBarProps) {
  const colorClasses = {
    blue: 'bg-blue-500',
    purple: 'bg-purple-500',
    green: 'bg-green-500',
  };

  const heightClasses = {
    sm: 'h-1',
    md: 'h-2',
  };

  return (
    <div className={`w-full bg-gray-200 rounded-full ${heightClasses[height]}`}>
      <div
        className={`${colorClasses[color]} ${heightClasses[height]} rounded-full transition-all duration-300`}
        style={{ width: `${Math.min(percent, 100)}%` }}
      ></div>
    </div>
  );
}
