interface StatusBadgeProps {
  isLoading: boolean;
  error: unknown;
  isHealthy?: boolean;
}

export function StatusBadge({ isLoading, error, isHealthy }: StatusBadgeProps) {
  const getStatusClass = () => {
    if (isLoading) {
      return 'bg-yellow-100 text-yellow-800 border border-yellow-200';
    }
    if (error) {
      return 'bg-red-100 text-red-800 border border-red-200';
    }
    if (isHealthy === false) {
      return 'bg-red-100 text-red-800 border border-red-200';
    }
    return 'bg-green-100 text-green-800 border border-green-200';
  };

  const getStatusText = () => {
    if (isLoading) return 'Checking...';
    if (error) return 'Error';
    if (isHealthy === false) return 'Unhealthy';
    return 'OK';
  };

  return (
    <span
      className={`px-2 py-1 rounded text-xs font-medium ${getStatusClass()}`}
    >
      {getStatusText()}
    </span>
  );
}
