import { createFileRoute } from '@tanstack/react-router';
import {
  StatusBadge,
  ProgressBar,
  MetricCard,
  MetricHeader,
} from '../components/diagnostics-components';
import {
  useUiHealthQuery,
  useSystemHealthQuery,
  useSystemMetricsQuery,
} from '../utils/useDiagnosticsQueries';

export const Route = createFileRoute('/diagnostics')({
  component: Diagnostics,
});

function Diagnostics() {
  const {
    data: healthData,
    isLoading: isHealthLoading,
    error: healthError,
  } = useUiHealthQuery();

  const {
    data: systemHealth,
    isLoading: isSystemHealthLoading,
    error: systemHealthError,
  } = useSystemHealthQuery();

  const { data: systemMetrics, error: systemMetricsError } =
    useSystemMetricsQuery();

  const platform = systemMetrics?.platform;
  const cpu = systemMetrics?.cpu;
  const cpuPerCore = systemMetrics?.cpu_per_core;
  const ram = systemMetrics?.ram;
  const disk = systemMetrics?.disk;

  return (
    <div className="p-2">
      <h1 className="text-2xl font-bold mb-6">Diagnostics</h1>

      {/* CoreUI Server Status */}
      <div className="mb-6 p-4 bg-white rounded-lg border border-gray-200 shadow-sm">
        <h2 className="text-lg font-semibold text-gray-900 mb-2">
          CoreUI Server Status
        </h2>
        <div className="flex items-center gap-2">
          <span className="text-sm text-gray-600">Health Check:</span>
          <span
            className={`px-2 py-1 rounded text-xs font-medium ${
              isHealthLoading
                ? 'bg-yellow-100 text-yellow-800 border border-yellow-200'
                : healthError
                  ? 'bg-red-100 text-red-800 border border-red-200'
                  : healthData?.status === 'ok'
                    ? 'bg-green-100 text-green-800 border border-green-200'
                    : 'bg-gray-100 text-gray-600 border border-gray-200'
            }`}
          >
            {isHealthLoading
              ? 'Checking...'
              : healthError
                ? 'Offline'
                : healthData?.status === 'ok'
                  ? 'Online'
                  : 'Unknown'}
          </span>
        </div>
        {healthData && (
          <div className="mt-2 text-xs text-gray-500 bg-gray-50 p-2 rounded border inline-block">
            Response: {JSON.stringify(healthData)}
          </div>
        )}
      </div>

      {/* System Metrics */}
      <div className="p-4 bg-white rounded-lg border border-gray-200 shadow-sm">
        <h2 className="text-lg font-semibold text-gray-900 mb-4">
          System Metrics
        </h2>

        <div className="space-y-4">
          {/* Health & Platform Row */}
          <div className="grid grid-cols-2 gap-4">
            {/* System Health */}
            <MetricCard>
              <MetricHeader
                title="System Health"
                badge={
                  <StatusBadge
                    isLoading={isSystemHealthLoading}
                    error={systemHealthError}
                    isHealthy={systemHealth?.status === 'healthy'}
                  />
                }
              />
              {systemHealth && (
                <div className="text-xs text-gray-600">
                  <div>Status: {systemHealth.status}</div>
                  <div>
                    DB:{' '}
                    {systemHealth.db_connected ? 'Connected' : 'Disconnected'}
                  </div>
                </div>
              )}
              {systemHealthError && (
                <div className="text-xs text-red-600">
                  Error: {systemHealthError.message}
                </div>
              )}
            </MetricCard>

            {/* Platform */}
            <MetricCard>
              <MetricHeader title="Platform" />
              {platform && (
                <div className="text-2xl font-bold text-gray-900">
                  {platform}
                </div>
              )}
              {systemMetricsError && (
                <div className="text-xs text-red-600">
                  Error: {systemMetricsError.message}
                </div>
              )}
            </MetricCard>
          </div>

          {/* CPU Section */}
          <MetricCard>
            <MetricHeader title="CPU Usage" />

            {cpu && (
              <div className="mb-3">
                <div className="flex items-center justify-between mb-1">
                  <span className="text-xs text-gray-600">Average</span>
                  <span className="text-lg font-bold text-gray-900">
                    {cpu.percent.toFixed(1)}%
                  </span>
                </div>
                <ProgressBar percent={cpu.percent} color="blue" />
              </div>
            )}

            {cpuPerCore && (
              <div>
                <div className="text-xs text-gray-600 mb-2">Per Core Usage</div>
                <div
                  className="grid gap-2"
                  style={{
                    gridTemplateColumns: `repeat(${cpuPerCore.percent.length === 14 ? 7 : Math.min(cpuPerCore.percent.length, 8)}, minmax(0, 1fr))`,
                  }}
                >
                  {cpuPerCore.percent.map(
                    (corePercent: number, index: number) => (
                      <div key={index} className="text-center">
                        <div className="text-xs text-gray-500 mb-1">
                          C{index}
                        </div>
                        <div className="text-xs font-semibold text-gray-900">
                          {corePercent.toFixed(1)}%
                        </div>
                        <div className="mt-1">
                          <ProgressBar
                            percent={corePercent}
                            color="blue"
                            height="sm"
                          />
                        </div>
                      </div>
                    )
                  )}
                </div>
              </div>
            )}

            {systemMetricsError && (
              <div className="text-xs text-red-600">
                Error: {systemMetricsError.message}
              </div>
            )}
          </MetricCard>

          {/* RAM & Disk Row */}
          <div className="grid grid-cols-2 gap-4">
            {/* RAM */}
            <MetricCard>
              <MetricHeader title="Memory" />
              {ram && (
                <div>
                  <div className="flex items-center justify-between mb-1">
                    <span className="text-xs text-gray-600">Usage</span>
                    <span className="text-lg font-bold text-gray-900">
                      {ram.percent.toFixed(1)}%
                    </span>
                  </div>
                  <div className="mb-2">
                    <ProgressBar percent={ram.percent} color="purple" />
                  </div>
                  <div className="text-xs text-gray-600">
                    {ram.used_gb.toFixed(2)} GB / {ram.total_gb.toFixed(2)} GB
                  </div>
                </div>
              )}
              {systemMetricsError && (
                <div className="text-xs text-red-600">
                  Error: {systemMetricsError.message}
                </div>
              )}
            </MetricCard>

            {/* Disk */}
            <MetricCard>
              <MetricHeader title="Disk" />
              {disk && (
                <div>
                  <div className="flex items-center justify-between mb-1">
                    <span className="text-xs text-gray-600">Usage</span>
                    <span className="text-lg font-bold text-gray-900">
                      {disk.percent.toFixed(1)}%
                    </span>
                  </div>
                  <div className="mb-2">
                    <ProgressBar percent={disk.percent} color="green" />
                  </div>
                  <div className="text-xs text-gray-600">
                    {disk.used_gb.toFixed(2)} GB / {disk.total_gb.toFixed(2)} GB
                  </div>
                </div>
              )}
              {systemMetricsError && (
                <div className="text-xs text-red-600">
                  Error: {systemMetricsError.message}
                </div>
              )}
            </MetricCard>
          </div>
        </div>
      </div>
    </div>
  );
}
