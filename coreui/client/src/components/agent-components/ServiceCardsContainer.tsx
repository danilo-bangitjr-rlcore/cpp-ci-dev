import DetailsCard from '../DetailsCard';
import type { ServiceStatus } from '../../types/agent-types';

type ServiceCardsContainerProps = {
  agentData: {
    agentId: string;
    configPath: string;
    corerl: ServiceStatus;
    coreio: ServiceStatus;
    isNeverStarted: boolean;
  };
  resolvedConfigPath?: string;
  selectedExistingIO: string;
  onToggleAgent: () => Promise<void>;
  onToggleIO: () => Promise<void>;
  isTogglingAgent: boolean;
  isTogglingIO: boolean;
  isLoadingConfigPath: boolean;
};

function getServiceState(
  service: ServiceStatus,
  isNeverStarted: boolean
): 'running' | 'stopped' | 'never-started' {
  if (isNeverStarted) return 'never-started';
  return service.state === 'running' ? 'running' : 'stopped';
}

function getServiceMetadata(
  service: ServiceStatus,
  resolvedConfigPath?: string
) {
  return [
    { label: 'Service ID', value: service.id || 'N/A' },
    { label: 'Intended State', value: service.intended_state || 'N/A' },
    {
      label: 'Config Path',
      value:
        resolvedConfigPath && resolvedConfigPath !== ''
          ? '.../' + resolvedConfigPath.split('/').slice(-2).join('/')
          : service.config_path
            ? '.../' + service.config_path.split('/').slice(-2).join('/')
            : 'N/A',
    },
  ];
}

export default function ServiceCardsContainer({
  agentData,
  resolvedConfigPath,
  selectedExistingIO,
  onToggleAgent,
  onToggleIO,
  isTogglingAgent,
  isTogglingIO,
  isLoadingConfigPath,
}: ServiceCardsContainerProps) {
  const agentName = agentData.agentId;
  const ioName = selectedExistingIO || agentData.coreio.id || 'I/O Service';
  const isStartingWithExistingIO =
    agentData.isNeverStarted && selectedExistingIO;

  return (
    <div className="flex flex-row gap-6 flex-wrap justify-center p-10">
      <DetailsCard
        entityName={agentName}
        state={getServiceState(agentData.corerl, agentData.isNeverStarted)}
        onToggleStatus={onToggleAgent}
        isLoading={
          isTogglingAgent || (agentData.isNeverStarted && isLoadingConfigPath)
        }
        metadata={getServiceMetadata(agentData.corerl, resolvedConfigPath)}
        metadataTitle="Agent Metadata"
        isFirstStart={agentData.isNeverStarted}
      />
      {!agentData.isNeverStarted && (
        <DetailsCard
          entityName={ioName}
          state={getServiceState(agentData.coreio, agentData.isNeverStarted)}
          onToggleStatus={onToggleIO}
          isLoading={
            isTogglingIO || (agentData.isNeverStarted && isLoadingConfigPath)
          }
          metadata={getServiceMetadata(agentData.coreio, resolvedConfigPath)}
          metadataTitle="I/O Metadata"
          isFirstStart={agentData.isNeverStarted && !selectedExistingIO}
          isUsingExisting={!!isStartingWithExistingIO}
        />
      )}
    </div>
  );
}
