export type AgentStatus =
  | 'on'
  | 'off'
  | 'error'
  | 'never-started'
  | 'config-missing';

export interface Agent {
  agentName: string;
  configName: string;
  status: AgentStatus;
}

export type ServiceStatus = {
  id: string;
  state: string;
  intended_state: string;
  config_path: string | null;
};

export type AgentStatusResponse = {
  state: string;
  config_path: string | null;
  service_statuses: {
    corerl?: ServiceStatus;
    coreio?: ServiceStatus;
  };
  id: string;
  version?: string;
  uptime?: string;
};
