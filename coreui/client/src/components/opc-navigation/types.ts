export interface NodeInfo {
  node_id: string;
  display_name: string;
  node_class: string;
  data_type?: string;
}

export interface NodeDetails {
  node_id: string;
  display_name: string;
  node_class: string;
  data_type?: string | null;
  value?: unknown;
  description?: string | null;
}

export interface NodeBadgeData {
  text: string;
  color: string;
}

export interface NodeBadge {
  text: string;
  color: string;
}
