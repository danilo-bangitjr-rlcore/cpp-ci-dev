import { useState } from 'react';
import { useQuery } from '@tanstack/react-query';
import { API_BASE_URL } from '../utils/api';

interface NodeInfo {
  node_id: string;
  display_name: string;
  node_class: string;
  data_type?: string;
}

interface OpcNavigationProps {
  className?: string;
}

export function OpcNavigation({ className = '' }: OpcNavigationProps) {
  const [expandedNodes, setExpandedNodes] = useState<Set<string>>(new Set());

  // Fetch root nodes
  const { data: rootNodes, isLoading: isLoadingRoot } = useQuery({
    queryKey: ['opc-browse-root'],
    queryFn: async (): Promise<NodeInfo[]> => {
      const response = await fetch(`${API_BASE_URL}/v1/opc/browse`);
      if (!response.ok) throw new Error('Failed to fetch root nodes');
      return response.json();
    },
    enabled: true,
  });

  // Fetch children of expanded nodes
  const getNodeChildren = (nodeId: string) => {
    return useQuery({
      queryKey: ['opc-browse-node', nodeId],
      queryFn: async (): Promise<NodeInfo[]> => {
        const response = await fetch(`${API_BASE_URL}/v1/opc/browse/${nodeId}`);
        if (!response.ok) throw new Error(`Failed to fetch children of ${nodeId}`);
        return response.json();
      },
      enabled: expandedNodes.has(nodeId),
    });
  };

  const toggleNode = (nodeId: string) => {
    setExpandedNodes(prev => {
      const newSet = new Set(prev);
      if (newSet.has(nodeId)) {
        newSet.delete(nodeId);
      } else {
        newSet.add(nodeId);
      }
      return newSet;
    });
  };

  const parseNodeId = (nodeId: string): string => {
    // Handle raw NodeId format: NodeId(Identifier=85, NamespaceIndex=0, NodeIdType=<NodeIdType.Numeric: 2>)
    const match = nodeId.match(/NodeId\(Identifier=(\w+),\s*NamespaceIndex=(\d+),\s*NodeIdType=.+\)/);
    if (match) {
      const [, identifier, namespaceIndex] = match;
      return `ns=${namespaceIndex};i=${identifier}`;
    }

    // If it's already in the correct format, return as-is
    if (nodeId.startsWith('ns=') && nodeId.includes(';i=')) {
      return nodeId;
    }

    // Fallback for other formats
    return nodeId;
  };

  const getNodeBadge = (nodeClass: string) => {
    switch (nodeClass.toLowerCase()) {
      case 'object': return { text: 'OBJ', color: 'text-gray-500 bg-gray-200' };
      case 'variable': return { text: 'VAR', color: 'text-green-600 bg-green-100' };
      case 'method': return { text: 'MTH', color: 'text-blue-600 bg-blue-100' };
      case 'objecttype': return { text: 'OTY', color: 'text-purple-600 bg-purple-100' };
      case 'variabletype': return { text: 'VTY', color: 'text-orange-600 bg-orange-100' };
      case 'referencetype': return { text: 'RTY', color: 'text-red-600 bg-red-100' };
      case 'datatype': return { text: 'DTY', color: 'text-indigo-600 bg-indigo-100' };
      default: return { text: 'UNK', color: 'text-gray-600 bg-gray-200' };
    }
  };

  const renderNode = (node: NodeInfo, level: number = 0) => {
    const isExpanded = expandedNodes.has(node.node_id);
    const childrenQuery = getNodeChildren(node.node_id);
    const badge = getNodeBadge(node.node_class);
    const indentClass = level === 0 ? '' : level === 1 ? 'pl-8' : 'pl-12';

    // Determine if node has children and expansion state
    const hasChildren = childrenQuery.data && childrenQuery.data.length > 0;
    const isLoadingChildren = childrenQuery.isLoading;
    const canExpand = hasChildren || isLoadingChildren || !childrenQuery.data; // Assume expandable if not yet checked

    const getChevronIcon = () => {
      if (isLoadingChildren) {
        return (
          <svg className="w-3 h-3 mr-1 text-gray-400 animate-spin" fill="none" viewBox="0 0 24 24">
            <circle className="opacity-25" cx="12" cy="12" r="10" stroke="currentColor" strokeWidth="4"></circle>
            <path className="opacity-75" fill="currentColor" d="M4 12a8 8 0 018-8V0C5.373 0 0 5.373 0 12h4zm2 5.291A7.962 7.962 0 014 12H0c0 3.042 1.135 5.824 3 7.938l3-2.647z"></path>
          </svg>
        );
      }

      if (isExpanded) {
        return (
          <svg className="w-3 h-3 mr-1 text-gray-600" fill="none" stroke="currentColor" viewBox="0 0 24 24">
            <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M19 9l-7 7-7-7" />
          </svg>
        );
      }

      if (canExpand) {
        return (
          <svg className="w-3 h-3 mr-1 text-gray-400" fill="none" stroke="currentColor" viewBox="0 0 24 24">
            <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M9 5l7 7-7 7" />
          </svg>
        );
      }

      return <span className="w-3 mr-1"></span>; // Empty space for leaf nodes
    };

    return (
      <div key={node.node_id}>
        <div
          className={`flex items-center px-1.5 py-1 ${indentClass} bg-white hover:bg-gray-50 ${canExpand ? 'cursor-pointer' : 'cursor-default'} ${!canExpand ? 'opacity-75' : ''}`}
          onClick={() => canExpand && toggleNode(node.node_id)}
        >
          {getChevronIcon()}
          <span className={`mr-1.5 text-xs font-mono ${badge.color} px-1.5 py-0.5 rounded`}>
            {badge.text}
          </span>
          <span className="font-medium text-gray-900 text-xs flex-1">{node.display_name}</span>
          <span className="ml-auto text-xs text-gray-500">{parseNodeId(node.node_id)}</span>
        </div>

        {isExpanded && childrenQuery.data && (
          <div>
            {childrenQuery.data.map((child: NodeInfo) => renderNode(child, level + 1))}
          </div>
        )}
      </div>
    );
  };

  return (
    <div className={`mt-4 p-4 bg-white rounded-lg border border-gray-200 shadow-sm ${className}`}>
      <h3 className="text-lg font-semibold text-gray-900 mb-4">
        OPC Server Navigation
      </h3>

      <div className="flex gap-6">
        {/* Node Tree Browser */}
        <div className="flex-1">
          <h4 className="text-md font-semibold text-gray-900 mb-3">Node Browser</h4>
          <div className="border border-gray-200 rounded-md overflow-hidden max-h-96 overflow-y-auto">
            {isLoadingRoot ? (
              <div className="p-4 text-center text-gray-500">Loading...</div>
            ) : rootNodes ? (
              <div className="divide-y divide-gray-100">
                {rootNodes.map((node: NodeInfo) => renderNode(node))}
              </div>
            ) : (
              <div className="p-4 text-center text-gray-500">No nodes found</div>
            )}
          </div>
        </div>

        {/* Node Details Panel */}
        <div className="flex-1">
          <h4 className="text-md font-semibold text-gray-900 mb-3">Node Details</h4>
          <div className="p-1.5 bg-gray-50 rounded-md border border-gray-200">
            <div className="space-y-0">
              <div className="flex justify-between px-1.5 py-1 bg-white rounded-sm">
                <strong className="text-gray-700 text-xs">Node ID:</strong>
                <span className="text-gray-900 text-xs">Select a node</span>
              </div>
              <div className="flex justify-between px-1.5 py-1 bg-gray-50 rounded-sm">
                <strong className="text-gray-700 text-xs">Display Name:</strong>
                <span className="text-gray-900 text-xs">-</span>
              </div>
              <div className="flex justify-between px-1.5 py-1 bg-white rounded-sm">
                <strong className="text-gray-700 text-xs">Node Class:</strong>
                <span className="text-gray-900 text-xs">-</span>
              </div>
              <div className="flex justify-between px-1.5 py-1 bg-gray-50 rounded-sm">
                <strong className="text-gray-700 text-xs">Data Type:</strong>
                <span className="text-gray-900 text-xs">-</span>
              </div>
            </div>
          </div>
        </div>
      </div>
    </div>
  );
}
