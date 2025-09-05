import { useState } from 'react';

export const useOpcNavigation = () => {
  const [expandedNodes, setExpandedNodes] = useState<Set<string>>(new Set());
  const [selectedNodeId, setSelectedNodeId] = useState<string | null>(null);

  const toggleNode = (nodeId: string) => {
    setExpandedNodes((prev) => {
      const newSet = new Set(prev);
      if (newSet.has(nodeId)) {
        newSet.delete(nodeId);
      } else {
        newSet.add(nodeId);
      }
      return newSet;
    });
  };

  const handleNodeClick = (nodeId: string) => {
    // Only update if it's a different node
    if (selectedNodeId !== nodeId) {
      setSelectedNodeId(nodeId);
    }
    // Always toggle expansion
    toggleNode(nodeId);
  };

  return {
    expandedNodes,
    selectedNodeId,
    toggleNode,
    handleNodeClick,
    setSelectedNodeId,
  };
};
