import { useState } from 'react';

export const useOpcNavigation = () => {
  const [expandedNodes, setExpandedNodes] = useState<Set<string>>(new Set());
  const [selectedNodeId, setSelectedNodeId] = useState<string | undefined>(
    undefined
  );

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
    // If the same node is clicked, deselect it (set to undefined)
    if (selectedNodeId === nodeId) {
      setSelectedNodeId(undefined);
    } else {
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
