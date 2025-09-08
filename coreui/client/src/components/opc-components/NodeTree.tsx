import type { NodeInfo } from './types';
import { NodeItem } from './NodeItem';

interface NodeTreeProps {
  nodes: NodeInfo[];
  expandedNodes: Set<string>;
  selectedNodeId?: string;
  childrenMap: Map<
    string,
    { data?: NodeInfo[]; isLoading?: boolean } | undefined
  >;
  onNodeClick: (nodeId: string) => void;
}

export const NodeTree = ({
  nodes,
  expandedNodes,
  selectedNodeId,
  childrenMap,
  onNodeClick,
}: NodeTreeProps) => {
  const renderNode = (node: NodeInfo, level: number = 0): React.ReactNode => {
    const isExpanded = expandedNodes.has(node.node_id);
    const childrenQuery = childrenMap.get(node.node_id);

    return (
      <NodeItem
        key={node.node_id}
        node={node}
        level={level}
        isExpanded={isExpanded}
        isSelected={selectedNodeId === node.node_id}
        childrenQuery={childrenQuery}
        onNodeClick={onNodeClick}
        renderChildren={(children, childLevel) =>
          children.map((child) => renderNode(child, childLevel))
        }
      />
    );
  };

  return (
    <div className="border border-gray-200 rounded-md overflow-hidden max-h-96 overflow-y-auto">
      <div className="divide-y divide-gray-100">
        {nodes.map((node) => renderNode(node))}
      </div>
    </div>
  );
};
