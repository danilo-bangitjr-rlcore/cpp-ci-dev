import type { NodeInfo } from './types';
import { NodeBadge } from './NodeBadge';
import { ChevronIcon } from './ChevronIcon';
import { parseNodeId } from './utils';

interface NodeItemProps {
  node: NodeInfo;
  level: number;
  isExpanded: boolean;
  isSelected: boolean;
  childrenQuery?: { data?: NodeInfo[]; isLoading?: boolean };
  onNodeClick: (nodeId: string) => void;
  renderChildren: (children: NodeInfo[], level: number) => React.ReactNode;
}

export const NodeItem = ({
  node,
  level,
  isExpanded,
  isSelected,
  childrenQuery,
  onNodeClick,
  renderChildren,
}: NodeItemProps) => {
  const indentClass = level === 0 ? '' : level === 1 ? 'pl-8' : 'pl-12';

  // Determine if node has children and expansion state
  const hasChildren = childrenQuery?.data && childrenQuery.data.length > 0;
  const isLoadingChildren = childrenQuery?.isLoading;
  const canExpand = hasChildren || isLoadingChildren || !childrenQuery?.data; // Assume expandable if not yet checked

  return (
    <div>
      <div
        className={`flex items-center px-1.5 py-1 ${indentClass} ${
          isSelected
            ? 'bg-blue-50 border-l-2 border-blue-500'
            : 'bg-white hover:bg-gray-50'
        } ${canExpand ? 'cursor-pointer' : 'cursor-default'} ${
          !canExpand ? 'opacity-75' : ''
        }`}
        onClick={() => onNodeClick(node.node_id)}
      >
        <ChevronIcon
          isExpanded={isExpanded}
          canExpand={canExpand}
          isLoading={isLoadingChildren || false}
        />
        <NodeBadge nodeClass={node.node_class} />
        <span className="font-medium text-gray-900 text-xs flex-1">
          {node.display_name}
        </span>
        <span className="ml-auto text-xs text-gray-500">
          {parseNodeId(node.node_id)}
        </span>
      </div>

      {isExpanded &&
        childrenQuery?.data &&
        renderChildren(childrenQuery.data, level + 1)}
    </div>
  );
};
