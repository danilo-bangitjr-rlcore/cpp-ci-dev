import {
  useOpcNavigation,
  useOpcData,
  NodeTree,
  NodeDetailsPanel,
} from './index';

interface OpcNavigationProps {
  className?: string;
}

export function OpcNavigation({ className = '' }: OpcNavigationProps) {
  const { expandedNodes, selectedNodeId, handleNodeClick } = useOpcNavigation();
  const {
    rootNodes,
    isLoadingRoot,
    childrenMap,
    selectedNodeDetails,
    isLoadingDetails,
  } = useOpcData(expandedNodes, selectedNodeId);

  return (
    <div
      className={`mt-4 p-4 bg-white rounded-lg border border-gray-200 shadow-sm ${className}`}
    >
      <h3 className="text-lg font-semibold text-gray-900 mb-4">
        OPC Server Navigation
      </h3>

      <div className="flex gap-6">
        {/* Node Tree Browser */}
        <div className="flex-1">
          <h4 className="text-md font-semibold text-gray-900 mb-3">
            Node Browser
          </h4>
          {isLoadingRoot ? (
            <div className="p-4 text-center text-gray-500">Loading...</div>
          ) : rootNodes ? (
            <NodeTree
              nodes={rootNodes}
              expandedNodes={expandedNodes}
              selectedNodeId={selectedNodeId}
              childrenMap={childrenMap}
              onNodeClick={handleNodeClick}
            />
          ) : (
            <div className="p-4 text-center text-gray-500">No nodes found</div>
          )}
        </div>

        {/* Node Details Panel */}
        <NodeDetailsPanel
          selectedNodeDetails={selectedNodeDetails}
          selectedNodeId={selectedNodeId}
          isLoading={isLoadingDetails}
        />
      </div>
    </div>
  );
}
