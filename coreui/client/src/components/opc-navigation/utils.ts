import type { NodeBadgeData } from './types';

export const parseNodeId = (nodeId: string): string => {
  // Handle raw NodeId format: NodeId(Identifier=85, NamespaceIndex=0, NodeIdType=<NodeIdType.Numeric: 2>)
  const match = nodeId.match(
    /NodeId\(Identifier=(\w+),\s*NamespaceIndex=(\d+),\s*NodeIdType=.+: (\d+)\)/
  );
  if (match) {
    const [, identifier, namespaceIndex, type] = match;
    let prefix = 'i'; // default to numeric
    switch (type) {
      case '2':
        prefix = 'i'; // Numeric
        break;
      case '3':
        prefix = 's'; // String
        break;
      case '4':
        prefix = 'g'; // GUID
        break;
      case '5':
        prefix = 'b'; // Opaque (ByteString)
        break;
      default:
        prefix = 'i'; // fallback
    }
    return `ns=${namespaceIndex};${prefix}=${identifier}`;
  }

  // If it's already in the correct format, return as-is
  if (
    nodeId.startsWith('ns=') &&
    (nodeId.includes(';i=') ||
      nodeId.includes(';s=') ||
      nodeId.includes(';g=') ||
      nodeId.includes(';b='))
  ) {
    return nodeId;
  }

  // Fallback for other formats
  return nodeId;
};

export const getNodeBadge = (nodeClass: string): NodeBadgeData => {
  switch (nodeClass) {
    case 'Unspecified':
      return { text: 'UNS', color: 'text-gray-600 bg-gray-200' };
    case 'Object':
      return { text: 'OBJ', color: 'text-gray-500 bg-gray-200' };
    case 'Variable':
      return { text: 'VAR', color: 'text-green-600 bg-green-100' };
    case 'Method':
      return { text: 'MTH', color: 'text-blue-600 bg-blue-100' };
    case 'ObjectType':
      return { text: 'OTY', color: 'text-purple-600 bg-purple-100' };
    case 'VariableType':
      return { text: 'VTY', color: 'text-orange-600 bg-orange-100' };
    case 'ReferenceType':
      return { text: 'RTY', color: 'text-red-600 bg-red-100' };
    case 'DataType':
      return { text: 'DTY', color: 'text-indigo-600 bg-indigo-100' };
    case 'View':
      return { text: 'VIEW', color: 'text-cyan-600 bg-cyan-100' };
    default:
      return { text: 'UNK', color: 'text-gray-600 bg-gray-200' };
  }
};
