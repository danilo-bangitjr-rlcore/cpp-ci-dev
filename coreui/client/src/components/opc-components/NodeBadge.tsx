import { getNodeBadge } from './utils';

interface NodeBadgeProps {
  nodeClass: string;
}

export const NodeBadge = ({ nodeClass }: NodeBadgeProps) => {
  const badge = getNodeBadge(nodeClass);

  return (
    <span
      className={`mr-1.5 text-xs font-mono ${badge.color} px-1.5 py-0.5 rounded`}
    >
      {badge.text}
    </span>
  );
};
