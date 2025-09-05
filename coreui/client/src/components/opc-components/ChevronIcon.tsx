interface ChevronIconProps {
  isExpanded: boolean;
  canExpand: boolean;
  isLoading: boolean;
}

export const ChevronIcon = ({
  isExpanded,
  canExpand,
  isLoading,
}: ChevronIconProps) => {
  if (isLoading) {
    return (
      <svg
        className="w-3 h-3 mr-1 text-gray-400 animate-spin"
        fill="none"
        viewBox="0 0 24 24"
      >
        <circle
          className="opacity-25"
          cx="12"
          cy="12"
          r="10"
          stroke="currentColor"
          strokeWidth="4"
        ></circle>
        <path
          className="opacity-75"
          fill="currentColor"
          d="M4 12a8 8 0 018-8V0C5.373 0 0 5.373 0 12h4zm2 5.291A7.962 7.962 0 014 12H0c0 3.042 1.135 5.824 3 7.938l3-2.647z"
        ></path>
      </svg>
    );
  }

  if (isExpanded) {
    return (
      <svg
        className="w-3 h-3 mr-1 text-gray-600"
        fill="none"
        stroke="currentColor"
        viewBox="0 0 24 24"
      >
        <path
          strokeLinecap="round"
          strokeLinejoin="round"
          strokeWidth={2}
          d="M19 9l-7 7-7-7"
        />
      </svg>
    );
  }

  if (canExpand) {
    return (
      <svg
        className="w-3 h-3 mr-1 text-gray-400"
        fill="none"
        stroke="currentColor"
        viewBox="0 0 24 24"
      >
        <path
          strokeLinecap="round"
          strokeLinejoin="round"
          strokeWidth={2}
          d="M9 5l7 7-7 7"
        />
      </svg>
    );
  }

  return <span className="w-3 mr-1"></span>; // Empty space for leaf nodes
};
