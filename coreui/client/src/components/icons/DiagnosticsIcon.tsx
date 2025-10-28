import * as React from 'react';

export interface DiagnosticsIconProps extends React.SVGProps<SVGSVGElement> {
  size?: number | string;
}

export const DiagnosticsIcon: React.FC<DiagnosticsIconProps> = ({
  size = 20,
  className = '',
  ...rest
}) => (
  <svg
    width={size}
    height={size}
    viewBox="0 0 24 24"
    fill="none"
    xmlns="http://www.w3.org/2000/svg"
    className={`text-gray-700 ${className}`}
    stroke="currentColor"
    strokeWidth={1.8}
    strokeLinecap="round"
    strokeLinejoin="round"
    {...rest}
  >
    <path d="M2 12 h4 l2 -7 l2 14 l2 -7 h4" />
  </svg>
);
