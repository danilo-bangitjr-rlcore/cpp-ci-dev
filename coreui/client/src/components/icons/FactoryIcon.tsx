import * as React from 'react';

export interface FactoryIconProps extends React.SVGProps<SVGSVGElement> {
  size?: number | string;
}

export const FactoryIcon: React.FC<FactoryIconProps> = ({
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
  <path d="M2.4 19.2 L2.4 4.8 L6 4.8 L6 12 L10.8 8.4 L10.8 12 L15.6 8.4 L15.6 19.2 L2.4 19.2 Z" />
</svg>
);
