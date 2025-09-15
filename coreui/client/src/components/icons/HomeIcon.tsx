import * as React from 'react';

export interface HomeIconProps extends React.SVGProps<SVGSVGElement> {
  size?: number | string;
}

export const HomeIcon: React.FC<HomeIconProps> = ({
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
    <path d="M3 11.5 12 4l9 7.5" />
    <path d="M5 10v10h14V10" />
    <path d="M9.5 20v-5.5h5V20" />
  </svg>
);

export default HomeIcon;
