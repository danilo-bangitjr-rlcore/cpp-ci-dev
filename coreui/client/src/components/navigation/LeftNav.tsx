import { Link } from '@tanstack/react-router';

type NavItem = {
  label: string;
  to?: string;
  onClick?: () => void;
  icon?: React.ReactNode;
};

type LeftNavProps = {
  items: NavItem[];
};

export function LeftNav({ items }: LeftNavProps) {
  return (
  <nav className="w-64 bg-gray-100 h-full p-4 flex-shrink-0 border-r border-gray-200 flex flex-col overflow-y-auto">
      <div className="space-y-2">
        {items.map((item) => {
          const baseClasses =
            'flex items-center w-full px-3 py-2 text-gray-900 rounded-md hover:bg-gray-200 hover:text-black transition-colors';
          const content = (
            <>
              {item.icon && <span className="mr-2">{item.icon}</span>}
              {item.label}
            </>
          );
          if (item.to) {
            return (
              <Link
                key={item.label}
                to={item.to}
                className={`${baseClasses} [&.active]:bg-gray-200 [&.active]:text-black`}
              >
                {content}
              </Link>
            );
          }
          return (
            <button
              key={item.label}
              onClick={item.onClick}
              className={baseClasses}
            >
              {content}
            </button>
          );
        })}
      </div>
    </nav>
  );
}
