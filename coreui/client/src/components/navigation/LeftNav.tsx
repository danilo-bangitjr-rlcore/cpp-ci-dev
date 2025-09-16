import { Link } from '@tanstack/react-router';

type NavItem = {
  label: string;
  to?: string;
  params?: Record<string, string | number>;
  onClick?: () => void;
  icon?: React.ReactNode;
  children?: NavItem[]; // for grouped nested links like agent subpages
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

          const renderItem = (navItem: NavItem, { isChild = false } = {}) => {
            const content = (
              <>
                {navItem.icon && (
                  <span
                    className="mr-2 flex items-center text-gray-500"
                    aria-hidden="true"
                  >
                    {navItem.icon}
                  </span>
                )}
                <span className={isChild ? 'text-sm text-gray-700' : undefined}>
                  {navItem.label}
                </span>
              </>
            );
            const className = `${baseClasses} ${isChild ? 'py-1.5' : ''}`;
            if (navItem.to) {
              return (
                <Link
                  key={navItem.label}
                  to={navItem.to}
                  params={navItem.params}
                  className={`${className} [&.active]:bg-gray-200 [&.active]:text-black`}
                >
                  {content}
                </Link>
              );
            }
            return (
              <button
                key={navItem.label}
                onClick={navItem.onClick}
                className={className}
              >
                {content}
              </button>
            );
          };

          // If the item has children, render group label and its children indented
          if (item.children && item.children.length) {
            return (
              <div key={item.label} className="space-y-0.5">
                {renderItem(item)}
                <div className="ml-4 pl-2 border-l-2 border-dashed border-gray-300/70 space-y-0.5">
                  {item.children.map((child) =>
                    renderItem(child, { isChild: true })
                  )}
                </div>
              </div>
            );
          }

          return renderItem(item);
        })}
      </div>
    </nav>
  );
}
