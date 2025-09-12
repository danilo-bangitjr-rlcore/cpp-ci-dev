import { Link } from '@tanstack/react-router';

type HeaderItem = {
  label: string;
  to?: string;
  onClick?: () => void;
  icon?: React.ReactNode;
};

type GlobalHeaderProps = {
  items: HeaderItem[];
};

export function GlobalHeader({ items }: GlobalHeaderProps) {
  return (
    <header className="bg-gray-100 border-b border-gray-200 px-4 py-3 flex-shrink-0">
      <div className="flex items-center justify-between">
        <div className="flex items-center gap-2">
          <img src="/app/assets/rlcore-logo.png" className="h-14 w-auto" />
        </div>
        <nav className="flex items-center gap-4">
          {items.map((item) => {
            const baseClasses =
              'px-3 py-2 text-gray-900 rounded-md hover:bg-gray-200 hover:text-black transition-colors';
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
        </nav>
      </div>
    </header>
  );
}
