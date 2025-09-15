import { createRootRoute, Outlet, useMatchRoute } from '@tanstack/react-router';
import { TanStackRouterDevtools } from '@tanstack/react-router-devtools';
import { GlobalHeader } from '../components/navigation/GlobalHeader';
import { LeftNav } from '../components/navigation/LeftNav';
import { HomeIcon } from '../components/icons/HomeIcon';

const baseNavItems = [
  {
    label: 'Home',
    to: '/',
    icon: <HomeIcon size={32} className="mr-1" />,
  },
  {
    label: 'Agents Overview',
    to: '/agents',
    icon: (
      <img
        src="/app/assets/corei.svg"
        className="h-8 w-8 object-contain"
        alt="Agents"
      />
    ),
  },
  { label: 'OPC Navigation', to: '/opc-navigation' },
  { label: 'About', to: '/about' },
  { label: 'Tags', to: '/tags' },
];

const headerItems = [
  { label: 'Settings', onClick: () => console.log('Settings clicked') },
];

function useAgentContextNav() {
  const matchRoute = useMatchRoute();
  const dynamicMatch = matchRoute({
    to: '/agents/$agent-name',
    fuzzy: true,
    pending: false,
  }) as any;
  const agentName: string | undefined =
    dynamicMatch?.['agent-name'] || dynamicMatch?.params?.['agent-name'];
  if (!agentName) return [];
  return [
    {
      label: agentName,
      to: '/agents/$agent-name',
      params: { 'agent-name': agentName },
      children: [
        {
          label: 'General Settings',
          to: '/agents/$agent-name/general-settings',
          params: { 'agent-name': agentName },
        },
        {
          label: 'Monitor',
          to: '/agents/$agent-name/monitor',
          params: { 'agent-name': agentName },
        },
      ],
    },
  ];
}

export const Route = createRootRoute({
  component: RootLayout,
});

function RootLayout() {
  const agentItems = useAgentContextNav();
  const navItems = baseNavItems.flatMap((item) => {
    if (item.label === 'Agents Overview' && agentItems.length) {
      return [item, ...agentItems];
    }
    return [item];
  });
  return (
    <div className="flex flex-col h-screen bg-white">
      <GlobalHeader items={headerItems} />
      <div className="flex flex-1 min-h-0">
        <LeftNav items={navItems} />
        <main className="flex-1 p-4 overflow-auto flex flex-col min-h-0">
          <Outlet />
        </main>
      </div>
      <TanStackRouterDevtools />
    </div>
  );
}
