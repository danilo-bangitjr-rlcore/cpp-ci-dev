import { createRootRoute, Outlet, useMatch } from '@tanstack/react-router';
import { TanStackRouterDevtools } from '@tanstack/react-router-devtools';
import { GlobalHeader } from '../components/navigation/GlobalHeader';
import { LeftNav } from '../components/navigation/LeftNav';
import { HomeIcon } from '../components/icons/HomeIcon';
import { useAgentNameQuery } from '../utils/useAgentQueries';

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
];

const headerItems = [
  { label: 'Settings', onClick: () => console.log('Settings clicked') },
];

function useAgentContextNav() {
  const match = useMatch({ from: '/agents/$config-name', shouldThrow: false });
  const configName = match?.params?.['config-name'];
  const { data: agentName } = useAgentNameQuery(configName);
  if (!configName) return [];
  return [
    {
      label: agentName || configName,
      to: '/agents/$config-name',
      params: { 'config-name': configName },
      children: [
        {
          label: 'General Settings',
          to: '/agents/$config-name/general-settings',
          params: { 'config-name': configName },
        },
        {
          label: 'Observation Tags',
          to: '/agents/$config-name/tags',
          params: { 'config-name': configName },
        },
        {
          label: 'Reward Configuration',
          to: '/agents/$config-name/reward',
          params: { 'config-name': configName },
        },
        {
          label: 'Monitor',
          to: '/agents/$config-name/monitor',
          params: { 'config-name': configName },
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
