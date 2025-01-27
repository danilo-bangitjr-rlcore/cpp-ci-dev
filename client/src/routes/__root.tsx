import { createRootRoute, Link, Outlet } from '@tanstack/react-router';
import { TanStackRouterDevtools } from '@tanstack/router-devtools';
import rlcoreLogo from '/RLCore_Stacked.svg';
import {
  QueryClient,
  QueryClientProvider,
} from '@tanstack/react-query'
import { ReactQueryDevtools } from '@tanstack/react-query-devtools'

const queryClient = new QueryClient()

export const Route = createRootRoute({
  component: () => (
    <QueryClientProvider client={queryClient}>
      <div
        className="
        min-h-screen min-w-screen
        flex flex-col
        "
      >
        <header className="row-span-1 col-span-2 h-24 bg-blue-100 flex items-center">
          <Link to="/">
            <img className="h-24 w-24" src={rlcoreLogo} alt="RLCore logo" />
          </Link>
          <span>GUI</span>
        </header>
        <div className="grow flex flex-col-reverse md:flex-row">
          <aside className="col-span-2 md:col-span-1 row-span-1 md:flex md:flex-col overflow-y-auto bg-gray-100">
            <nav className="flex md:flex-col md:h-full space-x-4 md:space-x-0">
              <Link
                to="/"
                className='block px-4 py-2 hover:bg-gray-200'
                activeOptions={{ exact: true }}
                activeProps={{ className: 'bg-gray-200' }}
              >
                Home
              </Link>
              <Link
                to="/about"
                className='block px-4 py-2 hover:bg-gray-200'
                activeProps={{ className: 'bg-gray-200' }}
              >
                About
              </Link>
            </nav>
          </aside>
          <main className="grow"><Outlet /></main>
        </div>
        <footer className="h-24 bg-blue-100 text-center flex items-center justify-center">
          <span><a href="https://rlcore.ai" target="_blank" rel="noreferrer">© 2025 RL Core Technologies.</a> All rights reserved.</span>
        </footer>
      </div>
      {!import.meta.env.PROD && <>
        <TanStackRouterDevtools />
        <ReactQueryDevtools />
      </>}
    </QueryClientProvider>
  ),
})
