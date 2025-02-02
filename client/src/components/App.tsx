import { Link, Outlet } from "@tanstack/react-router";
import rlcoreLogo from "/RLCore_Stacked.svg";
import NavLink from "./NavLink";

export const App = () => (
  <div className="min-h-screen min-w-screen flex flex-col">
    <header className="row-span-1 col-span-2 h-24 bg-blue-100 flex items-center">
      <Link to="/">
        <img className="h-24 w-24" src={rlcoreLogo} alt="RLCore logo" />
      </Link>
      <span>GUI</span>
    </header>
    <div className="grow flex flex-col-reverse md:flex-row">
      <aside className="col-span-2 md:col-span-1 row-span-1 md:flex md:flex-col overflow-y-auto bg-gray-100 w-max">
        <nav className="flex md:flex-col md:h-full space-x-4 md:space-x-0">
          <NavLink to="/">Home</NavLink>
          <NavLink to="/setup" activeOptions={{ exact: false }}>
            Setup
          </NavLink>
          <NavLink to="/about">About</NavLink>
        </nav>
      </aside>
      <main className="grow">
        <Outlet />
      </main>
    </div>
    <footer className="h-24 bg-blue-100 text-center flex items-center justify-center">
      <a href="https://rlcore.ai" target="_blank" rel="noreferrer">
        © 2025 RL Core Technologies.
      </a>
      <span>All rights reserved.</span>
    </footer>
  </div>
);

export default App;
