import { Link, Outlet } from "@tanstack/react-router";
import { Navbar } from "./navbar";
import {
  Sidebar,
  SidebarBody,
  SidebarHeader,
  SidebarItem,
  SidebarSection,
} from "./sidebar";
import { SidebarLayout } from "./sidebar-layout";
import rlcoreLogo from "/RLCore_Stacked.svg";

export const App = () => (
  <SidebarLayout
    sidebar={
      <Sidebar>
        <SidebarHeader>
          <Link to="/">
            <img className="h-24 w-24" src={rlcoreLogo} alt="RLCore logo" />
          </Link>
        </SidebarHeader>
        <SidebarBody>
          <SidebarSection>
            <SidebarItem to="/">Home</SidebarItem>
            <SidebarItem to="/setup" activeOptions={{ exact: false }}>
              Setup
            </SidebarItem>
            <SidebarItem to="/about">About</SidebarItem>
          </SidebarSection>
        </SidebarBody>
      </Sidebar>
    }
    navbar={<Navbar />}
  >
    <Outlet />
  </SidebarLayout>
);

export default App;
