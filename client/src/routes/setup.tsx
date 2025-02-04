import { createFileRoute, Outlet } from "@tanstack/react-router";
import { type components } from "../api-schema";
import { createContext } from "react";
import { useLocalForage } from "../utils/localstorage";

export const Route = createFileRoute("/setup")({
  component: RouteComponent,
});

type DeepPartial<T> = T extends object ? {
  [P in keyof T]?: DeepPartial<T[P]>;
} : T;

export const MainConfigContext = createContext<{
  mainConfig: DeepPartial<components["schemas"]["MainConfig"]>;
  setMainConfig: (value: DeepPartial<components["schemas"]["MainConfig"]>) => void;
}>({ mainConfig: {}, setMainConfig: () => {} });

function RouteComponent() {
  const [mainConfig, setMainConfig] = useLocalForage<
    DeepPartial<components["schemas"]["MainConfig"]>
  >("main_config", {});
  return (
    <MainConfigContext.Provider value={{ mainConfig, setMainConfig }}>
      <div className="border-b border-gray-200 pb-2 p-2">
        <div className="-mt-2 -ml-2 flex flex-wrap items-baseline">
          <h3 className="mt-2 ml-2 text-base font-semibold text-gray-900">
            Setup
          </h3>
          <p className="mt-1 ml-2 truncate text-sm text-gray-500">
            {mainConfig.experiment?.exp_name}
          </p>
        </div>
      </div>
      <Outlet />
    </MainConfigContext.Provider>
  );
}
