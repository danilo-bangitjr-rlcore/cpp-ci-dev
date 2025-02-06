import { createFileRoute, Outlet, useLocation } from "@tanstack/react-router";
import { type components } from "../api-schema";
import { useLocalForage } from "../utils/local-forage";
import { type DeepPartial, loadMainConfigHiddenDefaults, MainConfigContext } from "../utils/main-config";
import { ProgressBar, Step } from "../components/progress-bar";
import { useMemo } from "react";
import { Heading } from "../components/heading"
import { Text } from "../components/text"

export const Route = createFileRoute("/setup")({
  component: RouteComponent
});

function RouteComponent() {
  const [mainConfig, setMainConfig] = useLocalForage<
    DeepPartial<components["schemas"]["MainConfig"]>
  >("main_config", loadMainConfigHiddenDefaults({}));

  const { pathname: currentPathName } = useLocation()

  const setupSteps: Step[] = useMemo(() => {
    const steps: Step[] = [
      { name: "Start", to: "/setup", status: "complete" },
      { name: "Experiment Name", to: "/setup/name", status: "current" },
      { name: "Stub Required", to: "/setup/stub_required", status: "upcoming" },
      { name: "Finish", to: "/setup/finish", status: "upcoming" },
    ];

    const currentStepIdx = steps.findIndex((step) => step.to === currentPathName)
    for (let i = 0; i < steps.length; i++) {
      if (i > currentStepIdx) {
        steps[i].status = "upcoming";
      } else if (i == currentStepIdx) {
        steps[i].status = "current";
      } else {
        steps[i].status = "complete"
      }
    }
    return steps;
  }, [currentPathName]);

  return (
    <MainConfigContext.Provider value={{ mainConfig, setMainConfig }}>
      <div className="border-b border-gray-200 pb-2 p-2">
        <div className="-mt-2 -ml-2 flex flex-wrap items-baseline">
          <Heading level={2}>
            Setup
          </Heading>
          <Text className="mt-1 ml-2 truncate text-sm text-gray-500">
            {mainConfig.experiment?.exp_name}
          </Text>
          <ProgressBar steps={setupSteps} className="w-full"/>
        </div>
      </div>
      <Outlet />
    </MainConfigContext.Provider>
  );
}
