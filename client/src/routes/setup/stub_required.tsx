import { createFileRoute } from "@tanstack/react-router";
import { useContext } from "react";
import { Badge } from "../../components/badge";
import DurationInput from "../../components/duration";
import { Fieldset, Legend } from "../../components/fieldset";
import { SetupConfigNav } from "../../components/setup/setup-config-nav";
import { Text } from "../../components/text";
import { MainConfigContext } from "../../utils/main-config";

export const Route = createFileRoute("/setup/stub_required")({
  component: StubRequired,
});

function StubRequired() {
  const { mainConfig, setMainConfig } = useContext(MainConfigContext);

  const handleSubmit = (e: React.FormEvent) => {
    e.preventDefault();
  };

  const handleDurationChange = (isoDuration: string) => {
    setMainConfig({
      ...mainConfig,
      env: { ...mainConfig.env, obs_period: isoDuration },
    });
  };

  return (
    <div className="p-2">
      <form
        className="border border-gray-400 rounded-lg p-2 mb-2"
        onSubmit={handleSubmit}
      >
        <Fieldset>
          <Legend>Environment Observation Period</Legend>
          <Text>
            How much time should pass in-between sensor readings?
            <Badge className="ml-1">{mainConfig.env?.obs_period}</Badge>
          </Text>
          <DurationInput
            onChange={handleDurationChange}
            defaultValue={mainConfig.env?.obs_period ?? ""}
          />
        </Fieldset>
      </form>
      <SetupConfigNav />
    </div>
  );
}
