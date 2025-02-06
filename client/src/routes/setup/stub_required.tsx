import {
  createFileRoute,
  Link,
  useCanGoBack,
  useRouter,
} from "@tanstack/react-router";
import { useContext } from "react";
import { MainConfigContext } from "../../utils/main-config";
import DurationInput from "../../components/duration";
import { Fieldset, Legend } from "../../components/fieldset";
import { Text } from "../../components/text";
import { Badge } from "../../components/badge";

export const Route = createFileRoute("/setup/stub_required")({
  component: StubRequired,
});

function StubRequired() {
  const { mainConfig, setMainConfig } = useContext(MainConfigContext);
  const router = useRouter();
  const canGoBack = useCanGoBack();

  const handleSubmit = (e: React.FormEvent) => {
    e.preventDefault();
  };

  const handleDurationChange = (isoDuration: string) => {
    setMainConfig({...mainConfig, env: { ...mainConfig.env, obs_period: isoDuration}})
  };

  return (
    <div className="p-2">
      <form
        className="border border-gray-400 rounded-lg p-2 mb-2"
        onSubmit={handleSubmit}
      >
        <Fieldset>
          <Legend>Environment Observation Period</Legend>
          <Text>How much time should pass in-between sensor readings?<Badge className="ml-1">{mainConfig.env?.obs_period}</Badge></Text>
          <DurationInput
            onChange={handleDurationChange}
            defaultValue={mainConfig.env?.obs_period ?? ""}
          />
        </Fieldset>
      </form>
      <span className="isolate inline-flex rounded-md shadow-xs">
        {canGoBack ? (
          <button
            type="button"
            onClick={() => router.history.back()}
            className="cursor-pointer relative inline-flex items-center rounded-l-md bg-white px-3 py-2 text-sm font-semibold text-gray-900 ring-1 ring-gray-300 ring-inset hover:bg-gray-50 focus:z-10"
          >
            Go Back
          </button>
        ) : null}
        <Link
          to="/setup/finish"
          className="cursor-pointer relative -ml-px inline-flex items-center rounded-r-md bg-white px-3 py-2 text-sm font-semibold text-gray-900 ring-1 ring-gray-300 ring-inset hover:bg-gray-50 focus:z-10"
        >
          Go to /setup/finish
        </Link>
      </span>
    </div>
  );
}
