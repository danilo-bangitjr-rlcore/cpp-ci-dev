import {
  createFileRoute,
  Link,
  useCanGoBack,
  useRouter,
} from "@tanstack/react-router";
import { useContext } from "react";
import { MainConfigContext } from "../../utils/main-config";

export const Route = createFileRoute("/setup/name")({
  component: Name,
});

function Name() {
  const { mainConfig, setMainConfig } = useContext(MainConfigContext);
  const router = useRouter();
  const canGoBack = useCanGoBack();

  const handleExpNameChange = (e: React.ChangeEvent<HTMLInputElement>) => {
    const exp_name = e.target.value;

    setMainConfig({
      ...mainConfig,
      experiment: { ...mainConfig.experiment, exp_name },
    });
  };

  const handleSubmit = (e: React.FormEvent) => {
    e.preventDefault();
  };

  return (
    <div className="p-2">
      <form
        className="border border-gray-400 rounded-lg p-2 mb-2"
        onSubmit={handleSubmit}
      >
        <div className="sm:col-span-4">
          <label
            htmlFor="exp_name"
            className="block text-sm/6 font-medium text-gray-900"
          >
            Experiment Name
          </label>
          <div className="mt-2">
            <div className="flex items-center rounded-md bg-white pl-3 outline-1 -outline-offset-1 outline-gray-300 focus-within:outline-2 focus-within:-outline-offset-2 focus-within:outline-indigo-600">
              <input
                id="exp_name"
                name="exp_name"
                type="text"
                placeholder=""
                onChange={handleExpNameChange}
                value={mainConfig.experiment?.exp_name ?? ""}
                className="block min-w-0 grow py-1.5 pr-3 pl-1 text-base text-gray-900 placeholder:text-gray-400 focus:outline-none sm:text-sm/6"
              />
            </div>
          </div>
        </div>
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
          to="/setup/stub_required"
          className="cursor-pointer relative -ml-px inline-flex items-center rounded-r-md bg-white px-3 py-2 text-sm font-semibold text-gray-900 ring-1 ring-gray-300 ring-inset hover:bg-gray-50 focus:z-10"
        >
          Go to /setup/stub_required
        </Link>
      </span>
    </div>
  );
}
