import {
  createFileRoute,
  useRouter,
  useCanGoBack,
} from "@tanstack/react-router";
import { useContext, useState } from "react";
import { MainConfigContext } from "../setup";
import { fetchWithTimeout, getServerOrigin } from "../../utils/api";
import Alert from "../../components/Alert";

export const Route = createFileRoute("/setup/finish")({
  component: RouteComponent,
});

function RouteComponent() {
  const { mainConfig } = useContext(MainConfigContext);
  const [error, setError] = useState<string | undefined>(undefined);
  const router = useRouter();
  const canGoBack = useCanGoBack();

  const handleFormUpload = async (event: React.FormEvent<HTMLFormElement>) => {
    event.preventDefault();
    // NOTE: cannot use the OpenAPI client to make a request because of nested logic
    // that assumes content is always JSON, but we need to receive YAML
    const resp = await fetchWithTimeout(
      getServerOrigin() + "/api/configuration/file",
      {
        method: "POST",
        body: JSON.stringify(mainConfig),
        headers: {
          "Content-Type": "application/json",
          Accept: "application/yaml",
        },
      },
    );

    if (!resp.ok) {
      const error = await resp.json();
      setError(`Error generating configuration file: ${JSON.stringify(error)}`);
      console.error("Error generating configuration file");
      return;
    }

    // convert response yaml into a downloadable file
    const data = await resp.blob();
    const url = window.URL.createObjectURL(data);
    const a = document.createElement("a");
    a.href = url;
    a.download = `${mainConfig.experiment?.exp_name || "config"}.yaml`;
    document.body.appendChild(a);
    a.click();
    document.body.removeChild(a);
    window.URL.revokeObjectURL(url);
  };

  return (
    <div className="p-2">
      {!!error && Object.keys(error).length > 0 && (
        <Alert>
          <code>{JSON.stringify(error)}</code>
        </Alert>
      )}

      <form className="mt-2" onSubmit={handleFormUpload}>
        <label
          htmlFor="main_config_payload"
          className="block text-sm/6 font-medium text-gray-900"
        >
          Main Configuration Payload
        </label>
        <textarea
          id="main_config_payload"
          rows={4}
          className="block w-full rounded-md bg-white px-3 py-1.5 text-base text-gray-900 outline-1 -outline-offset-1 outline-gray-300 placeholder:text-gray-400 focus:outline-2 focus:-outline-offset-2 focus:outline-indigo-600 sm:text-sm/6 cursor-not-allowed"
          defaultValue={JSON.stringify(mainConfig, null, 2)}
          disabled={true}
        />

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
          <button
            type="submit"
            className="cursor-pointer relative -ml-px inline-flex items-center rounded-r-md bg-white px-3 py-2 text-sm font-semibold text-gray-900 ring-1 ring-gray-300 ring-inset hover:bg-gray-50 focus:z-10"
          >
            Generate Configuration YAML
          </button>
        </span>
      </form>
    </div>
  );
}
