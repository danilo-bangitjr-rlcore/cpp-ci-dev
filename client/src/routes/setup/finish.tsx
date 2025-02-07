import { createFileRoute } from "@tanstack/react-router";
import { FormEventHandler, useContext, useState } from "react";
import { Alert } from "../../components/alert";
import { fetchWithTimeout, getServerOrigin } from "../../utils/api";
import { MainConfigContext } from "../../utils/main-config";
import { Code } from "../../components/text";
import { Textarea } from "../../components/textarea";
import { SetupConfigNav } from "../../components/setup/setup-config-nav";
import { Button } from "../../components/button";
import { Field, Label } from "../../components/fieldset";

export const Route = createFileRoute("/setup/finish")({
  component: Finish,
});

function Finish() {
  const { mainConfig } = useContext(MainConfigContext);
  const [error, setError] = useState<string | undefined>(undefined);

  const handleFormUpload: FormEventHandler<HTMLFormElement> = (
    event: React.FormEvent<HTMLFormElement>,
  ) => {
    event.preventDefault();
    void (async () => {
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
        const error: unknown = await resp.json();
        setError(
          `Error generating configuration file: ${JSON.stringify(error)}`,
        );
        console.error("Error generating configuration file");
        return;
      }

      // convert response yaml into a downloadable file
      const data = await resp.blob();
      const url = window.URL.createObjectURL(data);
      const a = document.createElement("a");
      a.href = url;
      a.download = `${mainConfig.experiment?.exp_name ?? "config"}.yaml`;
      document.body.appendChild(a);
      a.click();
      document.body.removeChild(a);
      window.URL.revokeObjectURL(url);
    })();
  };

  return (
    <div className="p-2">
      <Alert
        open={!!error && Object.keys(error).length > 0}
        onClose={() => setError(undefined)}
      >
        <Code>{JSON.stringify(error)}</Code>
      </Alert>

      <form className="mt-2" onSubmit={handleFormUpload}>
        <Field>
          <Label
            htmlFor="main_config_payload"
            className="block text-sm/6 font-medium text-gray-900"
          >
            Main Configuration Payload
          </Label>
          <Textarea
            id="main_config_payload"
            rows={20}
            // defaultValue={JSON.stringify(mainConfig, null, 2)}
            value={JSON.stringify(mainConfig, null, 2)}
            disabled={true}
          />
        </Field>

        <SetupConfigNav />
        <Button className="cursor-pointer" type="submit" color="green">
          Generate Configuration YAML
        </Button>
      </form>
    </div>
  );
}
