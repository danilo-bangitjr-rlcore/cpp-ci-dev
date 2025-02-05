import { DocumentIcon } from "@heroicons/react/24/solid";
import { createFileRoute } from "@tanstack/react-router";
import createClient from "openapi-react-query";
import {
  ChangeEventHandler,
  FormEvent,
  FormEventHandler,
  MouseEventHandler,
  useContext,
  useState,
} from "react";
import { Alert } from "../../components/alert";
import { Badge, BadgeButton } from "../../components/badge";
import { Button } from "../../components/button";
import { Code, Text } from "../../components/text";
import { getApiFetchClient } from "../../utils/api";
import { classNames } from "../../utils/component";
import { MainConfigContext } from "../../utils/main-config";

export const Route = createFileRoute("/setup/")({
  component: RouteComponent,
});

function RouteComponent() {
  const client = createClient(getApiFetchClient());
  const [file, setFile] = useState<File | null>(null);
  const { mainConfig, setMainConfig } = useContext(MainConfigContext);

  const [isAlertOpen, setIsAlertOpen] = useState(false);

  const { mutate, data, error, reset } = client.useMutation(
    "post",
    "/api/configuration/file",
  );

  const uploadFile = async (rawFile: File) => {
    const rawFileContents = await rawFile.text();

    let body: unknown = rawFileContents;
    if (rawFile.type == "application/json") {
      body = JSON.parse(rawFileContents);
    }
    mutate(
      {
        /* eslint-disable-next-line @typescript-eslint/no-unsafe-assignment, @typescript-eslint/no-explicit-any */
        body: body as any,
        headers: { "Content-Type": rawFile.type, Accept: "application/json" },
      },
      {
        onSuccess: (data) => {
          setMainConfig(data);
        },
        onError: (error) => {
          console.error(error);
          setIsAlertOpen(true);
        },
      },
    );
  };

  const handleFormUpload: FormEventHandler<HTMLFormElement> = (
    event: FormEvent | null | undefined = undefined,
  ) => {
    if (event) {
      event.preventDefault();
    }
    if (file) {
      void uploadFile(file);
    }
  };

  const handleFileChange: ChangeEventHandler<HTMLInputElement> = (
    event: React.ChangeEvent<HTMLInputElement>,
  ) => {
    if (event.target.files?.[0]) {
      const file = event.target.files[0];
      setFile(file);
      void uploadFile(file);
    }
  };

  const clearMainConfig: MouseEventHandler<HTMLButtonElement> = () => {
    setMainConfig({});
    reset();
  };

  return (
    <div className="p-2">
      <Text>
        Welcome to the CoreRL setup wizard. Follow the steps below to create
        your YAML configuration file.
      </Text>
      <Text className="mb-2">
        If a yaml configuration file exists, you may upload it here to
        prepopulate the setup wizard defaults.
      </Text>

      <form
        className="border border-gray-400 rounded-lg p-2 mb-2"
        onSubmit={() => handleFormUpload}
      >
        <div className="col-span-full">
          <label
            htmlFor="cover-photo"
            className="block text-sm/6 font-medium text-gray-900"
          >
            Configuration File
          </label>

          <Alert open={isAlertOpen} onClose={() => setIsAlertOpen(false)}>
            <Code>{JSON.stringify(error)}</Code>
          </Alert>
          {!!error && (
            <BadgeButton
              onClick={() => setIsAlertOpen(true)}
              className="cursor-pointer"
            >
              There is an error with the provided file.
            </BadgeButton>
          )}
          {!!data && (
            <Badge color="green">
              Successfully loaded <code>{file?.name}</code>
            </Badge>
          )}
          <div className="mt-2 flex justify-center rounded-lg border border-dashed border-gray-900/25 px-6 py-10">
            <div className="text-center">
              <DocumentIcon
                aria-hidden="true"
                className="mx-auto size-12 text-gray-300"
              />
              <div className="mt-4 flex text-sm/6 text-gray-600">
                <label
                  htmlFor="file-upload"
                  className="relative cursor-pointer rounded-md bg-white font-semibold text-indigo-600 focus-within:ring-2 focus-within:ring-indigo-600 focus-within:ring-offset-2 focus-within:outline-hidden hover:text-indigo-500"
                >
                  <span>Upload a file</span>
                  <input
                    id="file-upload"
                    name="file-upload"
                    type="file"
                    className="sr-only"
                    accept=".yaml,.yml,.json"
                    onChange={handleFileChange}
                  />
                </label>
              </div>
              <p className="text-xs/5 text-gray-600">YAML, YML, or JSON</p>
            </div>
          </div>
        </div>
        <button
          type="submit"
          className={classNames(
            !file ? "cursor-not-allowed" : "cursor-pointer",
            "rounded-md bg-indigo-600 px-3.5 py-2.5 text-sm font-semibold text-white shadow-xs hover:bg-indigo-500 focus-visible:outline-2 focus-visible:outline-offset-2 focus-visible:outline-indigo-600",
          )}
          disabled={!file}
        >
          Upload
        </button>

        {!!Object.keys(mainConfig).length && (
          <button
            type="button"
            onClick={clearMainConfig}
            className="rounded-md bg-indigo-600 px-3.5 py-2.5 text-sm font-semibold text-white shadow-xs hover:bg-indigo-500 focus-visible:outline-2 focus-visible:outline-offset-2 focus-visible:outline-indigo-600"
          >
            Clear Setup Config
          </button>
        )}
      </form>
      <Button to="/setup/name">Go to /setup/name</Button>
    </div>
  );
}
