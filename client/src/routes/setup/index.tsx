import { FormEvent, useContext, useState } from "react";
import { createFileRoute, Link } from "@tanstack/react-router";
import createClient from "openapi-react-query";
import { getApiFetchClient } from "../../utils/api";
import { MainConfigContext } from "../setup";
import { DocumentIcon } from "@heroicons/react/24/solid";
import Alert from "../../components/Alert";
import { classNames } from "../../utils/component";

export const Route = createFileRoute("/setup/")({
  component: RouteComponent,
});

function RouteComponent() {
  const client = createClient(getApiFetchClient());
  const [file, setFile] = useState<File | null>(null);

  const { mainConfig, setMainConfig } = useContext(MainConfigContext);

  const { mutate, data, error } = client.useMutation(
    "post",
    "/api/configuration/file",
  );

  const uploadFile = async (rawFile: File) => {
    const rawFileContents = await rawFile.text();

    let body: any = rawFileContents;
    if (rawFile.type == "application/json") {
      body = JSON.parse(rawFileContents);
    }
    mutate(
      {
        body,
        headers: { "Content-Type": rawFile.type, Accept: "application/json" },
      },
      {
        onSuccess: (data) => {
          setMainConfig(data);
        },
      },
    );
  };

  const handleFormUpload = async (
    event: FormEvent | null | undefined = undefined,
  ) => {
    if (event) {
      event.preventDefault();
    }
    if (file) {
      await uploadFile(file);
    }
  };

  const handleFileChange = async (
    event: React.ChangeEvent<HTMLInputElement>,
  ) => {
    if (event.target.files && event.target.files[0]) {
      const file = event.target.files[0];
      setFile(file);
      await uploadFile(file);
    }
  };

  const clearMainConfig = () => {
    setMainConfig({});
  };

  return (
    <div className="p-2">
      <p>
        Welcome to the CoreRL setup wizard. Follow the steps below to create
        your YAML configuration file.
      </p>
      <p className="mb-2">
        If a yaml configuration file exists, you may upload it here to
        prepopulate the setup wizard defaults.
      </p>

      <form
        className="border border-gray-400 rounded-lg p-2 mb-2"
        onSubmit={handleFormUpload}
      >
        <div className="col-span-full">
          <label
            htmlFor="cover-photo"
            className="block text-sm/6 font-medium text-gray-900"
          >
            Configuration File
          </label>

          {!!error && (
            <Alert>
              <code>{JSON.stringify(error)}</code>
            </Alert>
          )}
          {!!data && (
            <span className="text-green-700">
              Successfully loaded <code>{file && file.name}</code>
            </span>
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
          onClick={handleFormUpload}
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
      <Link
        to="/setup/name"
        className="cursor-pointer relative -ml-px inline-flex items-center rounded-md bg-white px-3 py-2 text-sm font-semibold text-gray-900 ring-1 ring-gray-300 ring-inset hover:bg-gray-50 focus:z-10"
      >
        Go to /setup/name
      </Link>
    </div>
  );
}
