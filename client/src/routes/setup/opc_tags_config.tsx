import { MagnifyingGlassIcon } from "@heroicons/react/24/solid";
import { createFileRoute } from "@tanstack/react-router";
import createClient from "openapi-react-query";
import {
  MouseEventHandler,
  useContext,
  useEffect,
  useMemo,
  useState,
} from "react";
import { components } from "../../api-schema";
import { Badge } from "../../components/badge";
import { Fieldset, Legend } from "../../components/fieldset";
import { Input, InputGroup } from "../../components/input";
import { OPCNodesTable } from "../../components/setup/opc-nodes-table";
import { TagConfigDialog } from "../../components/setup/opc-tag-config-dialog";
import { SetupConfigNav } from "../../components/setup/setup-config-nav";
import { TagConfigsTable } from "../../components/setup/tag-configs-table";
import { Spinner } from "../../components/spinner";
import { getApiFetchClient } from "../../utils/api";
import { DeepPartial, MainConfigContext } from "../../utils/main-config";
import { Button } from "../../components/button";

export const Route = createFileRoute("/setup/opc_tags_config")({
  component: OPCTagsConfig,
});

function OPCTagsConfig() {
  const { mainConfig, setMainConfig } = useContext(MainConfigContext);

  const [dialogOpen, setDialogOpen] = useState<boolean>(false);
  const [tagConfigIndex, setTagConfigIndex] = useState<number | undefined>(
    undefined,
  );
  const [tagConfig, setTagConfig] = useState<
    DeepPartial<components["schemas"]["TagConfig"]>
  >({});

  // Always assume OPC environment is DepAsyncEnvConfig, retrieve OPC connection URL
  const env = mainConfig.env as DeepPartial<
    components["schemas"]["DepAsyncEnvConfig"]
  >;
  const opc_url = env.opc_conn_url ?? "";

  // search for OPC nodes on the client side
  const [globalFilterSearchInput, setGlobalFilterSearchInput] =
    useState<string>("");
  const [debouncedSearchString, setDebouncedSearchString] =
    useState<string>("");

  const client = createClient(getApiFetchClient());
  const { data, error, status } = client.useQuery("get", "/api/opc/nodes", {
    params: { query: { opc_url, query: "" } }, // query can be updated to make search occur on server side
  });

  useEffect(() => {
    const timeoutId = setTimeout(() => {
      setDebouncedSearchString(globalFilterSearchInput);
    }, 500);
    return () => clearTimeout(timeoutId);
  }, [globalFilterSearchInput]);

  const loading =
    debouncedSearchString !== globalFilterSearchInput || status === "pending";

  // ensures no undefined tags exist, should never be the case anyways but needed for types
  const tagConfigs = useMemo(
    () =>
      (mainConfig.pipeline?.tags ?? []).filter(Boolean) as DeepPartial<
        components["schemas"]["TagConfig"]
      >[],
    [mainConfig],
  );

  const handleManualAddTagConfiguration: MouseEventHandler<
    HTMLButtonElement
  > = () => {
    setTagConfigIndex(undefined);
    setTagConfig({});
    setDialogOpen(true);
  };

  const handleRowClick: (
    tableType: "opc" | "tag_config",
    key: string | number,
  ) => MouseEventHandler<HTMLTableRowElement> = (tableType, key) => () => {
    let clickedTagConfigIdx: number | undefined = undefined;
    let clickedTagConfig: DeepPartial<components["schemas"]["TagConfig"]> = {};

    if (tableType === "opc") {
      // clicked on OPC table
      const opcNodeIdx = data?.nodes.findIndex(({ nodeid }) => nodeid === key);

      if (opcNodeIdx === undefined) {
        throw new TypeError(`opcNodeIdx not found for nodeid ${key}`);
      }
      const opcNode = data?.nodes[opcNodeIdx];
      if (opcNode === undefined) {
        throw new TypeError(`opcNode not found for index ${opcNodeIdx}`);
      }

      // check if opc node is already in tag configs
      const findClickedTagConfigIdx = tagConfigs.findIndex(
        ({ node_identifier }) => opcNode.nodeid === node_identifier,
      );

      clickedTagConfig =
        findClickedTagConfigIdx >= 0
          ? tagConfigs[findClickedTagConfigIdx]
          : { node_identifier: opcNode.nodeid, name: opcNode.key };
      clickedTagConfigIdx =
        findClickedTagConfigIdx >= 0 ? findClickedTagConfigIdx : undefined;
    } else {
      // clicked on Tag Configs table
      if (typeof key === "string") {
        throw new TypeError(
          `tag_config table type called with string key ${key}`,
        );
      }
      clickedTagConfigIdx = key;
      clickedTagConfig = tagConfigs[key];
    }
    setTagConfigIndex(clickedTagConfigIdx);
    setTagConfig(clickedTagConfig);
    setDialogOpen(true);
  };

  const handleSubmitOPCNodeTagConfig = (
    updatedTagConfig: components["schemas"]["TagConfig"],
  ) => {
    const tags = structuredClone(mainConfig?.pipeline?.tags ?? []);

    if (tagConfigIndex !== undefined) {
      tags[tagConfigIndex] = updatedTagConfig;
    } else {
      tags.push(updatedTagConfig);
    }

    setMainConfig({
      ...mainConfig,
      pipeline: {
        ...mainConfig.pipeline,
        tags,
      },
    });
  };

  const handleDeleteOPCNodeTagConfig = (tagConfigIdx: number) => {
    setMainConfig((prevMainConfig) => {
      const newTags = structuredClone(tagConfigs);

      if (tagConfigIdx >= 0) {
        newTags.splice(tagConfigIdx, 1);
      }

      return {
        ...prevMainConfig,
        pipeline: { ...prevMainConfig.pipeline, tags: newTags },
      };
    });
  };

  return (
    <>
      <div className="p-2">
        <div className="mb-2">
          <TagConfigsTable data={tagConfigs} handleRowClick={handleRowClick} />
          <Button onClick={handleManualAddTagConfiguration}>
            Manual Add Tag Configuration
          </Button>
        </div>
        <Fieldset className="border border-gray-400 rounded-lg p-2 mb-2">
          <Legend>Search OPC Tags</Legend>
          {error && <Badge color="red">{JSON.stringify(error)}</Badge>}
          <InputGroup>
            <MagnifyingGlassIcon />
            <Input
              name="search"
              placeholder="Search&hellip;"
              aria-label="Search"
              type="text"
              defaultValue={globalFilterSearchInput}
              onChange={(e) => setGlobalFilterSearchInput(e.target.value)}
            />
          </InputGroup>
        </Fieldset>

        {loading ? (
          <Spinner />
        ) : (
          <OPCNodesTable
            data={data?.nodes ?? []}
            handleRowClick={handleRowClick}
            debouncedSearchString={debouncedSearchString}
            setDebouncedSearchString={setDebouncedSearchString}
          />
        )}

        <SetupConfigNav />
      </div>
      <TagConfigDialog
        dialogOpen={dialogOpen}
        setDialogOpen={setDialogOpen}
        selectedTagConfig={tagConfig}
        tagConfigIndex={tagConfigIndex}
        handleSubmittedOPCNodeTagConfig={handleSubmitOPCNodeTagConfig}
        handleDeleteOPCNodeTagConfig={handleDeleteOPCNodeTagConfig}
      />
    </>
  );
}
