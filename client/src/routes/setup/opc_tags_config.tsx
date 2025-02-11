import { MagnifyingGlassIcon } from "@heroicons/react/24/solid";
import { createFileRoute } from "@tanstack/react-router";
import createClient from "openapi-react-query";
import {
  memo,
  MouseEventHandler,
  useContext,
  useEffect,
  useMemo,
  useState,
} from "react";
import { Badge } from "../../components/badge";
import { Fieldset, Legend } from "../../components/fieldset";
import { Input, InputGroup } from "../../components/input";
import { SetupConfigNav } from "../../components/setup/setup-config-nav";
import {
  Table,
  TableBody,
  TableCell,
  TableHead,
  TableHeader,
  TableRow,
} from "../../components/table";
import { getApiFetchClient } from "../../utils/api";
import { Code, Text, TextSpan } from "../../components/text";
import { DeepPartial, MainConfigContext } from "../../utils/main-config";
import { components } from "../../api-schema";

import {
  useReactTable,
  getCoreRowModel,
  createColumnHelper,
  flexRender,
  getFilteredRowModel,
} from "@tanstack/react-table";
import { Spinner } from "../../components/spinner";
import { TagConfigDialog } from "../../components/setup/opc-tag-config-dialog";

export const Route = createFileRoute("/setup/opc_tags_config")({
  component: OPCTagsConfig,
});

const HighlightText = memo(
  ({ text, highlight }: { text: string; highlight: string }) => {
    // Split on highlight term and include term into parts, ignore case
    const parts = text.split(new RegExp(`(${highlight})`, "gi"));
    return (
      <Text>
        {parts.map((part, i) => (
          <span
            key={i}
            className={
              part.toLowerCase() === highlight.toLowerCase() ? "font-bold" : ""
            }
          >
            {part}
          </span>
        ))}
      </Text>
    );
  },
);
HighlightText.displayName = "HighlightText";

const opcSearchColumnHelper =
  createColumnHelper<components["schemas"]["OpcNodeDetail"]>();

const tagConfigColumnHelper =
  createColumnHelper<DeepPartial<components["schemas"]["TagConfig"]>>();

const renderRangeCell: (
  val: DeepPartial<[number | null, number | null] | null | undefined>,
) => React.ReactNode = (val) => {
  if (!val) {
    return <Code>null</Code>;
  }
  const low = val[0];
  const high = val[1];
  let renderedLow = <Code>null</Code>;
  if (typeof low === "number") {
    renderedLow = <TextSpan>{low}</TextSpan>;
  }
  let renderedHigh = <Code>null</Code>;
  if (typeof high === "number") {
    renderedHigh = <TextSpan>{high}</TextSpan>;
  }

  return (
    <span>
      {renderedLow} to {renderedHigh}
    </span>
  );
};

const tagConfigColumns = [
  tagConfigColumnHelper.accessor("node_identifier", {
    cell: (info) => info.getValue(),
    header: "OPC Node ID",
  }),
  tagConfigColumnHelper.accessor("action_constructor", {
    cell: (info) => (
      <input
        type="checkbox"
        checked={!!info.getValue()?.length}
        disabled={true}
      />
    ),
    header: "Is Setpoint",
  }),
  tagConfigColumnHelper.accessor("operating_range", {
    cell: (info) => renderRangeCell(info.getValue()),
    header: "Operating Range",
  }),
  tagConfigColumnHelper.accessor("yellow_bounds", {
    cell: (info) => renderRangeCell(info.getValue()),
    header: "Yellow Bounds",
  }),
  tagConfigColumnHelper.accessor("red_bounds", {
    cell: (info) => renderRangeCell(info.getValue()),
    header: "Red Bounds",
  }),
];

function OPCTagsConfig() {
  const [dialogOpen, setDialogOpen] = useState<boolean>(false);
  const [selectedNodeId, setSelectedNodeId] = useState<string>("");
  const { mainConfig, setMainConfig } = useContext(MainConfigContext);
  const env = mainConfig.env as DeepPartial<
    components["schemas"]["DepAsyncEnvConfig"]
  >;
  const opc_url = env.opc_conn_url ?? "";

  const [globalFilter, setGlobalFilter] = useState<string>("");
  const [debouncedSearchString, setDebouncedSearchString] =
    useState<string>("");

  const client = createClient(getApiFetchClient());

  const { data, error, status } = client.useQuery("get", "/api/opc/nodes", {
    params: { query: { opc_url, query: "" } }, // query can be updated to make search occur on server side
  });

  const tagConfigsTable = useReactTable({
    data: (mainConfig?.pipeline?.tags ?? []) as DeepPartial<
      components["schemas"]["TagConfig"]
    >[],
    columns: tagConfigColumns,
    getCoreRowModel: getCoreRowModel(),
  });

  const opcNodeSearchColumns = useMemo(
    () => [
      opcSearchColumnHelper.accessor("nodeid", {
        cell: (info) => (
          <HighlightText
            text={info.getValue()}
            highlight={debouncedSearchString}
          />
        ),
        header: "OPC Node ID",
      }),
      opcSearchColumnHelper.accessor("path", {
        cell: (info) => (
          <HighlightText
            text={info.getValue()}
            highlight={debouncedSearchString}
          />
        ),
        header: "Path",
      }),
      opcSearchColumnHelper.accessor("key", {
        cell: (info) => (
          <HighlightText
            text={info.getValue()}
            highlight={debouncedSearchString}
          />
        ),
        header: "Key",
      }),
      opcSearchColumnHelper.accessor("DataType", {
        cell: (info) => <Code>{info.getValue()}</Code>,
        header: "DataType",
        enableGlobalFilter: false,
      }),
    ],
    [debouncedSearchString],
  );

  const opcNodesSearchTable = useReactTable({
    data: data?.nodes ?? [],
    columns: opcNodeSearchColumns,
    getCoreRowModel: getCoreRowModel(),
    getFilteredRowModel: getFilteredRowModel(),
    globalFilterFn: "includesString",
    state: {
      globalFilter,
    },
    onGlobalFilterChange: setGlobalFilter,
  });

  const handleSearchChange: React.ChangeEventHandler<HTMLInputElement> = (
    e,
  ) => {
    setGlobalFilter(e.target.value);
    opcNodesSearchTable.setGlobalFilter(String(e.target.value));
  };

  useEffect(() => {
    const timeoutId = setTimeout(() => {
      setDebouncedSearchString(globalFilter);
    }, 500);
    return () => clearTimeout(timeoutId);
  }, [globalFilter]);

  const loading =
    debouncedSearchString !== globalFilter || status === "pending";

  const handleRowClick: (
    key: string,
  ) => MouseEventHandler<HTMLTableRowElement> = (key: string) => () => {
    setSelectedNodeId(key);
    setDialogOpen(true);
  };

  const selectedNode = useMemo(() => {
    const nodeIdx = data?.nodes.findIndex(
      ({ nodeid }) => nodeid === selectedNodeId,
    );
    if (nodeIdx === undefined) {
      return undefined;
    }
    return data?.nodes[nodeIdx];
  }, [selectedNodeId, data]);

  const selectedTagConfig = useMemo(() => {
    const tags = mainConfig.pipeline?.tags ?? [];
    const existingNodeIndex = tags.findIndex(
      (tagConfig) => tagConfig?.node_identifier === selectedNodeId,
    );

    if (existingNodeIndex >= 0) {
      return tags[existingNodeIndex];
    }
    return undefined;
  }, [selectedNodeId, mainConfig]);

  const handleSubmitOPCNodeTagConfig = (
    updatedTagConfig: components["schemas"]["TagConfig"],
  ) => {
    setMainConfig((prevMainConfig) => {
      const newMainConfig = structuredClone(prevMainConfig);
      const tags = newMainConfig?.pipeline?.tags ?? [];

      const existingNodeIndex = tags.findIndex(
        (tagConfig) =>
          tagConfig?.node_identifier === updatedTagConfig.node_identifier,
      );

      if (existingNodeIndex >= 0) {
        tags[existingNodeIndex] = updatedTagConfig;
      } else {
        tags.push(updatedTagConfig);
      }

      newMainConfig.pipeline = { ...prevMainConfig.pipeline, tags };
      return newMainConfig;
    });
  };

  const handleDeleteOPCNodeTagConfig = (nodeIdentifier: string | number) => {
    setMainConfig((prevMainConfig) => {
      const newMainConfig = structuredClone(prevMainConfig);
      const tags = newMainConfig?.pipeline?.tags ?? [];

      const existingNodeIndex = tags.findIndex(
        (tagConfig) => tagConfig?.node_identifier === nodeIdentifier,
      );

      if (existingNodeIndex >= 0) {
        tags.splice(existingNodeIndex, 1);
      }

      newMainConfig.pipeline = { ...prevMainConfig.pipeline, tags };
      return newMainConfig;
    });
  };

  return (
    <>
      <div className="p-2">
        <Table dense className="max-w-full mb-2">
          <TableHead>
            {tagConfigsTable.getHeaderGroups().map((headerGroup) => (
              <TableRow key={headerGroup.id}>
                {headerGroup.headers.map((header) => (
                  <TableHeader
                    key={header.id}
                    className="max-w-80 overflow-x-auto sticky"
                  >
                    {header.isPlaceholder
                      ? null
                      : flexRender(
                          header.column.columnDef.header,
                          header.getContext(),
                        )}
                  </TableHeader>
                ))}
              </TableRow>
            ))}
          </TableHead>
          <TableBody>
            {/* TODO: virtualize this to support large number of rows
          https://tanstack.com/table/latest/docs/framework/react/examples/virtualized-rows */}
            {tagConfigsTable.getRowModel().rows.map((row) => (
              <TableRow
                key={row.id}
                onClick={handleRowClick(
                  `${row.original.node_identifier ?? ""}`,
                )}
                href="#"
              >
                {row.getVisibleCells().map((cell) => (
                  <TableCell key={cell.id} className="max-w-80 overflow-x-auto">
                    {flexRender(cell.column.columnDef.cell, cell.getContext())}
                  </TableCell>
                ))}
              </TableRow>
            ))}
          </TableBody>
        </Table>
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
              defaultValue={globalFilter}
              onChange={handleSearchChange}
            />
          </InputGroup>
        </Fieldset>

        {loading ? (
          <Spinner />
        ) : (
          <Table dense className="max-w-full max-h-[70vh] mb-2">
            <TableHead>
              {opcNodesSearchTable.getHeaderGroups().map((headerGroup) => (
                <TableRow key={headerGroup.id}>
                  {headerGroup.headers.map((header) => (
                    <TableHeader
                      key={header.id}
                      className="max-w-80 overflow-x-auto sticky"
                    >
                      {header.isPlaceholder
                        ? null
                        : flexRender(
                            header.column.columnDef.header,
                            header.getContext(),
                          )}
                    </TableHeader>
                  ))}
                </TableRow>
              ))}
            </TableHead>
            <TableBody>
              {/* TODO: virtualize this to support large number of rows
              https://tanstack.com/table/latest/docs/framework/react/examples/virtualized-rows */}
              {opcNodesSearchTable.getRowModel().rows.map((row) => (
                <TableRow
                  key={row.id}
                  onClick={handleRowClick(row.original.nodeid)}
                  href="#"
                >
                  {row.getVisibleCells().map((cell) => (
                    <TableCell
                      key={cell.id}
                      className="max-w-80 overflow-x-auto"
                    >
                      {flexRender(
                        cell.column.columnDef.cell,
                        cell.getContext(),
                      )}
                    </TableCell>
                  ))}
                </TableRow>
              ))}
            </TableBody>
          </Table>
        )}
        <SetupConfigNav />
      </div>
      <TagConfigDialog
        dialogOpen={dialogOpen}
        setDialogOpen={setDialogOpen}
        selectedNode={selectedNode}
        selectedTagConfig={selectedTagConfig}
        handleSubmittedOPCNodeTagConfig={handleSubmitOPCNodeTagConfig}
        handleDeleteOPCNodeTagConfig={handleDeleteOPCNodeTagConfig}
      />
    </>
  );
}
