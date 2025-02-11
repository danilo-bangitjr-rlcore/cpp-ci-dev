import { MagnifyingGlassIcon } from "@heroicons/react/24/solid";
import { createFileRoute } from "@tanstack/react-router";
import createClient from "openapi-react-query";
import {
  memo,
  MouseEventHandler,
  useContext,
  useEffect,
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
import { Code, Text } from "../../components/text";
import { DeepPartial, MainConfigContext } from "../../utils/main-config";
import { components } from "../../api-schema";
import { Dialog, DialogTitle } from "../../components/dialog";
import {
  useReactTable,
  getCoreRowModel,
  createColumnHelper,
  flexRender,
  getFilteredRowModel,
} from "@tanstack/react-table";
import { Spinner } from "../../components/spinner";

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

const columnHelper =
  createColumnHelper<components["schemas"]["OpcNodeDetail"]>();

function OPCTagsConfig() {
  const [dialogOpen, setDialogOpen] = useState<boolean>(false);
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
    params: { query: { opc_url, query: "" } },
  });

  const columns = [
    columnHelper.accessor("nodeid", {
      cell: (info) => (
        <HighlightText
          text={info.getValue()}
          highlight={debouncedSearchString}
        />
      ),
      footer: (info) => info.column.id,
      header: "OPC Node ID",
    }),
    columnHelper.accessor("path", {
      cell: (info) => (
        <HighlightText
          text={info.getValue()}
          highlight={debouncedSearchString}
        />
      ),
      footer: (info) => info.column.id,
      header: "Path",
    }),
    columnHelper.accessor("key", {
      cell: (info) => (
        <HighlightText
          text={info.getValue()}
          highlight={debouncedSearchString}
        />
      ),
      footer: (info) => info.column.id,
      header: "Key",
    }),
    columnHelper.accessor("DataType", {
      cell: (info) => <Code>{info.getValue()}</Code>,
      footer: (info) => info.column.id,
      header: "DataType",
      enableGlobalFilter: false,
    }),
  ];

  const table = useReactTable({
    data: data?.nodes ?? [],
    columns,
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
    table.setGlobalFilter(String(e.target.value));
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
  ) => MouseEventHandler<HTMLTableRowElement> = (key: string) => (e) => {
    console.log(key, e);
    setDialogOpen(true);
  };

  console.log(new Date().toISOString());

  return (
    <>
      <div className="p-2">
        <Fieldset className="border border-gray-400 rounded-lg p-2 mb-2">
          <Legend>OPC Tags</Legend>
          {loading && <Badge>Loading...</Badge>}
          {error && <Badge color="red">{JSON.stringify(error)}</Badge>}
          {status && <Badge>{status}</Badge>}
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
              {table.getHeaderGroups().map((headerGroup) => (
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
              {table.getRowModel().rows.map((row) => (
                <TableRow
                  key={row.id}
                  onClick={handleRowClick(row.id)}
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
              {/* {data?.nodes.map((opc_node) => (
                <TableRow
                  key={opc_node.nodeid}
                  className="cursor-pointer hover:bg-gray-200"
                  onClick={handleRowClick(opc_node.nodeid)}
                  href="#"
                >
                  <TableCell>
                    <HighlightText
                      text={opc_node.nodeid}
                      highlight={debouncedSearchString}
                    />
                  </TableCell>
                  <TableCell className="max-w-80 overflow-x-auto">
                    <HighlightText
                      text={opc_node.path}
                      highlight={debouncedSearchString}
                    />
                  </TableCell>
                  <TableCell>
                    <HighlightText
                      text={opc_node.key}
                      highlight={debouncedSearchString}
                    />
                  </TableCell>
                  <TableCell className="max-w-80 overflow-x-auto">
                    <Code>{JSON.stringify(opc_node.DataType)}</Code>
                  </TableCell>
                </TableRow>
              ))} */}
            </TableBody>
          </Table>
        )}
        <SetupConfigNav />
      </div>
      <Dialog open={dialogOpen} onClose={setDialogOpen}>
        <DialogTitle>Configure Tag</DialogTitle>
      </Dialog>
    </>
  );
}
