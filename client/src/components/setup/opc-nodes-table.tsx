import {
  createColumnHelper,
  flexRender,
  getCoreRowModel,
  getFilteredRowModel,
  useReactTable,
} from "@tanstack/react-table";
// import { useVirtualizer } from "@tanstack/react-virtual";
import { MouseEventHandler, useMemo } from "react";
import { components } from "../../api-schema";
import {
  Table,
  TableBody,
  TableCell,
  TableHead,
  TableHeader,
  TableRow,
} from "../table";
import { Code, HighlightText } from "../text";

interface OPCNodesTableProps {
  data: components["schemas"]["OpcNodeDetail"][];
  handleRowClick: (
    tableType: "opc" | "tag_config",
    key: string | number,
  ) => MouseEventHandler<HTMLTableRowElement>;
  debouncedSearchString: string;
  setDebouncedSearchString: React.Dispatch<React.SetStateAction<string>>;
}

const opcSearchColumnHelper =
  createColumnHelper<components["schemas"]["OpcNodeDetail"]>();

export const OPCNodesTable = ({
  data,
  handleRowClick,
  debouncedSearchString,
  setDebouncedSearchString,
}: OPCNodesTableProps) => {
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
    data,
    columns: opcNodeSearchColumns,
    getCoreRowModel: getCoreRowModel(),
    getFilteredRowModel: getFilteredRowModel(),
    globalFilterFn: "includesString",
    state: {
      globalFilter: debouncedSearchString,
    },
    onGlobalFilterChange: setDebouncedSearchString,
  });

  // const { rows } = opcNodesSearchTable.getRowModel();
  // const scrollRootRef = useRef<HTMLDivElement>(null);
  // const virtualizer = useVirtualizer({
  //   count: rows.length,
  //   getScrollElement: () => scrollRootRef.current,
  //   estimateSize: () => 45,
  //   overscan: 10,
  //   debug: true
  // });
  // console.log(rows.length, virtualizer)
  // console.log(virtualizer.getVirtualItems().length)
  return (
    <Table
      dense
      className="max-w-full max-h-[70vh] mb-2"
      // scrollRootRef={scrollRootRef}
      // virtualScrollRootRefClassname="max-h-[70vh]"
      // virtualTableWrapStyle={{ height: `${virtualizer.getTotalSize()}px` }}
    >
      <TableHead>
        {opcNodesSearchTable.getHeaderGroups().map((headerGroup) => (
          <TableRow key={`opc-hg-${headerGroup.id}`}>
            {headerGroup.headers.map((header) => (
              <TableHeader
                key={`opc-h-${header.id}`}
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
        {opcNodesSearchTable.getRowModel().rows.map((row) => (
          <TableRow
            key={`opc-${row.id}`}
            onClick={handleRowClick("opc", row.original.nodeid)}
          >
            {row.getVisibleCells().map((cell) => (
              <TableCell key={cell.id} className="max-w-80 overflow-x-auto">
                {flexRender(cell.column.columnDef.cell, cell.getContext())}
              </TableCell>
            ))}
          </TableRow>
        ))}
        {/* {virtualizer.getVirtualItems().map((virtualRow, index) => {
          const row = rows[virtualRow.index];
          return (
            <TableRow
              key={`opc-r-${row.id}`}
              onClick={handleRowClick(row.original.nodeid)}
              style={{
                height: `${virtualRow.size}px`,
                transform: `translateY(${
                  virtualRow.start - index * virtualRow.size
                }px)`,
              }}
              href="#"
            >
              {row.getVisibleCells().map((cell) => (
                <TableCell key={`opc-${cell.id}`} className="max-w-80 overflow-x-auto">
                  {flexRender(cell.column.columnDef.cell, cell.getContext())}
                </TableCell>
              ))}
            </TableRow>
          );
        })} */}
      </TableBody>
    </Table>
  );
};
