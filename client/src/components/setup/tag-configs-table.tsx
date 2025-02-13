import {
  createColumnHelper,
  flexRender,
  getCoreRowModel,
  useReactTable,
} from "@tanstack/react-table";
import {
  Table,
  TableBody,
  TableCell,
  TableHead,
  TableHeader,
  TableRow,
} from "../table";
import { components } from "../../api-schema";
import { DeepPartial } from "../../utils/main-config";
import { Code, TextSpan } from "../text";
import { MouseEventHandler } from "react";

interface TagConfigsTableProps {
  data: DeepPartial<components["schemas"]["TagConfig"]>[];
  handleRowClick: (
    tableType: "opc" | "tag_config",
    key: string | number,
  ) => MouseEventHandler<HTMLTableRowElement>;
}

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

const tagConfigColumnHelper =
  createColumnHelper<DeepPartial<components["schemas"]["TagConfig"]>>();

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

export const TagConfigsTable = ({
  data,
  handleRowClick,
}: TagConfigsTableProps) => {
  // ensure no undefined or nulls in the data array

  const tagConfigsTable = useReactTable({
    data,
    columns: tagConfigColumns,
    getCoreRowModel: getCoreRowModel(),
  });

  return (
    <Table dense className="max-w-full mb-2">
      <TableHead>
        {tagConfigsTable.getHeaderGroups().map((headerGroup) => (
          <TableRow key={`tc-h-${headerGroup.id}`}>
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
        {tagConfigsTable.getRowModel().rows.map((row, row_index) => (
          <TableRow
            key={`tc-${row.id}`}
            onClick={handleRowClick("tag_config", row_index)}
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
  );
};
