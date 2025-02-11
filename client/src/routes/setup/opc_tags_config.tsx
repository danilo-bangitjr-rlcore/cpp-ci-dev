import { MagnifyingGlassIcon } from "@heroicons/react/24/solid";
import { createFileRoute } from "@tanstack/react-router";
import createClient from "openapi-react-query";
import { memo, useEffect, useState } from "react";
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

function OPCTagsConfig() {
  // const { mainConfig, setMainConfig } = useContext(MainConfigContext);
  const opc_url = "opc.tcp://admin@0.0.0.0:4840/rlcore/server/";

  const [searchString, setSearchString] = useState<string>("");
  const [debouncedSearchString, setDebouncedSearchString] =
    useState<string>("");

  const handleSearchStringChange: React.ChangeEventHandler<HTMLInputElement> = (
    e,
  ) => {
    setSearchString(e.target.value);
  };

  useEffect(() => {
    const timeoutId = setTimeout(() => {
      setDebouncedSearchString(searchString);
    }, 500);
    return () => clearTimeout(timeoutId);
  }, [searchString]);

  const client = createClient(getApiFetchClient());
  const { data, error, status } = client.useQuery("get", "/api/opc/nodes", {
    params: { query: { opc_url, query: debouncedSearchString } },
  });

  console.log(new Date().toISOString());

  const loading =
    debouncedSearchString !== searchString || status === "pending";

  return (
    <div className="p-2">
      <Fieldset className="border border-gray-400 rounded-lg p-2 mb-2">
        <Legend>OPC Tags</Legend>
        {debouncedSearchString != searchString && <Badge>Loading...</Badge>}
        {error && <Badge color="red">{JSON.stringify(error)}</Badge>}
        {status && <Badge>{status}</Badge>}
        <InputGroup>
          <MagnifyingGlassIcon />
          <Input
            name="search"
            placeholder="Search&hellip;"
            aria-label="Search"
            type="text"
            defaultValue={searchString}
            onChange={handleSearchStringChange}
          />
        </InputGroup>
      </Fieldset>

      {loading ? (
        <svg className="animate-spin mr-3 size-5">
          <path d="M10.14,1.16a11,11,0,0,0-9,8.92A1.59,1.59,0,0,0,2.46,12,1.52,1.52,0,0,0,4.11,10.7a8,8,0,0,1,6.66-6.61A1.42,1.42,0,0,0,12,2.69h0A1.57,1.57,0,0,0,10.14,1.16Z" />
        </svg>
      ) : (
        <Table dense className="max-w-full">
          <TableHead>
            <TableRow>
              <TableHeader>OPC ID</TableHeader>
              <TableHeader className="max-w-80">Path</TableHeader>
              <TableHeader>Key</TableHeader>
              <TableHeader className="max-w-80">Val</TableHeader>
            </TableRow>
          </TableHead>
          <TableBody>
            {data?.nodes.map((opc_node) => (
              <TableRow key={opc_node.nodeid}>
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
                  <Code>{JSON.stringify(opc_node.val)}</Code>
                </TableCell>
              </TableRow>
            ))}
          </TableBody>
        </Table>
      )}
      <SetupConfigNav />
    </div>
  );
}
