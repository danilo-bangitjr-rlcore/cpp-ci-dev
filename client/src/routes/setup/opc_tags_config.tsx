import { MagnifyingGlassIcon } from "@heroicons/react/24/solid";
import { createFileRoute } from "@tanstack/react-router";
import createClient from "openapi-react-query";
import { useEffect, useState } from "react";
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

export const Route = createFileRoute("/setup/opc_tags_config")({
  component: OPCTagsConfig,
});

function OPCTagsConfig() {
  // const { mainConfig, setMainConfig } = useContext(MainConfigContext);
  const opc_url = "opc.tcp://admin@0.0.0.0:4840/rlcore/server/"

  const [searchString, setSearchString] = useState<string>("");
  const [debouncedSearchString, setDebouncedSearchString] = useState<string>("")

  const handleSearchStringChange: React.ChangeEventHandler<HTMLInputElement> = (e) => {
    setSearchString(e.target.value);
  };

  useEffect(() => {
    const timeoutId = setTimeout(() => {
      setDebouncedSearchString(searchString);
    }, 500);
    return () => clearTimeout(timeoutId);
  }, [searchString])

  const client = createClient(getApiFetchClient());
  const {data, error, status} = client.useQuery("get", "/api/opc/nodes", {
    params: { query: { opc_url, query: debouncedSearchString } },
  });

  return (
    <div className="p-2">
      <Fieldset className="border border-gray-400 rounded-lg p-2 mb-2">
        <Legend>OPC Tags</Legend>
        {debouncedSearchString != searchString && <Badge>Loading...</Badge>}
        {error && <Badge>{JSON.stringify(error)}</Badge>}
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
              <TableCell>{opc_node.nodeid}</TableCell>
              <TableCell className="max-w-80 overflow-x-auto">{opc_node.path}</TableCell>
              <TableCell>{opc_node.key}</TableCell>
              <TableCell className="max-w-80 overflow-x-auto">{JSON.stringify(opc_node.val)}</TableCell>
            </TableRow>
          ))}
        </TableBody>
      </Table>
      <SetupConfigNav />
    </div>
  );
}
