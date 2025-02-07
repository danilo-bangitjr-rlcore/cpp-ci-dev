import { createFileRoute } from "@tanstack/react-router";
import { Fieldset, Legend } from "../../components/fieldset";
import { SetupConfigNav } from "../../components/setup/setup-config-nav";
import { Input, InputGroup } from "../../components/input";
import { MagnifyingGlassIcon } from "@heroicons/react/24/solid";
import {
  Table,
  TableBody,
  TableCell,
  TableHead,
  TableHeader,
  TableRow,
} from "../../components/table";
import { useContext, useEffect, useState } from "react";
import createClient from "openapi-react-query";
import { getApiFetchClient } from "../../utils/api";
import { MainConfigContext } from "../../utils/main-config";
import { Badge } from "../../components/badge";

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
        <Table dense className="[--gutter:--spacing(6)] sm:[--gutter:--spacing(8)] max-w-full">
          <TableHead>
            <TableRow>
            <TableHeader>OPC ID</TableHeader>
            <TableHeader>Path</TableHeader>
            <TableHeader>Key</TableHeader>
            <TableHeader>Val</TableHeader>
            </TableRow>
          </TableHead>
          <TableBody>
            {data?.nodes.map((opc_node) => (
              <TableRow key={opc_node.nodeid}>
                <TableCell>{opc_node.nodeid}</TableCell>
                <TableCell>{opc_node.path}</TableCell>
                <TableCell>{opc_node.key}</TableCell>
                <TableCell>{JSON.stringify(opc_node.val)}</TableCell>
              </TableRow>
            ))}
          </TableBody>
        </Table>
      </Fieldset>
      <SetupConfigNav />
    </div>
  );
}
