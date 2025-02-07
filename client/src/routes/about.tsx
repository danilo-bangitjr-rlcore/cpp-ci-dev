import { createFileRoute } from "@tanstack/react-router";
import createClient from "openapi-react-query";
import { getApiFetchClient } from "../utils/api";
import { Text } from "../components/text";
import { Heading } from "../components/heading";
import { Button } from "../components/button";

const About = () => {
  const client = createClient(getApiFetchClient());
  const { status, error, data, refetch } = client.useQuery("get", "/health", {
    headers: { Accept: "application/json" },
  });

  return (
    <div className="p-2">
      <Heading level={2}>About</Heading>
      <Text>
        About page. TBD: show core-rl version information, web client details,
        experiment/agent details.
      </Text>
      <div className="p-2">
        <Heading level={3}>Health</Heading>
        <Text>{status}</Text>
        {error && <div>Error: {`${error}`}</div>}
        {data && <Text>{JSON.stringify(data, null, 2)}</Text>}
        <Button
          className="mt-2 text-white bg-blue-500 px-4 py-2 rounded hover:bg-blue-600"
          onClick={() => {
            void refetch();
          }}
        >
          Re-fetch Health Status
        </Button>
      </div>
    </div>
  );
};

export const Route = createFileRoute("/about")({
  component: About,
});
