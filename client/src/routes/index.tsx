import { createFileRoute } from "@tanstack/react-router";
import { useState } from "react";
import { Heading } from "../components/heading";
import { Button } from "../components/button";

const Index = () => {
  const [count, setCount] = useState(0);

  return (
    <div className="p-2">
      <Heading level={2}>Home</Heading>
      <Button
        className="mt-2 text-white bg-blue-500 px-4 py-2 rounded hover:bg-blue-600"
        onClick={() => setCount((count) => count + 1)}
      >
        count is {count}
      </Button>
    </div>
  );
};

export const Route = createFileRoute("/")({
  component: Index,
});
