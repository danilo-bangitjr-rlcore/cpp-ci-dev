import { useState } from "react";
import { createFileRoute } from "@tanstack/react-router";

const Index = () => {
  const [count, setCount] = useState(0);

  return (
    <div className="p-2">
      <h2>Home</h2>
      <button onClick={() => setCount((count) => count + 1)}>
        count is {count}
      </button>
    </div>
  );
};

export const Route = createFileRoute("/")({
  component: Index,
});
