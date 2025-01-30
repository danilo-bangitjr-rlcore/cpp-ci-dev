import { useState } from "react";
import { createFileRoute } from "@tanstack/react-router";

const Index = () => {
  const [count, setCount] = useState(0);

  return (
    <div className="p-2">
      <h2>Home</h2>
      <button className="text-white bg-blue-500 px-4 py-2 rounded hover:bg-blue-600"
        onClick={() => setCount((count) => count + 1)}>
        count is {count}
      </button>
    </div>
  );
};

export const Route = createFileRoute("/")({
  component: Index,
});
