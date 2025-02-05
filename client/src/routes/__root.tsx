import { ReactQueryDevtools } from "@tanstack/react-query-devtools";
import { createRootRoute } from "@tanstack/react-router";
import { TanStackRouterDevtools } from "@tanstack/router-devtools";
import App from "../components/app";

export const Route = createRootRoute({
  component: () => (
    <>
      <App />
      {!import.meta.env.PROD && (
        <>
          <TanStackRouterDevtools />
          <ReactQueryDevtools />
        </>
      )}
    </>
  )
});
