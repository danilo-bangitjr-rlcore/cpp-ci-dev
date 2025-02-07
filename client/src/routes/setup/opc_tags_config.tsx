import { createFileRoute } from "@tanstack/react-router";
// import { useContext } from "react";
import { Field, Fieldset, Legend } from "../../components/fieldset";
import { SetupConfigNav } from "../../components/setup/setup-config-nav";
import { Text } from "../../components/text";
// import { MainConfigContext } from "../../utils/main-config";
import { Heading } from "../../components/heading";

export const Route = createFileRoute("/setup/opc_tags_config")({
  component: StubRequired,
});

function StubRequired() {
  // const { mainConfig, setMainConfig } = useContext(MainConfigContext);

  const handleSubmit = (e: React.FormEvent) => {
    e.preventDefault();
  };

  return (
    <div className="p-2">
      <form
        className="border border-gray-400 rounded-lg p-2 mb-2"
        onSubmit={handleSubmit}
      >
        <Field>
          <Heading level={3}>OPC Tags</Heading>
        </Field>
        <Fieldset>
          <Legend>TBD</Legend>
          <Text>...</Text>
        </Fieldset>
      </form>
      <SetupConfigNav />
    </div>
  );
}
