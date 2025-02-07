import { createFileRoute } from "@tanstack/react-router";
import { Field, Fieldset, Label, Legend } from "../../components/fieldset";
import { SetupConfigNav } from "../../components/setup/setup-config-nav";
import { Text } from "../../components/text";

export const Route = createFileRoute("/setup/opc_tags_config")({
  component: OPCTagsConfig,
});

function OPCTagsConfig() {
  return (
    <div className="p-2">
      <Fieldset className="border border-gray-400 rounded-lg p-2 mb-2">
        <Legend>OPC Tags</Legend>
        <Field>
          <Label>TBD</Label>
          <Text>...</Text>
        </Field>
      </Fieldset>
      <SetupConfigNav />
    </div>
  );
}
