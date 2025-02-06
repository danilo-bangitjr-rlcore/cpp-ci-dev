import { createFileRoute } from "@tanstack/react-router";
import { useContext } from "react";
import { type components } from "../../api-schema";
import { Field, Fieldset, Label } from "../../components/fieldset";
import { Heading } from "../../components/heading";
import { Input } from "../../components/input";
import { SetupConfigNav } from "../../components/setup/setup-config-nav";
import {
  type DeepPartial,
  type DeepPartialMainConfig,
  MainConfigContext,
  setValFromPath,
} from "../../utils/main-config";

export const Route = createFileRoute("/setup/name")({
  component: Name,
});

function Name() {
  const { mainConfig, setMainConfig } = useContext(MainConfigContext);
  // Type assertion of sub-parts of mainConfig
  const env = mainConfig.env as DeepPartial<
    components["schemas"]["DepAsyncEnvConfig"]
  >;
  const metrics = mainConfig.metrics as DeepPartial<
    components["schemas"]["MetricsDBConfig"]
  >;

  const handleInputChange = (e: React.ChangeEvent<HTMLInputElement>) => {
    const inputType = e.target.getAttribute("type")!;
    setMainConfig((prevMainConfig: DeepPartialMainConfig) => {
      const paths = e.target.name.split(",");
      let newConfig = structuredClone(prevMainConfig);
      for (const path of paths) {
        newConfig = setValFromPath(newConfig, path, e.target.value, inputType);
      }
      return newConfig;
    });
  };

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
          <Heading level={3}>Experiment</Heading>
        </Field>

        <Fieldset>
          <Field>
            <Label htmlFor="exp_name">Experiment Name</Label>
            <Input
              id="exp_name"
              name="experiment.exp_name"
              type="text"
              placeholder=""
              onChange={handleInputChange}
              defaultValue={mainConfig.experiment?.exp_name ?? ""}
            />
          </Field>
        </Fieldset>
      </form>

      <form
        className="border border-gray-400 rounded-lg p-2 mb-2"
        onSubmit={handleSubmit}
      >
        <Field>
          <Heading level={3}>Database</Heading>
        </Field>

        <Fieldset>
          <Field>
            <Label htmlFor="drivername">Driver name</Label>
            <Input
              id="drivername"
              name="pipeline.db.drivername,metrics.drivername,env.db.drivername"
              type="text"
              placeholder=""
              onChange={handleInputChange}
              defaultValue={mainConfig.pipeline?.db?.drivername ?? ""}
            />
          </Field>
        </Fieldset>

        <br></br>
        <Fieldset>
          <Field>
            <Label htmlFor="username">Username</Label>
            <Input
              id="username"
              name="pipeline.db.username,metrics.username,env.db.username"
              type="text"
              placeholder=""
              onChange={handleInputChange}
              defaultValue={mainConfig.pipeline?.db?.username ?? ""}
            />
          </Field>
        </Fieldset>

        <br></br>
        <Fieldset>
          <Field>
            <Label htmlFor="password">Password</Label>
            <Input
              id="password"
              name="pipeline.db.password,metrics.password,env.db.password"
              type="password"
              placeholder=""
              onChange={handleInputChange}
              defaultValue={mainConfig.pipeline?.db?.password ?? ""}
            />
          </Field>
        </Fieldset>

        <br></br>
        <Fieldset>
          <Field>
            <Label htmlFor="ip">IP</Label>
            <Input
              id="ip"
              name="pipeline.db.ip,metrics.ip,env.db.ip"
              type="text"
              placeholder=""
              onChange={handleInputChange}
              defaultValue={mainConfig.pipeline?.db?.ip ?? ""}
            />
          </Field>
        </Fieldset>

        <br></br>
        <Fieldset>
          <Field>
            <Label htmlFor="port">Port</Label>
            <Input
              id="port"
              name="pipeline.db.port,metrics.port,env.db.port"
              type="number"
              placeholder=""
              onChange={handleInputChange}
              defaultValue={mainConfig.pipeline?.db?.port ?? ""}
            />
          </Field>
        </Fieldset>

        <br></br>
        <Fieldset>
          <Field>
            <Label htmlFor="db_name">Database Name</Label>
            <Input
              id="db_name"
              name="pipeline.db.db_name,metrics.db_name,env.db.db_name"
              type="text"
              placeholder=""
              onChange={handleInputChange}
              defaultValue={mainConfig.pipeline?.db?.db_name ?? ""}
            />
          </Field>
        </Fieldset>

        <br></br>
        <Fieldset>
          <Field>
            <Label htmlFor="table_schema">Table Schema</Label>
            <Input
              id="table_schema"
              name="pipeline.db.table_schema,metrics.table_schema,env.db.table_schema"
              type="text"
              placeholder=""
              onChange={handleInputChange}
              defaultValue={mainConfig.pipeline?.db?.table_schema ?? ""}
            />
          </Field>
        </Fieldset>

        <br></br>
        <Fieldset>
          <Field>
            <Label htmlFor="table_name">Table Name</Label>
            <Input
              id="table_name"
              name="pipeline.db.table_name,env.db.table_name"
              type="text"
              placeholder=""
              onChange={handleInputChange}
              defaultValue={mainConfig.pipeline?.db?.table_name ?? ""}
            />
          </Field>
        </Fieldset>

        <br></br>
        <Fieldset>
          <Field>
            <Label htmlFor="metrics_name">Metrics Name</Label>
            <Input
              id="metrics_name"
              name="metrics.name"
              type="text"
              placeholder=""
              onChange={handleInputChange}
              defaultValue={mainConfig.metrics?.name ?? ""}
            />
          </Field>
        </Fieldset>

        <br></br>
        <Fieldset>
          <Field>
            <Label htmlFor="metrics_table_name">Metrics Table Name</Label>
            <Input
              id="metrics_table_name"
              name="metrics.table_name"
              type="text"
              placeholder=""
              onChange={handleInputChange}
              defaultValue={metrics?.table_name ?? ""}
            />
          </Field>
        </Fieldset>

        <br></br>
        <Fieldset>
          <Field>
            <Label htmlFor="metrics_lo_wm">Metrics Low Water Mark</Label>
            <Input
              id="metrics_lo_wm"
              name="metrics.lo_wm"
              type="number"
              placeholder=""
              onChange={handleInputChange}
              defaultValue={metrics?.lo_wm ?? ""}
            />
          </Field>
        </Fieldset>
      </form>

      <form
        className="border border-gray-400 rounded-lg p-2 mb-2"
        onSubmit={handleSubmit}
      >
        <Field>
          <Heading level={3}>OPC</Heading>
        </Field>

        <Fieldset>
          <Field>
            <Label htmlFor="opc_conn_url">OPC Connection URL</Label>
            <Input
              id="opc_conn_url"
              name="env.opc_conn_url"
              type="text"
              placeholder=""
              onChange={handleInputChange}
              defaultValue={env?.opc_conn_url ?? ""}
            />
          </Field>
        </Fieldset>

        <Fieldset>
          <Field>
            <Label htmlFor="opc_ns">OPC Namespace</Label>
            <Input
              id="opc_ns"
              name="env.opc_ns"
              type="number"
              placeholder=""
              onChange={handleInputChange}
              defaultValue={env?.opc_ns ?? ""}
            />
          </Field>
        </Fieldset>
      </form>
      <SetupConfigNav />
    </div>
  );
}
