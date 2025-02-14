import { createFileRoute } from "@tanstack/react-router";
import { useContext, useState } from "react";
import createClient from "openapi-react-query";
import { type components } from "../../api-schema";
import { Badge } from "../../components/badge";
import DurationInput from "../../components/duration";
import { Field, FieldGroup, Fieldset, Label } from "../../components/fieldset";
import { Heading } from "../../components/heading";
import { Input } from "../../components/input";
import { SetupConfigNav } from "../../components/setup/setup-config-nav";
import { Text } from "../../components/text";
import { Button } from "../../components/button";
import {
  type DeepPartial,
  type DeepPartialMainConfig,
  MainConfigContext,
  setValFromPath,
} from "../../utils/main-config";
import { getApiFetchClient } from "../../utils/api";

export const Route = createFileRoute("/setup/general_config")({
  component: GeneralConfig,
});

function GeneralConfig() {
  const { mainConfig, setMainConfig } = useContext(MainConfigContext);
  const [dbStatus, setDBStatus] = useState<
    components["schemas"]["DB_Status_Response"]
  >({
    db_status: false,
    table_status: false,
    has_connected: false,
  });

  const [opcStatus, setOPCStatus] = useState<
    components["schemas"]["OPC_Status_Response"]
  >({
    opc_status: false,
    has_connected: false,
  });

  type PartialDepAsyncEnvConfig = DeepPartial<
    components["schemas"]["DepAsyncEnvConfig"]
  >;

  type PartialMetricsDBConfig = DeepPartial<
    components["schemas"]["MetricsDBConfig"]
  >;

  const handleDurationChange =
    (path: string | string[]) => (isoDuration: string) => {
      setMainConfig((prevMainConfig: DeepPartialMainConfig) => {
        const paths = [];
        if (typeof path === "string") {
          paths.push(path);
        } else {
          paths.push(...path);
        }
        let newConfig = structuredClone(prevMainConfig);
        for (const p of paths) {
          newConfig = setValFromPath(newConfig, p, isoDuration, "duration");
        }
        return newConfig;
      });
    };

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

  const clearDB = () =>
    setDBStatus({
      db_status: false,
      table_status: false,
      has_connected: false,
    });

  const clearOPC = () =>
    setOPCStatus({
      opc_status: false,
      has_connected: false,
    });

  const client = createClient(getApiFetchClient());
  const { mutate: verifyConnectionDB, isPending: isPendingDB } =
    client.useMutation("post", "/api/verify-connection/db", {
      onSuccess: (data) => {
        setDBStatus(data);
      },
    });

  const { mutate: verifyConnectionOPC, isPending: isPendingOPC } =
    client.useMutation("post", "/api/verify-connection/opc", {
      onSuccess: (data) => {
        setOPCStatus(data);
      },
    });

  const checkDBStatus = () => {
    function areKeysDefined<T extends object>(
      obj: T,
      keys: (keyof T)[],
    ): boolean {
      return keys.every((key) => obj[key] !== undefined);
    }

    const keysToCheck: (keyof components["schemas"]["DBConfig"])[] = [
      "drivername",
      "username",
      "password",
      "ip",
      "port",
      "db_name",
    ];

    if (
      mainConfig.infra?.db !== undefined &&
      areKeysDefined(mainConfig.infra.db, keysToCheck)
    ) {
      console.log("All keys defined");
      verifyConnectionDB({
        body: {
          db_config: mainConfig.infra.db as components["schemas"]["DBConfig"],
          table_name:
            (mainConfig.env as PartialDepAsyncEnvConfig)?.db?.table_name ?? "",
        },
      });
    } else {
      alert("Make sure to fill in all the required database fields");
    }
  };

  const getDBLabel = () =>
    isPendingDB
      ? "Loading..."
      : dbStatus.has_connected
        ? dbStatus.db_status
          ? "Connected!"
          : "Couldn't connect"
        : "No status";

  const getDBStyle = () =>
    dbStatus.has_connected && !isPendingDB
      ? dbStatus.db_status
        ? "green"
        : "red"
      : "zinc";

  const getTableLabel = () =>
    isPendingDB
      ? "Loading..."
      : dbStatus.has_connected
        ? dbStatus.table_status
          ? "Table found!"
          : "Couldn't find table"
        : "No status";

  const getTableStyle = () =>
    dbStatus.has_connected && !isPendingDB
      ? dbStatus.table_status
        ? "green"
        : "red"
      : "zinc";

  const checkOPCStatus = () => {
    const opcUrl = (mainConfig.env as PartialDepAsyncEnvConfig)?.opc_conn_url;
    if (opcUrl) {
      const opc_req: components["schemas"]["OPC_Status_Request"] = {
        opc_url: opcUrl,
      };
      verifyConnectionOPC({ body: opc_req });
    } else {
      alert("Make sure to fill in all the required OPC fields");
    }
  };

  const getOPCLabel = () =>
    isPendingOPC
      ? "Loading..."
      : opcStatus.has_connected
        ? opcStatus.opc_status
          ? "Connected!"
          : "Couldn't connect"
        : "No status";

  const getOPCStyle = () =>
    opcStatus.has_connected && !isPendingOPC
      ? opcStatus.opc_status
        ? "green"
        : "red"
      : "zinc";

  return (
    <div className="p-2">
      <Fieldset className="border border-gray-400 rounded-lg p-2 mb-2">
        <Field>
          <Heading level={3}>Experiment</Heading>
        </Field>

        <FieldGroup className="mb-1">
          <Field>
            <Label htmlFor="exp_name">Experiment Name</Label>
            <Input
              id="exp_name"
              name="experiment.exp_name"
              type="text"
              placeholder=""
              onChange={handleInputChange}
              value={mainConfig.experiment?.exp_name ?? ""}
            />
          </Field>
        </FieldGroup>
      </Fieldset>

      <Fieldset className="border border-gray-400 rounded-lg p-2 mb-2">
        <Field>
          <Heading level={3}>Database</Heading>
        </Field>

        <FieldGroup className="mb-1">
          <Field>
            <Label htmlFor="drivername">Driver name</Label>
            <Input
              id="drivername"
              name="infra.db.drivername"
              type="text"
              placeholder=""
              onChange={(e: React.ChangeEvent<HTMLInputElement>) => {
                handleInputChange(e);
                clearDB();
              }}
              value={mainConfig.infra?.db?.drivername ?? ""}
            />
          </Field>
        </FieldGroup>

        <FieldGroup className="mb-1">
          <Field>
            <Label htmlFor="username">Username</Label>
            <Input
              id="username"
              name="infra.db.username"
              type="text"
              placeholder=""
              onChange={(e: React.ChangeEvent<HTMLInputElement>) => {
                handleInputChange(e);
                clearDB();
              }}
              value={mainConfig.infra?.db?.username ?? ""}
            />
          </Field>
        </FieldGroup>

        <FieldGroup className="mb-1">
          <Field>
            <Label htmlFor="password">Password</Label>
            <Input
              id="password"
              name="infra.db.password"
              type="password"
              autoComplete="off"
              placeholder=""
              onChange={(e: React.ChangeEvent<HTMLInputElement>) => {
                handleInputChange(e);
                clearDB();
              }}
              value={mainConfig.infra?.db?.password ?? ""}
            />
          </Field>
        </FieldGroup>

        <FieldGroup className="mb-1">
          <Field>
            <Label htmlFor="ip">IP</Label>
            <Input
              id="ip"
              name="infra.db.ip"
              type="text"
              placeholder=""
              onChange={(e: React.ChangeEvent<HTMLInputElement>) => {
                handleInputChange(e);
                clearDB();
              }}
              value={mainConfig.infra?.db?.ip ?? ""}
            />
          </Field>
        </FieldGroup>

        <FieldGroup className="mb-1">
          <Field>
            <Label htmlFor="port">Port</Label>
            <Input
              id="port"
              name="infra.db.port"
              type="number"
              placeholder=""
              onChange={(e: React.ChangeEvent<HTMLInputElement>) => {
                handleInputChange(e);
                clearDB();
              }}
              value={mainConfig.infra?.db?.port ?? ""}
            />
          </Field>
        </FieldGroup>

        <FieldGroup className="mb-1">
          <Field>
            <Label htmlFor="db_name">Database Name</Label>
            <Input
              id="db_name"
              name="infra.db.db_name"
              type="text"
              placeholder=""
              onChange={(e: React.ChangeEvent<HTMLInputElement>) => {
                handleInputChange(e);
                clearDB();
              }}
              value={mainConfig.infra?.db?.db_name ?? ""}
            />
          </Field>
        </FieldGroup>

        <FieldGroup className="mb-1">
          <Field>
            <Label htmlFor="table_name">Table Name</Label>
            <Input
              id="table_name"
              name="env.db.table_name"
              type="text"
              placeholder=""
              onChange={(e: React.ChangeEvent<HTMLInputElement>) => {
                handleInputChange(e);
                clearDB();
              }}
              value={
                (mainConfig.env as PartialDepAsyncEnvConfig)?.db?.table_name ??
                ""
              }
            />
          </Field>
        </FieldGroup>

        <FieldGroup className="mb-1">
          <Field>
            <Label htmlFor="metrics_table_name">Metrics Table Name</Label>
            <Input
              id="metrics_table_name"
              name="metrics.table_name"
              type="text"
              placeholder=""
              onChange={handleInputChange}
              value={
                (mainConfig.metrics as PartialMetricsDBConfig)?.table_name ?? ""
              }
            />
          </Field>
        </FieldGroup>

        <FieldGroup className="mb-1">
          <Field>
            <Label htmlFor="metrics_lo_wm">Metrics Low Water Mark</Label>
            <Input
              id="metrics_lo_wm"
              name="metrics.lo_wm"
              type="number"
              placeholder=""
              onChange={handleInputChange}
              value={
                (mainConfig.metrics as PartialMetricsDBConfig)?.lo_wm ?? ""
              }
            />
          </Field>
        </FieldGroup>

        <FieldGroup className="grid grid-cols-1 sm:grid-cols-2 sm:gap-x-4 sm:gap-y-0">
          <Field className="m-0 p-0">
            <Label className="mr-2">Check database connection:</Label>
            <Button onClick={checkDBStatus}>Check</Button>
          </Field>

          <Field className="m-0 p-0">
            <Label className="mr-2">Database status:</Label>
            <Badge color={getDBStyle()} className="mr-1">
              {getDBLabel()}
            </Badge>
          </Field>

          <Field className="sm:row-start-2 sm:col-start-2">
            <Label className="mr-8">Table status:</Label>
            <Badge color={getTableStyle()} className="mr-1">
              {getTableLabel()}
            </Badge>
          </Field>
        </FieldGroup>
      </Fieldset>

      <Fieldset className="border border-gray-400 rounded-lg p-2 mb-2">
        <Field>
          <Heading level={3}>OPC</Heading>
        </Field>

        <FieldGroup className="mb-1">
          <Field>
            <Label htmlFor="opc_conn_url">OPC Connection URL</Label>
            <Input
              id="opc_conn_url"
              name="env.opc_conn_url"
              type="text"
              placeholder=""
              onChange={(e: React.ChangeEvent<HTMLInputElement>) => {
                handleInputChange(e);
                clearOPC();
              }}
              value={
                (mainConfig.env as PartialDepAsyncEnvConfig)?.opc_conn_url ?? ""
              }
            />
          </Field>
        </FieldGroup>

        <FieldGroup className="mb-1">
          <Field>
            <Label htmlFor="opc_ns">OPC Namespace</Label>
            <Input
              id="opc_ns"
              name="env.opc_ns"
              type="number"
              placeholder=""
              onChange={handleInputChange}
              value={(mainConfig.env as PartialDepAsyncEnvConfig)?.opc_ns ?? ""}
            />
          </Field>
        </FieldGroup>

        <FieldGroup className="grid grid-cols-1 sm:grid-cols-2 sm:gap-x-4 sm:gap-y-0">
          <Field className="m-0 p-0">
            <Label className="mr-2">Check OPC connection:</Label>
            <Button onClick={checkOPCStatus}>Check</Button>
          </Field>

          <Field className="m-0 p-0">
            <Label className="mr-2">OPC status:</Label>
            <Badge color={getOPCStyle()} className="mr-1">
              {getOPCLabel()}
            </Badge>
          </Field>
        </FieldGroup>
      </Fieldset>

      <Fieldset className="border border-gray-400 rounded-lg p-2 mb-2">
        <Field>
          <Heading level={3}>Duration</Heading>
        </Field>

        <FieldGroup className="mb-1">
          <Field>
            <Label>Environment Observation Period</Label>
            <Text>
              How much time should pass in-between sensor readings?
              <Badge className="ml-1">{mainConfig.env?.obs_period}</Badge>
            </Text>
            <DurationInput
              onChange={handleDurationChange([
                "env.obs_period",
                "interaction.obs_period",
              ])}
              defaultValue={mainConfig.env?.obs_period ?? ""}
            />
          </Field>
        </FieldGroup>

        <FieldGroup className="mb-1">
          <Field>
            <Label>Agent Update Period</Label>
            <Text>
              How often should the agent perform a learning update?
              <Badge className="ml-1">{mainConfig.env?.update_period}</Badge>
            </Text>
            <DurationInput
              onChange={handleDurationChange("env.update_period")}
              defaultValue={mainConfig.env?.update_period ?? ""}
            />
          </Field>
        </FieldGroup>

        <FieldGroup className="mb-1">
          <Field>
            <Label>Action Period</Label>
            <Text>
              How often should the agent emit a new set point?
              <Badge className="ml-1">{mainConfig.env?.action_period}</Badge>
            </Text>
            <DurationInput
              onChange={handleDurationChange([
                "env.action_period",
                "interaction.action_period",
              ])}
              defaultValue={mainConfig.env?.action_period ?? ""}
            />
          </Field>
        </FieldGroup>

        <FieldGroup className="mb-1">
          <Field>
            <Label>Observation Staleness Tolearance</Label>
            <Text>
              How old can an observation be to reliably use for taking actions?
              <Badge className="ml-1">
                {(mainConfig.env as PartialDepAsyncEnvConfig)?.action_tolerance}
              </Badge>
            </Text>
            <DurationInput
              onChange={handleDurationChange("env.action_tolerance")}
              defaultValue={
                (mainConfig.env as PartialDepAsyncEnvConfig)
                  ?.action_tolerance ?? ""
              }
            />
          </Field>
        </FieldGroup>
      </Fieldset>
      <SetupConfigNav />
    </div>
  );
}
