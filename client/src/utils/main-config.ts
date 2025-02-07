import { LinkComponentProps } from "@tanstack/react-router";
import { createContext } from "react";
import { type components } from "../api-schema";

export type DeepPartial<T> = T extends object
  ? {
      [P in keyof T]?: DeepPartial<T[P]>;
    }
  : T;

export type DeepPartialMainConfig = DeepPartial<
  components["schemas"]["MainConfig"]
>;

export const MainConfigContext = createContext<{
  mainConfig: DeepPartialMainConfig;
  setMainConfig: (
    value:
      | DeepPartialMainConfig
      | ((val: DeepPartialMainConfig) => DeepPartialMainConfig),
  ) => void;
}>({
  mainConfig: {},
  setMainConfig: () => {
    /* Does nothing, should define a setState compatible hook within a provider */
  },
});

export const setValFromPath = (
  /**
   * Current limitations:
   * - Only works when value is number, bool, string values (Not arrays)
   * - Keys must be strings (i.e. my_obj.a.b.c, not tested for my_obj.a.b[1])
   */
  config: DeepPartialMainConfig,
  path: string,
  value: string,
  inputType: string,
): DeepPartialMainConfig => {
  const keys = path.split(".");
  const newMainConfig = structuredClone(config);
  let current = newMainConfig as Record<string, unknown>;

  for (let i = 0; i < keys.length - 1; i++) {
    const key = keys[i];

    if (typeof current[key] !== "object" || current[key] === null) {
      current[key] = {};
    }

    current = current[key] as Record<string, unknown>;
  }

  const lastKey = keys[keys.length - 1];

  let conv_value;

  if (inputType == "number") {
    conv_value = Number(value);
  } else {
    conv_value = value;
  }

  current[lastKey] = conv_value;
  return newMainConfig;
};

/**
 * Provide default values for our MainConfig inputs. This should be sufficient to generate our structurally valid MainConfig yaml.
 * Refer to: corerl/config.py for baseline structure pydantic model.
 * @param prevMainConfig
 * @returns
 */
export const loadMainConfigDefaults = (
  prevMainConfig: DeepPartialMainConfig = {},
) => {
  const newMainConfig = structuredClone(prevMainConfig);

  type DatabaseCommon = Omit<components["schemas"]["TagDBConfig"], "data_agg">;

  // shared default database stub
  const database_common: DatabaseCommon = {
    enabled: true,
    drivername: "postgresql+psycopg2",
    username: "postgres",
    password: "password",
    ip: "localhost",
    port: 5432,
    db_name: "postgres",
    table_schema: "public",
    table_name: "opc_ua",
  };

  // corerl/interaction/factory.py
  newMainConfig.interaction = {
    ...(newMainConfig.interaction ?? {}),
    name: "dep_interaction",
    action_period: "PT5M",
    obs_period: "PT5M",
  } as DeepPartial<components["schemas"]["DepInteractionConfig"]>;

  // corerl/eval/metrics.py
  newMainConfig.metrics = {
    ...(newMainConfig.metrics ?? {}),
    ...database_common,
    name: "db",
    table_name: "metrics",
    lo_wm: 5,
  } as DeepPartial<components["schemas"]["MetricsDBConfig"]>;

  // corerl/environment/async_env/factory.py
  // client only supports generating DepAsyncEnvConfig
  newMainConfig.env = {
    ...(newMainConfig.env ?? {}),
    discrete_control: false,
    db: database_common,
    opc_conn_url: "opc.tcp://admin@0.0.0.0:4840/rlcore/server/",
    opc_ns: 2,
    obs_period: "PT5M",
    update_period: "PT5M",
    action_period: "PT5M",
    action_tolerance: "PT5M",
  } as DeepPartial<components["schemas"]["DepAsyncEnvConfig"]>;

  // corerl/agent/__init__.py
  newMainConfig.agent = {
    name: "greedy_ac",
    n_critic_updates: 1,
    n_actor_updates: 1,
    n_sampler_updates: 1,
    num_samples: 128,
    uniform_sampling_percentage: 0.8,
    discrete_control: false,
    seed: 1,

    critic: {
      buffer: {
        name: "uniform",
        seed: 1,
      },
      critic_optimizer: {
        lr: 0.01,
      },
    },

    actor: {
      buffer: {
        name: "uniform",
        seed: 1,
      },
      actor_optimizer: {
        lr: 0.01,
      },
    },
  } as DeepPartial<components["schemas"]["GreedyACConfig"]>;

  // corerl/experiment/config.py
  newMainConfig.experiment = {
    ...(newMainConfig.experiment ?? {}),
    exp_name: "CoreRL_Experiment",
  } as DeepPartial<components["schemas"]["ExperimentConfig"]>;

  // corerl/data_pipeline/pipeline.py
  newMainConfig.pipeline = {
    ...(newMainConfig.pipeline ?? {}),
    db: { ...(newMainConfig.pipeline?.db ?? {}), ...database_common },
    state_constructor: { defaults: [] },
    transition_creator: { name: "anytime" },
  } as DeepPartial<components["schemas"]["PipelineConfig"]>;

  return newMainConfig;
};

/**
 * Helper context for forward and backwards during Main Config setup wizard
 */
export const MainConfigStepsContext = createContext<{
  nextStep: LinkComponentProps["to"] | null;
  prevStep: LinkComponentProps["to"] | null;
}>({ nextStep: null, prevStep: null });
