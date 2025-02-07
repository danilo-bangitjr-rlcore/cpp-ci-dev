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

export const loadMainConfigHiddenDefaults = (
  prevMainConfig: DeepPartialMainConfig,
) => {
  const newMainConfig = structuredClone(prevMainConfig);

  // Hidden defaults
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
  };

  newMainConfig.metrics = {
    ...(newMainConfig.metrics ?? {}),
    name: "db",
    enabled: true,
  };
  newMainConfig.env = { ...(newMainConfig.env ?? {}), discrete_control: false };
  newMainConfig.interaction = {
    ...(newMainConfig.interaction ?? {}),
    name: "dep_interaction",
  };
  newMainConfig.pipeline = {
    ...(newMainConfig.pipeline ?? {}),
    state_constructor: { defaults: [] },
    transition_creator: { name: "anytime" },
  };

  // User modifiable defaults
  newMainConfig.experiment = {
    ...(newMainConfig.experiment ?? {}),
    exp_name: "CoreRL_Experiment",
  };
  type DeepPartialDatabase = DeepPartial<components["schemas"]["TagDBConfig"]>;
  const database_common: DeepPartialDatabase = {
    drivername: "postgresql+psycopg2",
    username: "postgres",
    password: "password",
    ip: "localhost",
    port: 5432,
    db_name: "postgres",
    table_schema: "public",
  };

  newMainConfig.pipeline.db = {
    ...(newMainConfig.pipeline.db ?? {}),
    ...database_common,
    table_name: "opc_ua",
  };
  newMainConfig.metrics = {
    ...(newMainConfig.metrics ?? {}),
    ...database_common,
    table_name: "metrics",
    lo_wm: 5,
  };

  let env = newMainConfig.env as DeepPartial<
    components["schemas"]["DepAsyncEnvConfig"]
  >;

  env = {
    ...(env ?? {}),
    db: database_common,
    opc_conn_url: "opc.tcp://admin@0.0.0.0:4840/rlcore/server/",
    opc_ns: 2,
    obs_period: "PT5M",
    update_period: "PT5M",
    action_period: "PT5M",
    action_tolerance: "PT5M",
  };

  newMainConfig.env = env;
  return newMainConfig;
};

/**
 * Helper context for forward and backwards during Main Config setup wizard
 */
export const MainConfigStepsContext = createContext<{
  nextStep: LinkComponentProps["to"] | null;
  prevStep: LinkComponentProps["to"] | null;
}>({ nextStep: null, prevStep: null });
