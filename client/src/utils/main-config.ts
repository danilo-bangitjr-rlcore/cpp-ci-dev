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

  newMainConfig.env = { ...(newMainConfig.env ?? {}), discrete_control: false };
  newMainConfig.pipeline = {
    ...(newMainConfig.pipeline ?? {}),
    state_constructor: { defaults: [] },
    transition_creator: { name: "anytime" },
  };

  return newMainConfig;
};

/**
 * Helper context for forward and backwards during Main Config setup wizard
 */
export const MainConfigStepsContext = createContext<{
  nextStep: LinkComponentProps["to"] | null;
  prevStep: LinkComponentProps["to"] | null;
}>({ nextStep: null, prevStep: null });
