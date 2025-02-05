import { createContext } from "react";
import { type components } from "../api-schema";

export type DeepPartial<T> = T extends object
  ? {
      [P in keyof T]?: DeepPartial<T[P]>;
    }
  : T;

export const MainConfigContext = createContext<{
  mainConfig: DeepPartial<components["schemas"]["MainConfig"]>;
  setMainConfig: (
    value:
      | DeepPartial<components["schemas"]["MainConfig"]>
      | ((
          value: DeepPartial<components["schemas"]["MainConfig"]>,
        ) => DeepPartial<components["schemas"]["MainConfig"]>),
  ) => void;
}>({
  mainConfig: {},
  setMainConfig: () => {
    /* Does nothing, should define a setState compatible hook within a provider */
  },
});
