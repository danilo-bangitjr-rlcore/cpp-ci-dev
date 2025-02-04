import localforage from "localforage";
import { useCallback, useEffect, useState } from "react";

/**
 * React hook for using local forage persistence
 * @param key localstorage key
 * @param initialValue localstorage initial and default value
 * @returns [storedValue, setValue] where storedValue is the current value and setValue is a function to update the value
 */
export function useLocalForage<T>(
  key: string,
  initialValue: T,
): [T, (value: T | ((val: T) => T)) => void] {
  const [storedValue, setStoredValue] = useState<T>(initialValue);

  useEffect(() => {
    async function get() {
      try {
        const value = await localforage.getItem<T>(key);
        setStoredValue(value ?? initialValue);
      } catch (error) {
        console.error(error);
      }
    }
    void get();
  }, [key, initialValue]);

  // Return a wrapped version of useState's setter function that ...
  // ... persists the new value to localForage.
  const setValue = useCallback(
    (value: T | ((val: T) => T)) => {
      async function set() {
        // Allow value to be a function so we have the same API as useState
        const valueToStore =
          value instanceof Function ? value(storedValue) : value;
        try {
          await localforage.setItem(key, valueToStore);
          setStoredValue(value);
        } catch (error) {
          console.error(error);
        }
      }
      void set();
    },
    [key, storedValue],
  );

  return [storedValue, setValue] as const;
}
