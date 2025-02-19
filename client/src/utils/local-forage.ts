import localforage from "localforage";
import { useCallback, useEffect, useState } from "react";

export function deepEquals(a: unknown, b: unknown) {
  if (a === b) return true;

  if (
    typeof a !== "object" ||
    a === null ||
    typeof b !== "object" ||
    b === null
  ) {
    return false;
  }

  const keysA = Object.keys(a);
  const keysB = Object.keys(b);

  if (keysA.length !== keysB.length) return false;

  for (const key of keysA) {
    if (
      !keysB.includes(key) ||
      !deepEquals(
        (a as Record<string, unknown>)[key],
        (b as Record<string, unknown>)[key],
      )
    ) {
      return false;
    }
  }

  return true;
}

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
  const [initialLoaded, setInitialLoaded] = useState(false);

  useEffect(() => {
    async function get() {
      if (initialLoaded) {
        return;
      }
      try {
        const value = await localforage.getItem<T>(key);
        if (value == null) {
          setStoredValue(initialValue);
        } else if (!deepEquals(value, storedValue)) {
          setStoredValue(value);
        }
      } catch (error) {
        console.error(error);
      } finally {
        setInitialLoaded(true);
      }
    }
    void get();
  }, [key, storedValue, initialValue, initialLoaded, setInitialLoaded]);

  // Return a wrapped version of useState's setter function that persists the new value to localForage.
  const setValue = useCallback(
    (value: T | ((val: T) => T)) => {
      async function set() {
        // Allow value to be a function so we have the same API as useState
        const valueToStore =
          value instanceof Function ? value(storedValue) : value;
        // console.log(valueToStore)
        try {
          setStoredValue(valueToStore);
          await localforage.setItem(key, valueToStore);
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
