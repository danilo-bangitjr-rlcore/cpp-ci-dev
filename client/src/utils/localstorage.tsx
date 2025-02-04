import {
  useEffect,
  useState,
} from "react";
import localforage from "localforage";

/**
 * React hook for using local forage persistence
 * @param key localstorage key
 * @param initialValue localstorage initial and default value
 * @returns [storedValue, setValue] where storedValue is the current value in localstorage and setValue is a function to update the value in localstorage
 */
export function useLocalForage<T>(
  key: string,
  initialValue: T,
): [T, (value: T | ((val: T) => T)) => void] {
  const [storedValue, setStoredValue] = useState<T>(initialValue);

  useEffect(() => {
    (async function () {
      try {
        const value = await localforage.getItem<T>(key);
        if (value) {
          setStoredValue(value);
        }
      } catch (error) {
        console.error(error);
        return initialValue;
      }
    })();
  }, [initialValue, storedValue, key])

  // Return a wrapped version of useState's setter function that ...
  // ... persists the new value to localForage.
  const setValue = (value: T | ((val: T) => T)) => {
    (async function () {
      // Allow value to be a function so we have the same API as useState
      const valueToStore =
        value instanceof Function ? value(storedValue) : value;
      try {
        await localforage.setItem(key, valueToStore);
        setStoredValue(value);
      } catch (error) {
        console.error(error);
        return initialValue;
      }
    })()
  }

  return [storedValue, setValue];
}
