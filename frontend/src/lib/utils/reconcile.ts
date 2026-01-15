/**
 * Reconciliation utilities for efficient reactive state updates in Svelte 5.
 *
 * These utilities compare data and mutate arrays/collections in-place,
 * leveraging Svelte 5's fine-grained reactivity through proxies.
 * Only items that have actually changed trigger re-renders.
 */

/**
 * Options for array reconciliation
 */
export interface ReconcileArrayOptions<T> {
  /** Function to extract a unique key from an item */
  getKey: (item: T) => string | number;
  /** Optional function to check if two items are equal */
  isEqual?: (a: T, b: T) => boolean;
}

/**
 * Shallow equality check for objects.
 * Compares all enumerable own properties.
 */
export function shallowEqual<T extends object>(a: T, b: T): boolean {
  const keysA = Object.keys(a) as (keyof T)[];
  const keysB = Object.keys(b) as (keyof T)[];

  if (keysA.length !== keysB.length) return false;

  for (const key of keysA) {
    if (a[key] !== b[key]) return false;
  }
  return true;
}

/**
 * Reconcile an existing array with new data, mutating in-place.
 * Only updates items that have actually changed.
 *
 * @param target - The existing reactive array to update
 * @param source - The new data from the API
 * @param options - Configuration for key extraction and equality checking
 * @returns true if any changes were made, false otherwise
 *
 * @example
 * ```ts
 * const changed = reconcileArray(this.servers, newServers, {
 *   getKey: (s) => s.profile_id,
 *   isEqual: (a, b) => a.pid === b.pid && a.memory_mb === b.memory_mb
 * });
 * ```
 */
export function reconcileArray<T extends object>(
  target: T[],
  source: T[],
  options: ReconcileArrayOptions<T>,
): boolean {
  const { getKey, isEqual = shallowEqual } = options;

  // Build a map of existing items by key for O(1) lookup
  const existingByKey = new Map<string | number, { item: T; index: number }>();
  target.forEach((item, index) => {
    existingByKey.set(getKey(item), { item, index });
  });

  // Build set of keys in source for removal detection
  const sourceKeys = new Set(source.map(getKey));

  // Track the new order of items
  const newOrder: T[] = [];
  let changed = false;

  for (const sourceItem of source) {
    const key = getKey(sourceItem);
    const existing = existingByKey.get(key);

    if (existing) {
      // Item exists - check if it needs updating
      if (!isEqual(existing.item, sourceItem)) {
        // Update properties in-place (Svelte 5 proxies detect this)
        Object.assign(existing.item, sourceItem);
        changed = true;
      }
      newOrder.push(existing.item);
    } else {
      // New item - add it
      newOrder.push(sourceItem);
      changed = true;
    }
  }

  // Check for removals
  for (const [key] of existingByKey) {
    if (!sourceKeys.has(key)) {
      changed = true;
      break;
    }
  }

  // Check if order changed (even if no additions/removals/updates)
  if (!changed && target.length === newOrder.length) {
    for (let i = 0; i < target.length; i++) {
      if (getKey(target[i]) !== getKey(newOrder[i])) {
        changed = true;
        break;
      }
    }
  }

  // Only update the array if there were actual changes
  if (changed || target.length !== newOrder.length) {
    target.length = 0;
    target.push(...newOrder);
    return true;
  }

  return false;
}

/**
 * Update a reactive Set in-place, only modifying what changed.
 *
 * @param target - The existing reactive Set
 * @param values - The new values to set
 * @returns true if any changes were made
 *
 * @example
 * ```ts
 * const changed = reconcileSet(this.startingProfiles, [1, 2, 3]);
 * ```
 */
export function reconcileSet<T>(target: Set<T>, values: Iterable<T>): boolean {
  const newValues = new Set(values);
  let changed = false;

  // Remove items not in new set
  for (const item of target) {
    if (!newValues.has(item)) {
      target.delete(item);
      changed = true;
    }
  }

  // Add items not in current set
  for (const item of newValues) {
    if (!target.has(item)) {
      target.add(item);
      changed = true;
    }
  }

  return changed;
}

/**
 * Update a reactive Map in-place, only modifying what changed.
 *
 * @param target - The existing reactive Map
 * @param entries - The new entries to set
 * @param isEqual - Optional equality function for values
 * @returns true if any changes were made
 *
 * @example
 * ```ts
 * const changed = reconcileMap(this.failedProfiles, newFailures, (a, b) =>
 *   a.error === b.error && a.details === b.details
 * );
 * ```
 */
export function reconcileMap<K, V>(
  target: Map<K, V>,
  entries: Iterable<[K, V]>,
  isEqual: (a: V, b: V) => boolean = (a, b) => a === b,
): boolean {
  const newEntries = new Map(entries);
  let changed = false;

  // Remove keys not in new map
  for (const key of target.keys()) {
    if (!newEntries.has(key)) {
      target.delete(key);
      changed = true;
    }
  }

  // Add or update entries
  for (const [key, value] of newEntries) {
    const existing = target.get(key);
    if (existing === undefined) {
      target.set(key, value);
      changed = true;
    } else if (!isEqual(existing, value)) {
      target.set(key, value);
      changed = true;
    }
  }

  return changed;
}

/**
 * Deep equality check for objects (one level deep).
 * Useful for comparing items with nested objects.
 */
export function deepEqual<T>(a: T, b: T): boolean {
  if (a === b) return true;
  if (a == null || b == null) return false;
  if (typeof a !== "object" || typeof b !== "object") return a === b;

  const objA = a as Record<string, unknown>;
  const objB = b as Record<string, unknown>;
  const keysA = Object.keys(objA);
  const keysB = Object.keys(objB);

  if (keysA.length !== keysB.length) return false;

  for (const key of keysA) {
    const valA = objA[key];
    const valB = objB[key];

    if (
      typeof valA === "object" &&
      valA !== null &&
      typeof valB === "object" &&
      valB !== null
    ) {
      if (!shallowEqual(valA as object, valB as object)) return false;
    } else if (valA !== valB) {
      return false;
    }
  }

  return true;
}
