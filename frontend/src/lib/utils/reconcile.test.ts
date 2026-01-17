import { describe, it, expect } from "vitest";
import {
  reconcileArray,
  reconcileSet,
  reconcileMap,
  shallowEqual,
  deepEqual,
} from "./reconcile";

describe("shallowEqual", () => {
  it("returns true for identical objects", () => {
    const obj = { a: 1, b: "test" };
    expect(shallowEqual(obj, obj)).toBe(true);
  });

  it("returns true for equal shallow objects", () => {
    expect(shallowEqual({ a: 1, b: 2 }, { a: 1, b: 2 })).toBe(true);
    expect(shallowEqual({ x: "hello" }, { x: "hello" })).toBe(true);
  });

  it("returns false for different values", () => {
    expect(shallowEqual({ a: 1 }, { a: 2 })).toBe(false);
  });

  it("returns false for different keys", () => {
    expect(shallowEqual({ a: 1 }, { b: 1 } as unknown as { a: number })).toBe(
      false,
    );
  });

  it("returns false for different number of keys", () => {
    expect(
      shallowEqual({ a: 1 }, { a: 1, b: 2 } as unknown as { a: number }),
    ).toBe(false);
    expect(
      shallowEqual({ a: 1, b: 2 }, { a: 1 } as unknown as {
        a: number;
        b: number;
      }),
    ).toBe(false);
  });

  it("uses reference equality for nested objects", () => {
    const nested = { x: 1 };
    expect(shallowEqual({ a: nested }, { a: nested })).toBe(true);
    expect(shallowEqual({ a: { x: 1 } }, { a: { x: 1 } })).toBe(false);
  });
});

describe("deepEqual", () => {
  it("returns true for identical primitives", () => {
    expect(deepEqual(1, 1)).toBe(true);
    expect(deepEqual("a", "a")).toBe(true);
    expect(deepEqual(true, true)).toBe(true);
    expect(deepEqual(null, null)).toBe(true);
  });

  it("returns false for different primitives", () => {
    expect(deepEqual(1, 2)).toBe(false);
    expect(deepEqual("a", "b")).toBe(false);
  });

  it("returns true for equal shallow objects", () => {
    expect(deepEqual({ a: 1, b: 2 }, { a: 1, b: 2 })).toBe(true);
  });

  it("returns false for objects with different primitive values", () => {
    expect(deepEqual({ a: 1, b: 2 }, { a: 1, b: 3 })).toBe(false);
    expect(deepEqual({ x: "hello" }, { x: "world" })).toBe(false);
  });

  it("returns false for objects with different number of keys", () => {
    expect(deepEqual({ a: 1, b: 2 }, { a: 1 })).toBe(false);
    expect(deepEqual({ a: 1 }, { a: 1, b: 2 })).toBe(false);
  });

  it("returns true for equal nested objects (one level)", () => {
    expect(deepEqual({ a: { x: 1, y: 2 } }, { a: { x: 1, y: 2 } })).toBe(true);
  });

  it("returns false for different nested objects", () => {
    expect(deepEqual({ a: { x: 1, y: 2 } }, { a: { x: 1, y: 3 } })).toBe(false);
  });

  it("handles null values", () => {
    expect(deepEqual(null, { a: 1 })).toBe(false);
    expect(deepEqual({ a: 1 }, null)).toBe(false);
  });
});

describe("reconcileArray", () => {
  interface TestItem {
    id: number;
    name: string;
    value: number;
  }

  const options = {
    getKey: (item: TestItem) => item.id,
  };

  it("returns false when arrays are identical", () => {
    const target: TestItem[] = [
      { id: 1, name: "one", value: 1 },
      { id: 2, name: "two", value: 2 },
    ];
    const source: TestItem[] = [
      { id: 1, name: "one", value: 1 },
      { id: 2, name: "two", value: 2 },
    ];

    const changed = reconcileArray(target, source, options);

    expect(changed).toBe(false);
    expect(target).toHaveLength(2);
  });

  it("updates changed items in-place and preserves reference", () => {
    const original = { id: 1, name: "one", value: 1 };
    const target: TestItem[] = [original];
    const source: TestItem[] = [{ id: 1, name: "one updated", value: 10 }];

    const changed = reconcileArray(target, source, options);

    expect(changed).toBe(true);
    expect(target[0]).toBe(original); // Same reference preserved
    expect(target[0].name).toBe("one updated");
    expect(target[0].value).toBe(10);
  });

  it("adds new items", () => {
    const target: TestItem[] = [{ id: 1, name: "one", value: 1 }];
    const source: TestItem[] = [
      { id: 1, name: "one", value: 1 },
      { id: 2, name: "two", value: 2 },
    ];

    const changed = reconcileArray(target, source, options);

    expect(changed).toBe(true);
    expect(target).toHaveLength(2);
    expect(target[1]).toEqual({ id: 2, name: "two", value: 2 });
  });

  it("removes items not in source", () => {
    const target: TestItem[] = [
      { id: 1, name: "one", value: 1 },
      { id: 2, name: "two", value: 2 },
    ];
    const source: TestItem[] = [{ id: 1, name: "one", value: 1 }];

    const changed = reconcileArray(target, source, options);

    expect(changed).toBe(true);
    expect(target).toHaveLength(1);
    expect(target[0].id).toBe(1);
  });

  it("handles reordering", () => {
    const item1 = { id: 1, name: "one", value: 1 };
    const item2 = { id: 2, name: "two", value: 2 };
    const target: TestItem[] = [item1, item2];
    const source: TestItem[] = [
      { id: 2, name: "two", value: 2 },
      { id: 1, name: "one", value: 1 },
    ];

    const changed = reconcileArray(target, source, options);

    expect(changed).toBe(true);
    expect(target[0].id).toBe(2);
    expect(target[1].id).toBe(1);
    // References should be preserved
    expect(target[0]).toBe(item2);
    expect(target[1]).toBe(item1);
  });

  it("handles empty arrays", () => {
    const target: TestItem[] = [];
    const source: TestItem[] = [];

    const changed = reconcileArray(target, source, options);

    expect(changed).toBe(false);
    expect(target).toHaveLength(0);
  });

  it("handles clearing array", () => {
    const target: TestItem[] = [{ id: 1, name: "one", value: 1 }];
    const source: TestItem[] = [];

    const changed = reconcileArray(target, source, options);

    expect(changed).toBe(true);
    expect(target).toHaveLength(0);
  });

  it("handles populating empty array", () => {
    const target: TestItem[] = [];
    const source: TestItem[] = [
      { id: 1, name: "one", value: 1 },
      { id: 2, name: "two", value: 2 },
    ];

    const changed = reconcileArray(target, source, options);

    expect(changed).toBe(true);
    expect(target).toHaveLength(2);
  });

  it("uses custom isEqual function", () => {
    const target: TestItem[] = [{ id: 1, name: "one", value: 1 }];
    const source: TestItem[] = [{ id: 1, name: "ONE", value: 1 }]; // name changed

    // Custom isEqual that only compares id and value, not name
    const changed = reconcileArray(target, source, {
      getKey: (item) => item.id,
      isEqual: (a, b) => a.id === b.id && a.value === b.value,
    });

    // Should not report change since name is ignored
    expect(changed).toBe(false);
    expect(target[0].name).toBe("one"); // Original name preserved
  });

  it("handles complex update scenario", () => {
    const target: TestItem[] = [
      { id: 1, name: "one", value: 1 },
      { id: 2, name: "two", value: 2 },
      { id: 3, name: "three", value: 3 },
    ];
    const source: TestItem[] = [
      { id: 2, name: "two updated", value: 20 }, // updated and moved to first
      { id: 4, name: "four", value: 4 }, // new
      { id: 1, name: "one", value: 1 }, // moved to last
      // id: 3 removed
    ];

    const changed = reconcileArray(target, source, options);

    expect(changed).toBe(true);
    expect(target).toHaveLength(3);
    expect(target[0].id).toBe(2);
    expect(target[0].name).toBe("two updated");
    expect(target[0].value).toBe(20);
    expect(target[1].id).toBe(4);
    expect(target[2].id).toBe(1);
  });
});

describe("reconcileSet", () => {
  it("returns false when sets are identical", () => {
    const target = new Set([1, 2, 3]);

    const changed = reconcileSet(target, [1, 2, 3]);

    expect(changed).toBe(false);
    expect([...target]).toEqual([1, 2, 3]);
  });

  it("adds new items", () => {
    const target = new Set([1, 2]);

    const changed = reconcileSet(target, [1, 2, 3]);

    expect(changed).toBe(true);
    expect(target.has(3)).toBe(true);
    expect(target.size).toBe(3);
  });

  it("removes items not in source", () => {
    const target = new Set([1, 2, 3]);

    const changed = reconcileSet(target, [1, 2]);

    expect(changed).toBe(true);
    expect(target.has(3)).toBe(false);
    expect(target.size).toBe(2);
  });

  it("handles empty set", () => {
    const target = new Set<number>();

    const changed = reconcileSet(target, [1, 2]);

    expect(changed).toBe(true);
    expect(target.size).toBe(2);
  });

  it("handles clearing set", () => {
    const target = new Set([1, 2, 3]);

    const changed = reconcileSet(target, []);

    expect(changed).toBe(true);
    expect(target.size).toBe(0);
  });

  it("handles same items different order", () => {
    const target = new Set([1, 2, 3]);

    const changed = reconcileSet(target, [3, 2, 1]);

    expect(changed).toBe(false);
    expect(target.size).toBe(3);
  });
});

describe("reconcileMap", () => {
  it("returns false when maps are identical", () => {
    const target = new Map<number, string>([
      [1, "one"],
      [2, "two"],
    ]);

    const changed = reconcileMap(target, [
      [1, "one"],
      [2, "two"],
    ]);

    expect(changed).toBe(false);
  });

  it("adds new entries", () => {
    const target = new Map<number, string>([[1, "one"]]);

    const changed = reconcileMap(target, [
      [1, "one"],
      [2, "two"],
    ]);

    expect(changed).toBe(true);
    expect(target.get(2)).toBe("two");
    expect(target.size).toBe(2);
  });

  it("updates existing entries", () => {
    const target = new Map<number, string>([[1, "one"]]);

    const changed = reconcileMap(target, [[1, "ONE"]]);

    expect(changed).toBe(true);
    expect(target.get(1)).toBe("ONE");
  });

  it("removes entries not in source", () => {
    const target = new Map<number, string>([
      [1, "one"],
      [2, "two"],
    ]);

    const changed = reconcileMap(target, [[1, "one"]]);

    expect(changed).toBe(true);
    expect(target.has(2)).toBe(false);
    expect(target.size).toBe(1);
  });

  it("handles empty map", () => {
    const target = new Map<number, string>();

    const changed = reconcileMap(target, [
      [1, "one"],
      [2, "two"],
    ]);

    expect(changed).toBe(true);
    expect(target.size).toBe(2);
  });

  it("handles clearing map", () => {
    const target = new Map<number, string>([
      [1, "one"],
      [2, "two"],
    ]);

    const changed = reconcileMap(target, []);

    expect(changed).toBe(true);
    expect(target.size).toBe(0);
  });

  it("uses custom isEqual for values", () => {
    interface Value {
      x: number;
      y: number;
    }

    const target = new Map<number, Value>([[1, { x: 1, y: 1 }]]);

    // Custom isEqual that only compares x
    const changed = reconcileMap(
      target,
      [[1, { x: 1, y: 999 }]], // y changed but we ignore it
      (a, b) => a.x === b.x,
    );

    expect(changed).toBe(false);
  });

  it("handles object values with custom equality", () => {
    interface Status {
      error: string;
      details: string | null;
    }

    const target = new Map<number, Status>([
      [1, { error: "Error 1", details: "details" }],
    ]);

    // Same error/details, should not change
    const changed = reconcileMap(
      target,
      [[1, { error: "Error 1", details: "details" }]],
      (a, b) => a.error === b.error && a.details === b.details,
    );

    expect(changed).toBe(false);
  });
});
