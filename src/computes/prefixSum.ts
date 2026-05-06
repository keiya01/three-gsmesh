import { BufferAttribute } from "three/webgpu";
import {
  Fn,
  If,
  invocationLocalIndex,
  Loop,
  storage,
  uint,
  workgroupArray,
  workgroupBarrier,
  workgroupId,
} from "three/tsl";
import type { ComputeNode, StorageBufferNode } from "three/webgpu";

export type PrefixLevel = {
  reduce: ComputeNode;
  add: ComputeNode | null;
};

function wgElement(buffer: ReturnType<typeof workgroupArray>, index: any) {
  return buffer.element(index) as any;
}

type PrefixComputeOptions = {
  items: StorageBufferNode<"uint">;
  blockSums: StorageBufferNode<"uint">;
  count: number;
  workgroupSize: number;
  level: number;
};

function createPrefixReduceCompute({
  items,
  blockSums,
  count,
  workgroupSize,
  level,
}: PrefixComputeOptions): ComputeNode {
  const itemsPerWorkgroup = workgroupSize * 2;
  const temp = workgroupArray("uint", itemsPerWorkgroup);
  const pairCount = Math.ceil(count / 2);

  return Fn(() => {
    const tid = invocationLocalIndex;
    const local0 = tid.mul(2);
    const local1 = local0.add(1);
    const item0 = workgroupId.x.mul(uint(itemsPerWorkgroup)).add(local0);
    const item1 = item0.add(1);
    const v0 = uint(0).toVar();
    const v1 = uint(0).toVar();

    If(item0.lessThan(uint(count)), () => {
      v0.assign(items.element(item0));
    });

    If(item1.lessThan(uint(count)), () => {
      v1.assign(items.element(item1));
    });

    wgElement(temp, local0).assign(v0);
    wgElement(temp, local1).assign(v1);

    const offset = uint(1).toVar();

    // Up-sweep (reduce) phase
    Loop(
      { start: uint(itemsPerWorkgroup >> 1), end: uint(0), type: "uint", condition: ">", update: ">>= 1" },
      ({ i }) => {
        workgroupBarrier();

        If(tid.lessThan(i), () => {
          const ai = offset.mul(local0.add(1)).sub(1);
          const bi = offset.mul(local0.add(2)).sub(1);

          wgElement(temp, bi).addAssign(wgElement(temp, ai));
        });

        offset.mulAssign(2);
      },
    );

    // Save workgroup sum and clear last element
    If(tid.equal(0), () => {
      const last = uint(itemsPerWorkgroup - 1);

      blockSums.element(workgroupId.x).assign(wgElement(temp, last));
      wgElement(temp, last).assign(0);
    });

    // Down-sweep phase
    Loop(
      { start: uint(1), end: uint(itemsPerWorkgroup), type: "uint", condition: "<", update: "<<= 1" },
      ({ i }) => {
        offset.divAssign(2);
        workgroupBarrier();

        If(tid.lessThan(i), () => {
          const ai = offset.mul(local0.add(1)).sub(1);
          const bi = offset.mul(local0.add(2)).sub(1);
          const t = wgElement(temp, ai).toVar();

          wgElement(temp, ai).assign(wgElement(temp, bi));
          wgElement(temp, bi).addAssign(t);
        });
      },
    );
    workgroupBarrier();

    If(item0.lessThan(uint(count)), () => {
      items.element(item0).assign(wgElement(temp, local0));
    });

    If(item1.lessThan(uint(count)), () => {
      items.element(item1).assign(wgElement(temp, local1));
    });
  })()
    .compute(pairCount, [workgroupSize])
    .setName(`Prefix Reduce ${level}`);
}

function createPrefixAddCompute({
  items,
  blockSums,
  count,
  workgroupSize,
  level,
}: PrefixComputeOptions): ComputeNode {
  const itemsPerWorkgroup = workgroupSize * 2;
  const pairCount = Math.ceil(count / 2);

  return Fn(() => {
    const tid = invocationLocalIndex;
    const item0 = workgroupId.x.mul(uint(itemsPerWorkgroup)).add(tid.mul(2));
    const item1 = item0.add(1);
    const blockSum = blockSums.element(workgroupId.x);

    If(item0.lessThan(uint(count)), () => {
      items.element(item0).addAssign(blockSum);
    });

    If(item1.lessThan(uint(count)), () => {
      items.element(item1).addAssign(blockSum);
    });
  })()
    .compute(pairCount, [workgroupSize])
    .setName(`Prefix Add ${level}`);
}

export function createPrefixSumLevels(
  items: StorageBufferNode<"uint">,
  count: number,
  workgroupSize: number,
): PrefixLevel[] {
  const levels: PrefixLevel[] = [];
  const itemsPerWorkgroup = workgroupSize * 2;
  let levelItems = items;
  let levelCount = count;
  let level = 0;

  while (levelCount > 1) {
    const blockCount = Math.ceil(levelCount / itemsPerWorkgroup);
    const nextItems = storage(
      new BufferAttribute(new Uint32Array(blockCount), 1),
      "uint",
      blockCount,
    );

    levels.push({
      reduce: createPrefixReduceCompute({
        items: levelItems,
        blockSums: nextItems,
        count: levelCount,
        workgroupSize,
        level,
      }),
      add: blockCount > 1
        ? createPrefixAddCompute({
          items: levelItems,
          blockSums: nextItems,
          count: levelCount,
          workgroupSize,
          level,
        })
        : null,
    });

    levelItems = nextItems;
    levelCount = blockCount;
    level++;
  }

  return levels;
}
