import type ComputeNode from "three/src/nodes/gpgpu/ComputeNode.js";
import type StorageBufferNode from "three/src/nodes/accessors/StorageBufferNode.js";
import {
  Fn,
  If,
  invocationLocalIndex,
  uint,
  workgroupArray,
  workgroupBarrier,
  workgroupId,
} from "three/tsl";
import type { Node } from "three/webgpu";

export const RADIX_BITS_PER_PASS = 2;
export const RADIX_BUCKETS = 1 << RADIX_BITS_PER_PASS;
export const RADIX_ITEMS_PER_THREAD = 1;

function wgElement(buffer: ReturnType<typeof workgroupArray>, index: Node<"uint">) {
  return buffer.element(index) as Node<"uint">;
}

type RadixBlockComputeOptions = {
  inputKey: StorageBufferNode<"uint">;
  localPrefix: StorageBufferNode<"uint">;
  blockSums: StorageBufferNode<"uint">;
  count: number;
  workgroupSize: number;
  workgroupCount: number;
  bit: number;
};

type RadixReorderComputeOptions = {
  inputKey: StorageBufferNode<"uint">;
  inputIndex: StorageBufferNode<"uint">;
  outputKey: StorageBufferNode<"uint">;
  outputIndex: StorageBufferNode<"uint">;
  localPrefix: StorageBufferNode<"uint">;
  prefixBlockSums: StorageBufferNode<"uint">;
  count: number;
  workgroupSize: number;
  workgroupCount: number;
  bit: number;
};

function assignBucketFlags(
  bucket0: ReturnType<typeof workgroupArray>,
  bucket1: ReturnType<typeof workgroupArray>,
  bucket2: ReturnType<typeof workgroupArray>,
  bucket3: ReturnType<typeof workgroupArray>,
  localIndex: Node<"uint">,
  active: Node<"bool">,
  bucket: Node<"uint">,
) {
  wgElement(bucket0, localIndex).assign(active.and(bucket.equal(0)).select(uint(1), uint(0)));
  wgElement(bucket1, localIndex).assign(active.and(bucket.equal(1)).select(uint(1), uint(0)));
  wgElement(bucket2, localIndex).assign(active.and(bucket.equal(2)).select(uint(1), uint(0)));
  wgElement(bucket3, localIndex).assign(active.and(bucket.equal(3)).select(uint(1), uint(0)));
}

function assignLocalPrefix(
  bucket0: ReturnType<typeof workgroupArray>,
  bucket1: ReturnType<typeof workgroupArray>,
  bucket2: ReturnType<typeof workgroupArray>,
  bucket3: ReturnType<typeof workgroupArray>,
  localIndex: Node<"uint">,
  bucket: Node<"uint">,
  prefix: Node<"uint">,
) {
  If(bucket.equal(0), () => {
    prefix.assign(wgElement(bucket0, localIndex));
  }).ElseIf(bucket.equal(1), () => {
    prefix.assign(wgElement(bucket1, localIndex));
  }).ElseIf(bucket.equal(2), () => {
    prefix.assign(wgElement(bucket2, localIndex));
  }).Else(() => {
    prefix.assign(wgElement(bucket3, localIndex));
  });
}

function scanBucketLane(
  src: ReturnType<typeof workgroupArray>,
  dst: ReturnType<typeof workgroupArray>,
  localIndex: Node<"uint">,
  offset: Node<"uint">,
) {
  const value = wgElement(src, localIndex).toVar();

  If(localIndex.greaterThanEqual(offset), () => {
    value.addAssign(wgElement(src, localIndex.sub(offset)));
  });

  wgElement(dst, localIndex).assign(value);
}

export function createRadixBlockCompute({
  inputKey,
  localPrefix,
  blockSums,
  count,
  workgroupSize,
  workgroupCount,
  bit,
}: RadixBlockComputeOptions): ComputeNode {
  const itemsPerWorkgroup = workgroupSize * RADIX_ITEMS_PER_THREAD;
  const scanStageCount = Math.ceil(Math.log2(itemsPerWorkgroup));
  const dispatchCount = Math.ceil(count / RADIX_ITEMS_PER_THREAD);
  const bucket0 = workgroupArray("uint", itemsPerWorkgroup);
  const bucket1 = workgroupArray("uint", itemsPerWorkgroup);
  const bucket2 = workgroupArray("uint", itemsPerWorkgroup);
  const bucket3 = workgroupArray("uint", itemsPerWorkgroup);
  const nextBucket0 = workgroupArray("uint", itemsPerWorkgroup);
  const nextBucket1 = workgroupArray("uint", itemsPerWorkgroup);
  const nextBucket2 = workgroupArray("uint", itemsPerWorkgroup);
  const nextBucket3 = workgroupArray("uint", itemsPerWorkgroup);

  return Fn(() => {
    const tid = invocationLocalIndex;
    const local0 = tid.mul(uint(RADIX_ITEMS_PER_THREAD));
    const gid0 = workgroupId.x.mul(uint(itemsPerWorkgroup)).add(local0);
    const keyValue0 = uint(0).toVar();

    If(gid0.lessThan(uint(count)), () => {
      keyValue0.assign(inputKey.element(gid0));
    });

    const bucket0Value = keyValue0.shiftRight(uint(bit)).bitAnd(uint(RADIX_BUCKETS - 1));
    const active0 = gid0.lessThan(uint(count));

    // Lanes past the real splat count still participate in the fixed-size workgroup scan,
    // but contribute zero to every radix bucket so they never affect prefix counts.
    assignBucketFlags(bucket0, bucket1, bucket2, bucket3, local0, active0, bucket0Value);
    workgroupBarrier();

    // Hillis-Steele inclusive scan, adapted from the reference radix block.
    // The radix bucket arrays use ping-pong storage so each scan stage reads a
    // complete previous stage and writes the next stage without in-place hazards.
    for (let offset = 1, stage = 0; offset < itemsPerWorkgroup; offset <<= 1, stage++) {
      const src0 = stage % 2 === 0 ? bucket0 : nextBucket0;
      const src1 = stage % 2 === 0 ? bucket1 : nextBucket1;
      const src2 = stage % 2 === 0 ? bucket2 : nextBucket2;
      const src3 = stage % 2 === 0 ? bucket3 : nextBucket3;
      const dst0 = stage % 2 === 0 ? nextBucket0 : bucket0;
      const dst1 = stage % 2 === 0 ? nextBucket1 : bucket1;
      const dst2 = stage % 2 === 0 ? nextBucket2 : bucket2;
      const dst3 = stage % 2 === 0 ? nextBucket3 : bucket3;
      const stageOffset = uint(offset);

      scanBucketLane(src0, dst0, local0, stageOffset);
      scanBucketLane(src1, dst1, local0, stageOffset);
      scanBucketLane(src2, dst2, local0, stageOffset);
      scanBucketLane(src3, dst3, local0, stageOffset);
      workgroupBarrier();
    }

    If(tid.equal(0), () => {
      const last = uint(itemsPerWorkgroup - 1);
      const blockOffset = workgroupId.x;

      if (scanStageCount % 2 === 1) {
        blockSums.element(blockOffset).assign(wgElement(nextBucket0, last));
        blockSums.element(uint(workgroupCount).add(blockOffset)).assign(wgElement(nextBucket1, last));
        blockSums.element(uint(workgroupCount * 2).add(blockOffset)).assign(wgElement(nextBucket2, last));
        blockSums.element(uint(workgroupCount * 3).add(blockOffset)).assign(wgElement(nextBucket3, last));
      } else {
        blockSums.element(blockOffset).assign(wgElement(bucket0, last));
        blockSums.element(uint(workgroupCount).add(blockOffset)).assign(wgElement(bucket1, last));
        blockSums.element(uint(workgroupCount * 2).add(blockOffset)).assign(wgElement(bucket2, last));
        blockSums.element(uint(workgroupCount * 3).add(blockOffset)).assign(wgElement(bucket3, last));
      }
    });

    If(active0, () => {
      const prefix = uint(0).toVar();

      if (scanStageCount % 2 === 1) {
        assignLocalPrefix(nextBucket0, nextBucket1, nextBucket2, nextBucket3, local0, bucket0Value, prefix);
      } else {
        assignLocalPrefix(bucket0, bucket1, bucket2, bucket3, local0, bucket0Value, prefix);
      }

      localPrefix.element(gid0).assign(prefix.sub(1));
    });
  })()
    .compute(dispatchCount, [workgroupSize])
    .setName(`Radix Block Prefix ${bit}`);
}

export function createRadixReorderCompute({
  inputKey,
  inputIndex,
  outputKey,
  outputIndex,
  localPrefix,
  prefixBlockSums,
  count,
  workgroupSize,
  workgroupCount,
  bit,
}: RadixReorderComputeOptions): ComputeNode {
  const itemsPerWorkgroup = workgroupSize * RADIX_ITEMS_PER_THREAD;
  const dispatchCount = Math.ceil(count / RADIX_ITEMS_PER_THREAD);

  return Fn(() => {
    const tid = invocationLocalIndex;
    const local0 = tid.mul(uint(RADIX_ITEMS_PER_THREAD));
    const gid0 = workgroupId.x.mul(uint(itemsPerWorkgroup)).add(local0);

    // Inactive padded lanes were scanned as zeros above; skip their writes here so the
    // output buffers stay compact and contain only real splats.
    If(gid0.lessThan(uint(count)), () => {
      const k = inputKey.element(gid0);
      const bucket = k.shiftRight(uint(bit)).bitAnd(uint(RADIX_BUCKETS - 1));
      const prefixIndex = bucket.mul(uint(workgroupCount)).add(workgroupId.x);
      const sortedPosition = prefixBlockSums.element(prefixIndex).add(localPrefix.element(gid0));

      outputKey.element(sortedPosition).assign(k);
      outputIndex.element(sortedPosition).assign(inputIndex.element(gid0));
    });
  })()
    .compute(dispatchCount, [workgroupSize])
    .setName(`Radix Reorder ${bit}`);
}
