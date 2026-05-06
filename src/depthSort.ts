import { BufferAttribute, ComputeNode, Node, StorageBufferNode, type WebGPURenderer } from "three/webgpu";
import { Fn, instanceIndex, storage } from "three/tsl";
import { createDepthKeyCompute } from "./computes/depthKey";
import { createPrefixSumLevels, type PrefixLevel } from "./computes/prefixSum";
import {
  createRadixBlockCompute,
  createRadixReorderCompute,
  RADIX_BITS_PER_PASS,
  RADIX_BUCKETS,
  RADIX_ITEMS_PER_THREAD,
} from "./computes/radixSort";

const DEFAULT_WORKGROUP_SIZE = 256;
export const DEFAULT_RADIX_BIT_COUNT = 20;

type DepthSortOptions = {
  center: StorageBufferNode<"vec4">;
  positionNode?: (position: Node<"vec3">) => Node<"vec3">;
  modelViewMatrix: Node<"mat4">;
  count: number;
  radixBitCount?: number;
};

export class SplatDepthSorter {
  readonly keyCompute: ComputeNode;
  readonly clearBlockSumsCompute: ComputeNode;
  readonly radixPasses: ComputeNode[] = [];
  readonly prefixLevels: PrefixLevel[][] = [];

  sortedIndex: StorageBufferNode<"uint">;

  constructor(options: DepthSortOptions) {
    const workgroupSize = DEFAULT_WORKGROUP_SIZE;
    const workgroupCount = Math.ceil(options.count / (workgroupSize * RADIX_ITEMS_PER_THREAD));
    const radixBitCount = normalizeRadixBitCount(options.radixBitCount ?? DEFAULT_RADIX_BIT_COUNT);
    const maxDepthKey = 2 ** radixBitCount - 1;

    const {
      sortedIndex,
      tmpSortedIndex,
      depthKey,
      tmpDepthKey,
      blockSums,
      localPrefix,
    } = this.createStorages(options.count, workgroupSize);

    this.sortedIndex = sortedIndex;

    const center = options.center.element(instanceIndex).xyz;
    const sortCenter = options.positionNode ? options.positionNode(center) : center;

    this.keyCompute = createDepthKeyCompute({
      ...options,
      center: sortCenter,
      workgroupSize,
      maxDepthKey,
      key: depthKey,
      index: sortedIndex
    });
    this.clearBlockSumsCompute = Fn(() => {
      blockSums.element(instanceIndex).assign(0);
    })()
      .compute(workgroupCount * RADIX_BUCKETS, [workgroupSize])
      .setName("Clear Radix Block Sums");

    for (let bit = 0; bit < radixBitCount; bit += RADIX_BITS_PER_PASS) {
      const passIndex = bit / RADIX_BITS_PER_PASS;
      const even = passIndex % 2 === 0;
      const inputKey = even ? depthKey : tmpDepthKey;
      const inputIndex = even ? sortedIndex : tmpSortedIndex;
      const outputKey = even ? tmpDepthKey : depthKey;
      const outputIndex = even ? tmpSortedIndex : sortedIndex;

      this.radixPasses.push(createRadixBlockCompute({
        inputKey,
        localPrefix,
        blockSums,
        count: options.count,
        workgroupSize,
        workgroupCount,
        bit,
      }));

      this.prefixLevels.push(createPrefixSumLevels(
        blockSums,
        workgroupCount * RADIX_BUCKETS,
        workgroupSize,
      ));

      this.radixPasses.push(createRadixReorderCompute({
        inputKey,
        inputIndex,
        outputKey,
        outputIndex,
        localPrefix,
        prefixBlockSums: blockSums,
        count: options.count,
        workgroupSize,
        workgroupCount,
        bit,
      }));
    }
  }

  createStorages(splatCount: number, workgroupSize: number) {
        const sortedIndexAttribute = new BufferAttribute(new Uint32Array(splatCount), 1);
        const depthKeyAttribute = new BufferAttribute(new Uint32Array(splatCount), 1);
    
        const sortedIndex = storage(
          sortedIndexAttribute,
          "uint",
          splatCount,
        );
        const depthKey = storage(
          depthKeyAttribute,
          "uint",
          splatCount,
        );

        const sortItemsPerWorkgroup = workgroupSize * RADIX_ITEMS_PER_THREAD;

        const tmpSortedIndex = storage(
          new BufferAttribute(new Uint32Array(splatCount), 1),
          "uint",
          splatCount,
        );
        const tmpDepthKey = storage(
          new BufferAttribute(new Uint32Array(splatCount), 1),
          "uint",
          splatCount,
        );
        const localPrefix = storage(
          new BufferAttribute(new Uint32Array(splatCount), 1),
          "uint",
          splatCount,
        );
        const sortWorkgroupCount = Math.ceil(splatCount / sortItemsPerWorkgroup);
        const blockSums = storage(
          new BufferAttribute(new Uint32Array(sortWorkgroupCount * RADIX_BUCKETS), 1),
          "uint",
          sortWorkgroupCount * RADIX_BUCKETS,
        );

        return {
          sortedIndex,
          depthKey,
          tmpSortedIndex,
          tmpDepthKey,
          localPrefix,
          blockSums,
        }
  }

  compute(renderer: WebGPURenderer) {
    const computeNodes: ComputeNode[] = [this.keyCompute];

    for (let pass = 0; pass < this.radixPasses.length; pass += 2) {
      const radixPass = pass / 2;

      computeNodes.push(this.clearBlockSumsCompute);
      computeNodes.push(this.radixPasses[pass]);

      for (const level of this.prefixLevels[radixPass]) {
        computeNodes.push(level.reduce);
      }

      for (let levelIndex = this.prefixLevels[radixPass].length - 1; levelIndex >= 0; levelIndex--) {
        const add = this.prefixLevels[radixPass][levelIndex].add;

        if (add !== null) {
          computeNodes.push(add);
        }
      }

      computeNodes.push(this.radixPasses[pass + 1]);
    }

    renderer.compute(computeNodes);
  }
}

function normalizeRadixBitCount(bitCount: number): number {
  const clamped = Math.min(Math.max(Math.ceil(bitCount), RADIX_BITS_PER_PASS), 32);
  const roundedToPass = Math.ceil(clamped / RADIX_BITS_PER_PASS) * RADIX_BITS_PER_PASS;
  const passCount = roundedToPass / RADIX_BITS_PER_PASS;

  return passCount % 2 === 0
    ? roundedToPass
    : Math.min(roundedToPass + RADIX_BITS_PER_PASS, 32);
}
