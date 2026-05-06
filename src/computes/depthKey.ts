import {
  float,
  Fn,
  If,
  instanceIndex,
  max,
  min,
  uint,
  vec4,
} from "three/tsl";
import type { ComputeNode, Node, StorageBufferNode } from "three/webgpu";

const DEPTH_KEY_SUBDIVISIONS_PER_UNIT = 1024;

type DepthKeyComputeOptions = {
  center: Node<"vec3">;
  key: StorageBufferNode<"uint">;
  index: StorageBufferNode<"uint">;
  modelViewMatrix: Node<"mat4">;
  count: number;
  workgroupSize: number;
  maxDepthKey: number;
};

export function createDepthKeyCompute({
  center,
  key,
  index,
  modelViewMatrix,
  count,
  workgroupSize,
  maxDepthKey,
}: DepthKeyComputeOptions): ComputeNode {
  return Fn(() => {
    const sortIndex = instanceIndex;
    const activeLimit = uint(count);
    const active = sortIndex.lessThan(activeLimit);

    If(active, () => {
      const centerView = modelViewMatrix.mul(vec4(center, 1));
      const depth = max(centerView.z.negate(), 0);
      const quantizedDepth = uint(min(depth.mul(float(DEPTH_KEY_SUBDIVISIONS_PER_UNIT)), float(maxDepthKey)));

      key.element(sortIndex).assign(uint(maxDepthKey).sub(quantizedDepth));
      index.element(sortIndex).assign(sortIndex);
    });
  })()
    .compute(count, [workgroupSize])
    .setName("Compute Splat Depth Keys");
}
