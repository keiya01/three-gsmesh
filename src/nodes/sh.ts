// Ref: https://github.com/sparkjsdev/spark/blob/3cf9fa15adb7ac7c47a1e962740db97b9e8a9fdf/src/PackedSplats.ts

import type StorageBufferNode from "three/src/nodes/accessors/StorageBufferNode.js";
import {
  float,
  int,
  uint,
  uvec2,
  uvec4,
  vec3,
} from "three/tsl";
import type { Node } from "three/webgpu";

type PackedSHNodeOptions = {
  sh1?: StorageBufferNode<"uint">;
  sh2?: StorageBufferNode<"uint">;
  sh3?: StorageBufferNode<"uint">;
  splatIndex: Node<"uint">;
  viewDir: Node<"vec3">;
  shMax?: {
    sh1?: number;
    sh2?: number;
    sh3?: number;
  };
};

function sint(packed: Node<"uint">, leftShift: number, rightShift: number) {
  return int(packed.shiftLeft(uint(leftShift))).shiftRight(uint(rightShift)).toFloat();
}

function sintJoined(
  low: Node<"uint">,
  lowRightShift: number,
  high: Node<"uint">,
  highLeftShift: number,
  rightShift: number,
) {
  return int(low.shiftRight(uint(lowRightShift)).bitOr(high.shiftLeft(uint(highLeftShift))))
    .shiftRight(uint(rightShift))
    .toFloat();
}

export function evaluatePackedSH1(
  packed: Node<"uvec2">,
  viewDir: Node<"vec3">,
  sh1Max: Node<"float"> = float(1),
) {
  const sh1_0 = vec3(
    sint(packed.x, 25, 25),
    sint(packed.x, 18, 25),
    sint(packed.x, 11, 25),
  );
  const sh1_1 = vec3(
    sint(packed.x, 4, 25),
    sintJoined(packed.x, 3, packed.y, 29, 25),
    sint(packed.y, 22, 25),
  );
  const sh1_2 = vec3(
    sint(packed.y, 15, 25),
    sint(packed.y, 8, 25),
    sint(packed.y, 1, 25),
  );

  return sh1_0.mul(float(-0.4886025).mul(viewDir.y))
    .add(sh1_1.mul(float(0.4886025).mul(viewDir.z)))
    .add(sh1_2.mul(float(-0.4886025).mul(viewDir.x)))
    .mul(sh1Max.div(63));
}

export function evaluatePackedSH2(
  packed: Node<"uvec4">,
  viewDir: Node<"vec3">,
  sh2Max: Node<"float"> = float(1),
) {
  const sh2_0 = vec3(
    sint(packed.x, 24, 24),
    sint(packed.x, 16, 24),
    sint(packed.x, 8, 24),
  );
  const sh2_1 = vec3(
    int(packed.x).shiftRight(uint(24)).toFloat(),
    sint(packed.y, 24, 24),
    sint(packed.y, 16, 24),
  );
  const sh2_2 = vec3(
    sint(packed.y, 8, 24),
    int(packed.y).shiftRight(uint(24)).toFloat(),
    sint(packed.z, 24, 24),
  );
  const sh2_3 = vec3(
    sint(packed.z, 16, 24),
    sint(packed.z, 8, 24),
    int(packed.z).shiftRight(uint(24)).toFloat(),
  );
  const sh2_4 = vec3(
    sint(packed.w, 24, 24),
    sint(packed.w, 16, 24),
    sint(packed.w, 8, 24),
  );

  const x = viewDir.x;
  const y = viewDir.y;
  const z = viewDir.z;

  return sh2_0.mul(float(1.0925484).mul(x).mul(y))
    .add(sh2_1.mul(float(-1.0925484).mul(y).mul(z)))
    .add(sh2_2.mul(float(0.3153915).mul(float(2).mul(z).mul(z).sub(x.mul(x)).sub(y.mul(y)))))
    .add(sh2_3.mul(float(-1.0925484).mul(x).mul(z)))
    .add(sh2_4.mul(float(0.5462742).mul(x.mul(x).sub(y.mul(y)))))
    .mul(sh2Max.div(127));
}

export function evaluatePackedSH3(
  packed: Node<"uvec4">,
  viewDir: Node<"vec3">,
  sh3Max: Node<"float"> = float(1),
) {
  const sh3_0 = vec3(
    sint(packed.x, 26, 26),
    sint(packed.x, 20, 26),
    sint(packed.x, 14, 26),
  );
  const sh3_1 = vec3(
    sint(packed.x, 8, 26),
    sint(packed.x, 2, 26),
    sintJoined(packed.x, 4, packed.y, 28, 26),
  );
  const sh3_2 = vec3(
    sint(packed.y, 22, 26),
    sint(packed.y, 16, 26),
    sint(packed.y, 10, 26),
  );
  const sh3_3 = vec3(
    sint(packed.y, 4, 26),
    sintJoined(packed.y, 2, packed.z, 30, 26),
    sint(packed.z, 24, 26),
  );
  const sh3_4 = vec3(
    sint(packed.z, 18, 26),
    sint(packed.z, 12, 26),
    sint(packed.z, 6, 26),
  );
  const sh3_5 = vec3(
    int(packed.z).shiftRight(uint(26)).toFloat(),
    sint(packed.w, 26, 26),
    sint(packed.w, 20, 26),
  );
  const sh3_6 = vec3(
    sint(packed.w, 14, 26),
    sint(packed.w, 8, 26),
    sint(packed.w, 2, 26),
  );

  const x = viewDir.x;
  const y = viewDir.y;
  const z = viewDir.z;
  const xx = x.mul(x);
  const yy = y.mul(y);
  const zz = z.mul(z);
  const xy = x.mul(y);

  return sh3_0.mul(float(-0.5900436).mul(y).mul(float(3).mul(xx).sub(yy)))
    .add(sh3_1.mul(float(2.8906114).mul(xy).mul(z)))
    .add(sh3_2.mul(float(-0.4570458).mul(y).mul(float(4).mul(zz).sub(xx).sub(yy))))
    .add(sh3_3.mul(float(0.3731763).mul(z).mul(float(2).mul(zz).sub(float(3).mul(xx)).sub(float(3).mul(yy)))))
    .add(sh3_4.mul(float(-0.4570458).mul(x).mul(float(4).mul(zz).sub(xx).sub(yy))))
    .add(sh3_5.mul(float(1.4453057).mul(z).mul(xx.sub(yy))))
    .add(sh3_6.mul(float(-0.5900436).mul(x).mul(xx.sub(float(3).mul(yy)))))
    .mul(sh3Max.div(31));
}

export function createPackedSHNode({
  sh1,
  sh2,
  sh3,
  splatIndex,
  viewDir,
  shMax = {},
}: PackedSHNodeOptions) {
  let rgb: Node<"vec3"> = vec3(0);

  if (sh1) {
    const base = splatIndex.mul(uint(2));
    const packed = uvec2(
      sh1.element(base),
      sh1.element(base.add(uint(1))),
    );

    rgb = rgb.add(evaluatePackedSH1(packed, viewDir, float(shMax.sh1 ?? 1)));
  }
  if (sh2) {
    const base = splatIndex.mul(uint(4));
    const packed = uvec4(
      sh2.element(base),
      sh2.element(base.add(uint(1))),
      sh2.element(base.add(uint(2))),
      sh2.element(base.add(uint(3))),
    );

    rgb = rgb.add(evaluatePackedSH2(packed, viewDir, float(shMax.sh2 ?? 1)));
  }
  if (sh3) {
    const base = splatIndex.mul(uint(4));
    const packed = uvec4(
      sh3.element(base),
      sh3.element(base.add(uint(1))),
      sh3.element(base.add(uint(2))),
      sh3.element(base.add(uint(3))),
    );

    rgb = rgb.add(evaluatePackedSH3(packed, viewDir, float(shMax.sh3 ?? 1)));
  }

  return rgb;
}
