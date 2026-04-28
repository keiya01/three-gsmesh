import type ComputeNode from "three/src/nodes/gpgpu/ComputeNode.js";
import type StorageBufferNode from "three/src/nodes/accessors/StorageBufferNode.js";
import {
  dot,
  float,
  Fn,
  If,
  instanceIndex,
  log,
  max,
  min,
  normalize,
  select,
  sqrt,
  uint,
  vec2,
  vec3,
  vec4,
} from "three/tsl";
import type { Matrix4, Node, UniformNode, Vector2 } from "three/webgpu";

type BasisComputeOptions = {
  center: StorageBufferNode<"vec4">;
  covariance: StorageBufferNode<"vec4">;
  basis: StorageBufferNode<"vec4">;
  modelViewMatrix: Node<"mat4">;
  projectionMatrix: UniformNode<"mat4", Matrix4>;
  screenSize: UniformNode<"vec2", Vector2>;
  count: number;
  basisScale: any;
  maxPixelRadius: number;
  enableCulling?: boolean;
};

function covarianceMul(c0: any, c1: any, v: any) {
  return vec3(
    c0.x.mul(v.x).add(c0.y.mul(v.y)).add(c0.z.mul(v.z)),
    c0.y.mul(v.x).add(c1.x.mul(v.y)).add(c1.y.mul(v.z)),
    c0.z.mul(v.x).add(c1.y.mul(v.y)).add(c1.z.mul(v.z)),
  );
}

export function createSplatProjectionNode({
  blurAmount = 0,
  center,
  centerNode,
  covariance,
  splatIndex,
  modelViewMatrix,
  projectionMatrix,
  screenSize,
  basisScale,
  alpha,
  minAlpha,
  maxPixelRadius,
}: {
  blurAmount?: number;
  center: StorageBufferNode<"vec4">;
  centerNode?: Node<"vec3">;
  covariance: StorageBufferNode<"vec4">;
  splatIndex: Node<"uint">;
  modelViewMatrix: Node<"mat4">;
  projectionMatrix: UniformNode<"mat4", Matrix4>;
  screenSize: UniformNode<"vec2", Vector2>;
  basisScale: Node<"float">;
  alpha?: Node<"float">;
  minAlpha?: number;
  maxPixelRadius: number;
}) {
  const mv = modelViewMatrix as any;
  const projection = projectionMatrix as any;
  const screen = screenSize as any;
  const covarianceIndex = splatIndex.mul(2);
  const c0 = covariance.element(covarianceIndex);
  const c1 = covariance.element(covarianceIndex.add(1));

  const projectionCenter = centerNode ?? center.element(splatIndex).xyz;
  const centerView = mv.mul(vec4(projectionCenter, 1));
  const viewX = centerView.x;
  const viewZ = centerView.z;
  const viewZ2 = viewZ.mul(viewZ);
  const focalX = projection.element(uint(0)).x.mul(screen.x).mul(0.5);
  const focalY = projection.element(uint(1)).y.mul(screen.y).mul(0.5);

  const modelViewColumn0 = mv.element(uint(0));
  const modelViewColumn1 = mv.element(uint(1));
  const modelViewColumn2 = mv.element(uint(2));
  const viewRow0 = vec3(modelViewColumn0.x, modelViewColumn1.x, modelViewColumn2.x);
  const viewRow1 = vec3(modelViewColumn0.y, modelViewColumn1.y, modelViewColumn2.y);
  const viewRow2 = vec3(modelViewColumn0.z, modelViewColumn1.z, modelViewColumn2.z);

  const covRow0 = covarianceMul(c0, c1, viewRow0);
  const covRow1 = covarianceMul(c0, c1, viewRow1);
  const covRow2 = covarianceMul(c0, c1, viewRow2);
  const viewCov0 = vec3(dot(viewRow0, covRow0), dot(viewRow0, covRow1), dot(viewRow0, covRow2));
  const viewCov1 = vec3(dot(viewRow1, covRow0), dot(viewRow1, covRow1), dot(viewRow1, covRow2));
  const viewCov2 = vec3(dot(viewRow2, covRow0), dot(viewRow2, covRow1), dot(viewRow2, covRow2));

  const jacobianX = vec3(
    focalX.div(viewZ),
    0,
    focalX.mul(viewX).negate().div(viewZ2),
  );
  const jacobianY = vec3(
    0,
    focalY.div(viewZ),
    centerView.y.mul(focalY).negate().div(viewZ2),
  );

  const covJacobianX = vec3(
    dot(viewCov0, jacobianX),
    dot(viewCov1, jacobianX),
    dot(viewCov2, jacobianX),
  );
  const covJacobianY = vec3(
    dot(viewCov0, jacobianY),
    dot(viewCov1, jacobianY),
    dot(viewCov2, jacobianY),
  );

  const aOrig = dot(jacobianX, covJacobianX);
  const b = dot(jacobianX, covJacobianY);
  const dOrig = dot(jacobianY, covJacobianY);
  const detOrig = max(aOrig.mul(dOrig).sub(b.mul(b)), 0);
  const a = aOrig.add(float(blurAmount));
  const d = dOrig.add(float(blurAmount));
  const det = max(a.mul(d).sub(b.mul(b)), 0.000001);
  const blurAdjust = sqrt(detOrig.div(det));
  const baseStdDev = typeof basisScale === "number" ? float(basisScale) : basisScale;
  const renderAlpha = alpha ? alpha.mul(blurAdjust) : null;
  const edgeAlpha = float(minAlpha ?? 0.5 / 255).mul(0.5);
  const adjustedStdDev = renderAlpha
    ? max(
      baseStdDev,
      sqrt(max(log(edgeAlpha.div(max(renderAlpha, 0.000001))).mul(-2), 0)),
    )
    : baseStdDev;

  const halfTrace = a.add(d).mul(0.5);
  const halfDiff = a.sub(d).mul(0.5);
  const radius = sqrt(max(halfDiff.mul(halfDiff).add(b.mul(b)), 0.0));
  const lambda1 = max(halfTrace.add(radius), 0.000001);
  const lambda2 = max(halfTrace.sub(radius), 0.000001);

  const rawMajor = vec2(b, lambda1.sub(a));
  const fallbackMajor = select(a.greaterThanEqual(d), vec2(1, 0), vec2(0, 1));
  const safeMajor = select(b.abs().greaterThan(0.001), rawMajor, fallbackMajor);
  const major = normalize(safeMajor);
  const minor = vec2(major.y.negate(), major.x);
  const majorRadius = min(float(maxPixelRadius), sqrt(lambda1).mul(baseStdDev));
  const minorRadius = min(float(maxPixelRadius), sqrt(lambda2).mul(baseStdDev));
  const majorBasis = major.mul(majorRadius);
  const minorBasis = minor.mul(minorRadius);

  return {
    basis: vec4(majorBasis.x, majorBasis.y, minorBasis.x, minorBasis.y),
    blurAdjust,
    adjustedStdDev,
  };
}

export function createSplatBasisNode(options: {
  blurAmount?: number;
  center: StorageBufferNode<"vec4">;
  covariance: StorageBufferNode<"vec4">;
  splatIndex: Node<"uint">;
  modelViewMatrix: Node<"mat4">;
  projectionMatrix: UniformNode<"mat4", Matrix4>;
  screenSize: UniformNode<"vec2", Vector2>;
  basisScale: any;
  maxPixelRadius: number;
}) {
  return createSplatProjectionNode(options).basis;
}

export function createBasisCompute({
  center,
  covariance,
  basis,
  modelViewMatrix,
  projectionMatrix,
  screenSize,
  count,
  basisScale,
  maxPixelRadius,
  enableCulling = false,
}: BasisComputeOptions): ComputeNode {
  return Fn(() => {
    const splatIndex = instanceIndex;
    const centerView = modelViewMatrix.mul(vec4(center.element(splatIndex).xyz, 1));
    const clipCenter = projectionMatrix.mul(centerView);
    const ndcCenter = clipCenter.xyz.div(clipCenter.w);
    const splatBasis = createSplatBasisNode({
      center,
      covariance,
      splatIndex,
      modelViewMatrix,
      projectionMatrix,
      screenSize,
      basisScale,
      maxPixelRadius,
    });
    const halfExtentPixels = vec2(
      splatBasis.x.abs().add(splatBasis.z.abs()),
      splatBasis.y.abs().add(splatBasis.w.abs()),
    );
    const halfExtentNdc = halfExtentPixels.mul(vec2(2).div(screenSize as any));
    const visible = clipCenter.w.greaterThan(0)
      .and(ndcCenter.x.add(halfExtentNdc.x).greaterThanEqual(-1))
      .and(ndcCenter.x.sub(halfExtentNdc.x).lessThanEqual(1))
      .and(ndcCenter.y.add(halfExtentNdc.y).greaterThanEqual(-1))
      .and(ndcCenter.y.sub(halfExtentNdc.y).lessThanEqual(1))
      .and(ndcCenter.z.greaterThanEqual(-1))
      .and(ndcCenter.z.lessThanEqual(1));

    if (enableCulling) {
      If(visible, () => {
        basis.element(splatIndex).assign(splatBasis);
      }).Else(() => {
        basis.element(splatIndex).assign(vec4(0));
      });
    } else {
      basis.element(splatIndex).assign(splatBasis);
    }
  })()
    .compute(count)
    .setName("Compute Splat Basis");
}
