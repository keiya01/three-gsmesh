import { PlyReader } from "@sparkjsdev/spark";
import { BufferAttribute, BufferGeometry, InstancedBufferGeometry, Matrix4, Vector2 } from "three/webgpu";
import { Mesh, MeshBasicNodeMaterial, Node, Object3D, Points, PointsNodeMaterial, WebGPURenderer } from "three/webgpu";
import {
  attribute,
  dot,
  exp,
  Fn,
  float,
  instanceIndex,
  normalize,
  storage,
  uniform,
  vec4,
  modelWorldMatrix,
  modelWorldMatrixInverse,
  cameraPosition,
  sRGBTransferEOTF,
} from "three/tsl";
import { createSplatProjectionNode } from "./nodes/basis";
import { createPackedSHNode } from "./nodes/sh";
import { DEFAULT_RADIX_BIT_COUNT, SplatDepthSorter } from "./depthSort";
import type { SplatData } from "./loaders";

export type RenderMode = "point" | "billboard";

const DEFAULT_MAX_STD_DEV = Math.sqrt(8);

export type GSMeshOptions = {
  /**
   * Rendering primitive used for splats. `"billboard"` renders oriented Gaussian quads,
   * while `"point"` is a simpler point-cloud fallback.
   */
  renderMode?: RenderMode;
  /**
   * Optional node hook for transforming splat centers before rendering.
   */
  positionNode?: (position: Node<"vec3">) => Node<"vec3">;
  /**
   * Optional node hook for transforming loaded base RGBA colors before rendering.
   */
  colorNode?: (color: Node<"vec4">) => Node<"vec4">;
  /**
   * Maximum standard deviations from the center to render.
   */
  basisScale?: number;
  /**
   * Maximum projected radius in physical pixels for each basis axis. This prevents
   * extremely large splats from dominating the frame with excessive overdraw.
   */
  maxPixelRadius?: number;
  /**
   * Gaussian blur amount added to projected 2D covariance for antialiasing.
   */
  blurAmount?: number;
  /**
   * Fragment alpha cutoff.
   */
  minAlpha?: number;
  /**
   * Enables elliptical billboard rendering. Disable to draw rectangular splat quads.
   */
  useBasis?: boolean;
  /**
   * Displays the full projected ellipsoid footprint without Gaussian alpha falloff.
   * Useful for debugging basis size and culling behavior.
   */
  showEllipsoid?: boolean;
  /**
   * Number of quantized depth bits used by the radix sorter. Lower values reduce sort
   * passes but may introduce more ordering artifacts.
   */
  depthSortBits?: number;
  /**
   * Center-based frustum clipping margin. Spark uses 1.4 by default; lower values
   * cull more off-screen splats before they rasterize.
   */
  clipXY?: number;
  /**
   * Maximum absolute SH coefficient ranges used when unpacking Spark-packed SH data.
   */
  shMax?: {
    sh1?: number;
    sh2?: number;
    sh3?: number;
  };
};

export class GSMesh extends Object3D {
  plyReader!: PlyReader;
  options: Required<Omit<GSMeshOptions, "positionNode" | "colorNode" | "shMax">>
    & Pick<GSMeshOptions, "positionNode" | "colorNode">
    & { shMax: Required<NonNullable<GSMeshOptions["shMax"]>> };
  depthSorter: SplatDepthSorter | null = null;

  private lastSortModelViewMatrix = new Matrix4();
  private needsSort = true;

  constructor(data: SplatData, options: GSMeshOptions = {}) {
    super();
    this.options = {
      renderMode: options.renderMode ?? "billboard",
      positionNode: options.positionNode,
      colorNode: options.colorNode,
      basisScale: options.basisScale ?? DEFAULT_MAX_STD_DEV,
      maxPixelRadius: options.maxPixelRadius ?? 512,
      blurAmount: options.blurAmount ?? 0.3,
      minAlpha: options.minAlpha ?? 0.5 / 255,
      useBasis: options.useBasis ?? true,
      showEllipsoid: options.showEllipsoid ?? false,
      depthSortBits: options.depthSortBits ?? DEFAULT_RADIX_BIT_COUNT,
      clipXY: options.clipXY ?? 1,
      shMax: {
        sh1: options.shMax?.sh1 ?? 1,
        sh2: options.shMax?.sh2 ?? 1,
        sh3: options.shMax?.sh3 ?? 1,
      },
    };
    this.setup(data);
  }

  createPositionNode(position: Node<"vec3">): Node<"vec3"> {
    return this.options.positionNode ? this.options.positionNode(position) : position;
  }

  createColorNode(color: Node<"vec4">): Node<"vec4"> {
    return this.options.colorNode ? this.options.colorNode(color) : color;
  }

  setup = (bufs: SplatData) => {
    const m = this.createMesh(bufs);
    this.add(m);
  }

  createMesh(bufs: SplatData) {
    if (this.options.renderMode === "billboard") {
      return this.createBillboard(bufs);
    }

    return this.createPoint(bufs);
  }

  createPoint(bufs: SplatData) {
    const geometry = new BufferGeometry();
    geometry.setAttribute("position", new BufferAttribute(bufs.position, 4));
    geometry.setAttribute("color", new BufferAttribute(bufs.color, 4));

    const material = new PointsNodeMaterial({
      transparent: true,
    });

    material.positionNode = this.createPositionNode(attribute<"vec4">("position").xyz);
    material.colorNode = this.createColorNode(attribute<"vec4">("color"));

    return new Points(geometry, material);
  }

  createBillboard(bufs: SplatData) {
    const splatCount = bufs.position.length / 4;
    const geometry = new InstancedBufferGeometry();
    geometry.instanceCount = splatCount;
    geometry.setIndex(new BufferAttribute(new Uint16Array([0, 1, 2, 2, 1, 3]), 1));
    geometry.setAttribute("position", new BufferAttribute(new Float32Array([
      -1, 1, 0,
      -1, -1, 0,
      1, 1, 0,
      1, -1, 0,
    ]), 3));
    geometry.setAttribute("corner", new BufferAttribute(new Float32Array([
      -1, 1,
      -1, -1,
      1, 1,
      1, -1,
    ]), 2));

    const material = new MeshBasicNodeMaterial({
      depthWrite: false,
      premultipliedAlpha: true,
      transparent: true,
      lights: false,
      fog: false,
    });
    const USE_BASIS = this.options.useBasis;
    const SHOW_ELLIPSOID = this.options.showEllipsoid;

    const corner = attribute<"vec2">("corner");

    const covariance = storage(
      new BufferAttribute(bufs.covariance, 4),
      "vec4",
      splatCount * 2,
    ).toReadOnly();
    const centerBuffer = storage(
      new BufferAttribute(bufs.position, 4),
      "vec4",
      splatCount,
    ).toReadOnly();
    const colorBuffer = storage(
      new BufferAttribute(bufs.color, 4),
      "vec4",
      splatCount,
    ).toReadOnly();
    const sh1Buffer = bufs.extra.sh1
      ? storage(
        new BufferAttribute(bufs.extra.sh1, 1),
        "uint",
        splatCount * 2,
      ).toReadOnly()
      : undefined;
    const sh2Buffer = bufs.extra.sh2
      ? storage(
        new BufferAttribute(bufs.extra.sh2, 1),
        "uint",
        splatCount * 4,
      ).toReadOnly()
      : undefined;
    const sh3Buffer = bufs.extra.sh3
      ? storage(
        new BufferAttribute(bufs.extra.sh3, 1),
        "uint",
        splatCount * 4,
      ).toReadOnly()
      : undefined;

    const basisModelViewMatrix = uniform(new Matrix4()).setName("basisModelViewMatrix");

    this.depthSorter = new SplatDepthSorter({
      center: centerBuffer,
      positionNode: this.options.positionNode,
      modelViewMatrix: basisModelViewMatrix,
      count: splatCount,
      radixBitCount: this.options.depthSortBits,
    });

    const basisProjectionMatrix = uniform(new Matrix4()).setName("basisProjectionMatrix");
    const basisScreenSize = uniform(new Vector2()).setName("basisScreenSize");
    const basisScreenToClip = uniform(new Vector2()).setName("basisScreenToClip");
    const sortedSplatIndex = this.depthSorter.sortedIndex.element(instanceIndex);
    const splatCenter = this.createPositionNode(centerBuffer.element(sortedSplatIndex).xyz);
    const splatColor = this.createColorNode(colorBuffer.element(sortedSplatIndex));
    const decodedAlpha = splatColor.a;
    const splatProjection = createSplatProjectionNode({
      center: centerBuffer,
      centerNode: splatCenter,
      covariance,
      splatIndex: sortedSplatIndex,
      modelViewMatrix: basisModelViewMatrix,
      projectionMatrix: basisProjectionMatrix,
      screenSize: basisScreenSize,
      basisScale: float(this.options.basisScale),
      alpha: decodedAlpha,
      minAlpha: this.options.minAlpha,
      blurAmount: this.options.blurAmount,
      maxPixelRadius: this.options.maxPixelRadius,
    });
    const splatRgb = Fn(() => {
      const worldCenter = modelWorldMatrix.mul(vec4(splatCenter, 1));
      const worldViewDir = normalize(worldCenter.xyz.sub(cameraPosition));
      const viewDir = worldViewDir.transformDirection(modelWorldMatrixInverse);
      const shRgb = createPackedSHNode({
        sh1: sh1Buffer,
        sh2: sh2Buffer,
        sh3: sh3Buffer,
        splatIndex: sortedSplatIndex,
        viewDir,
        shMax: this.options.shMax,
      });

      return splatColor.rgb.add(shRgb);
    })().toVarying("vSplatRgb");
    const splatAlpha = decodedAlpha.mul(splatProjection.blurAdjust).toVarying("vSplatAlpha");
    const adjustedStdDev = splatProjection.adjustedStdDev.toVarying("vAdjustedStdDev");

    material.vertexNode = Fn(() => {
      const centerView = basisModelViewMatrix.mul(vec4(splatCenter, 1));
      const splatBasis = splatProjection.basis;
      const clipCenter = basisProjectionMatrix.mul(centerView);
      const pixelOffset = splatBasis.xy.mul(corner.x).add(splatBasis.zw.mul(corner.y));
      const clipOffset = pixelOffset.mul(basisScreenToClip).mul(clipCenter.w);
      const clipPosition = vec4(clipCenter.xy.add(clipOffset), clipCenter.z, clipCenter.w);

      return clipPosition;
    })();

    // Ref: https://github.com/sparkjsdev/spark/blob/915c474795e0c78f7cd1b7f4eb97695028b495c0/src/shaders/splatFragment.glsl
    material.colorNode = Fn(() => {
      let gaussianAlpha: Node<"float"> = splatAlpha;
      if (USE_BASIS) {
        const splatUv = corner.mul(adjustedStdDev);
        const z2 = dot(splatUv, splatUv);
        const maxZ2 = adjustedStdDev.mul(adjustedStdDev);
  
        z2.greaterThan(maxZ2).discard();
  
        if(!SHOW_ELLIPSOID) {
          const gaussian = exp(z2.mul(-0.5));
          const lowAlpha = splatAlpha.mul(gaussian);

          const opacity = exp(splatAlpha.mul(splatAlpha).sub(1).div(Math.E));
          const highAlpha = float(1).sub(float(1).sub(gaussian).pow(opacity));
          gaussianAlpha = splatAlpha.greaterThan(1).select(highAlpha, lowAlpha);
    
          gaussianAlpha.lessThan(this.options.minAlpha).discard();
        }
      }

      return vec4(sRGBTransferEOTF(splatRgb) as Node<"vec3">, gaussianAlpha);
    })();

    const mesh = new Mesh(geometry, material);

    mesh.onBeforeRender = (renderer, _scene, camera) => {
      basisModelViewMatrix.value.multiplyMatrices(camera.matrixWorldInverse, mesh.matrixWorld);
      basisProjectionMatrix.value.copy(camera.projectionMatrix);
      renderer.getDrawingBufferSize(basisScreenSize.value);
      basisScreenToClip.value.set(2 / basisScreenSize.value.x, 2 / basisScreenSize.value.y);

      const shouldSort = this.needsSort
        || !basisModelViewMatrix.value.equals(this.lastSortModelViewMatrix);

      const webgpuRenderer = renderer as unknown as WebGPURenderer;

      if (!shouldSort) {
        return;
      }

      this.depthSorter!.compute(webgpuRenderer);

      this.lastSortModelViewMatrix.copy(basisModelViewMatrix.value);
      this.needsSort = false;
    };

    return mesh;
  }
}
