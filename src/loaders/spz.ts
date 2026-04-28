import { SpzReader, utils } from "@sparkjsdev/spark";
import { Loader, LoadingManager } from "three";
import { computeCovariance } from "../covariance";
import type { SplatData, SplatLoaderOptions } from "./constants";

export class SpzLoader extends Loader<SplatData> {
  private options: Required<SplatLoaderOptions>;

  constructor(manager?: LoadingManager, options?: SplatLoaderOptions);
  constructor(options?: SplatLoaderOptions);
  constructor(
    managerOrOptions?: LoadingManager | SplatLoaderOptions,
    options: SplatLoaderOptions = {},
  ) {
    const hasLoaderOptions = managerOrOptions && "includeSH" in managerOrOptions;

    super(hasLoaderOptions ? undefined : managerOrOptions as LoadingManager | undefined);

    const loaderOptions = hasLoaderOptions
      ? managerOrOptions as SplatLoaderOptions
      : options;

    this.options = {
      includeSH: loaderOptions.includeSH ?? true,
    };
  }

  async load(url: string, onLoad: (data: SplatData) => void, _onProgress?: (event: ProgressEvent) => void, onError?: (err: unknown) => void): Promise<void> {
    const fileBytes = await fetch(url).then(r => r.arrayBuffer()).catch(onError);
    if(!fileBytes) return;
    return await this.parseData(fileBytes).then(onLoad).catch(onError);
  }

  async parseData(fileBytes: ArrayBuffer) {
    const includeSH = this.options.includeSH;
    const reader = new SpzReader({ fileBytes })
    await reader.parseHeader();
    const numSplats = reader.numSplats;
    const position = new Float32Array(numSplats * 4);
    const color = new Float32Array(numSplats * 4);
    const covariance = new Float32Array(numSplats * 8);
    const extra: { sh1?: Uint32Array, sh2?: Uint32Array, sh3?: Uint32Array } = {};

    const scaleQuats = new Float32Array(numSplats * 7);

    await reader.parseSplats(
       (index, x, y, z) => {
        const i = index * 4;
        // Positions
        position[i] = x;
        position[i + 1] = y;
        position[i + 2] = z;
       },
      (index, alpha) => {
        const i = index * 4;
        color[i + 3] = alpha;
      },
      (index, r, g, b) => {
        const i = index * 4;
        color[i] = r;
        color[i + 1] = g;
        color[i + 2] = b;
      },
      (index, scaleX, scaleY, scaleZ) => {
        const i = index * 7;
        scaleQuats[i] = scaleX;
        scaleQuats[i + 1] = scaleY;
        scaleQuats[i + 2] = scaleZ;
      },
      (index, quatX, quatY, quatZ, quatW) => {
        const i = index * 7;
        scaleQuats[i + 3] = quatX;
        scaleQuats[i + 4] = quatY;
        scaleQuats[i + 5] = quatZ;
        scaleQuats[i + 6] = quatW;
      },
      // Pack SH coefficients
      // Ref: https://github.com/sparkjsdev/spark/blob/3cf9fa15adb7ac7c47a1e962740db97b9e8a9fdf/src/oldWorker.ts#L356
      (index, sh1, sh2, sh3) => {
        if (!includeSH) return;

        if (sh1) {
          if (!extra.sh1) {
            extra.sh1 = new Uint32Array(numSplats * 2);
          }
          utils.encodeSh1Rgb(extra.sh1, index, sh1);
        }
        if (sh2) {
          if (!extra.sh2) {
            extra.sh2 = new Uint32Array(numSplats * 4);
          }
          utils.encodeSh2Rgb(extra.sh2, index, sh2);
        }
        if (sh3) {
          if (!extra.sh3) {
            extra.sh3 = new Uint32Array(numSplats * 4);
          }
          utils.encodeSh3Rgb(extra.sh3, index, sh3);
        }
      },
    );

    for (let i = 0; i < numSplats; i++) {
      const scaleQuatOffset = i * 7;
      const scaleX = scaleQuats[scaleQuatOffset];
      const scaleY = scaleQuats[scaleQuatOffset + 1];
      const scaleZ = scaleQuats[scaleQuatOffset + 2];
      const quatX = scaleQuats[scaleQuatOffset + 3];
      const quatY = scaleQuats[scaleQuatOffset + 4];
      const quatZ = scaleQuats[scaleQuatOffset + 5];
      const quatW = scaleQuats[scaleQuatOffset + 6];

      const covarianceOffset = i * 8;
      const [
        covariance00,
        covariance10,
        covariance20,
        covariance11,
        covariance21,
        covariance22,
      ] = computeCovariance(scaleX, scaleY, scaleZ, quatX, quatY, quatZ, quatW);
  
      covariance[covarianceOffset] = covariance00;
      covariance[covarianceOffset + 1] = covariance10;
      covariance[covarianceOffset + 2] = covariance20;
      covariance[covarianceOffset + 4] = covariance11;
      covariance[covarianceOffset + 5] = covariance21;
      covariance[covarianceOffset + 6] = covariance22;
    }

    return {
      position,
      color,
      covariance,
      extra,
    };
  }
}
