import { PlyReader, utils } from "@sparkjsdev/spark";
import { Loader, LoadingManager } from "three";
import { computeCovariance } from "../covariance";
import type { SplatData, SplatLoaderOptions } from "./constants";

export class PlyLoader extends Loader<SplatData> {
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
    const reader = new PlyReader({ fileBytes })
    await reader.parseHeader();
    const numSplats = reader.numSplats;
    const position = new Float32Array(numSplats * 4);
    const color = new Float32Array(numSplats * 4);
    const covariance = new Float32Array(numSplats * 8);
    const extra: { sh1?: Uint32Array, sh2?: Uint32Array, sh3?: Uint32Array } = {};

    reader.parseSplats(
      (
        index,
        x,
        y,
        z,
        scaleX,
        scaleY,
        scaleZ,
        quatX,
        quatY,
        quatZ,
        quatW,
        opacity,
        r,
        g,
        b,
      ) => {
        const i = index * 4;
        // Positions
        position[i] = x;
        position[i + 1] = y;
        position[i + 2] = z;

        // Colors
        color[i] = r;
        color[i + 1] = g;
        color[i + 2] = b;
        color[i + 3] = opacity;

        const covarianceOffset = index * 8;
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

    return {
      position,
      color,
      covariance,
      extra,
    };
  }
}
