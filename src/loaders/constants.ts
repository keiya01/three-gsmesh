export type Extra = { sh1?: Uint32Array, sh2?: Uint32Array, sh3?: Uint32Array };

export type SplatLoaderOptions = {
  includeSH?: boolean;
};

export type SplatData = {
  position: Float32Array;
  color: Float32Array;
  covariance: Float32Array;
  extra: Extra;
};
