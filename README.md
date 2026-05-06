# GSMesh for TSL

![plant](assets/plant.png)

This library provides `GSMesh`, which renders 3D Gaussian Splatting data with Three.js Shading Language (TSL).

Note that this is mostly for learning purposes, so the performance is still not good.

## Overview

This is basically a project for learning purpose, but you can use this as a library like below.

Note that this library only works on `three/webgpu`.

```ts
import { GSMesh, SpzLoader } from "three-gsmesh";

const splatLoader = new SpzLoader();
// const splatLoader = new PlyLoader();
const data = await splatLoader.loadAsync("/CHERRY BLOSSOMS/scene.spz")
const gsmesh = new GSMesh(data);
scene.add(gsmesh);
```

## How this works

I implemented 3D Gaussian Splatting with TSL shaders and WebGPU compute shaders.

Rendering 3DGS involves the following steps:

1. Parse 3D GS data.
2. Compute covariance matrices.
3. Sort splats so alpha blending works correctly.
4. Project covariance matrices to screen-space billboard bases.
5. Clip billboards to the projected ellipses.
6. Render the splats.

## Features

- PLY support
- Radix sort with compute shaders
- 3DGS model rendering
- Extra SH properties defined in some datasets

## Limitations

- Other data formats
- Additional radix sort optimizations, such as tile-based binning/rasterization used in the original paper.
- LOD support like [Spark](https://sparkjs.dev/docs/new-spark-renderer/).

## Next steps

This paper might be helpful for optimization: [WebSplatter: Enabling Cross-Device Efficient Gaussian Splatting in Web Browsers via WebGPU](https://arxiv.org/pdf/2602.03207)

## References

- https://github.com/sparkjsdev/spark
- https://shi-yan.github.io/webgpuunleashed/Advanced/gaussian_splatting.html
- https://github.com/graphdeco-inria/gaussian-splatting
- https://github.com/kishimisu/WebGPU-Radix-Sort
- https://developer.nvidia.com/gpugems/gpugems3/part-vi-gpu-computing/chapter-39-parallel-prefix-sum-scan-cuda
