import { defineConfig } from "vite";
import { resolve } from "path";
import dts from "unplugin-dts/vite";

export default defineConfig((env) => ({
  plugins: [dts({ bundleTypes: true })],
  build: {
    lib: {
      entry: resolve(import.meta.dirname, 'src/lib.ts'),
      name: 'ThreeGSMesh',
      // the proper extensions will be added
      fileName: 'three-gsmesh',
    },
    rolldownOptions: {
      external: ['three', '@sparkjsdev/spark'],
      output: {
        globals: {
          three: 'THREE',
          spark: 'SPARK',
        },
      },
    },
    copyPublicDir: false,
  },
  resolve: {
    alias: env.command === "serve" ? {
      "three-gsmesh": resolve(__dirname, "./src/lib.ts"),
    } : undefined,
  },
}));
