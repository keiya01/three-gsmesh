import { defineConfig } from "vite";
import { resolve } from "path";
import dts from "unplugin-dts/vite";

const externalPackages = [
  /^three(?:\/.*)?$/,
  /^@sparkjsdev\/spark(?:\/.*)?$/,
];

export default defineConfig((env) => ({
  plugins: [dts({ bundleTypes: true })],
  build: {
    lib: {
      entry: resolve(import.meta.dirname, 'src/lib.ts'),
      name: 'ThreeGSMesh',
      formats: ['es'],
      // the proper extensions will be added
      fileName: 'three-gsmesh',
    },
    rolldownOptions: {
      external: externalPackages,
    },
    copyPublicDir: false,
  }
}));
