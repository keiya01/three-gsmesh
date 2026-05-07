import { resolve } from "node:path";
import { defineConfig } from "vite";

export default defineConfig((env) => ({
  base: "/three-gsmesh",
  resolve: {
    alias: {
      "three-gsmesh": resolve(import.meta.dirname, 'src/lib.ts'),
    },
  },
}));
