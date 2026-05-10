// One-time vendor bundle for the Memory Inspector pipeline tab.
// Outputs ESM JS + plain CSS into static/inspector/vendor/.

import * as esbuild from "esbuild";
import { fileURLToPath } from "node:url";
import path from "node:path";

const __dirname = path.dirname(fileURLToPath(import.meta.url));
const outDir = path.resolve(__dirname, "../plugmem/api/static/inspector/vendor");

await esbuild.build({
  entryPoints: [path.join(__dirname, "entry.js")],
  bundle: true,
  format: "esm",
  outfile: path.join(outDir, "xyflow-bundle.js"),
  minify: true,
  target: ["es2022"],
  define: { "process.env.NODE_ENV": JSON.stringify("production") },
  legalComments: "none",
});

await esbuild.build({
  entryPoints: [path.join(__dirname, "entry.css")],
  bundle: true,
  outfile: path.join(outDir, "xyflow-bundle.css"),
  minify: true,
  legalComments: "none",
});

console.log("vendor bundle built →", outDir);
