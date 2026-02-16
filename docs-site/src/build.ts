import { app, routes } from "./app";
import { mkdir, writeFile } from "fs/promises";
import { join } from "path";

const BASE = "llama-rs";
const OUT_DIR = join(import.meta.dir, "..", "dist", BASE);

async function build() {
  console.log("Building static docs site...\n");

  for (const route of routes) {
    const urlPath = `/${BASE}${route === "/" ? "/" : route}`;
    const req = new Request(`http://localhost${urlPath}`);
    const res = await app.fetch(req);
    const html = await res.text();

    // Determine output file path
    const filePath =
      route === "/"
        ? join(OUT_DIR, "index.html")
        : join(OUT_DIR, ...route.split("/").filter(Boolean), "index.html");

    const dir = filePath.replace(/\/index\.html$/, "");
    await mkdir(dir, { recursive: true });
    await writeFile(filePath, html, "utf-8");

    console.log(`  ${urlPath} -> ${filePath.replace(join(import.meta.dir, "..") + "/", "")}`);
  }

  console.log(`\nBuilt ${routes.length} pages to dist/${BASE}/`);
}

build().catch((err) => {
  console.error("Build failed:", err);
  process.exit(1);
});
