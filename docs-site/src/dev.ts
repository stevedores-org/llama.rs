import { app } from "./app";

const port = 3000;

export default {
  port,
  fetch: app.fetch,
};

console.log(`Docs server running at http://localhost:${port}/llama-rs/`);
