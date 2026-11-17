// trapcheck worker: static frontend + /api/verdict proxy to HF Inference.
// The browser never sees the HF token; it lives in this Worker's secrets.

import { onRequestPost } from "./verdict-handler.js";

export default {
  async fetch(request, env, ctx) {
    const url = new URL(request.url);
    if (url.pathname === "/api/verdict" && request.method === "POST") {
      return onRequestPost({ env, request });
    }
    return env.ASSETS.fetch(request);
  },
};
