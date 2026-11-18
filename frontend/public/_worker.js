// Pages advanced-mode worker: static assets + /api/verdict proxy.
import { onRequestPost } from "./verdict-handler.js";

export default {
  async fetch(request, env) {
    const url = new URL(request.url);
    if (url.pathname === "/api/verdict" && request.method === "POST") {
      return onRequestPost({ env, request });
    }
    return env.ASSETS.fetch(request);
  },
};
