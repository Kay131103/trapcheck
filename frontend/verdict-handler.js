// Cloudflare Pages Function: POST /api/verdict
// Proxies the fine-tuned scam-detector on HF Inference so the browser never
// needs an HF token. Set HF_TOKEN in the Pages project env vars.

const SYSTEM = (
  "You are a scam-detection assistant for Indian SMS/WhatsApp messages. "
  + "Reply with ONLY a JSON verdict: "
  + '{"risk": <0-100>, "scam_type": <family or null>, '
  + '"red_flags": [<exact phrases from the message>], "action": "<one line>"}'
);

const jsonRe = /\{[\s\S]*\}/;

function extractVerdict(text) {
  const match = (text || "").match(jsonRe);
  if (!match) return null;
  try {
    return JSON.parse(match[0]);
  } catch {
    return null;
  }
}

export async function onRequestPost(context) {
  const { HF_TOKEN, MODEL_REPO } = context.env;
  const cors = {
    "Access-Control-Allow-Origin": "*",
    "Content-Type": "application/json",
  };
  if (!HF_TOKEN) {
    return new Response(JSON.stringify({ error: "HF_TOKEN not set in Pages env vars" }), { status: 500, headers: cors });
  }
  let message = "";
  try {
    const body = await context.request.json();
    message = (body.message || "").slice(0, 1000);
  } catch {
    return new Response(JSON.stringify({ error: "send {\"message\": \"...\"}" }), { status: 400, headers: cors });
  }
  if (!message.trim()) {
    return new Response(JSON.stringify({ error: "empty message" }), { status: 400, headers: cors });
  }

  const MODELS = ["kitty1901/trapcheck-0.6b", "kitty1901/scam-radar-0.6b"];
  let upstream = null;
  let lastError = "";
  for (const model of MODELS) {
    const res = await fetch("https://router.huggingface.co/v1/chat/completions", {
      method: "POST",
      headers: {
        Authorization: `Bearer ${HF_TOKEN}`,
        "Content-Type": "application/json",
      },
      body: JSON.stringify({
        model,
        messages: [
          { role: "system", content: SYSTEM },
          { role: "user", content: message },
        ],
        max_tokens: 200,
        temperature: 0,
      }),
    });
    if (res.ok) {
      upstream = res;
      break;
    }
    lastError = `${model}: ${res.status}`;
    // 404 = model not (yet) available under this name — try the next one
    if (res.status !== 404) break;
  }

  if (!upstream || !upstream.ok) {
    const detail = upstream ? await upstream.text() : lastError;
    return new Response(JSON.stringify({ error: `HF inference failed (${lastError})`, detail: detail.slice(0, 300) }), {
      status: 502, headers: cors,
    });
  }

  const data = await upstream.json();
  const raw = data.choices?.[0]?.message?.content ?? "";
  const verdict = extractVerdict(raw);
  if (!verdict) {
    return new Response(JSON.stringify({ error: "unparseable verdict", raw: raw.slice(0, 200) }), { status: 502, headers: cors });
  }
  return new Response(JSON.stringify(verdict), { headers: cors });
}
