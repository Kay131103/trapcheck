// scam-lab frontend: paste message → /api/verdict (Cloudflare Worker → HF
// Inference) → verdict card. No keys in the browser, no Space needed.

const $ = (id) => document.getElementById(id);

function esc(s) {
  return s.replace(/&/g, "&amp;").replace(/</g, "&lt;").replace(/>/g, "&gt;");
}

function renderVerdict(v) {
  const risk = Math.max(0, Math.min(100, Number(v.risk) || 0));
  const color = risk >= 70 ? "var(--bad)" : risk >= 40 ? "var(--warn)" : "var(--accent)";
  const level = risk >= 70 ? "HIGH RISK" : risk >= 40 ? "MEDIUM RISK" : "LOOKS SAFE";
  const family =
    { kyc_expiry: "Fake KYC expiry", digital_arrest: "Digital arrest / fake police",
      utility_bill: "Electricity bill cut-off", lottery: "Lottery / prize / grant",
      loan_app: "Loan app / instant credit", job_scam: "Job / WFH / exam paper",
      courier_customs: "Courier at customs", otp_fraud: "OTP / CVV harvesting",
      investment_tips: "Trading / crypto tips", tax_refund: "Tax / subsidy refund",
      emergency_scam: "Emergency / sextortion" }[v.scam_type] || "—";

  const flags = (v.red_flags || []).filter((f) => typeof f === "string");
  const flagged = flags.reduce(
    (acc, f) => acc.replace(f, `<mark style="background:#7f1d1d;color:#fecaca;padding:0 3px;border-radius:3px">${esc(f)}</mark>`),
    `<span class="${risk >= 50 ? "scam" : "safe"}">${esc($("msg").value)}</span>`,
  );

  $("verdict").innerHTML = `
    <div class="meter"><div style="width:${risk}%;background:${color}"></div></div>
    <div class="lv" style="color:${color}">${level} — ${risk}/100</div>
    <div class="cat">Category: <b style="color:var(--text)">${esc(family)}</b></div>
    <div class="act">Action: <b>${esc(v.action || "—")}</b></div>
    ${flags.length ? `<div class="flags">🚩 Red flags: ${flags.map(esc).join(" · ")}</div><div class="flags">${flagged}</div>` : ""}
  `;
}

let busy = false;
async function analyze() {
  const message = $("msg").value.trim();
  if (!message || busy) return;
  busy = true;
  $("go").disabled = true;
  $("verdict").innerHTML = `<div class="cat">analyzing…</div>`;
  try {
    const res = await fetch("/api/verdict", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ message }),
    });
    const data = await res.json();
    if (!res.ok) throw new Error(data.error || `HTTP ${res.status}`);
    renderVerdict(data);
  } catch (err) {
    $("verdict").innerHTML = `<div class="act">${esc(err instanceof Error ? err.message : String(err))}</div>`;
  }
  busy = false;
  $("go").disabled = false;
}

$("go").addEventListener("click", analyze);
$("msg").addEventListener("keydown", (e) => { if ((e.ctrlKey || e.metaKey) && e.key === "Enter") analyze(); });
$("clear").addEventListener("click", () => { $("msg").value = ""; $("verdict").innerHTML = ""; });
for (const btn of document.querySelectorAll(".examples button")) {
  btn.addEventListener("click", () => { $("msg").value = btn.dataset.q; analyze(); });
}
