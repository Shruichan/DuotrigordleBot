const SERVER = "http://127.0.0.1:8765/suggest";
const DEFAULTS = { mode: "auto", pool: "default", topk: 5, alpha: 1.0 };

chrome.runtime.onMessage.addListener((msg, _sender, send) => {
  if (msg && msg.type === "suggest") {
    chrome.storage.local.get(DEFAULTS).then((settings) => {
      const payload = {
        boards: msg.state.boards,
        top_k: settings.topk,
        alpha: settings.alpha,
        mode: settings.mode,
        pool: settings.pool,
      };
      fetch(SERVER, {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify(payload),
      })
        .then((r) => r.json().then((data) => ({ ok: r.ok, status: r.status, data })))
        .then(({ ok, status, data }) => {
          if (!ok) send({ ok: false, error: `HTTP ${status}: ${JSON.stringify(data)}` });
          else if (data && data.error) send({ ok: false, error: data.error });
          else send({ ok: true, data });
        })
        .catch((e) => send({ ok: false, error: String(e) }));
    });
    return true;
  }
});
