const DEFAULTS = { mode: "auto", pool: "default", topk: 5, alpha: 300 };

const $ = (id) => document.getElementById(id);

async function load() {
  const stored = await chrome.storage.local.get(DEFAULTS);
  $("mode").value = stored.mode;
  $("pool").value = stored.pool;
  $("topk").value = stored.topk;
  $("alpha").value = stored.alpha;
  $("alpha-val").textContent = String(Math.round(Number(stored.alpha)));
}

function save() {
  const settings = {
    mode: $("mode").value,
    pool: $("pool").value,
    topk: Math.max(1, Math.min(20, parseInt($("topk").value, 10) || 5)),
    alpha: parseFloat($("alpha").value) || 150,
  };
  chrome.storage.local.set(settings);
}

for (const id of ["mode", "pool", "topk", "alpha"]) {
  document.addEventListener("DOMContentLoaded", () => {
    $(id).addEventListener("change", save);
    $(id).addEventListener("input", () => {
      if (id === "alpha") $("alpha-val").textContent = String(Math.round(Number($("alpha").value)));
      save();
    });
  });
}
document.addEventListener("DOMContentLoaded", load);

(async () => {
  const el = $("status");
  try {
    const r = await fetch("http://127.0.0.1:8765/health");
    const j = await r.json();
    el.className = "ok";
    el.textContent = `solver reachable (pid ${j.worker_pid})`;
  } catch {
    el.className = "bad";
    el.textContent = "solver not running at 127.0.0.1:8765";
  }
})();
