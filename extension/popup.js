const el = document.getElementById("status");
fetch("http://127.0.0.1:8765/health")
  .then((r) => r.json())
  .then((j) => { el.className = "ok"; el.textContent = `solver reachable (pid ${j.worker_pid})`; })
  .catch(() => { el.className = "bad"; el.textContent = "solver not running at 127.0.0.1:8765"; });
