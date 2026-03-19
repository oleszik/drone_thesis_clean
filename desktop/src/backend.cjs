const { spawn } = require("node:child_process");
const http = require("node:http");

function log(msg) {
  console.log(`[desktop-backend] ${msg}`);
}

function httpGetJson(url) {
  return new Promise((resolve, reject) => {
    const req = http.get(url, (res) => {
      let body = "";
      res.setEncoding("utf8");
      res.on("data", (chunk) => {
        body += chunk;
      });
      res.on("end", () => {
        if (res.statusCode && res.statusCode >= 200 && res.statusCode < 300) {
          try {
            resolve(body ? JSON.parse(body) : {});
          } catch (err) {
            reject(err);
          }
          return;
        }
        reject(new Error(`HTTP ${res.statusCode || "?"}`));
      });
    });
    req.on("error", reject);
    req.setTimeout(2500, () => {
      req.destroy(new Error("request timeout"));
    });
  });
}

async function waitForHealth(healthUrl, timeoutMs) {
  const start = Date.now();
  while (Date.now() - start < timeoutMs) {
    try {
      const payload = await httpGetJson(healthUrl);
      if (payload && payload.ok) {
        return payload;
      }
    } catch (_) {
      // keep polling until timeout
    }
    await new Promise((resolve) => setTimeout(resolve, 500));
  }
  throw new Error(`backend health check timed out after ${timeoutMs}ms`);
}

function startBackendProcess(config) {
  const args = [
    "-m",
    "uvicorn",
    "backend.app.main:app",
    "--host",
    config.host,
    "--port",
    String(config.port),
  ];
  log(`starting managed backend: ${config.pythonCommand} ${args.join(" ")}`);
  const child = spawn(config.pythonCommand, args, {
    cwd: config.cwd,
    stdio: ["ignore", "pipe", "pipe"],
    env: {
      ...process.env,
      PYTHONUNBUFFERED: "1",
    },
  });
  child.stdout.on("data", (chunk) => process.stdout.write(`[backend] ${chunk}`));
  child.stderr.on("data", (chunk) => process.stderr.write(`[backend] ${chunk}`));
  child.on("exit", (code, signal) => {
    log(`managed backend exited with code=${code} signal=${signal}`);
  });
  child.on("error", (err) => {
    log(`managed backend failed to start: ${err.message}`);
  });
  return child;
}

async function ensureBackend(config) {
  try {
    const payload = await waitForHealth(config.healthUrl, 2000);
    log(`reusing existing backend at ${config.baseUrl}`);
    return {
      baseUrl: config.baseUrl,
      managed: false,
      health: payload,
      child: null,
    };
  } catch (_) {
    if (!config.manageBackend) {
      throw new Error(`backend is not reachable at ${config.baseUrl} and DESKTOP_MANAGE_BACKEND=0`);
    }
  }

  const child = startBackendProcess(config);
  const health = await waitForHealth(config.healthUrl, config.startupTimeoutMs);
  log(`managed backend is ready at ${config.baseUrl}`);
  return {
    baseUrl: config.baseUrl,
    managed: true,
    health,
    child,
  };
}

async function stopBackend(handle) {
  if (!handle || !handle.managed || !handle.child) return;
  if (handle.child.exitCode !== null || handle.child.killed) return;
  log("stopping managed backend");
  handle.child.kill("SIGTERM");
  await new Promise((resolve) => setTimeout(resolve, 1500));
  if (handle.child.exitCode === null && !handle.child.killed) {
    log("backend did not exit after SIGTERM, sending SIGKILL");
    handle.child.kill("SIGKILL");
  }
}

module.exports = {
  ensureBackend,
  stopBackend,
};
