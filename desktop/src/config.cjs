const path = require("node:path");
const { app } = require("electron");

function envBool(name, fallback) {
  const raw = process.env[name];
  if (raw === undefined) return fallback;
  return !["0", "false", "False", "FALSE", ""].includes(String(raw));
}

function resolveAppRoot() {
  if (app.isPackaged) {
    return process.resourcesPath;
  }
  return path.resolve(__dirname, "..", "..");
}

function resolveFrontendEntry(appRoot) {
  if (!app.isPackaged) {
    return path.join(appRoot, "frontend", "dist", "index.html");
  }
  return path.join(process.resourcesPath, "frontend-dist", "index.html");
}

function resolveBackendCwd(appRoot) {
  return app.isPackaged ? process.resourcesPath : appRoot;
}

function createDesktopConfig() {
  const appRoot = resolveAppRoot();
  const backendHost = process.env.DESKTOP_BACKEND_HOST || "127.0.0.1";
  const backendPort = Number(process.env.DESKTOP_BACKEND_PORT || "8000");
  const frontendDevUrl = process.env.DESKTOP_FRONTEND_URL || "http://127.0.0.1:5173";
  const pythonCommand = process.env.DESKTOP_PYTHON || "python3";
  const manageBackend = envBool("DESKTOP_MANAGE_BACKEND", true);

  return {
    appRoot,
    backend: {
      host: backendHost,
      port: backendPort,
      baseUrl: `http://${backendHost}:${backendPort}`,
      healthUrl: `http://${backendHost}:${backendPort}/api/health`,
      pythonCommand,
      manageBackend,
      cwd: resolveBackendCwd(appRoot),
      startupTimeoutMs: Number(process.env.DESKTOP_BACKEND_TIMEOUT_MS || "30000"),
    },
    frontend: {
      devUrl: frontendDevUrl,
      prodEntry: resolveFrontendEntry(appRoot),
    },
    window: {
      width: Number(process.env.DESKTOP_WINDOW_WIDTH || "1540"),
      height: Number(process.env.DESKTOP_WINDOW_HEIGHT || "980"),
    },
  };
}

module.exports = {
  createDesktopConfig,
};
