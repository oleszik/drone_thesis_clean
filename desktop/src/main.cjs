const path = require("node:path");
const { app, BrowserWindow, dialog } = require("electron");

const { ensureBackend, stopBackend } = require("./backend.cjs");
const { createDesktopConfig } = require("./config.cjs");

let mainWindow = null;
let backendHandle = null;

function log(msg) {
  console.log(`[desktop] ${msg}`);
}

function createWindow(config) {
  mainWindow = new BrowserWindow({
    width: config.window.width,
    height: config.window.height,
    minWidth: 1200,
    minHeight: 760,
    autoHideMenuBar: true,
    title: "Drone Thesis Desktop",
    webPreferences: {
      preload: path.join(__dirname, "preload.cjs"),
      contextIsolation: true,
      nodeIntegration: false,
      sandbox: false,
    },
  });

  mainWindow.on("closed", () => {
    mainWindow = null;
  });
}

async function loadRenderer(config) {
  if (!mainWindow) return;
  if (!app.isPackaged) {
    log(`loading frontend dev server ${config.frontend.devUrl}`);
    await mainWindow.loadURL(config.frontend.devUrl);
    mainWindow.webContents.openDevTools({ mode: "detach" });
    return;
  }
  log(`loading built frontend ${config.frontend.prodEntry}`);
  await mainWindow.loadFile(config.frontend.prodEntry);
}

async function bootstrap() {
  const config = createDesktopConfig();
  log(`app root: ${config.appRoot}`);
  log(`backend target: ${config.backend.baseUrl}`);

  try {
    backendHandle = await ensureBackend(config.backend);
    createWindow(config);
    await loadRenderer(config);
  } catch (err) {
    log(`bootstrap failed: ${err.stack || err.message}`);
    await dialog.showMessageBox({
      type: "error",
      title: "Desktop startup failed",
      message: "Could not start the Drone Thesis desktop shell.",
      detail: String(err.stack || err.message || err),
    });
    app.quit();
  }
}

app.whenReady().then(bootstrap);

app.on("window-all-closed", async () => {
  await stopBackend(backendHandle);
  if (process.platform !== "darwin") {
    app.quit();
  }
});

app.on("before-quit", async () => {
  await stopBackend(backendHandle);
});

app.on("activate", async () => {
  if (BrowserWindow.getAllWindows().length === 0) {
    const config = createDesktopConfig();
    createWindow(config);
    await loadRenderer(config);
  }
});
