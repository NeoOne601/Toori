const fs = require("fs");
const path = require("path");
const { app, BrowserWindow, dialog, ipcMain, session, shell, systemPreferences } = require("electron");

const RUNTIME_URL = process.env.TOORI_RUNTIME_URL || "http://127.0.0.1:7777";

app.commandLine.appendSwitch("autoplay-policy", "no-user-gesture-required");
app.setName("Toori Lens Assistant");
app.setAppLogsPath();

function logFilePath() {
  try {
    return path.join(app.getPath("logs"), "toori-electron.log");
  } catch {
    return path.join(process.cwd(), "toori-electron.log");
  }
}

function logEvent(label, details) {
  const message = `[${new Date().toISOString()}] ${label}${details ? ` ${details}` : ""}\n`;
  try {
    fs.appendFileSync(logFilePath(), message);
  } catch {
    // Avoid crashing the app because logging failed.
  }
}

function getCameraAccessState() {
  if (process.platform !== "darwin") {
    return { status: "granted", granted: true, canPrompt: false };
  }
  const status = systemPreferences.getMediaAccessStatus("camera");
  return {
    status,
    granted: status === "granted",
    canPrompt: status === "not-determined",
  };
}

async function requestCameraAccess() {
  if (process.platform !== "darwin") {
    return getCameraAccessState();
  }
  const current = getCameraAccessState();
  if (current.granted || !current.canPrompt) {
    return current;
  }
  const granted = await systemPreferences.askForMediaAccess("camera");
  return {
    ...getCameraAccessState(),
    granted,
  };
}

function configureSessionPermissions() {
  const defaultSession = session.defaultSession;
  defaultSession.setPermissionCheckHandler((_webContents, permission) => {
    if (permission === "media" || permission === "camera" || permission === "microphone") {
      return getCameraAccessState().granted;
    }
    return true;
  });
  defaultSession.setPermissionRequestHandler((webContents, permission, callback) => {
    if (permission === "media" || permission === "camera" || permission === "microphone") {
      requestCameraAccess()
        .then((state) => callback(state.granted))
        .catch(() => callback(false));
      return;
    }
    callback(true);
  });
  if (typeof defaultSession.setDevicePermissionHandler === "function") {
    defaultSession.setDevicePermissionHandler((details) => {
      return details.deviceType === "videoCapture" || details.deviceType === "audioCapture";
    });
  }
}

const isDev = Boolean(process.env.VITE_DEV_SERVER_URL);

// ─── SECTION 1: GPU / Tile memory fix ────────────────────────────────────────
app.commandLine.appendSwitch('--disable-gpu-sandbox');
app.commandLine.appendSwitch('--enable-gpu-rasterization');
app.commandLine.appendSwitch('--enable-zero-copy');
app.commandLine.appendSwitch('--gpu-memory-buffer-size', '134217728'); // 128MB
// FAST FALLBACK for persistent "tile memory limits exceeded"
app.commandLine.appendSwitch('--disable-gpu-compositing');
app.commandLine.appendSwitch('force-device-scale-factor', '1');

function createWindow() {
  const isDev = Boolean(process.env.VITE_DEV_SERVER_URL);

  const win = new BrowserWindow({
    width: 1560,
    height: 980,
    minWidth: 1280,
    minHeight: 820,
    backgroundColor: "#07111b",
    title: "Toori Lens Assistant",
    webPreferences: {
      preload: path.join(__dirname, "preload.js"),
      nodeIntegration: false,
      contextIsolation: true,
      webSecurity: true,
      allowRunningInsecureContent: false,
      sandbox: true,
      backgroundThrottling: false,
    },
  });

  win.webContents.session.webRequest.onHeadersReceived((details, callback) => {
    callback({
      responseHeaders: {
        ...details.responseHeaders,
        'Content-Security-Policy': [
          [
            "default-src 'self' file: data: blob:;",
            "script-src 'self' 'unsafe-inline' file:;",
            "style-src 'self' 'unsafe-inline' file: https://fonts.googleapis.com;",
            "font-src 'self' file: data: https://fonts.gstatic.com;",
            "connect-src 'self' http://127.0.0.1:7777 ws://127.0.0.1:7777;",
            "img-src 'self' file: data: blob: http://127.0.0.1:7777;",
            "media-src 'self' file: blob:;",
            "worker-src 'self' blob:;",
          ].join(' ')
        ]
      }
    });
  });

  if (process.env.VITE_DEV_SERVER_URL) {
    win.loadURL(process.env.VITE_DEV_SERVER_URL);
    win.webContents.openDevTools({ mode: "detach" });
  } else {
    win.loadFile(path.join(__dirname, "dist", "index.html"));
    // Uncomment to debug production build issues:
    // if (isDev) win.webContents.openDevTools({ mode: "detach" });
  }

  win.webContents.on("did-fail-load", (_event, code, description, validatedURL) => {
    logEvent("did-fail-load", `${code} ${description} ${validatedURL}`);
  });

  win.webContents.on("render-process-gone", (_event, details) => {
    logEvent("render-process-gone", JSON.stringify(details));
  });
}


app.whenReady().then(() => {
  logEvent("app-ready");
  configureSessionPermissions();
  createWindow();

  app.on("activate", () => {
    if (BrowserWindow.getAllWindows().length === 0) {
      createWindow();
    }
  });
});

process.on("uncaughtException", (error) => {
  logEvent("uncaught-exception", error?.stack || String(error));
});

process.on("unhandledRejection", (error) => {
  logEvent("unhandled-rejection", error?.stack || String(error));
});

app.on("child-process-gone", (_event, details) => {
  logEvent("child-process-gone", JSON.stringify(details));
});

app.on("window-all-closed", () => {
  if (process.platform !== "darwin") {
    app.quit();
  }
});

ipcMain.handle("runtime:request", async (_, payload) => {
  const requestPath = payload.path.startsWith("/") ? payload.path : `/${payload.path}`;
  try {
    const response = await fetch(`${RUNTIME_URL}${requestPath}`, {
      method: payload.method || "GET",
      headers: {
        "Content-Type": "application/json",
        ...(payload.headers || {}),
      },
      body: payload.body ? JSON.stringify(payload.body) : undefined,
    });

    const contentType = response.headers.get("content-type") || "";
    const data = contentType.includes("application/json")
      ? await response.json()
      : await response.text();

    return {
      ok: response.ok,
      status: response.status,
      data,
    };
  } catch (error) {
    return {
      ok: false,
      status: 503,
      data: { error: "Runtime unreachable", details: String(error) }
    };
  }
});

ipcMain.handle("runtime:pick-file", async () => {
  const result = await dialog.showOpenDialog({
    properties: ["openFile"],
    filters: [
      { name: "Images", extensions: ["png", "jpg", "jpeg", "webp"] },
      { name: "All Files", extensions: ["*"] },
    ],
  });
  if (result.canceled || result.filePaths.length === 0) {
    return null;
  }
  return result.filePaths[0];
});

ipcMain.handle("runtime:pick-folder", async () => {
  const result = await dialog.showOpenDialog({
    properties: ["openDirectory", "createDirectory"],
    title: "Choose Smriti Storage Location",
    buttonLabel: "Select Folder",
    message: "Choose where Smriti should store media indexes, embeddings, and thumbnails.",
  });
  if (result.canceled || result.filePaths.length === 0) {
    return null;
  }
  return result.filePaths[0];
});

ipcMain.handle("runtime:camera-access", async () => getCameraAccessState());
ipcMain.handle("runtime:request-camera-access", async () => requestCameraAccess());

ipcMain.handle("runtime:open-camera-settings", async () => {
  if (process.platform === "darwin") {
    await shell.openExternal("x-apple.systempreferences:com.apple.preference.security?Privacy_Camera");
    return true;
  }
  return false;
});

ipcMain.handle("runtime:open-path", async (_event, targetPath) => {
  if (!targetPath) {
    return "Missing path";
  }
  return await shell.openPath(targetPath);
});
