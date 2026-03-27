const { contextBridge, ipcRenderer } = require("electron");

contextBridge.exposeInMainWorld("tooriDesktop", {
  request: (path, options = {}) => ipcRenderer.invoke("runtime:request", { path, ...options }),
  pickFile: () => ipcRenderer.invoke("runtime:pick-file"),
  getCameraAccess: () => ipcRenderer.invoke("runtime:camera-access"),
  requestCameraAccess: () => ipcRenderer.invoke("runtime:request-camera-access"),
  openCameraSettings: () => ipcRenderer.invoke("runtime:open-camera-settings"),
  openPath: (targetPath) => ipcRenderer.invoke("runtime:open-path", targetPath),
});

const _origWarn = console.warn.bind(console);
console.warn = (...args) => {
  if (typeof args[0] === 'string' && args[0].includes('electron-compat')) return;
  _origWarn(...args);
};
const _origLog = console.log.bind(console);
console.log = (...args) => {
  if (typeof args[0] === 'string' && args[0].includes('Stripped crossorigin')) return;
  _origLog(...args);
};
