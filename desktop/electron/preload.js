const { contextBridge, ipcRenderer } = require("electron");

contextBridge.exposeInMainWorld("tooriDesktop", {
  request: (path, options = {}) => ipcRenderer.invoke("runtime:request", { path, ...options }),
  pickFile: () => ipcRenderer.invoke("runtime:pick-file"),
  getCameraAccess: () => ipcRenderer.invoke("runtime:camera-access"),
  requestCameraAccess: () => ipcRenderer.invoke("runtime:request-camera-access"),
  openCameraSettings: () => ipcRenderer.invoke("runtime:open-camera-settings"),
});
