const { contextBridge, ipcRenderer } = require('electron');

contextBridge.exposeInMainWorld('electronAPI', {
  search: (embedding) => ipcRenderer.invoke('search-api', embedding)
});
