export type DesktopBridge = {
  request: (
    path: string,
    options?: {
      method?: string;
      headers?: Record<string, string>;
      body?: unknown;
    },
  ) => Promise<{ ok: boolean; status: number; data: any }>;
  pickFile?: () => Promise<string | null>;
  getCameraAccess?: () => Promise<{ status: string; granted: boolean; canPrompt: boolean }>;
  requestCameraAccess?: () => Promise<{ status: string; granted: boolean; canPrompt: boolean }>;
  openCameraSettings?: () => Promise<boolean>;
};

export const BROWSER_RUNTIME_URL = "http://127.0.0.1:7777";

export function browserBridge(): DesktopBridge {
  return {
    async request(path, options = {}) {
      const requestPath = path.startsWith("/") ? path : `/${path}`;
      const response = await fetch(`${BROWSER_RUNTIME_URL}${requestPath}`, {
        method: options.method || "GET",
        headers: {
          "Content-Type": "application/json",
          ...(options.headers || {}),
        },
        body: options.body ? JSON.stringify(options.body) : undefined,
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
    },
  };
}

export function getDesktopBridge(): DesktopBridge {
  return (window as Window & { tooriDesktop?: DesktopBridge }).tooriDesktop || browserBridge();
}

export function isDesktopBridgeAvailable(): boolean {
  return Boolean((window as Window & { tooriDesktop?: DesktopBridge }).tooriDesktop);
}

export function assetUrl(filePath: string): string {
  if (isDesktopBridgeAvailable()) {
    return `file://${filePath}`;
  }
  return `${BROWSER_RUNTIME_URL}/v1/file?path=${encodeURIComponent(filePath)}`;
}

async function readFileAsBase64(file: File): Promise<string> {
  return await new Promise((resolve, reject) => {
    const reader = new FileReader();
    reader.onerror = () => reject(new Error("Unable to read file"));
    reader.onload = () => {
      const result = String(reader.result || "");
      const [, payload = ""] = result.split(",", 2);
      resolve(payload);
    };
    reader.readAsDataURL(file);
  });
}

export async function pickImagePayload(): Promise<{ filePath?: string; imageBase64?: string } | null> {
  const bridge = getDesktopBridge();
  if (bridge.pickFile) {
    const filePath = await bridge.pickFile();
    return filePath ? { filePath } : null;
  }
  const input = document.createElement("input");
  input.type = "file";
  input.accept = "image/png,image/jpeg,image/webp";
  return await new Promise((resolve) => {
    input.onchange = async () => {
      const file = input.files?.[0];
      if (!file) {
        resolve(null);
        return;
      }
      resolve({ imageBase64: await readFileAsBase64(file) });
    };
    input.click();
  });
}

export async function runtimeRequest<T>(path: string, method = "GET", body?: unknown): Promise<T> {
  const response = await getDesktopBridge().request(path, { method, body });
  if (!response.ok) {
    const message =
      typeof response.data === "string"
        ? response.data
        : response.data?.detail || `Request failed with status ${response.status}`;
    throw new Error(message);
  }
  return response.data as T;
}

export function useRuntimeBridge() {
  const bridge = getDesktopBridge();
  return {
    bridge,
    assetUrl,
    runtimeRequest,
    pickImagePayload,
  };
}
