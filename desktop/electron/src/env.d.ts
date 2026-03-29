export {};

declare global {
  interface Window {
    tooriDesktop?: {
      request: (
        path: string,
        options?: {
          method?: string;
          headers?: Record<string, string>;
          body?: unknown;
        },
      ) => Promise<{ ok: boolean; status: number; data: any }>;
      pickFile: () => Promise<string | null>;
      pickFolder: () => Promise<string | null>;
      getCameraAccess: () => Promise<{ status: string; granted: boolean; canPrompt: boolean }>;
      requestCameraAccess: () => Promise<{ status: string; granted: boolean; canPrompt: boolean }>;
      openCameraSettings: () => Promise<boolean>;
    };
  }
}
