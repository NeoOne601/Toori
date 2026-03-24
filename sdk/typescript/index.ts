export type TooriAnalyzeRequest = {
  imageBase64?: string;
  filePath?: string;
  sessionId?: string;
  query?: string;
  decodeMode?: "off" | "auto" | "force";
  topK?: number;
};

export class TooriClient {
  constructor(
    private readonly baseUrl = "http://127.0.0.1:7777",
    private readonly apiKey?: string,
  ) {}

  private async request(path: string, init?: RequestInit) {
    const response = await fetch(`${this.baseUrl}${path}`, {
      ...init,
      headers: {
        "Content-Type": "application/json",
        ...(this.apiKey ? { "X-API-Key": this.apiKey } : {}),
        ...(init?.headers || {}),
      },
    });
    if (!response.ok) {
      throw new Error(await response.text());
    }
    return response.json();
  }

  analyze(payload: TooriAnalyzeRequest) {
    return this.request("/v1/analyze", {
      method: "POST",
      body: JSON.stringify({
        image_base64: payload.imageBase64,
        file_path: payload.filePath,
        session_id: payload.sessionId ?? "default",
        query: payload.query,
        decode_mode: payload.decodeMode ?? "auto",
        top_k: payload.topK ?? 6,
      }),
    });
  }

  query(query: string, sessionId = "default", topK = 6) {
    return this.request("/v1/query", {
      method: "POST",
      body: JSON.stringify({ query, session_id: sessionId, top_k: topK }),
    });
  }

  settings() {
    return this.request("/v1/settings");
  }
}
