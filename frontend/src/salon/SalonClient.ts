import type {
  FurnishingAssetsListResponse,
  UploadImageResponse,
  WecomSimulateResponse,
} from "./types";

export type SalonClientOptions = {
  /** e.g. `https://host/salon` or `/salon` */
  apiBase: string;
  /** `SALON_SIMULATE_TOKEN` — dev only; replace with real auth in production */
  simulateToken: string;
  /** `SALON_INTERNAL_BOOKING_TOKEN` 之一；未配置则不调 internal 素材接口 */
  internalToken?: string;
};

export type ChatAction = "generate_preview" | "handoff_yes" | "handoff_no" | "show_assets";

/** Bump when API / 素材字段变更，避免旧 localStorage 缓存长期挡住新字段（如 price、dimensions）。 */
const FURNISHING_ASSETS_CACHE_PREFIX = "furnishing-assets:v2:";
const FURNISHING_ASSETS_CACHE_TTL_MS = 24 * 60 * 60 * 1000;

type CacheEnvelope<T> = {
  expiresAt: number;
  data: T;
};

function readCache<T>(key: string): T | null {
  try {
    const raw = localStorage.getItem(key);
    if (!raw) return null;
    const parsed = JSON.parse(raw) as CacheEnvelope<T>;
    if (!parsed || typeof parsed.expiresAt !== "number" || Date.now() > parsed.expiresAt) {
      localStorage.removeItem(key);
      return null;
    }
    return parsed.data;
  } catch {
    localStorage.removeItem(key);
    return null;
  }
}

function writeCache<T>(key: string, data: T) {
  try {
    localStorage.setItem(
      key,
      JSON.stringify({
        expiresAt: Date.now() + FURNISHING_ASSETS_CACHE_TTL_MS,
        data,
      } satisfies CacheEnvelope<T>),
    );
  } catch {
    // Storage can be unavailable or full; keep network behavior unchanged.
  }
}

/**
 * Thin HTTP wrapper around `salon_gateway` simulate routes.
 * Swap implementation or extend methods when you add `/api/v1/chat` + JWT.
 */
export class SalonClient {
  constructor(private readonly opts: SalonClientOptions) {}

  private authHeaders(json = true): HeadersInit {
    const h: Record<string, string> = {
      Authorization: `Bearer ${this.opts.simulateToken}`,
    };
    if (json) h["Content-Type"] = "application/json";
    return h;
  }

  private internalAuthHeaders(): HeadersInit {
    const t = (this.opts.internalToken || "").trim();
    if (!t) throw new Error("internal token not configured");
    return { Authorization: `Bearer ${t}` };
  }

  private url(path: string): string {
    const base = this.opts.apiBase.replace(/\/$/, "");
    return `${base}${path.startsWith("/") ? path : `/${path}`}`;
  }

  uploadedImagePreviewUrl(upload_file_id: string): string {
    const sp = new URLSearchParams({ token: this.opts.simulateToken });
    return `${this.url(`/simulate/upload-image/${encodeURIComponent(upload_file_id)}`)}?${sp}`;
  }

  async sendWecomText(body: {
    from_user: string;
    content: string;
    upload_file_id?: string | null;
    image_url?: string | null;
    selected_asset_ids?: string[];
    action?: ChatAction;
  }): Promise<WecomSimulateResponse> {
    const res = await fetch(this.url("/simulate/wecom-text"), {
      method: "POST",
      headers: this.authHeaders(true),
      body: JSON.stringify({
        content: body.content,
        from_user: body.from_user,
        to_user: "web",
        upload_file_id: body.upload_file_id || undefined,
        image_url: body.image_url || undefined,
        selected_asset_ids: (body.selected_asset_ids || []).slice(0, 4),
        action: body.action || undefined,
      }),
    });
    if (!res.ok) {
      const text = await res.text().catch(() => "");
      throw new Error(text || `HTTP ${res.status}`);
    }
    const data = (await res.json()) as WecomSimulateResponse;
    return { ...data, reply: (data.reply || "").trim() || "（无回复）" };
  }

  /**
   * Dify 流式：``POST /simulate/wecom-text-stream``，解析 ``data: {json}`` 行。
   * 网关透传 Dify SSE；反代需关闭 buffering（如 ``X-Accel-Buffering: no``）。
   */
  async sendWecomTextStream(
    body: {
      from_user: string;
      content: string;
      upload_file_id?: string | null;
      image_url?: string | null;
      selected_asset_ids?: string[];
      action?: ChatAction;
    },
    onData: (obj: Record<string, unknown>) => void,
    options?: { signal?: AbortSignal },
  ): Promise<void> {
    const res = await fetch(this.url("/simulate/wecom-text-stream"), {
      method: "POST",
      headers: this.authHeaders(true),
      body: JSON.stringify({
        content: body.content,
        from_user: body.from_user,
        to_user: "web",
        upload_file_id: body.upload_file_id || undefined,
        image_url: body.image_url || undefined,
        selected_asset_ids: (body.selected_asset_ids || []).slice(0, 4),
        action: body.action || undefined,
      }),
      signal: options?.signal,
    });
    if (!res.ok) {
      const text = await res.text().catch(() => "");
      throw new Error(text || `HTTP ${res.status}`);
    }
    if (!res.body) throw new Error("empty response body");
    const reader = res.body.getReader();
    const decoder = new TextDecoder();
    let buffer = "";
    const flushBlock = (block: string) => {
      for (const rawLine of block.split("\n")) {
        const line = rawLine.replace(/\r$/, "");
        if (!line.startsWith("data:")) continue;
        const payload = line.slice(5).trim();
        if (!payload || payload === "[DONE]") continue;
        try {
          onData(JSON.parse(payload) as Record<string, unknown>);
        } catch {
          /* 非 JSON 行忽略 */
        }
      }
    };
    while (true) {
      const { done, value } = await reader.read();
      if (done) break;
      buffer += decoder.decode(value, { stream: true });
      let sep: number;
      while ((sep = buffer.indexOf("\n\n")) !== -1) {
        const block = buffer.slice(0, sep);
        buffer = buffer.slice(sep + 2);
        flushBlock(block);
      }
    }
    if (buffer.trim()) flushBlock(buffer);
  }

  async uploadImage(file: File, fromUser: string): Promise<UploadImageResponse> {
    const fd = new FormData();
    fd.append("file", file, file.name);
    const q = new URLSearchParams({ from_user: fromUser });
    const res = await fetch(`${this.url("/simulate/upload-image")}?${q}`, {
      method: "POST",
      headers: this.authHeaders(false),
      body: fd,
    });
    if (!res.ok) {
      const text = await res.text().catch(() => "");
      throw new Error(text || `HTTP ${res.status}`);
    }
    return (await res.json()) as UploadImageResponse;
  }

  async listFurnishingAssets(params?: {
    q?: string;
    category?: string;
    limit?: number;
  }): Promise<FurnishingAssetsListResponse> {
    const sp = new URLSearchParams();
    if (params?.q?.trim()) sp.set("q", params.q.trim());
    if (params?.category?.trim()) sp.set("category", params.category.trim());
    sp.set("limit", String(params?.limit ?? 100));
    const path = `/internal/furnishing-assets?${sp}`;
    const cacheKey = `${FURNISHING_ASSETS_CACHE_PREFIX}${this.opts.apiBase}|${path}`;
    const cached = readCache<FurnishingAssetsListResponse>(cacheKey);
    if (cached) return cached;
    const res = await fetch(this.url(path), {
      method: "GET",
      headers: this.internalAuthHeaders(),
    });
    if (!res.ok) {
      const text = await res.text().catch(() => "");
      throw new Error(text || `HTTP ${res.status}`);
    }
    const data = (await res.json()) as FurnishingAssetsListResponse;
    writeCache(cacheKey, data);
    return data;
  }
}
