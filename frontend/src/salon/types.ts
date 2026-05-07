export type ChatRole = "user" | "assistant" | "system";

export type ChatMessage = {
  id: string;
  role: ChatRole;
  content: string;
  imagePreviewUrl?: string;
  uiActions?: Array<"generate_preview" | "handoff_yes" | "handoff_no">;
  /** 流式生成中（最后一条助手气泡） */
  streaming?: boolean;
  /** Dify 工作流 / 节点等过程提示 */
  streamHint?: string;
};

export type WecomSimulateResponse = {
  reply: string;
  state?: string;
  actions?: Array<"generate_preview" | "handoff_yes" | "handoff_no" | "show_assets">;
};

export type UploadImageResponse = {
  upload_file_id: string;
  filename: string;
  dify_user: string;
  preview_url?: string;
};

export type FurnishingAsset = {
  id: string;
  category: string;
  name: string;
  image_url: string;
  tags: string[];
  /** 展示用，如 ¥3280 */
  price?: string;
  /** 展示用，如 220×90×75cm；JSON 也可用 size 字段由后端映射 */
  dimensions?: string;
};

export type FurnishingAssetsListResponse = {
  items: FurnishingAsset[];
  total: number;
};
