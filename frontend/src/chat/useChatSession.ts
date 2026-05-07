import { useCallback, useEffect, useState } from "react";
import type { ChatMessage } from "@/salon/types";
import { useSalonClient } from "@/salon/SalonClientContext";
import type { ChatAction } from "@/salon/SalonClient";
import { mergeAnswerChunk, pickStreamHint } from "./difyStream";
import { loadOrCreateFromUser, saveFromUser } from "./ids";

const CHAT_CACHE_KEY = "furnishing-chat-session:v1";
const CHAT_CACHE_TTL_MS = 24 * 60 * 60 * 1000;

type ChatCache = {
  expiresAt: number;
  fromUser: string;
  messages: ChatMessage[];
  handoffAnswered: boolean;
  hasRoomImage: boolean;
  hasSelectedAssets: boolean;
};

function uid() {
  return crypto.randomUUID();
}

function normalizePreviewUrl(
  rawUrl: string | undefined,
  uploadFileId: string,
  fallback: (upload_file_id: string) => string,
): string {
  const u = (rawUrl || "").trim();
  if (!u) return fallback(uploadFileId);
  if (/^https?:\/\//i.test(u)) return u;
  if (u.startsWith("/salon/simulate/upload-image/")) return u;
  if (u.startsWith("/simulate/upload-image/")) {
    return u.replace(/^\/simulate\/upload-image\//, "/salon/simulate/upload-image/");
  }
  return fallback(uploadFileId);
}

function readChatCache(): Omit<ChatCache, "expiresAt"> | null {
  try {
    const raw = localStorage.getItem(CHAT_CACHE_KEY);
    if (!raw) return null;
    const parsed = JSON.parse(raw) as ChatCache;
    if (!parsed || typeof parsed.expiresAt !== "number" || Date.now() > parsed.expiresAt) {
      localStorage.removeItem(CHAT_CACHE_KEY);
      return null;
    }
    return {
      fromUser: parsed.fromUser,
      messages: (parsed.messages || []).map((m) => ({
        ...m,
        streaming: false,
        streamHint: undefined,
      })),
      handoffAnswered: Boolean(parsed.handoffAnswered),
      hasRoomImage: Boolean(parsed.hasRoomImage),
      hasSelectedAssets: Boolean(parsed.hasSelectedAssets),
    };
  } catch {
    localStorage.removeItem(CHAT_CACHE_KEY);
    return null;
  }
}

function writeChatCache(data: Omit<ChatCache, "expiresAt">) {
  try {
    localStorage.setItem(
      CHAT_CACHE_KEY,
      JSON.stringify({
        ...data,
        messages: data.messages.map((m) => ({
          ...m,
          streaming: false,
          streamHint: undefined,
        })),
        expiresAt: Date.now() + CHAT_CACHE_TTL_MS,
      } satisfies ChatCache),
    );
  } catch {
    // localStorage can be full or disabled; chat still works without persistence.
  }
}

/**
 * Chat state + send/upload。
 * 优先走 Dify 流式（与 Dify 网页类似的增量输出）；失败时回退阻塞接口。
 */
export function useChatSession() {
  const client = useSalonClient();
  const cached = readChatCache();
  const [fromUser, setFromUserState] = useState(cached?.fromUser || loadOrCreateFromUser);
  const [messages, setMessages] = useState<ChatMessage[]>(cached?.messages || []);
  const [pendingFileId, setPendingFileId] = useState<string | null>(null);
  const [pendingImagePreviewUrl, setPendingImagePreviewUrl] = useState<string | null>(null);
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState<string | null>(null);
  const [handoffAnswered, setHandoffAnswered] = useState(cached?.handoffAnswered || false);
  const [hasRoomImage, setHasRoomImage] = useState(cached?.hasRoomImage || false);
  const [hasSelectedAssets, setHasSelectedAssets] = useState(cached?.hasSelectedAssets || false);

  useEffect(() => {
    setMessages((prev) =>
      prev.map((m) => {
        if (m.role !== "user" || !m.imagePreviewUrl) return m;
        const mm = m.imagePreviewUrl.match(/\/(?:salon\/)?simulate\/upload-image\/([^?\/]+)/);
        if (!mm?.[1]) return m;
        const fixed = normalizePreviewUrl(m.imagePreviewUrl, decodeURIComponent(mm[1]), (id) =>
          client.uploadedImagePreviewUrl(id),
        );
        return fixed === m.imagePreviewUrl ? m : { ...m, imagePreviewUrl: fixed };
      }),
    );
  }, [client]);

  useEffect(() => {
    writeChatCache({
      fromUser,
      messages,
      handoffAnswered,
      hasRoomImage,
      hasSelectedAssets,
    });
  }, [fromUser, messages, handoffAnswered, hasRoomImage, hasSelectedAssets]);

  const setUserId = useCallback((id: string) => {
    const t = id.trim();
    setFromUserState(t);
    saveFromUser(t);
  }, []);

  const send = useCallback(
    async (content: string, selectedAssetIds: string[] = [], action?: ChatAction) => {
      const trimmed = content.trim();
      const hasSelectedAssets = selectedAssetIds.length > 0;
      if (!trimmed && !pendingFileId && !hasSelectedAssets && !action) return;
      const assetOnlyText = hasSelectedAssets
        ? `我选择了这些素材：${selectedAssetIds.join("、")}`
        : "";
      const actionText =
        action === "generate_preview"
          ? "生成效果图"
          : action === "handoff_yes"
            ? "需要转人工"
            : action === "handoff_no"
              ? "暂不转人工"
              : action === "show_assets"
                ? "查看可选素材"
                : "";
      const outgoingContent = trimmed || assetOnlyText || actionText;
      setError(null);
      if (action === "handoff_yes" || action === "handoff_no") {
        setHandoffAnswered(true);
      }
      const userLine: ChatMessage = {
        id: uid(),
        role: "user",
        content: outgoingContent,
        imagePreviewUrl: pendingImagePreviewUrl || undefined,
      };
      const assistantId = uid();
      const assistantPlaceholder: ChatMessage = {
        id: assistantId,
        role: "assistant",
        content: "",
        streaming: true,
        streamHint: "稍等…",
      };
      setMessages((m) => [...m, userLine, assistantPlaceholder]);
      setLoading(true);
      const fileId = pendingFileId;
      setPendingFileId(null);
      setPendingImagePreviewUrl(null);
      if (fileId) setHasRoomImage(true);
      if (selectedAssetIds.length > 0) setHasSelectedAssets(true);

      const applyAssistant = (patch: Partial<ChatMessage>) => {
        setMessages((prev) =>
          prev.map((msg) => (msg.id === assistantId ? { ...msg, ...patch } : msg)),
        );
      };

      try {
        let accumulated = "";
        let serverUiActions: ChatMessage["uiActions"] = [];
        const errBox: { err: Error | null } = { err: null };
        try {
          await client.sendWecomTextStream(
            {
              from_user: fromUser,
              content: outgoingContent,
              upload_file_id: fileId,
              selected_asset_ids: selectedAssetIds,
              action,
            },
            (d) => {
              if (d.event === "error") {
                errBox.err = new Error(String(d.message || "Dify 流式错误"));
                return;
              }
              if (d.event === "ui_actions" && Array.isArray(d.actions)) {
                serverUiActions = d.actions.filter(
                  (x): x is NonNullable<ChatMessage["uiActions"]>[number] =>
                    x === "generate_preview" || x === "handoff_yes" || x === "handoff_no",
                );
                return;
              }
              const hint = pickStreamHint(d);
              if (typeof d.answer === "string") {
                accumulated = mergeAnswerChunk(accumulated, d.answer);
                applyAssistant({
                  content: accumulated,
                  ...(hint ? { streamHint: hint } : {}),
                });
              } else if (hint) {
                applyAssistant({ streamHint: hint });
              }
            },
          );
          if (errBox.err) throw errBox.err;
          applyAssistant({
            content: (accumulated || "").trim() || "（无回复）",
            streaming: false,
            streamHint: undefined,
            uiActions: serverUiActions,
          });
        } catch {
          const data = await client.sendWecomText({
            from_user: fromUser,
            content: outgoingContent,
            upload_file_id: fileId,
            selected_asset_ids: selectedAssetIds,
            action,
          });
          applyAssistant({
            content: (data.reply || "").trim() || "（无回复）",
            streaming: false,
            streamHint: undefined,
            uiActions: (data.actions || []).filter(
              (x): x is NonNullable<ChatMessage["uiActions"]>[number] =>
                x === "generate_preview" || x === "handoff_yes" || x === "handoff_no",
            ),
          });
        }
      } catch (e) {
        const msg = e instanceof Error ? e.message : String(e);
        setError(msg);
        setMessages((m) => [
          ...m.filter((x) => x.id !== assistantId),
          { id: uid(), role: "system", content: `发送失败：${msg}` },
        ]);
      } finally {
        setLoading(false);
      }
    },
    [client, fromUser, pendingFileId, pendingImagePreviewUrl],
  );

  const userMessageCount = messages.filter((m) => m.role === "user").length;
  const handoffPrompted =
    !handoffAnswered &&
    userMessageCount >= 2;

  const chooseHandoff = useCallback(
    async (needHandoff: boolean) => {
      setError(null);
      setLoading(true);
      const action: ChatAction = needHandoff ? "handoff_yes" : "handoff_no";
      const userText = needHandoff ? "需要转人工" : "暂不转人工";
      setMessages((m) => [...m, { id: uid(), role: "user", content: userText }]);
      try {
        const data = await client.sendWecomText({
          from_user: fromUser,
          content: "",
          action,
        });
        const reply = (data.reply || "").trim();
        const fallback = needHandoff ? "已登记转人工。" : "已记录你的选择。";
        const uiActions = (data.actions || []).filter(
          (x): x is NonNullable<ChatMessage["uiActions"]>[number] =>
            x === "generate_preview" || x === "handoff_yes" || x === "handoff_no",
        );
        setMessages((m) => [
          ...m,
          {
            id: uid(),
            role: "assistant",
            content: reply || fallback,
            ...(uiActions.length ? { uiActions } : {}),
          },
        ]);
      } catch (e) {
        const msg = e instanceof Error ? e.message : String(e);
        setError(msg);
        setMessages((m) => [...m, { id: uid(), role: "system", content: `记录失败：${msg}` }]);
      } finally {
        setLoading(false);
      }
      setHandoffAnswered(true);
    },
    [client, fromUser],
  );

  const pickImage = useCallback(
    async (file: File | null) => {
      if (!file) return;
      setError(null);
      setLoading(true);
      try {
        const { upload_file_id, preview_url } = await client.uploadImage(file, fromUser);
        setPendingFileId(upload_file_id);
        setPendingImagePreviewUrl(
          normalizePreviewUrl(preview_url, upload_file_id, (id) => client.uploadedImagePreviewUrl(id)),
        );
      } catch (e) {
        const msg = e instanceof Error ? e.message : String(e);
        setError(msg);
      } finally {
        setLoading(false);
      }
    },
    [client, fromUser],
  );

  const clearPendingFile = useCallback(() => {
    setPendingFileId(null);
    setPendingImagePreviewUrl(null);
  }, []);

  return {
    fromUser,
    setUserId,
    messages,
    pendingFileId,
    hasRoomImage,
    hasSelectedAssets,
    loading,
    error,
    handoffPrompted,
    chooseHandoff,
    send,
    pickImage,
    clearPendingFile,
  };
}
