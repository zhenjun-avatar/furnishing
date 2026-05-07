import Markdown from "react-markdown";
import remarkGfm from "remark-gfm";
import { useEffect, useRef } from "react";
import type { ChatMessage, FurnishingAsset } from "@/salon/types";
import {
  absolutizeMediaUrl,
  assistantContentForMd,
  markdownUrlTransform,
  MarkdownImg,
} from "./assistantMarkdown";
import { formatAssetPriceDimensions } from "@/salon/formatAssetSpec";
import styles from "./MessageList.module.css";

function canonicalAssetId(raw: string, availableAssetIds: string[]): string | null {
  const lower = raw.toLowerCase();
  const exact = availableAssetIds.find((id) => id.toLowerCase() === lower);
  if (exact) return exact;
  const m = raw.match(/^(.+-)(\d+)$/i);
  if (!m) return availableAssetIds.length ? null : raw;
  const [, prefix, num] = m;
  const n = Number(num);
  const matches = availableAssetIds.filter((id) => {
    const im = id.match(/^(.+-)(\d+)$/i);
    return im && im[1].toLowerCase() === prefix.toLowerCase() && Number(im[2]) === n;
  });
  if (matches.length === 1) return matches[0];
  return availableAssetIds.length ? null : raw;
}

function extractAssetIds(text: string, availableAssetIds: string[]): string[] {
  const found = text.match(/\bdemo-[a-z0-9-]+\b/gi) || [];
  return Array.from(
    new Set(
      found
        .map((x) => canonicalAssetId(x.trim(), availableAssetIds))
        .filter((x): x is string => Boolean(x)),
    ),
  ).slice(0, 12);
}

function Bubble({
  m,
  selectedAssetIds,
  availableAssetsById,
  showHandoffActions,
  showGenerateAction,
  onToggleAssetId,
  onChooseHandoff,
  onGeneratePreview,
}: {
  m: ChatMessage;
  selectedAssetIds: string[];
  availableAssetsById: Record<string, FurnishingAsset>;
  showHandoffActions: boolean;
  showGenerateAction: boolean;
  onToggleAssetId: (id: string) => void;
  onChooseHandoff: (needHandoff: boolean) => void;
  onGeneratePreview: () => void;
}) {
  const isUser = m.role === "user";
  const isSystem = m.role === "system";
  const isAssistant = m.role === "assistant";
  const availableAssetIds = Object.keys(availableAssetsById);
  const assetIds = isAssistant ? extractAssetIds(m.content, availableAssetIds) : [];
  const selectedSet = new Set(selectedAssetIds);
  return (
    <div
      className={`${styles.row} ${isUser ? styles.rowUser : ""} ${isSystem ? styles.rowSystem : ""}`}
    >
      <div
        className={`${styles.bubble} ${isUser ? styles.bubbleUser : ""} ${isSystem ? styles.bubbleSystem : ""} ${isAssistant && m.streaming ? styles.bubbleStreaming : ""}`}
      >
        {isAssistant ? (
          <>
            {m.streamHint ? <div className={styles.streamHint}>{m.streamHint}</div> : null}
            {m.streaming && !m.content.trim() ? (
              <div className={styles.typingRow} aria-live="polite">
                <span className={styles.typingDot} />
                <span className={styles.typingDot} />
                <span className={styles.typingDot} />
              </div>
            ) : null}
            {m.content.trim() ? (
              <div className={styles.md}>
                <Markdown
                  remarkPlugins={[remarkGfm]}
                  urlTransform={markdownUrlTransform}
                  components={{ img: MarkdownImg }}
                >
                  {assistantContentForMd(m.content)}
                </Markdown>
              </div>
            ) : null}
            {assetIds.length > 0 ? (
              <div className={styles.assetActions}>
                <span className={styles.assetActionLabel}>选择素材</span>
                {assetIds.map((id) => {
                  const meta = availableAssetsById[id];
                  const selected = selectedSet.has(id);
                  const spec = meta ? formatAssetPriceDimensions(meta) : null;
                  return (
                    <div key={id} className={`${styles.assetCard} ${selected ? styles.assetCardOn : ""}`}>
                      {meta?.image_url ? (
                        <img
                          className={styles.assetThumb}
                          src={absolutizeMediaUrl(meta.image_url)}
                          alt={meta.name || id}
                          loading="lazy"
                          decoding="async"
                          referrerPolicy="no-referrer"
                        />
                      ) : (
                        <div className={styles.assetThumbFallback} aria-hidden />
                      )}
                      <div className={styles.assetMeta}>
                        <div className={styles.assetName}>{meta?.name || id}</div>
                        {spec ? <div className={styles.assetSpec}>{spec}</div> : null}
                        <div className={styles.assetId}>{id}</div>
                      </div>
                      <button
                        type="button"
                        className={`${styles.assetActionBtn} ${selected ? styles.assetActionBtnOn : ""}`}
                        disabled={!selected && selectedAssetIds.length >= 4}
                        onClick={() => onToggleAssetId(id)}
                        aria-label={selected ? `取消选择 ${id}` : `选择 ${id}`}
                      >
                        <span className={`${styles.assetCheck} ${selected ? styles.assetCheckOn : ""}`} aria-hidden>
                          {selected ? "✓" : ""}
                        </span>
                        <span className={styles.assetActionText}></span>
                      </button>
                    </div>
                  );
                })}
                <div className={styles.assetSidebarHint} aria-live="polite">
                  <span className={styles.assetSidebarArrow}>←</span>
                  选择更多素材请到左侧边栏
                </div>
              </div>
            ) : null}
            {showHandoffActions || showGenerateAction ? (
              <div className={styles.replyActions}>
                {showGenerateAction ? (
                  <button type="button" className={styles.primaryReplyAction} onClick={onGeneratePreview}>
                    生成效果图
                  </button>
                ) : null}
                {showHandoffActions ? (
                  <>
                    <button
                      type="button"
                      className={styles.replyAction}
                      onClick={() => onChooseHandoff(true)}
                    >
                      转人工
                    </button>
                    <button
                      type="button"
                      className={styles.replyActionGhost}
                      onClick={() => onChooseHandoff(false)}
                    >
                      暂不转人工
                    </button>
                  </>
                ) : null}
              </div>
            ) : null}
          </>
        ) : (
          <>
            {m.imagePreviewUrl ? (
              <div className={styles.userImageFrame}>
                <img className={styles.userImagePreview} src={m.imagePreviewUrl} alt="上传实景图" />
              </div>
            ) : null}
            {m.content.trim() ? <div className={styles.plain}>{m.content}</div> : null}
          </>
        )}
      </div>
    </div>
  );
}

export function MessageList({
  messages,
  selectedAssetIds,
  availableAssetsById,
  actionMessageId,
  showHandoffActions,
  showGenerateAction,
  onToggleAssetId,
  onChooseHandoff,
  onGeneratePreview,
}: {
  messages: ChatMessage[];
  selectedAssetIds: string[];
  availableAssetsById: Record<string, FurnishingAsset>;
  actionMessageId: string | null;
  showHandoffActions: boolean;
  showGenerateAction: boolean;
  onToggleAssetId: (id: string) => void;
  onChooseHandoff: (needHandoff: boolean) => void;
  onGeneratePreview: () => void;
}) {
  const bottomRef = useRef<HTMLDivElement | null>(null);
  const firstScrollDoneRef = useRef(false);

  useEffect(() => {
    if (!bottomRef.current) return;
    bottomRef.current.scrollIntoView({
      block: "end",
      behavior: firstScrollDoneRef.current ? "smooth" : "auto",
    });
    firstScrollDoneRef.current = true;
  }, [messages.length]);

  return (
    <div className={styles.list}>
      {messages.length === 0 ? (
        <p className={styles.empty}>发送消息开始对话</p>
      ) : (
        messages.map((m) => (
          <Bubble
            key={m.id}
            m={m}
            selectedAssetIds={selectedAssetIds}
            availableAssetsById={availableAssetsById}
            showHandoffActions={m.id === actionMessageId && showHandoffActions}
            showGenerateAction={m.id === actionMessageId && showGenerateAction}
            onToggleAssetId={onToggleAssetId}
            onChooseHandoff={onChooseHandoff}
            onGeneratePreview={onGeneratePreview}
          />
        ))
      )}
      <div ref={bottomRef} />
    </div>
  );
}
