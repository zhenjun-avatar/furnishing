import { useCallback, useState } from "react";
import { FurnishingPicker } from "@/furnishing/FurnishingPicker";
import { ProductCatalog } from "@/furnishing/ProductCatalog";
import { resolveInternalToken } from "@/salon/config";
import type { FurnishingAsset } from "@/salon/types";
import { MessageList } from "./MessageList";
import { Composer } from "./Composer";
import { useChatSession } from "./useChatSession";
import styles from "./ChatScreen.module.css";

type MainView = "chat" | "products";

export function ChatScreen() {
  const internalOk = Boolean(resolveInternalToken());
  const [mainView, setMainView] = useState<MainView>("chat");
  const [selectedAssetIds, setSelectedAssetIds] = useState<string[]>([]);
  const [availableAssetsById, setAvailableAssetsById] = useState<Record<string, FurnishingAsset>>({});
  const [dismissedActionMessageIds, setDismissedActionMessageIds] = useState<string[]>([]);
  const {
    messages,
    pendingFileId,
    hasRoomImage,
    loading,
    error,
    chooseHandoff,
    send,
    pickImage,
    clearPendingFile,
  } = useChatSession();

  const sendWithAssets = useCallback(
    (text: string) => {
      void send(text, selectedAssetIds);
      setSelectedAssetIds([]);
    },
    [send, selectedAssetIds],
  );

  const latestAssistant = [...messages].reverse().find((m) => m.role === "assistant" && !m.streaming);
  const latestAssistantId = latestAssistant?.id ?? null;
  const latestActionsDismissed = latestAssistantId
    ? dismissedActionMessageIds.includes(latestAssistantId)
    : true;
  const latestUiActions = latestActionsDismissed ? [] : (latestAssistant?.uiActions || []);

  const dismissLatestActions = useCallback(() => {
    if (!latestAssistantId) return;
    setDismissedActionMessageIds((prev) =>
      prev.includes(latestAssistantId) ? prev : [...prev, latestAssistantId],
    );
  }, [latestAssistantId]);

  const generatePreview = useCallback(() => {
    dismissLatestActions();
    void send("", selectedAssetIds, "generate_preview");
    setSelectedAssetIds([]);
  }, [dismissLatestActions, send, selectedAssetIds]);

  const chooseHandoffFromReply = useCallback(
    (needHandoff: boolean) => {
      dismissLatestActions();
      chooseHandoff(needHandoff);
    },
    [chooseHandoff, dismissLatestActions],
  );

  const toggleAssetId = useCallback((id: string) => {
    setSelectedAssetIds((prev) => {
      if (prev.includes(id)) return prev.filter((x) => x !== id);
      if (prev.length >= 4) return prev;
      return [...prev, id];
    });
  }, []);

  const rememberAvailableAssets = useCallback((items: FurnishingAsset[]) => {
    setAvailableAssetsById((prev) => {
      const next = { ...prev };
      for (const it of items) {
        if (it?.id) next[it.id] = it;
      }
      return next;
    });
  }, []);

  const canQuickGenerate = !loading && selectedAssetIds.length > 0 && (hasRoomImage || Boolean(pendingFileId));

  return (
    <div className={styles.shell}>
      <header className={styles.header}>
        <div className={styles.headerMain}>
          <h1 className={styles.title}>AI代理 · 家居</h1>
          <nav className={styles.tabs} role="tablist" aria-label="功能切换">
            <button
              type="button"
              role="tab"
              aria-selected={mainView === "chat"}
              className={`${styles.tab} ${mainView === "chat" ? styles.tabActive : ""}`}
              onClick={() => setMainView("chat")}
            >
              对话
            </button>
            <button
              type="button"
              role="tab"
              aria-selected={mainView === "products"}
              className={`${styles.tab} ${mainView === "products" ? styles.tabActive : ""}`}
              onClick={() => setMainView("products")}
            >
              产品列表
            </button>
          </nav>
        </div>
        {!internalOk ? (
          <div className={styles.headerRight}>
            <p className={styles.internalHint}>
              配置 <code>VITE_SALON_INTERNAL_TOKEN</code>（与网关 <code>SALON_INTERNAL_BOOKING_TOKEN</code>{" "}
              一致）可启用对话侧栏与产品列表数据。
            </p>
          </div>
        ) : null}
      </header>
      {error && mainView === "chat" ? <div className={styles.banner}>{error}</div> : null}
      <main className={styles.main}>
        {mainView === "products" ? (
          <div className={styles.catalogHost}>
            <ProductCatalog />
          </div>
        ) : (
          <div className={`${styles.layout} ${internalOk ? "" : styles.layoutSingle}`}>
            {internalOk ? (
              <div className={styles.pickerCol}>
                <FurnishingPicker
                  selectedIds={selectedAssetIds}
                  onChangeSelectedIds={setSelectedAssetIds}
                  onAssetsLoaded={rememberAvailableAssets}
                />
              </div>
            ) : null}
            <div className={styles.chatCol}>
              <MessageList
                messages={messages}
                selectedAssetIds={selectedAssetIds}
                availableAssetsById={availableAssetsById}
                actionMessageId={latestActionsDismissed ? null : latestAssistantId}
                showHandoffActions={
                  !loading &&
                  latestUiActions.includes("handoff_yes") &&
                  latestUiActions.includes("handoff_no")
                }
                showGenerateAction={
                  (!loading && latestUiActions.includes("generate_preview")) ||
                  (canQuickGenerate && Boolean(latestAssistantId))
                }
                onToggleAssetId={toggleAssetId}
                onChooseHandoff={chooseHandoffFromReply}
                onGeneratePreview={generatePreview}
              />
              <Composer
                disabled={loading}
                pendingFileId={pendingFileId}
                attachedIds={selectedAssetIds}
                onSend={sendWithAssets}
                onPickFile={pickImage}
                onClearFile={clearPendingFile}
              />
            </div>
          </div>
        )}
      </main>
    </div>
  );
}
