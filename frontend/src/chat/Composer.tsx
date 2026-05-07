import { useRef, useState, type FormEvent } from "react";
import styles from "./Composer.module.css";

function IconImage() {
  return (
    <svg viewBox="0 0 24 24" aria-hidden="true" className={styles.btnIcon}>
      <path
        d="M4.5 6.25A1.75 1.75 0 0 1 6.25 4.5h11.5a1.75 1.75 0 0 1 1.75 1.75v11.5a1.75 1.75 0 0 1-1.75 1.75H6.25a1.75 1.75 0 0 1-1.75-1.75V6.25Z"
        fill="none"
        stroke="currentColor"
        strokeWidth="1.6"
      />
      <circle cx="9" cy="9" r="1.7" fill="currentColor" />
      <path
        d="m19.3 15.8-3.7-3.9a1 1 0 0 0-1.44.03l-4.3 4.8-1.8-1.7a1 1 0 0 0-1.4.02l-2.1 2.2"
        fill="none"
        stroke="currentColor"
        strokeWidth="1.6"
        strokeLinecap="round"
        strokeLinejoin="round"
      />
    </svg>
  );
}

function IconSend() {
  return (
    <svg viewBox="0 0 24 24" aria-hidden="true" className={styles.btnIcon}>
      <path
        d="m20.4 3.6-16 7.1a.8.8 0 0 0 .06 1.48l6.44 2.14 2.14 6.44a.8.8 0 0 0 1.48.05l7.08-16a.8.8 0 0 0-1.2-1.01Z"
        fill="currentColor"
      />
      <path d="m10.85 14.34 9.04-9.04" fill="none" stroke="#fff" strokeWidth="1.5" />
    </svg>
  );
}

type Props = {
  disabled: boolean;
  pendingFileId: string | null;
  attachedIds: string[];
  onSend: (text: string) => void;
  onPickFile: (file: File | null) => void;
  onClearFile: () => void;
};

export function Composer({
  disabled,
  pendingFileId,
  attachedIds,
  onSend,
  onPickFile,
  onClearFile,
}: Props) {
  const [text, setText] = useState("");
  const fileRef = useRef<HTMLInputElement>(null);

  function submit(e: FormEvent) {
    e.preventDefault();
    if (disabled) return;
    const t = text.trim();
    if (!t && !pendingFileId && attachedIds.length === 0) return;
    onSend(text);
    setText("");
  }

  return (
    <form className={styles.form} onSubmit={submit}>
      <div className={styles.bar}>
      <input
        ref={fileRef}
        type="file"
        accept="image/*"
        className={styles.fileHidden}
        onChange={(e) => {
          onPickFile(e.target.files?.[0] ?? null);
          e.target.value = "";
        }}
      />
      <button
        type="button"
        className={styles.iconBtn}
        disabled={disabled}
        onClick={() => fileRef.current?.click()}
        title="上传图片"
        aria-label="上传图片"
      >
        <IconImage />
      </button>
      <textarea
        className={styles.input}
        rows={2}
        placeholder="输入消息…"
        value={text}
        disabled={disabled}
        onChange={(e) => setText(e.target.value)}
        onKeyDown={(e) => {
          if (e.key === "Enter" && !e.shiftKey) {
            e.preventDefault();
            submit(e);
          }
        }}
      />
      <button type="submit" className={styles.send} disabled={disabled} aria-label="发送">
        <IconSend />
      </button>
      {pendingFileId ? (
        <span className={styles.chip}>
          已选图
          <button type="button" className={styles.chipX} onClick={onClearFile}>
            ×
          </button>
        </span>
      ) : null}
      </div>
    </form>
  );
}
