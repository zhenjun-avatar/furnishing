import type { FurnishingAsset } from "./types";

function asText(v: unknown): string {
  if (v == null) return "";
  return String(v).trim();
}

/** 一行展示价格与尺寸；无则返回 null。 */
export function formatAssetPriceDimensions(a: Pick<FurnishingAsset, "price" | "dimensions">): string | null {
  const p = asText(a.price);
  const d = asText(a.dimensions);
  if (!p && !d) return null;
  return [p, d].filter(Boolean).join(" · ");
}
