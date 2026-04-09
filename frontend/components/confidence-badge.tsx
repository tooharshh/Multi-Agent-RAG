"use client";

import { type FC } from "react";
import { ShieldCheckIcon, AlertTriangleIcon } from "lucide-react";

export const ConfidenceBadge: FC<{ score: number; flagged: string[] }> = ({
  score,
  flagged,
}) => {
  const isHigh = score >= 0.8;

  return (
    <div className="mt-3 flex items-center gap-2">
      {isHigh ? (
        <div className="flex items-center gap-1.5 rounded-full border border-neutral-200 bg-neutral-50 px-2.5 py-1 text-xs text-neutral-600 dark:border-neutral-700 dark:bg-neutral-800 dark:text-neutral-300">
          <ShieldCheckIcon className="size-3" />
          Verified — {(score * 100).toFixed(0)}%
        </div>
      ) : (
        <div className="flex items-center gap-1.5 rounded-full border border-neutral-300 bg-neutral-100 px-2.5 py-1 text-xs text-neutral-600 dark:border-neutral-600 dark:bg-neutral-800 dark:text-neutral-300">
          <AlertTriangleIcon className="size-3" />
          Review — {(score * 100).toFixed(0)}%
        </div>
      )}
      {flagged.length > 0 && (
        <span className="text-xs text-neutral-400 dark:text-neutral-500">
          {flagged.length} claim{flagged.length > 1 ? "s" : ""} flagged
        </span>
      )}
    </div>
  );
};

export const ConfidenceBadgeFromData: FC<{ data: unknown[] }> = ({ data }) => {
  if (!Array.isArray(data)) return null;

  for (const item of data) {
    if (
      typeof item === "object" &&
      item !== null &&
      "type" in item &&
      (item as Record<string, unknown>).type === "critic"
    ) {
      const d = item as Record<string, unknown>;
      return (
        <ConfidenceBadge
          score={(d.confidence_score as number) || 0}
          flagged={(d.flagged_claims as string[]) || []}
        />
      );
    }
  }

  return null;
};
