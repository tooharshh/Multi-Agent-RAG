"use client";

import { type FC } from "react";
import { FileTextIcon } from "lucide-react";

interface Source {
  doc_id: string;
  title: string;
  excerpt: string;
}

export const SourceCards: FC<{ sources: Source[] }> = ({ sources }) => {
  if (!sources || sources.length === 0) return null;

  return (
    <div className="mt-4 space-y-2">
      <p className="text-xs font-medium uppercase tracking-wider text-neutral-400 dark:text-neutral-500">
        Sources
      </p>
      <div className="grid gap-2 sm:grid-cols-2">
        {sources.map((source) => (
          <div
            key={source.doc_id}
            className="rounded-lg border border-neutral-200 bg-neutral-50 px-3 py-2.5 transition-colors hover:bg-neutral-100 dark:border-neutral-800 dark:bg-neutral-900 dark:hover:bg-neutral-800"
          >
            <div className="flex items-center gap-2">
              <FileTextIcon className="size-3.5 shrink-0 text-neutral-400 dark:text-neutral-500" />
              <span className="text-xs font-mono font-medium text-neutral-600 dark:text-neutral-300">
                {source.doc_id}
              </span>
            </div>
            {source.title && (
              <p className="mt-1 text-xs leading-relaxed text-neutral-500 dark:text-neutral-400 line-clamp-2">
                {source.title}
              </p>
            )}
          </div>
        ))}
      </div>
    </div>
  );
};

export const SourceCardsFromData: FC<{ data: unknown[] }> = ({ data }) => {
  if (!Array.isArray(data)) return null;

  const sources: Source[] = [];
  for (const item of data) {
    if (
      typeof item === "object" &&
      item !== null &&
      "type" in item &&
      (item as Record<string, unknown>).type === "sources"
    ) {
      const d = item as Record<string, unknown>;
      const s = d.sources as Source[];
      if (Array.isArray(s)) {
        sources.push(...s);
      }
    }
  }

  return <SourceCards sources={sources} />;
};
