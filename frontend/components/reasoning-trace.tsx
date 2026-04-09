"use client";

import { type FC } from "react";
import { ChevronDownIcon, ChevronUpIcon } from "lucide-react";
import { useState } from "react";

export const ReasoningTrace: FC<{ text: string }> = ({ text }) => {
  const [isOpen, setIsOpen] = useState(false);

  if (!text || text === "No explicit chain of thought generated.") return null;

  // Clean and format the reasoning text
  const lines = text
    .split("\n")
    .map((l) => l.trim())
    .filter(Boolean);

  return (
    <div className="mt-3 mb-1">
      <button
        type="button"
        onClick={() => setIsOpen(!isOpen)}
        className="flex items-center gap-1.5 text-xs font-medium text-neutral-400 transition-colors hover:text-neutral-600 dark:text-neutral-500 dark:hover:text-neutral-300"
      >
        {isOpen ? (
          <ChevronUpIcon className="size-3" />
        ) : (
          <ChevronDownIcon className="size-3" />
        )}
        Reasoning trace
      </button>
      {isOpen && (
        <div className="mt-2 rounded-lg border border-neutral-200 bg-neutral-50 px-3 py-2.5 text-xs leading-relaxed text-neutral-500 dark:border-neutral-800 dark:bg-neutral-900 dark:text-neutral-400">
          {lines.map((line, i) => (
            <p key={i} className="mb-1 last:mb-0">
              {line}
            </p>
          ))}
        </div>
      )}
    </div>
  );
};

export const ReasoningTraceFromData: FC<{ data: unknown[] }> = ({ data }) => {
  if (!Array.isArray(data)) return null;

  for (const item of data) {
    if (
      typeof item === "object" &&
      item !== null &&
      "type" in item &&
      (item as Record<string, unknown>).type === "reasoning"
    ) {
      const d = item as Record<string, unknown>;
      return <ReasoningTrace text={(d.text as string) || ""} />;
    }
  }

  return null;
};
