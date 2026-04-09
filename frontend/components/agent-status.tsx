"use client";

import { type FC, useState } from "react";
import { SearchIcon, BrainIcon, PenLineIcon, CheckIcon, ChevronDownIcon } from "lucide-react";

interface AgentStep {
  agent: "research" | "analysis" | "writer";
  status: "active" | "complete" | "pending";
  message: string;
  route?: string;
  sub_queries?: string[];
}

const agentConfig = {
  research: { label: "Research Agent", icon: SearchIcon },
  analysis: { label: "Analysis Agent", icon: BrainIcon },
  writer: { label: "Writer Agent", icon: PenLineIcon },
};

export const AgentSteps: FC<{ steps: AgentStep[] }> = ({ steps }) => {
  const [showQueries, setShowQueries] = useState(false);
  if (steps.length === 0) return null;

  const allAgents: Array<"research" | "analysis" | "writer"> = [
    "research",
    "analysis",
    "writer",
  ];

  return (
    <div className="mb-4 space-y-1">
      {allAgents.map((agentKey) => {
        const step = steps.find((s) => s.agent === agentKey);
        const config = agentConfig[agentKey];
        const Icon = config.icon;

        if (!step) {
          return (
            <div
              key={agentKey}
              className="flex items-center gap-2.5 rounded-lg px-3 py-2 text-sm text-neutral-600 dark:text-neutral-500"
            >
              <Icon className="size-3.5" />
              <span>{config.label}</span>
            </div>
          );
        }

        const isActive = step.status === "active";
        const isComplete = step.status === "complete";
        const hasSubQueries = step.sub_queries && step.sub_queries.length > 1;

        return (
          <div key={agentKey}>
            <div
              className={`flex items-center gap-2.5 rounded-lg px-3 py-2 text-sm transition-all duration-300 ${
                isActive
                  ? "bg-neutral-100 text-neutral-900 dark:bg-neutral-800 dark:text-neutral-100"
                  : isComplete
                    ? "text-neutral-500 dark:text-neutral-400"
                    : "text-neutral-400 dark:text-neutral-600"
              }`}
            >
              {isComplete ? (
                <CheckIcon className="size-3.5 text-neutral-400 dark:text-neutral-500" />
              ) : isActive ? (
                <div className="relative flex size-3.5 items-center justify-center">
                  <div className="absolute size-3.5 animate-ping rounded-full bg-neutral-400 opacity-30 dark:bg-neutral-500" />
                  <Icon className="relative size-3.5" />
                </div>
              ) : (
                <Icon className="size-3.5" />
              )}
              <span className="font-medium">{config.label}</span>
              {step.message && (
                <span className="ml-1 truncate text-xs text-neutral-400 dark:text-neutral-500">
                  — {step.message}
                </span>
              )}
              {hasSubQueries && isComplete && (
                <button
                  onClick={() => setShowQueries((v) => !v)}
                  className="ml-auto flex items-center gap-0.5 text-xs text-neutral-400 hover:text-neutral-300 transition-colors"
                >
                  <ChevronDownIcon
                    className={`size-3 transition-transform duration-200 ${showQueries ? "rotate-180" : ""}`}
                  />
                </button>
              )}
            </div>
            {hasSubQueries && isComplete && showQueries && (
              <div className="ml-9 mt-1 mb-1 space-y-1">
                {step.sub_queries!.map((q, i) => (
                  <div
                    key={i}
                    className="flex items-start gap-2 rounded-md bg-neutral-50 px-2.5 py-1.5 text-xs text-neutral-500 dark:bg-neutral-800/60 dark:text-neutral-400"
                  >
                    <span className="mt-px shrink-0 font-mono text-[10px] text-neutral-400 dark:text-neutral-500">
                      Q{i + 1}
                    </span>
                    <span>{q}</span>
                  </div>
                ))}
              </div>
            )}
          </div>
        );
      })}
    </div>
  );
};

export const AgentStepsFromData: FC<{ data: unknown[] }> = ({ data }) => {
  const steps: AgentStep[] = [];
  if (!Array.isArray(data)) return null;

  for (const item of data) {
    if (
      typeof item === "object" &&
      item !== null &&
      "type" in item &&
      (item as Record<string, unknown>).type === "agent_status"
    ) {
      const d = item as Record<string, unknown>;
      const agent = d.agent as "research" | "analysis" | "writer";
      const existing = steps.find((s) => s.agent === agent);
      if (existing) {
        existing.status = d.status as "active" | "complete";
        existing.message = (d.message as string) || "";
        if (d.route) existing.route = d.route as string;
        if (d.sub_queries) existing.sub_queries = d.sub_queries as string[];
      } else {
        steps.push({
          agent,
          status: (d.status as "active" | "complete") || "pending",
          message: (d.message as string) || "",
          route: d.route as string | undefined,
          sub_queries: d.sub_queries as string[] | undefined,
        });
      }
    }
  }

  return <AgentSteps steps={steps} />;
};
