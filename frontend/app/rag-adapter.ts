"use client";

import type { ChatModelAdapter } from "@assistant-ui/react";

export type AgentStep = {
  agent: "research" | "analysis" | "writer";
  status: "active" | "complete" | "pending";
  message: string;
  route?: string;
  sub_queries?: string[];
};

export type RagSource = {
  doc_id: string;
  title: string;
  excerpt: string;
};

export type RagCritic = {
  confidence_score: number;
  flagged_claims: string[];
};

export type RagFollowUp = {
  detected: boolean;
  rewritten_question: string | null;
  original_question: string;
};

export type RagCustomMetadata = {
  agentSteps: AgentStep[];
  reasoning: string;
  sources: RagSource[];
  critic: RagCritic | null;
  followUp: RagFollowUp | null;
};

export const ragAdapter: ChatModelAdapter = {
  async *run({ messages, abortSignal }) {
    const response = await fetch("/api/chat", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({
        messages: messages.map((m) => ({
          role: m.role,
          content: m.content
            .filter((p): p is { type: "text"; text: string } => p.type === "text")
            .map((p) => ({ type: "text", text: p.text })),
        })),
      }),
      signal: abortSignal,
    });

    if (!response.ok || !response.body) {
      throw new Error(`Backend error: ${response.status}`);
    }

    const reader = response.body.getReader();
    const decoder = new TextDecoder();
    let buffer = "";
    let text = "";
    const meta: RagCustomMetadata = {
      agentSteps: [],
      reasoning: "",
      sources: [],
      critic: null,
      followUp: null,
    };

    while (true) {
      const { done, value } = await reader.read();
      if (done) break;

      buffer += decoder.decode(value, { stream: true });
      const lines = buffer.split("\n");
      buffer = lines.pop() || "";

      for (const line of lines) {
        if (!line.trim()) continue;

        // Text token: 0:"token"
        if (line.startsWith("0:")) {
          try {
            const token = JSON.parse(line.slice(2)) as string;
            text += token;
            yield {
              content: [{ type: "text" as const, text }],
              metadata: { custom: structuredClone(meta) },
            };
          } catch {
            // skip malformed text line
          }
        }
        // Data: 2:[{...}]
        else if (line.startsWith("2:")) {
          try {
            const dataItems = JSON.parse(line.slice(2)) as Record<string, unknown>[];
            for (const item of dataItems) {
              if (item.type === "agent_status") {
                const agent = item.agent as AgentStep["agent"];
                const existing = meta.agentSteps.find((s) => s.agent === agent);
                if (existing) {
                  existing.status = item.status as AgentStep["status"];
                  existing.message = (item.message as string) || "";
                  if (item.route) existing.route = item.route as string;
                  if (item.sub_queries)
                    existing.sub_queries = item.sub_queries as string[];
                } else {
                  meta.agentSteps.push({
                    agent,
                    status: (item.status as AgentStep["status"]) || "active",
                    message: (item.message as string) || "",
                    route: item.route as string | undefined,
                    sub_queries: item.sub_queries as string[] | undefined,
                  });
                }
              } else if (item.type === "reasoning") {
                meta.reasoning = (item.text as string) || "";
              } else if (item.type === "sources") {
                meta.sources = (item.sources as RagSource[]) || [];
              } else if (item.type === "critic") {
                meta.critic = {
                  confidence_score: (item.confidence_score as number) || 0,
                  flagged_claims: (item.flagged_claims as string[]) || [],
                };
              } else if (item.type === "follow_up_info") {
                meta.followUp = {
                  detected: (item.detected as boolean) || false,
                  rewritten_question: (item.rewritten_question as string) || null,
                  original_question: (item.original_question as string) || "",
                };
              }
            }
            yield {
              content: text ? [{ type: "text" as const, text }] : [],
              metadata: { custom: structuredClone(meta) },
            };
          } catch {
            // skip malformed data line
          }
        }
        // Finish: d:{...} — handled by loop exit
      }
    }

    // Final yield
    yield {
      content: text ? [{ type: "text" as const, text }] : [],
      status: { type: "complete" as const, reason: "stop" as const },
      metadata: { custom: structuredClone(meta) },
    };
  },
};
