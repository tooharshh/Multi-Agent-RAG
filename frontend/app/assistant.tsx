"use client";

import {
  AssistantRuntimeProvider,
  useExternalStoreRuntime,
} from "@assistant-ui/react";
import type { ThreadMessageLike } from "@assistant-ui/react";
import { Thread } from "@/components/assistant-ui/thread-rag";
import { useCallback, useEffect, useMemo, useRef, useState } from "react";
import {
  MenuIcon,
  PlusIcon,
  XIcon,
  MessageSquareIcon,
  Trash2Icon,
} from "lucide-react";
import { useChatStore } from "@/lib/chat-store";
import type { RagCustomMetadata } from "./rag-adapter";

export const Assistant = () => {
  const [sidebarOpen, setSidebarOpen] = useState(false);
  const store = useChatStore();
  const abortRef = useRef<AbortController | null>(null);

  // Hydrate on mount
  useEffect(() => {
    store.hydrate();
  }, []); // eslint-disable-line react-hooks/exhaustive-deps

  // Convert stored messages to ThreadMessageLike for the runtime
  const activeThread = store.getActiveThread();
  const threadMessages: ThreadMessageLike[] = useMemo(() => {
    if (!activeThread) return [];
    return activeThread.messages.map((m) => ({
      role: m.role as "user" | "assistant",
      id: m.id,
      content: [{ type: "text" as const, text: m.text }],
      ...(m.role === "assistant"
        ? {
            status: { type: "complete" as const, reason: "stop" as const },
            metadata: m.metadata ? { custom: m.metadata } : undefined,
          }
        : {}),
    }));
  }, [activeThread]);

  // Handler called when user sends a new message
  const onNew = useCallback(
    async (message: { content: readonly { type: string; text?: string }[] }) => {
      const userText = (message.content as { type: string; text?: string }[])
        .filter((p) => p.type === "text" && p.text)
        .map((p) => p.text!)
        .join(" ");

      if (!userText.trim()) return;

      // Add user message to store
      store.addMessage({
        id: crypto.randomUUID(),
        role: "user",
        text: userText,
        createdAt: Date.now(),
      });

      // Add placeholder assistant message
      store.addMessage({
        id: crypto.randomUUID(),
        role: "assistant",
        text: "",
        createdAt: Date.now(),
      });
      store.setIsRunning(true);

      // Build conversation history for the API
      const thread = store.getActiveThread();
      const apiMessages = (thread?.messages ?? [])
        .filter((m) => m.text.trim())
        .map((m) => ({
          role: m.role,
          content: [{ type: "text", text: m.text }],
        }));

      // Stream from backend
      const controller = new AbortController();
      abortRef.current = controller;
      let text = "";

      try {
        const response = await fetch("/api/chat", {
          method: "POST",
          headers: { "Content-Type": "application/json" },
          body: JSON.stringify({ messages: apiMessages }),
          signal: controller.signal,
        });

        if (!response.ok || !response.body) {
          store.updateLastAssistant("Error: Backend unavailable.");
          store.setIsRunning(false);
          return;
        }

        const reader = response.body.getReader();
        const decoder = new TextDecoder();
        let buffer = "";
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

            if (line.startsWith("0:")) {
              try {
                const token = JSON.parse(line.slice(2)) as string;
                text += token;
                store.updateLastAssistant(text, structuredClone(meta));
              } catch {
                /* skip malformed */
              }
            } else if (line.startsWith("2:")) {
              try {
                const items = JSON.parse(line.slice(2)) as Record<string, unknown>[];
                for (const item of items) {
                  if (item.type === "agent_status") {
                    const agent = item.agent as "research" | "analysis" | "writer";
                    const existing = meta.agentSteps.find((s) => s.agent === agent);
                    if (existing) {
                      existing.status = item.status as "active" | "complete";
                      existing.message = (item.message as string) || "";
                      if (item.route) existing.route = item.route as string;
                      if (item.sub_queries) existing.sub_queries = item.sub_queries as string[];
                    } else {
                      meta.agentSteps.push({
                        agent,
                        status: (item.status as "active" | "complete") || "active",
                        message: (item.message as string) || "",
                        route: item.route as string | undefined,
                        sub_queries: item.sub_queries as string[] | undefined,
                      });
                    }
                  } else if (item.type === "reasoning") {
                    meta.reasoning = (item.text as string) || "";
                  } else if (item.type === "sources") {
                    meta.sources = (item.sources as RagCustomMetadata["sources"]) || [];
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
                store.updateLastAssistant(text, structuredClone(meta));
              } catch {
                /* skip malformed */
              }
            }
          }
        }

        // Final update
        store.updateLastAssistant(text, structuredClone(meta));
      } catch (err: unknown) {
        if ((err as Error)?.name !== "AbortError") {
          store.updateLastAssistant(text || "Error: Something went wrong.");
        }
      } finally {
        store.setIsRunning(false);
        abortRef.current = null;
      }
    },
    [store],
  );

  const onCancel = useCallback(async () => {
    abortRef.current?.abort();
    store.setIsRunning(false);
  }, [store]);

  const runtime = useExternalStoreRuntime({
    messages: threadMessages,
    isRunning: store.isRunning,
    onNew,
    onCancel,
    convertMessage: (m: ThreadMessageLike) => m,
  });

  return (
    <AssistantRuntimeProvider runtime={runtime}>
      <div className="flex h-dvh bg-background">
        {/* Sidebar */}
        <div
          className={`fixed inset-y-0 left-0 z-40 flex w-64 flex-col border-r border-neutral-200 bg-neutral-50 transition-transform duration-200 dark:border-neutral-800 dark:bg-neutral-950 md:relative md:translate-x-0 ${
            sidebarOpen ? "translate-x-0" : "-translate-x-full"
          }`}
        >
          <div className="flex items-center justify-between border-b border-neutral-200 px-4 py-3 dark:border-neutral-800">
            <span className="text-sm font-semibold tracking-tight">
              Multi-Agent RAG
            </span>
            <button
              type="button"
              onClick={() => setSidebarOpen(false)}
              className="rounded-md p-1 text-neutral-400 hover:text-neutral-600 md:hidden dark:hover:text-neutral-300"
            >
              <XIcon className="size-4" />
            </button>
          </div>

          <div className="px-3 pt-3">
            <button
              type="button"
              className="flex w-full items-center gap-2 rounded-lg border border-neutral-200 px-3 py-2 text-sm transition-colors hover:bg-neutral-100 dark:border-neutral-800 dark:hover:bg-neutral-900"
              onClick={() => {
                store.createThread();
                setSidebarOpen(false);
              }}
            >
              <PlusIcon className="size-4" />
              New chat
            </button>
          </div>

          <div className="flex-1 overflow-y-auto px-3 pt-4">
            {store.hydrated && store.threads.length > 0 && (
              <>
                <p className="mb-2 text-xs font-medium uppercase tracking-wider text-neutral-400 dark:text-neutral-500">
                  Recent chats
                </p>
                <div className="space-y-0.5">
                  {store.threads.map((t) => {
                    const isCurrent = t.id === store.activeThreadId;
                    return (
                      <div
                        key={t.id}
                        className={`group flex cursor-pointer items-center gap-2 rounded-md px-2 py-1.5 text-xs transition-colors ${
                          isCurrent
                            ? "bg-neutral-200 text-neutral-900 dark:bg-neutral-800 dark:text-neutral-100"
                            : "text-neutral-500 hover:bg-neutral-100 dark:text-neutral-400 dark:hover:bg-neutral-900"
                        }`}
                        onClick={() => {
                          store.switchThread(t.id);
                          setSidebarOpen(false);
                        }}
                      >
                        <MessageSquareIcon className="size-3 shrink-0" />
                        <span className="flex-1 truncate">{t.title}</span>
                        <button
                          type="button"
                          className="hidden shrink-0 rounded p-0.5 text-neutral-400 hover:text-neutral-600 group-hover:block dark:hover:text-neutral-300"
                          onClick={(e) => {
                            e.stopPropagation();
                            store.deleteThread(t.id);
                          }}
                        >
                          <Trash2Icon className="size-3" />
                        </button>
                      </div>
                    );
                  })}
                </div>
              </>
            )}
          </div>
        </div>

        {/* Overlay for mobile sidebar */}
        {sidebarOpen && (
          <button
            type="button"
            className="fixed inset-0 z-30 bg-black/20 md:hidden"
            onClick={() => setSidebarOpen(false)}
            aria-label="Close sidebar"
          />
        )}

        {/* Main content */}
        <div className="flex min-h-0 flex-1 flex-col">
          {/* Mobile menu button */}
          <div className="flex items-center px-4 py-2 md:hidden">
            <button
              type="button"
              onClick={() => setSidebarOpen(true)}
              className="rounded-md p-1.5 text-neutral-400 transition-colors hover:text-neutral-600 dark:hover:text-neutral-300"
            >
              <MenuIcon className="size-4" />
            </button>
          </div>

          {/* Thread */}
          <div className="min-h-0 flex-1">
            <Thread />
          </div>
        </div>
      </div>
    </AssistantRuntimeProvider>
  );
};
