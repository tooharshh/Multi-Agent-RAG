"use client";

import { create } from "zustand";
import type { RagCustomMetadata } from "@/app/rag-adapter";

// Simplified message format for storage
export interface StoredMessage {
  id: string;
  role: "user" | "assistant";
  text: string;
  metadata?: RagCustomMetadata;
  createdAt: number;
}

export interface ChatThread {
  id: string;
  title: string;
  messages: StoredMessage[];
  createdAt: number;
}

interface ChatStoreState {
  threads: ChatThread[];
  activeThreadId: string | null;
  hydrated: boolean;
  isRunning: boolean;

  hydrate: () => void;
  createThread: () => string;
  switchThread: (id: string) => void;
  deleteThread: (id: string) => void;
  getActiveThread: () => ChatThread | null;
  addMessage: (msg: StoredMessage) => void;
  updateLastAssistant: (text: string, metadata?: RagCustomMetadata) => void;
  setIsRunning: (v: boolean) => void;
}

const STORAGE_KEY = "rag-chat-threads";

function loadThreads(): ChatThread[] {
  try {
    const raw = localStorage.getItem(STORAGE_KEY);
    return raw ? (JSON.parse(raw) as ChatThread[]) : [];
  } catch {
    return [];
  }
}

function persist(threads: ChatThread[]) {
  try {
    localStorage.setItem(STORAGE_KEY, JSON.stringify(threads));
  } catch {
    // quota
  }
}

export const useChatStore = create<ChatStoreState>((set, get) => ({
  threads: [],
  activeThreadId: null,
  hydrated: false,
  isRunning: false,

  hydrate() {
    if (get().hydrated) return;
    const threads = loadThreads();
    set({ threads, hydrated: true });
  },

  createThread() {
    const id = crypto.randomUUID();
    const thread: ChatThread = {
      id,
      title: "New chat",
      messages: [],
      createdAt: Date.now(),
    };
    const updated = [thread, ...get().threads];
    persist(updated);
    set({ threads: updated, activeThreadId: id });
    return id;
  },

  switchThread(id: string) {
    set({ activeThreadId: id, isRunning: false });
  },

  deleteThread(id: string) {
    const updated = get().threads.filter((t) => t.id !== id);
    persist(updated);
    const newActive =
      get().activeThreadId === id
        ? updated[0]?.id ?? null
        : get().activeThreadId;
    set({ threads: updated, activeThreadId: newActive });
  },

  getActiveThread() {
    const { threads, activeThreadId } = get();
    return threads.find((t) => t.id === activeThreadId) ?? null;
  },

  addMessage(msg: StoredMessage) {
    const { threads, activeThreadId } = get();
    // Auto-create thread if none active
    let threadId = activeThreadId;
    let updatedThreads = [...threads];

    if (!threadId) {
      threadId = crypto.randomUUID();
      updatedThreads.unshift({
        id: threadId,
        title: "New chat",
        messages: [],
        createdAt: Date.now(),
      });
    }

    updatedThreads = updatedThreads.map((t) => {
      if (t.id !== threadId) return t;
      const newMessages = [...t.messages, msg];
      // Update title from first user message
      let title = t.title;
      if (msg.role === "user" && t.messages.filter((m) => m.role === "user").length === 0) {
        title = msg.text.length > 50 ? msg.text.slice(0, 47) + "..." : msg.text;
      }
      return { ...t, messages: newMessages, title };
    });

    persist(updatedThreads);
    set({ threads: updatedThreads, activeThreadId: threadId });
  },

  updateLastAssistant(text: string, metadata?: RagCustomMetadata) {
    const { threads, activeThreadId } = get();
    const updated = threads.map((t) => {
      if (t.id !== activeThreadId) return t;
      const msgs = [...t.messages];
      const lastIdx = msgs.findLastIndex((m) => m.role === "assistant");
      if (lastIdx >= 0) {
        msgs[lastIdx] = { ...msgs[lastIdx], text, metadata };
      }
      return { ...t, messages: msgs };
    });
    persist(updated);
    set({ threads: updated });
  },

  setIsRunning(v: boolean) {
    set({ isRunning: v });
  },
}));
