import { MarkdownText } from "@/components/assistant-ui/markdown-text";
import { TooltipIconButton } from "@/components/assistant-ui/tooltip-icon-button";
import { Button } from "@/components/ui/button";
import { cn } from "@/lib/utils";
import { AgentSteps } from "@/components/agent-status";
import { SourceCards } from "@/components/source-cards";
import { ReasoningTrace } from "@/components/reasoning-trace";
import { ConfidenceBadge } from "@/components/confidence-badge";
import type { RagCustomMetadata } from "@/app/rag-adapter";
import {
  ActionBarMorePrimitive,
  ActionBarPrimitive,
  AuiIf,
  BranchPickerPrimitive,
  ComposerPrimitive,
  ErrorPrimitive,
  MessagePrimitive,
  SuggestionPrimitive,
  ThreadPrimitive,
  useAuiState,
  useMessage,
  useThreadRuntime,
} from "@assistant-ui/react";
import {
  ArrowDownIcon,
  ArrowUpIcon,
  CheckIcon,
  ChevronLeftIcon,
  ChevronRightIcon,
  CopyIcon,
  DownloadIcon,
  MoreHorizontalIcon,
  PencilIcon,
  RefreshCwIcon,
  SendIcon,
  SquareIcon,
} from "lucide-react";
import { type FC, useState } from "react";

export const Thread: FC = () => {
  return (
    <ThreadPrimitive.Root
      className="aui-root aui-thread-root @container flex h-full flex-col bg-background"
      style={{
        ["--thread-max-width" as string]: "48rem",
        ["--composer-radius" as string]: "24px",
        ["--composer-padding" as string]: "10px",
      }}
    >
      {/* Scrollable messages area */}
      <ThreadPrimitive.Viewport
        autoScroll
        className="aui-thread-viewport relative flex min-h-0 flex-1 flex-col overflow-y-auto scroll-smooth px-4 pt-4"
      >
        <AuiIf condition={(s) => s.thread.isEmpty}>
          <ThreadWelcome />
        </AuiIf>

        <ThreadPrimitive.Messages>
          {() => <ThreadMessage />}
        </ThreadPrimitive.Messages>

        {/* Spacer so last message isn't hidden behind composer */}
        <div className="min-h-4 shrink-0" />
      </ThreadPrimitive.Viewport>

      {/* Fixed composer at bottom — outside scroll area */}
      <div className="mx-auto w-full max-w-(--thread-max-width) shrink-0 border-t border-neutral-100 bg-background px-4 pb-4 pt-2 dark:border-neutral-900">
        <div className="relative">
          <ThreadScrollToBottom />
          <Composer />
        </div>
      </div>
    </ThreadPrimitive.Root>
  );
};

const ThreadMessage: FC = () => {
  const role = useAuiState((s) => s.message.role);
  const isEditing = useAuiState((s) => s.message.composer.isEditing);
  if (isEditing) return <EditComposer />;
  if (role === "user") return <UserMessage />;
  return <AssistantMessage />;
};

const ThreadScrollToBottom: FC = () => {
  return (
    <ThreadPrimitive.ScrollToBottom asChild>
      <TooltipIconButton
        tooltip="Scroll to bottom"
        variant="outline"
        className="aui-thread-scroll-to-bottom absolute -top-14 left-1/2 z-10 -translate-x-1/2 rounded-full p-3 disabled:invisible dark:border-neutral-700 dark:bg-background dark:hover:bg-neutral-800"
      >
        <ArrowDownIcon />
      </TooltipIconButton>
    </ThreadPrimitive.ScrollToBottom>
  );
};

const ThreadWelcome: FC = () => {
  return (
    <div className="aui-thread-welcome-root mx-auto my-auto flex w-full max-w-(--thread-max-width) grow flex-col">
      <div className="aui-thread-welcome-center flex w-full grow flex-col items-center justify-center">
        <div className="aui-thread-welcome-message flex size-full flex-col items-center justify-center px-4 text-center">
          <h1 className="aui-thread-welcome-message-inner fade-in slide-in-from-bottom-2 animate-in fill-mode-both font-semibold text-3xl tracking-tight duration-300 md:text-4xl">
            Multi-Agent RAG
          </h1>
          <p className="aui-thread-welcome-message-inner fade-in slide-in-from-bottom-2 animate-in fill-mode-both mt-2 text-neutral-500 text-base delay-100 duration-300 dark:text-neutral-400">
            AI in Healthcare Knowledge Base
          </p>
        </div>
      </div>
      <ThreadSuggestions />
    </div>
  );
};

const SAMPLE_QUESTIONS = [
  { label: "Healthcare AI adoption rates", question: "What percentage of healthcare organisations had implemented domain-specific AI tools as of 2025, and how does this compare to the broader enterprise market?" },
  { label: "Ambient Notes adoption", question: "In a Fall 2024 survey of 43 US non-profit health systems, which AI use case was the only one with 100% adoption activity, and what success rate did respondents report for it?" },
  { label: "ChatRWD vs standard LLMs", question: "What share of clinical questions did the agentic system ChatRWD answer usefully, compared to standard large language models such as ChatGPT and Gemini?" },
  { label: "FDA-cleared AI devices", question: "How many FDA-cleared AI/ML-enabled medical devices existed in the United States by mid-2024, and which clinical specialty accounts for the largest share of those approvals?" },
  { label: "ISM001-055 & drug discovery", question: "What Phase II milestone did Insilico Medicine reach with its AI-designed drug ISM001-055 (Rentosertib), what disease does it target, and what company merger in the same period reshaped the AI drug discovery competitive landscape?" },
  { label: "Ethical concerns in clinical AI", question: "A 2025 PLOS Digital Health paper identifies five critical ethical concerns in clinical AI integration. What are they, and which concern does the paper call out as most likely to worsen existing healthcare inequities?" },
  { label: "LLM sycophancy in therapy", question: "The Hastings Center identifies a specific behavioural tendency of large language models that makes them unsuitable as standalone therapists. What is it, and why does it pose a particular risk in mental health contexts?" },
  { label: "FDA demographic reporting gap", question: "A 2024 cross-sectional study of FDA-cleared AI devices found a critical gap in demographic reporting. What specific data was missing from the vast majority of device submissions, and what percentage of devices reported it?" },
  { label: "Imaging AI success divergence", question: "Imaging AI has high deployment rates across US health systems but only 19% of deployers report high success. Using what you know about FDA validation gaps and algorithmic bias literature, construct the most likely causal mechanism for this divergence." },
  { label: "EU AI Act vs other frameworks", question: "Two frameworks for governing post-deployment AI in healthcare propose different monitoring mechanisms. One focuses on operationalised clinic-level audit steps; the other calls for adaptive regulatory oversight replacing static approval. What does the EU AI Act add that neither framework alone provides?" },
  { label: "Kaiser Permanente strike (live)", question: "On March 18 2026, mental health care providers at Kaiser Permanente went on strike. How many workers participated, what specific triage headcount change at one facility triggered the action, and what UK AI company was KP confirmed to be evaluating?" },
];

const ThreadSuggestions: FC = () => {
  const [showAll, setShowAll] = useState(false);
  const threadRuntime = useThreadRuntime();
  const visible = showAll ? SAMPLE_QUESTIONS : SAMPLE_QUESTIONS.slice(0, 4);

  const send = (text: string) => {
    threadRuntime.append({
      role: "user",
      content: [{ type: "text", text }],
    });
  };

  return (
    <div className="aui-thread-welcome-suggestions w-full pb-4">
      <p className="mb-2 text-xs text-neutral-500 dark:text-neutral-400 px-1">Try an eval question</p>
      <div className="grid @md:grid-cols-2 gap-2">
        {visible.map((item) => (
          <Button
            key={item.label}
            variant="ghost"
            onClick={() => send(item.question)}
            className="group h-auto w-full items-start justify-start gap-1 rounded-2xl border border-neutral-200 bg-background px-4 py-3 text-left text-sm transition-colors hover:bg-neutral-50 dark:border-neutral-800 dark:hover:bg-neutral-900"
          >
            <span className="font-medium">{item.label}</span>
            <SendIcon className="ml-auto size-3.5 shrink-0 text-neutral-400 opacity-0 transition-opacity group-hover:opacity-100" />
          </Button>
        ))}
      </div>
      {!showAll && (
        <button
          onClick={() => setShowAll(true)}
          className="mt-2 w-full text-center text-xs text-neutral-500 hover:text-neutral-300 transition-colors"
        >
          Show all {SAMPLE_QUESTIONS.length} eval questions
        </button>
      )}
    </div>
  );
};

const ThreadSuggestionItem: FC = () => {
  return (
    <div className="aui-thread-welcome-suggestion-display fade-in slide-in-from-bottom-2 @md:nth-[n+3]:block nth-[n+3]:hidden animate-in fill-mode-both duration-200">
      <SuggestionPrimitive.Trigger send asChild>
        <Button
          variant="ghost"
          className="aui-thread-welcome-suggestion h-auto w-full @md:flex-col flex-wrap items-start justify-start gap-1 rounded-2xl border border-neutral-200 bg-background px-4 py-3 text-left text-sm transition-colors hover:bg-neutral-50 dark:border-neutral-800 dark:hover:bg-neutral-900"
        >
          <SuggestionPrimitive.Title className="aui-thread-welcome-suggestion-text-1 font-medium" />
          <SuggestionPrimitive.Description className="aui-thread-welcome-suggestion-text-2 text-neutral-500 dark:text-neutral-400 empty:hidden" />
        </Button>
      </SuggestionPrimitive.Trigger>
    </div>
  );
};

const Composer: FC = () => {
  return (
    <ComposerPrimitive.Root className="aui-composer-root relative flex w-full flex-col">
      <div
        data-slot="composer-shell"
        className="flex w-full flex-col gap-2 rounded-(--composer-radius) border border-neutral-200 bg-background p-(--composer-padding) transition-shadow focus-within:border-neutral-400 focus-within:ring-2 focus-within:ring-neutral-200 dark:border-neutral-800 dark:focus-within:border-neutral-600 dark:focus-within:ring-neutral-800"
      >
        <ComposerPrimitive.Input
          placeholder="Ask about AI in healthcare..."
          className="aui-composer-input max-h-32 min-h-10 w-full resize-none bg-transparent px-1.75 py-1 text-sm outline-none placeholder:text-neutral-400 dark:placeholder:text-neutral-500"
          rows={1}
          autoFocus
          aria-label="Message input"
        />
        <ComposerAction />
      </div>
    </ComposerPrimitive.Root>
  );
};

const ComposerAction: FC = () => {
  return (
    <div className="aui-composer-action-wrapper relative flex items-center justify-end">
      <AuiIf condition={(s) => !s.thread.isRunning}>
        <ComposerPrimitive.Send asChild>
          <TooltipIconButton
            tooltip="Send message"
            side="bottom"
            type="button"
            variant="default"
            size="icon"
            className="aui-composer-send size-8 rounded-full bg-neutral-900 text-white hover:bg-neutral-700 dark:bg-white dark:text-black dark:hover:bg-neutral-200"
            aria-label="Send message"
          >
            <ArrowUpIcon className="aui-composer-send-icon size-4" />
          </TooltipIconButton>
        </ComposerPrimitive.Send>
      </AuiIf>
      <AuiIf condition={(s) => s.thread.isRunning}>
        <ComposerPrimitive.Cancel asChild>
          <Button
            type="button"
            variant="default"
            size="icon"
            className="aui-composer-cancel size-8 rounded-full bg-neutral-900 text-white hover:bg-neutral-700 dark:bg-white dark:text-black dark:hover:bg-neutral-200"
            aria-label="Stop generating"
          >
            <SquareIcon className="aui-composer-cancel-icon size-3 fill-current" />
          </Button>
        </ComposerPrimitive.Cancel>
      </AuiIf>
    </div>
  );
};

const MessageError: FC = () => {
  return (
    <MessagePrimitive.Error>
      <ErrorPrimitive.Root className="aui-message-error-root mt-2 rounded-md border border-neutral-300 bg-neutral-50 p-3 text-neutral-700 text-sm dark:border-neutral-700 dark:bg-neutral-900 dark:text-neutral-300">
        <ErrorPrimitive.Message className="aui-message-error-message line-clamp-2" />
      </ErrorPrimitive.Root>
    </MessagePrimitive.Error>
  );
};

const AssistantMessage: FC = () => {
  const message = useMessage();
  const custom = (message?.metadata?.custom ?? null) as RagCustomMetadata | null;

  return (
    <MessagePrimitive.Root
      className="aui-assistant-message-root fade-in slide-in-from-bottom-1 relative mx-auto w-full max-w-(--thread-max-width) animate-in py-3 duration-150"
      data-role="assistant"
    >
      {/* Agent progression steps */}
      {custom?.agentSteps && custom.agentSteps.length > 0 && (
        <AgentSteps steps={custom.agentSteps} />
      )}

      <div className="aui-assistant-message-content wrap-break-word px-2 text-foreground leading-relaxed">
        <MessagePrimitive.Parts>
          {({ part }) => {
            if (part.type === "text") return <MarkdownText />;
            if (part.type === "tool-call") return null;
            return null;
          }}
        </MessagePrimitive.Parts>
        <MessageError />
      </div>

      {/* Reasoning trace */}
      {custom?.reasoning && (
        <div className="px-2">
          <ReasoningTrace text={custom.reasoning} />
        </div>
      )}

      {/* Confidence badge */}
      {custom?.critic && (
        <div className="px-2">
          <ConfidenceBadge
            score={custom.critic.confidence_score}
            flagged={custom.critic.flagged_claims}
          />
        </div>
      )}

      {/* Source cards */}
      {custom?.sources && custom.sources.length > 0 && (
        <div className="px-2">
          <SourceCards sources={custom.sources} />
        </div>
      )}

      <div className="aui-assistant-message-footer mt-2 ml-2 flex">
        <AssistantActionBar />
      </div>
    </MessagePrimitive.Root>
  );
};

const AssistantActionBar: FC = () => {
  return (
    <ActionBarPrimitive.Root
      hideWhenRunning
      autohide="not-last"
      autohideFloat="single-branch"
      className="aui-assistant-action-bar-root col-start-3 row-start-2 -ml-1 flex gap-1 text-neutral-400 dark:text-neutral-500 data-floating:absolute data-floating:rounded-md data-floating:border data-floating:border-neutral-200 data-floating:bg-background data-floating:p-1 data-floating:shadow-sm dark:data-floating:border-neutral-800"
    >
      <ActionBarPrimitive.Copy asChild>
        <TooltipIconButton tooltip="Copy">
          <AuiIf condition={(s) => s.message.isCopied}>
            <CheckIcon />
          </AuiIf>
          <AuiIf condition={(s) => !s.message.isCopied}>
            <CopyIcon />
          </AuiIf>
        </TooltipIconButton>
      </ActionBarPrimitive.Copy>
      <ActionBarMorePrimitive.Root>
        <ActionBarMorePrimitive.Trigger asChild>
          <TooltipIconButton
            tooltip="More"
            className="data-[state=open]:bg-neutral-100 dark:data-[state=open]:bg-neutral-800"
          >
            <MoreHorizontalIcon />
          </TooltipIconButton>
        </ActionBarMorePrimitive.Trigger>
        <ActionBarMorePrimitive.Content
          side="bottom"
          align="start"
          className="aui-action-bar-more-content z-50 min-w-32 overflow-hidden rounded-md border border-neutral-200 bg-background p-1 shadow-md dark:border-neutral-800"
        >
          <ActionBarPrimitive.ExportMarkdown asChild>
            <ActionBarMorePrimitive.Item className="aui-action-bar-more-item flex cursor-pointer select-none items-center gap-2 rounded-sm px-2 py-1.5 text-sm outline-none hover:bg-neutral-100 dark:hover:bg-neutral-800">
              <DownloadIcon className="size-4" />
              Export as Markdown
            </ActionBarMorePrimitive.Item>
          </ActionBarPrimitive.ExportMarkdown>
        </ActionBarMorePrimitive.Content>
      </ActionBarMorePrimitive.Root>
    </ActionBarPrimitive.Root>
  );
};

const UserMessage: FC = () => {
  return (
    <MessagePrimitive.Root
      className="aui-user-message-root fade-in slide-in-from-bottom-1 mx-auto grid w-full max-w-(--thread-max-width) animate-in auto-rows-auto grid-cols-[minmax(72px,1fr)_auto] content-start gap-y-2 px-2 py-3 duration-150 [&:where(>*)]:col-start-2"
      data-role="user"
    >
      <div className="aui-user-message-content-wrapper relative col-start-2 min-w-0">
        <div className="aui-user-message-content wrap-break-word peer rounded-2xl bg-neutral-100 px-4 py-2.5 text-foreground empty:hidden dark:bg-neutral-800">
          <MessagePrimitive.Parts />
        </div>
      </div>
    </MessagePrimitive.Root>
  );
};


const EditComposer: FC = () => {
  return (
    <MessagePrimitive.Root className="aui-edit-composer-wrapper mx-auto flex w-full max-w-(--thread-max-width) flex-col px-2 py-3">
      <ComposerPrimitive.Root className="aui-edit-composer-root ml-auto flex w-full max-w-[85%] flex-col rounded-2xl bg-neutral-100 dark:bg-neutral-800">
        <ComposerPrimitive.Input
          className="aui-edit-composer-input min-h-14 w-full resize-none bg-transparent p-4 text-foreground text-sm outline-none"
          autoFocus
        />
        <div className="aui-edit-composer-footer mx-3 mb-3 flex items-center gap-2 self-end">
          <ComposerPrimitive.Cancel asChild>
            <Button variant="ghost" size="sm">
              Cancel
            </Button>
          </ComposerPrimitive.Cancel>
          <ComposerPrimitive.Send asChild>
            <Button size="sm">Update</Button>
          </ComposerPrimitive.Send>
        </div>
      </ComposerPrimitive.Root>
    </MessagePrimitive.Root>
  );
};

const BranchPicker: FC<BranchPickerPrimitive.Root.Props> = ({
  className,
  ...rest
}) => {
  return (
    <BranchPickerPrimitive.Root
      hideWhenSingleBranch
      className={cn(
        "aui-branch-picker-root mr-2 -ml-2 inline-flex items-center text-neutral-400 text-xs dark:text-neutral-500",
        className,
      )}
      {...rest}
    >
      <BranchPickerPrimitive.Previous asChild>
        <TooltipIconButton tooltip="Previous">
          <ChevronLeftIcon />
        </TooltipIconButton>
      </BranchPickerPrimitive.Previous>
      <span className="aui-branch-picker-state font-medium">
        <BranchPickerPrimitive.Number /> / <BranchPickerPrimitive.Count />
      </span>
      <BranchPickerPrimitive.Next asChild>
        <TooltipIconButton tooltip="Next">
          <ChevronRightIcon />
        </TooltipIconButton>
      </BranchPickerPrimitive.Next>
    </BranchPickerPrimitive.Root>
  );
};
