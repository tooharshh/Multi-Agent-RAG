"""
Shared conversation history compression utilities.
Provides topic-line extraction and history trimming for token-budget management.
"""

import re
import tiktoken

# Lazily initialized tokenizer
_enc = None

def _get_encoder():
    global _enc
    if _enc is None:
        _enc = tiktoken.get_encoding("cl100k_base")
    return _enc


def _count_tokens(text: str) -> int:
    return len(_get_encoder().encode(text))


def _truncate_to_tokens(text: str, max_tokens: int) -> str:
    enc = _get_encoder()
    tokens = enc.encode(text)
    if len(tokens) <= max_tokens:
        return text
    return enc.decode(tokens[:max_tokens]) + "..."


# Domain terms that should be captured in topic lines even if they aren't
# bold, capitalized, or a stat. Kept in sync with CONCEPT_CROSS_TERMS keys.
_DOMAIN_TERMS = {
    "sycophancy", "algorithmic bias", "conformity assessment", "binding enforcement",
    "ce marking", "adaptive regulatory", "post-market surveillance", "scoping review",
    "cross-sectional", "phase ii", "phase iia", "idiopathic pulmonary fibrosis",
    "chatrwd", "openevidence", "eu ai act", "fda", "limbic", "kaiser permanente",
    "recursion", "exscientia", "rentosertib", "ism001-055",
    "drug discovery", "clinical ai", "wearables", "radiology",
    "diabetic retinopathy", "mental health", "neonatal", "oncology",
    "genomics", "robotic surgery", "sepsis", "mammography",
    "retrieval-augmented generation", "rag",
}


def extract_topic_line(answer: str, max_tokens: int = 80) -> str:
    """
    Extract key topics, entities, statistics, and doc IDs from a full assistant
    answer.  Returns a compact string suitable for conversation-context injection
    into a follow-up rewrite prompt.

    Scans the ENTIRE answer (not just the first N sentences) so that content
    from any position is captured.
    """
    if not answer or not answer.strip():
        return ""

    parts: list[str] = []

    # 1. Cited doc IDs
    doc_ids = sorted(set(re.findall(r'\[DOC-\d{3}\]', answer)))
    if doc_ids:
        # Strip brackets for compactness
        ids = [d.strip("[]") for d in doc_ids[:8]]
        parts.append(f"Sources: {', '.join(ids)}")

    # 2. Bold terms (markdown **term**)
    bold_terms = re.findall(r'\*\*([^*]+)\*\*', answer)
    if bold_terms:
        # Deduplicate preserving order
        seen = set()
        unique = []
        for t in bold_terms:
            low = t.lower()
            if low not in seen:
                seen.add(low)
                unique.append(t)
        parts.append(f"Topics: {', '.join(unique[:6])}")

    # 3. Percentages and dollar figures
    stats = re.findall(r'~?\d+(?:[.,]\d+)?%|\$[\d.,]+[BMK]?', answer)
    if stats:
        unique_stats = list(dict.fromkeys(stats))[:5]
        parts.append(f"Stats: {', '.join(unique_stats)}")

    # 4. Capitalized multi-word phrases (proper nouns / named entities)
    entities = re.findall(r'(?:[A-Z][a-z]+(?:\s+[A-Z][a-z]+)+)', answer)
    if entities:
        seen = set()
        unique = []
        for e in entities:
            low = e.lower()
            if low not in seen:
                seen.add(low)
                unique.append(e)
        parts.append(f"Entities: {', '.join(unique[:5])}")

    # 5. Domain-specific terms found anywhere in the answer
    answer_lower = answer.lower()
    found_domain = [t for t in _DOMAIN_TERMS if t in answer_lower]
    if found_domain:
        parts.append(f"Domain: {', '.join(sorted(found_domain)[:5])}")

    topic_line = "; ".join(parts)

    # Trim to budget
    return _truncate_to_tokens(topic_line, max_tokens)


def compress_history(
    conversation_history: list[dict],
    max_turns: int = 5,
    topic_tokens: int = 80,
    fallback_answer_tokens: int = 100,
) -> list[dict]:
    """
    Compress conversation history for agent consumption.

    Each compressed turn contains:
      - "user": original user question (kept verbatim — typically short)
      - "topic_line": compact extraction of key topics from assistant answer
      - "assistant_answer": truncated assistant answer (fallback for Agent 2)

    Args:
        conversation_history: Raw list of {"user": ..., "assistant_answer": ...}
        max_turns: Sliding window size
        topic_tokens: Max tokens for topic_line per turn
        fallback_answer_tokens: Max tokens for truncated assistant answer

    Returns:
        List of compressed turn dicts (most recent last).
    """
    # Take the last N turns
    recent = conversation_history[-max_turns:] if conversation_history else []

    compressed = []
    for turn in recent:
        user_q = turn.get("user", "")
        full_answer = turn.get("assistant_answer", "")

        compressed.append({
            "user": user_q,
            "topic_line": extract_topic_line(full_answer, max_tokens=topic_tokens),
            "assistant_answer": _truncate_to_tokens(full_answer, fallback_answer_tokens),
        })

    return compressed


def history_budget_check(
    compressed_history: list[dict],
    max_total_tokens: int = 1000,
    mode: str = "topic",
) -> list[dict]:
    """
    Trim compressed history to fit within a token budget, dropping oldest first.

    Args:
        compressed_history: Output of compress_history()
        max_total_tokens: Hard ceiling
        mode: "topic" uses topic_line field, "answer" uses assistant_answer field

    Returns:
        Trimmed list (newest turns preserved).
    """
    field = "topic_line" if mode == "topic" else "assistant_answer"
    result = []
    total = 0

    # Walk newest-first, accumulate
    for turn in reversed(compressed_history):
        text = turn.get("user", "") + " " + turn.get(field, "")
        tokens = _count_tokens(text)
        if total + tokens > max_total_tokens:
            break
        result.insert(0, turn)
        total += tokens

    return result
