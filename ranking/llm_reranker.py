"""
LLM-based reranker using OpenAI-compatible API.
Rescores candidate items based on user history and item metadata.
"""

import os
import json
from openai import OpenAI


def build_rerank_prompt(user_history_items, candidate_items, top_k=10):
    """
    Build a prompt for the LLM to rerank candidate items.

    Args:
        user_history_items: list of dicts with item metadata
            [{"title": "...", "category": "..."}, ...]
        candidate_items: list of dicts with item metadata and current rank
            [{"id": 0, "title": "...", "category": "...", "score": 0.9}, ...]
        top_k: number of items to return
    """
    history_str = "\n".join(
        f"- {item.get('title', 'Item')} (Category: {item.get('category', 'Unknown')})"
        for item in user_history_items[-10:]  # last 10 interactions
    )

    candidates_str = "\n".join(
        f"{i+1}. [ID:{item['id']}] {item.get('title', 'Item')} "
        f"(Category: {item.get('category', 'Unknown')}, Score: {item.get('score', 0):.3f})"
        for i, item in enumerate(candidate_items)
    )

    prompt = f"""You are a product recommendation expert. Based on the user's purchase/browsing history, rerank the candidate products by relevance.

## User's Recent History:
{history_str}

## Candidate Products (currently ranked by model score):
{candidates_str}

## Task:
Rerank the top {top_k} most relevant products for this user. Consider:
1. Category preferences from history
2. Complementary products the user might need
3. Diversity of recommendations

Return ONLY a JSON array of product IDs in order of relevance, like: [3, 1, 7, ...]
Do not include any other text."""

    return prompt


def rerank_with_llm(user_history_items, candidate_items, top_k=10,
                    api_key=None, model="gpt-3.5-turbo"):
    """
    Use LLM to rerank candidates.

    Args:
        user_history_items: list of dicts with item info
        candidate_items: list of dicts with item info and scores
        top_k: number to return
        api_key: OpenAI API key (or set OPENAI_API_KEY env var)
        model: LLM model name
    Returns:
        reranked list of item IDs
    """
    if api_key is None:
        api_key = os.environ.get("OPENAI_API_KEY")

    if not api_key:
        # Fallback: return original order if no API key
        print("Warning: No OPENAI_API_KEY set. Skipping LLM reranking.")
        return [item["id"] for item in candidate_items[:top_k]]

    client = OpenAI(api_key=api_key)
    prompt = build_rerank_prompt(user_history_items, candidate_items, top_k)

    response = client.chat.completions.create(
        model=model,
        messages=[{"role": "user", "content": prompt}],
        temperature=0.0,
        max_tokens=256,
    )

    # Parse response
    content = response.choices[0].message.content.strip()
    try:
        reranked_ids = json.loads(content)
        # Validate IDs
        valid_ids = {item["id"] for item in candidate_items}
        reranked_ids = [rid for rid in reranked_ids if rid in valid_ids]

        # Append any missing IDs at the end
        for item in candidate_items:
            if item["id"] not in reranked_ids:
                reranked_ids.append(item["id"])

        return reranked_ids[:top_k]
    except (json.JSONDecodeError, TypeError):
        print(f"Warning: Failed to parse LLM response: {content}")
        return [item["id"] for item in candidate_items[:top_k]]
