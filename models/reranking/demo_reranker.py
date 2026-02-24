"""
Demo script for LLM Reranker.

This demonstrates how to use the LLM reranker without requiring
actual trained models or item metadata.
"""

import sys
from pathlib import Path

sys.path.append(str(Path(__file__).parent.parent.parent))

from models.reranking.llm_reranker import SimpleLLMReranker


def demo_reranker():
    """Demonstrate LLM reranker with example data."""
    print("="*80)
    print("LLM Reranker Demo")
    print("="*80)

    # Create reranker
    print("\nInitializing LLM reranker...")
    print("(This will download a small ~80MB model on first run)")

    reranker = SimpleLLMReranker(
        model_name="sentence-transformers/all-MiniLM-L6-v2"
    )

    print("[OK] Reranker initialized!\n")

    # Example user history
    user_history = [
        "Laptop computer 15 inch for programming",
        "Wireless mouse gaming RGB",
        "Mechanical keyboard Cherry MX switches"
    ]

    # Example candidates from ranking model
    candidates = [
        "USB-C hub adapter for laptop with multiple ports",
        "Gaming headset with microphone noise cancelling",
        "Laptop cooling pad with fans",
        "Office chair ergonomic adjustable",
        "Monitor 27 inch 4K gaming",
        "Desk lamp LED adjustable brightness",
        "External hard drive 1TB portable",
        "Webcam 1080p for video calls",
        "Laptop backpack waterproof with USB charging port",
        "Portable SSD 500GB high speed"
    ]

    # Previous scores from ranking model (simulated)
    previous_scores = [0.65, 0.72, 0.68, 0.45, 0.81, 0.42, 0.58, 0.51, 0.63, 0.70]

    print("User History:")
    for i, item in enumerate(user_history, 1):
        print(f"  {i}. {item}")

    print("\nCandidates from Ranking Model (with scores):")
    for i, (cand, score) in enumerate(zip(candidates, previous_scores), 1):
        print(f"  {i}. [{score:.2f}] {cand}")

    # Rerank with LLM
    print("\nReranking with LLM (alpha=0.5)...")
    top_indices, top_scores = reranker.rerank(
        user_history=user_history,
        candidates=candidates,
        candidate_scores=previous_scores,
        top_k=5,
        alpha=0.5  # 50% LLM, 50% previous ranker
    )

    print("\nTop-5 After LLM Reranking:")
    for i, (idx, score) in enumerate(zip(top_indices, top_scores), 1):
        prev_score = previous_scores[idx]
        improvement = score - prev_score
        print(f"  {i}. [{score:.2f}] {candidates[idx]}")
        print(f"      (Previous: {prev_score:.2f}, Change: {improvement:+.2f})")

    # Compare with LLM only
    print("\n" + "="*80)
    print("Comparison: LLM Only (alpha=1.0)")
    print("="*80)

    top_indices_llm, top_scores_llm = reranker.rerank(
        user_history=user_history,
        candidates=candidates,
        candidate_scores=None,  # No previous scores
        top_k=5,
        alpha=1.0
    )

    print("\nTop-5 with LLM Only:")
    for i, (idx, score) in enumerate(zip(top_indices_llm, top_scores_llm), 1):
        print(f"  {i}. [{score:.2f}] {candidates[idx]}")

    print("\n" + "="*80)
    print("Demo Complete!")
    print("="*80)
    print("\nKey Observations:")
    print("1. LLM captures semantic similarity (laptop accessories ranked higher)")
    print("2. Combining LLM + ranking model gives balanced results")
    print("3. LLM helps with cold-start and semantic understanding")
    print("\nNote: This is a lightweight model (~80MB). For production,")
    print("      consider larger models like sentence-transformers/all-mpnet-base-v2")


if __name__ == "__main__":
    try:
        demo_reranker()
    except ImportError as e:
        print(f"\nError: {e}")
        print("\nPlease install transformers:")
        print("  pip install transformers")
    except Exception as e:
        print(f"\nError: {e}")
        print("\nIf you don't have internet, the model will fail to download.")
        print("The LLM reranker is optional and can be skipped for now.")
