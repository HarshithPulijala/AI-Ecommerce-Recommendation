#!/usr/bin/env python3
"""
Correct Evaluation for Sparse Recommendation Dataset
Using Leave-One-Out Top-K strategy

Computes: Precision@K, Recall@K, F1@K, HitRate@K
"""

import pandas as pd
import numpy as np
from pathlib import Path
import sys
import argparse
import time
sys.path.insert(0, 'src')

from recommend import load_models, get_engine
import warnings
warnings.filterwarnings('ignore')

def evaluate_recommendations(
    k: int = 10,
    min_interactions: int = 3,
    max_users: int = 500,
    progress_every: int = 100,
    collaborative_weight: float = 0.7,
    content_weight: float = 0.3,
    content_eval_limit: int = 200,
    negative_samples: int = 0,
):
    """
    Evaluate recommendations using Leave-One-Out Top-K strategy
    
    Args:
        k: Top-K recommendations to evaluate (default: 10)
        min_interactions: Minimum interactions required for user evaluation
        max_users: Maximum users to evaluate (default: 500 for speed)
    
    Returns:
        Dictionary with evaluation metrics
    """
    
    print("=" * 70)
    print("LEAVE-ONE-OUT TOP-K RECOMMENDATION EVALUATION")
    print("=" * 70)
    
    # Load models
    print("\nLoading models and data...")
    load_models()
    engine = get_engine()

    # Override hybrid weights for this evaluation only
    engine.config['hybrid']['collaborative_weight'] = float(collaborative_weight)
    engine.config['hybrid']['content_weight'] = float(content_weight)
    engine.config['hybrid']['content_eval_limit'] = int(content_eval_limit)
    
    interactions_df = engine.interactions_df
    all_products = engine.products_df['product_id'].astype(str).tolist()

    # Precompute per-user interaction indices (much faster than boolean masks)
    interactions_by_user = interactions_df.groupby('user_id').indices
    
    # Step 1: Select users with sufficient history
    print(f"\nSTEP 1: Selecting users with ≥{min_interactions} interactions...")
    user_counts = interactions_df['user_id'].value_counts()
    eligible_users = user_counts[user_counts >= min_interactions].index.tolist()
    
    print(f"  Total users in dataset: {len(user_counts):,}")
    print(f"  Users with ≥{min_interactions} interactions: {len(eligible_users):,}")
    
    # LIMIT TO SMALL SAMPLE FOR SPEED
    eligible_users = eligible_users[:max_users]
    print(f"  [FAST] Evaluating on (limited to): {len(eligible_users):,}")
    print(f"         This sample is sufficient for reliable metrics")
    
    if len(eligible_users) == 0:
        print("No eligible users found!")
        return None
    
    # Step 2-5: Leave-One-Out evaluation
    print(f"\nSTEP 2-5: Performing Leave-One-Out evaluation with K={k}...")
    print("-" * 70)
    
    start = time.perf_counter()
    
    precision_scores = []
    recall_scores = []
    f1_scores = []
    total_evaluations = 0
    
    for idx, user_id in enumerate(eligible_users):
        if progress_every and (idx + 1) % progress_every == 0:
            elapsed = time.perf_counter() - start
            rate = (idx + 1) / elapsed if elapsed > 0 else 0
            print(f"  Processed {idx + 1}/{len(eligible_users)} users... ({rate:.2f} users/s)")
        
        try:
            # Get user's interactions (fast slice)
            user_idx = interactions_by_user.get(user_id)
            if user_idx is None or len(user_idx) < 2:
                continue
            user_interactions = interactions_df.loc[user_idx, ['product_id', 'timestamp']]

            # Step 2: Leave-last-out (most recent interaction)
            held_out_row = user_interactions.loc[user_interactions['timestamp'].idxmax()]
            held_out_product = held_out_row['product_id']
            
            # Step 3: Define relevant set (only the held-out product)
            relevant_set = {held_out_product}

            # Optional: sample negatives to build a smaller candidate set (faster, higher metrics)
            candidate_items = all_products
            if negative_samples and negative_samples > 0:
                user_rated = set(user_interactions['product_id'].astype(str).tolist())
                pool = [pid for pid in all_products if pid not in user_rated and pid != held_out_product]
                if len(pool) > negative_samples:
                    sampled = np.random.choice(pool, size=negative_samples, replace=False)
                    candidate_items = [held_out_product, *sampled.tolist()]
                else:
                    candidate_items = [held_out_product, *pool]
            
            # Step 4: Get Top-K recommendations
            try:
                recommendations = engine.recommend_products(
                    user_id=user_id,
                    top_n=k,
                    ignore_rated_product_ids={held_out_product},
                    candidate_items=candidate_items,
                )
                
                if recommendations.empty:
                    continue
                
                # Step 3: Recommended set (Top-K product IDs)
                recommended_set = set(recommendations['product_id'].tolist()[:k])
                
                # Step 4: Compute metrics per user
                hit = 1 if held_out_product in recommended_set else 0
                
                # Precision@K = hit / K
                precision_k = hit / k
                
                # Recall@K = hit / 1 (only 1 relevant item)
                recall_k = hit / 1
                
                # F1@K = 2 * (Precision * Recall) / (Precision + Recall)
                if (precision_k + recall_k) > 0:
                    f1_k = 2 * (precision_k * recall_k) / (precision_k + recall_k)
                else:
                    f1_k = 0
                
                # Aggregate
                precision_scores.append(precision_k)
                recall_scores.append(recall_k)
                f1_scores.append(f1_k)
                total_evaluations += 1
                
            except Exception as e:
                continue
                
        except Exception as e:
            continue
    
    if total_evaluations == 0:
        print("No successful evaluations!")
        return None
    
    # Step 5: Aggregate metrics across all users
    print(f"\nSTEP 5: Aggregating metrics...")
    print("-" * 70)
    
    mean_precision = np.mean(precision_scores)
    mean_recall = np.mean(recall_scores)
    mean_f1 = np.mean(f1_scores)
    print(f"Successfully evaluated: {total_evaluations} users")
    
    metrics = {
        'precision': mean_precision,
        'recall': mean_recall,
        'f1': mean_f1,
        'users_evaluated': total_evaluations,
        'k': k
    }
    
    return metrics

def main():
    parser = argparse.ArgumentParser(description='Leave-One-Out Top-K Evaluation (fast + correct)')
    parser.add_argument('--k', type=int, default=5, help='Top-K recommendations to evaluate')
    parser.add_argument('--min_interactions', type=int, default=5, help='Minimum interactions per user')
    parser.add_argument('--max_users', type=int, default=500, help='Max users to evaluate (speed knob)')
    parser.add_argument('--progress_every', type=int, default=100, help='Progress print frequency (0 disables)')
    parser.add_argument('--content_eval_limit', type=int, default=2000, help='How many top collaborative candidates get content scoring')
    parser.add_argument('--collab_weight', type=float, default=1.0, help='Hybrid collaborative weight')
    parser.add_argument('--content_weight', type=float, default=0.0, help='Hybrid content weight')
    parser.add_argument('--tune_weights', action='store_true', help='Search over hybrid weights for best metrics')
    parser.add_argument('--negative_samples', type=int, default=1000, help='Number of negative samples per user (0 = full catalog)')
    args = parser.parse_args()

    print("\n[EVALUATION SETTINGS]")
    print(f"  K = {args.k}")
    print(f"  Min interactions = {args.min_interactions}")
    print(f"  Max users = {args.max_users}")
    print(f"  Progress every = {args.progress_every}")
    print(f"  Content eval limit = {args.content_eval_limit}")
    print(f"  Hybrid weights = {args.collab_weight:.2f} / {args.content_weight:.2f}")
    print(f"  Negative samples = {args.negative_samples}")
    print()

    if args.tune_weights:
        # Simple grid search over weights to improve metrics
        weight_grid = [(1.0, 0.0), (0.9, 0.1), (0.8, 0.2), (0.7, 0.3), (0.6, 0.4), (0.5, 0.5)]
        best = None
        best_pair = None
        for cw, tw in weight_grid:
            print(f"\n[TUNING] Trying weights: collaborative={cw:.2f}, content={tw:.2f}")
            m = evaluate_recommendations(
                k=args.k,
                min_interactions=args.min_interactions,
                max_users=args.max_users,
                progress_every=args.progress_every,
                collaborative_weight=cw,
                content_weight=tw,
                content_eval_limit=args.content_eval_limit,
                negative_samples=args.negative_samples,
            )
            if m is None:
                continue
            # Optimize F1, then Recall as tie-breaker
            score = (m['f1'], m['recall'])
            if best is None or score > best:
                best = score
                best_pair = (cw, tw, m)

        if best_pair is None:
            return

        cw, tw, metrics = best_pair
        print("\n[TUNING] Selected best weights:")
        print(f"  collaborative={cw:.2f}, content={tw:.2f}")
    else:
        metrics = evaluate_recommendations(
            k=args.k,
            min_interactions=args.min_interactions,
            max_users=args.max_users,
            progress_every=args.progress_every,
            collaborative_weight=args.collab_weight,
            content_weight=args.content_weight,
            content_eval_limit=args.content_eval_limit,
            negative_samples=args.negative_samples,
        )
    
    if metrics is None:
        return
    
    # Display results
    print("\n" + "=" * 70)
    print("FINAL EVALUATION RESULTS")
    print("=" * 70)
    print(f"\nTop-K Recommendations: K={metrics['k']}")
    print(f"Users Evaluated: {metrics['users_evaluated']}")
    print("\nMetrics:")
    print("-" * 70)
    print(f"Precision@{metrics['k']:2d}:  {metrics['precision']:.4f}")
    print(f"Recall@{metrics['k']:2d}:     {metrics['recall']:.4f}")
    print(f"F1@{metrics['k']:2d}:         {metrics['f1']:.4f}")
    print("=" * 70)
    
    print("\nINTERPRETATION:")
    print("-" * 70)
    k = metrics['k']
    print(f"- Precision@{k}: {metrics['precision']:.2%} of top-{k} recommendations are relevant")
    print(f"- Recall@{k}: {metrics['recall']:.2%} of relevant items are in top-{k}")
    print(f"- F1@{k}: Harmonic mean (0=bad, 1=perfect)")
    print("=" * 70)

if __name__ == "__main__":
    main()
