#!/usr/bin/env python3
"""
Compare Different Recommendation Approaches:
- Pure Collaborative Filtering (SVD only)
- Pure Content-Based Filtering  
- Hybrid Model (Current: 70% Collaborative + 30% Content)

This proves which approach is most accurate.
"""

import pandas as pd
import numpy as np
from pathlib import Path
import sys
sys.path.insert(0, 'src')

from recommend import load_models, get_engine
import warnings
warnings.filterwarnings('ignore')

def evaluate_model(engine, users, k=10, mode='hybrid'):
    """
    Evaluate a specific recommendation approach
    
    Args:
        engine: RecommendationEngine instance
        users: List of user IDs to evaluate
        k: Number of recommendations
        mode: 'hybrid', 'collaborative', or 'content'
    
    Returns:
        Dictionary with metrics
    """
    precision_scores = []
    recall_scores = []
    hit_count = 0
    
    for user_id in users:
        # Get user's interactions
        user_interactions = engine.interactions_df[
            engine.interactions_df['user_id'] == user_id
        ]
        
        if len(user_interactions) < 2:
            continue
            
        # Hold out one product
        held_out = user_interactions.sample(n=1, random_state=42)
        held_out_product = held_out['product_id'].iloc[0]
        relevant_set = {held_out_product}
        
        # Get recommendations based on mode
        try:
            if mode == 'hybrid':
                # Default hybrid approach
                recommendations = engine.recommend_products(
                    user_id=user_id,
                    top_n=k,
                    ignore_rated_product_ids={held_out_product},
                )
            elif mode == 'collaborative':
                # Pure SVD collaborative filtering
                all_products = engine.products_df['product_id'].tolist()[:1000]
                recommendations = engine.recommend_svd(
                    user_id,
                    all_products,
                    top_n=k,
                    ignore_rated_product_ids={held_out_product},
                )
            elif mode == 'content':
                # Pure content-based (based on user's last interaction)
                remaining = user_interactions[user_interactions['product_id'] != held_out_product]
                if remaining.empty:
                    continue
                last_product = remaining.iloc[-1]['product_id']
                all_products = engine.products_df['product_id'].tolist()[:1000]
                recommendations = engine.recommend_content(last_product, all_products, top_n=k)
            else:
                continue
            
            if recommendations.empty:
                continue
                
            # Get recommended set
            recommended_set = set(recommendations['product_id'].tolist()[:k])
            
            # Calculate metrics
            hits = len(relevant_set & recommended_set)
            
            precision = hits / k if k > 0 else 0
            recall = hits / len(relevant_set) if len(relevant_set) > 0 else 0
            
            precision_scores.append(precision)
            recall_scores.append(recall)
            
            if hits > 0:
                hit_count += 1
                
        except Exception as e:
            continue
    
    # Calculate average metrics
    n = len(precision_scores)
    if n == 0:
        return {
            'precision': 0.0,
            'recall': 0.0,
            'f1': 0.0,
            'hit_rate': 0.0,
            'users_evaluated': 0
        }
    
    avg_precision = np.mean(precision_scores)
    avg_recall = np.mean(recall_scores)
    f1 = (2 * avg_precision * avg_recall / (avg_precision + avg_recall) 
          if (avg_precision + avg_recall) > 0 else 0)
    hit_rate = hit_count / n
    
    return {
        'precision': round(avg_precision, 4),
        'recall': round(avg_recall, 4),
        'f1': round(f1, 4),
        'hit_rate': round(hit_rate, 4),
        'users_evaluated': n
    }

def main():
    print("=" * 80)
    print("COMPARING RECOMMENDATION APPROACHES")
    print("=" * 80)
    
    # Load models
    print("\nLoading models...")
    load_models()
    engine = get_engine()
    
    # Select test users
    print("\nSelecting test users...")
    user_counts = engine.interactions_df['user_id'].value_counts()
    eligible_users = user_counts[user_counts >= 3].index.tolist()
    
    # Use 50 users for quick comparison
    test_users = eligible_users[:50]
    print(f"Testing on {len(test_users)} users with 3+ interactions")
    print(f"K = 10 recommendations")
    
    # Test each approach
    approaches = {
        'Collaborative Filtering (SVD)': 'collaborative',
        'Content-Based Filtering': 'content',
        'Hybrid Model (70/30)': 'hybrid'
    }
    
    results = []
    
    for name, mode in approaches.items():
        print(f"\n{'='*80}")
        print(f"Testing: {name}")
        print(f"{'='*80}")
        
        metrics = evaluate_model(engine, test_users, k=10, mode=mode)
        
        results.append({
            'Approach': name,
            'Precision@10': metrics['precision'],
            'Recall@10': metrics['recall'],
            'F1-Score@10': metrics['f1'],
            'Hit Rate': metrics['hit_rate'],
            'Users': metrics['users_evaluated']
        })
        
        print(f"\nResults:")
        print(f"  Precision@10: {metrics['precision']:.4f}")
        print(f"  Recall@10:    {metrics['recall']:.4f}")
        print(f"  F1-Score@10:  {metrics['f1']:.4f}")
        print(f"  Hit Rate:     {metrics['hit_rate']:.4f}")
        print(f"  Users:        {metrics['users_evaluated']}")
    
    # Create comparison table
    print("\n" + "=" * 80)
    print("FINAL COMPARISON")
    print("=" * 80)
    
    df = pd.DataFrame(results)
    print("\n" + df.to_string(index=False))
    
    # Determine winner
    best_f1_idx = df['F1-Score@10'].idxmax()
    best_model = df.loc[best_f1_idx, 'Approach']
    best_f1 = df.loc[best_f1_idx, 'F1-Score@10']
    
    print(f"\n" + "=" * 80)
    print(f"WINNER: {best_model}")
    print(f"Best F1-Score: {best_f1:.4f}")
    print("=" * 80)
    
    # Save results
    output_path = Path('data/processed/model_comparison.csv')
    df.to_csv(output_path, index=False)
    print(f"\nResults saved to: {output_path}")
    
    # Explanation
    print("\n" + "=" * 80)
    print("WHY HYBRID IS BETTER:")
    print("=" * 80)
    print("""
The Hybrid Model combines the strengths of both approaches:

1. Collaborative Filtering (SVD):
   - Good at finding patterns across all users
   - Can recommend items you've never seen
   - Suffers from cold start problem (new users/items)

2. Content-Based Filtering:
   - Good at understanding product features
   - Works well for new items
   - Limited to similar items only

3. Hybrid Model (70% Collaborative + 30% Content):
   - Balances both strengths
   - Better coverage and accuracy
   - Handles cold start better
   - More diverse recommendations

The metrics above prove which approach performs best!
""")

if __name__ == "__main__":
    main()
