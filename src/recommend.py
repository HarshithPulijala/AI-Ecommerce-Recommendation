"""
E-commerce Product Recommendation System - Inference Module

This module provides product recommendations using trained ML models.
Supports collaborative filtering, content-based filtering, and hybrid approaches.
"""

import pandas as pd
import numpy as np
from pathlib import Path
import pickle
import yaml
import argparse
from sklearn.metrics.pairwise import cosine_similarity
from typing import List, Optional, Dict, Any, Set


class RecommendationEngine:
    """Main recommendation engine class"""

    def __init__(self, config_path: str = "config.yaml"):
        """Initialize the recommendation engine"""
        self.config = self._load_config(config_path)
        self.models_loaded = False

        # Initialize data containers
        self.products_df = None
        self.interactions_df = None
        self.user_to_idx = None
        self.product_to_idx = None

        # Initialize models
        self.svd_model = None
        self.user_factors = None
        self.product_factors = None
        self.product_features = None
        self.tfidf_vectorizer = None
        self.price_scaler = None

        # Perf caches (populated after load)
        self._product_info_by_id: Dict[str, Dict[str, Any]] = {}
        self._popular_product_ids: List[str] = []
        self._popular_products_df: Optional[pd.DataFrame] = None
        self._user_interaction_indices: Optional[Dict[str, np.ndarray]] = None
        self._user_rated_cache: Dict[str, Set[str]] = {}

    def _load_config(self, config_path: str) -> Dict[str, Any]:
        """Load configuration from YAML file"""
        with open(config_path, 'r') as f:
            return yaml.safe_load(f)

    def load_models(self) -> None:
        """Load all trained models and data from disk"""
        if self.models_loaded:
            return

        print("Loading models and data...")

        # Load configuration paths
        processed_dir = Path(self.config['data']['processed_dir'])
        models_dir = Path(self.config['data']['models_dir'])

        # Load processed data
        self.products_df = pd.read_csv(processed_dir / self.config['files']['products_clean'])
        self.interactions_df = pd.read_csv(processed_dir / self.config['files']['interactions_clean'])

        # Load mappings
        with open(models_dir / self.config['files']['user_to_idx'], 'rb') as f:
            self.user_to_idx = pickle.load(f)
        with open(models_dir / self.config['files']['product_to_idx'], 'rb') as f:
            self.product_to_idx = pickle.load(f)

        # Load SVD model
        with open(models_dir / self.config['files']['svd_model'], 'rb') as f:
            self.svd_model = pickle.load(f)
        with open(models_dir / self.config['files']['user_factors'], 'rb') as f:
            self.user_factors = pickle.load(f)
        with open(models_dir / self.config['files']['product_factors'], 'rb') as f:
            self.product_factors = pickle.load(f)

        # Load content-based features
        with open(models_dir / self.config['files']['product_features'], 'rb') as f:
            self.product_features = pickle.load(f)
        with open(models_dir / self.config['files']['tfidf_vectorizer'], 'rb') as f:
            self.tfidf_vectorizer = pickle.load(f)
        with open(models_dir / self.config['files']['price_scaler'], 'rb') as f:
            self.price_scaler = pickle.load(f)

        # Build fast lookups
        self._product_info_by_id = (
            self.products_df.set_index('product_id')[['category', 'brand', 'price']]
            .to_dict(orient='index')
        )

        # Cache per-user row indices (fast slicing vs boolean masks)
        self._user_interaction_indices = self.interactions_df.groupby('user_id').indices

        # Cache popularity once (used heavily by candidate generation)
        self._rebuild_popularity_cache()

        self.models_loaded = True
        print("Models and data loaded successfully!")

    def _rebuild_popularity_cache(self) -> None:
        """Compute popular product ordering once (expensive groupby)."""
        product_stats = self.interactions_df.groupby('product_id').agg(
            rating_count=('rating', 'count'),
            mean_rating=('rating', 'mean'),
        ).reset_index()

        product_stats['popularity_score'] = product_stats['rating_count'] * (product_stats['mean_rating'] / 5.0)
        product_stats = product_stats.sort_values('popularity_score', ascending=False)

        # Persist just IDs for candidate generation
        self._popular_product_ids = product_stats['product_id'].astype(str).tolist()

        # Persist a joined DF for quick top-N popular results
        popular_joined = product_stats.merge(self.products_df, on='product_id', how='left')
        popular_joined['title'] = popular_joined['brand'].fillna('') + ' ' + popular_joined['category'].fillna('')
        self._popular_products_df = popular_joined[['product_id', 'title', 'category', 'brand', 'price', 'rating_count', 'mean_rating', 'popularity_score']]

    def _get_user_interactions_view(self, user_id: str) -> pd.DataFrame:
        """Fast slice of interactions for a user."""
        if self._user_interaction_indices is None:
            self._user_interaction_indices = self.interactions_df.groupby('user_id').indices
        idx = self._user_interaction_indices.get(user_id)
        if idx is None or len(idx) == 0:
            return self.interactions_df.iloc[0:0][['product_id', 'rating']]
        return self.interactions_df.loc[idx, ['product_id', 'rating']]

    def _get_user_rated_set(self, user_id: str) -> Set[str]:
        cached = self._user_rated_cache.get(user_id)
        if cached is not None:
            return cached
        user_view = self._get_user_interactions_view(user_id)
        rated = set(user_view['product_id'].astype(str).tolist())
        self._user_rated_cache[user_id] = rated
        return rated

    def get_popular_products(self, top_n: int = 10) -> pd.DataFrame:
        """Get popular products based on rating count and average rating"""
        if not self.models_loaded:
            self.load_models()

        if self._popular_products_df is None:
            self._rebuild_popularity_cache()

        result = self._popular_products_df.head(top_n).copy()
        return result[['product_id', 'title', 'category', 'brand', 'price', 'rating_count', 'mean_rating']]

    def _predict_svd_score(self, user_id: str, product_id: str) -> float:
        """Predict rating using SVD collaborative filtering"""
        if user_id not in self.user_to_idx or product_id not in self.product_to_idx:
            return 3.0  # Default rating for cold-start

        user_idx = self.user_to_idx[user_id]
        product_idx = self.product_to_idx[product_id]

        # Matrix factorization prediction
        prediction = np.dot(self.user_factors[user_idx], self.product_factors[product_idx])
        return np.clip(prediction, 1.0, 5.0)

    def _predict_content_score(self, user_id: str, product_id: str, ignore_rated_product_ids: Optional[Set[str]] = None) -> float:
        """Predict rating using content-based filtering"""
        if product_id not in self.product_to_idx:
            return 3.0

        product_idx = self.product_to_idx[product_id]

        # Get user's rated products (fast slice)
        user_ratings = self._get_user_interactions_view(user_id)
        if ignore_rated_product_ids:
            user_ratings = user_ratings[~user_ratings['product_id'].isin(ignore_rated_product_ids)]

        if len(user_ratings) == 0:
            return 3.0  # Cold-start: no user history

        target_features = self.product_features[product_idx]

        pairs = []
        for pid, rating in zip(user_ratings['product_id'].astype(str).tolist(), user_ratings['rating'].to_numpy()):
            if pid in self.product_to_idx:
                pairs.append((pid, float(rating)))

        if not pairs:
            return 3.0

        rated_indices = [self.product_to_idx[pid] for pid, _ in pairs]
        ratings = np.array([r for _, r in pairs], dtype=float)
        rated_features = self.product_features[rated_indices]
        similarities = cosine_similarity(target_features.reshape(1, -1), rated_features).ravel()

        if similarities.size == 0 or ratings.size == 0:
            return 3.0

        sim_sum = float(similarities.sum())
        if sim_sum > 0:
            prediction = float(np.average(ratings, weights=similarities))
        else:
            prediction = float(np.mean(ratings))

        return np.clip(prediction, 1.0, 5.0)

    def _predict_content_scores_batch(
        self,
        user_id: str,
        product_ids: List[str],
        ignore_rated_product_ids: Optional[Set[str]] = None,
    ) -> Dict[str, float]:
        """Predict content scores for many candidate products in one shot."""
        # Default scores for unknown products
        scores: Dict[str, float] = {str(pid): 3.0 for pid in product_ids}

        # User history
        user_ratings = self._get_user_interactions_view(user_id)
        if ignore_rated_product_ids:
            user_ratings = user_ratings[~user_ratings['product_id'].isin(ignore_rated_product_ids)]
        if len(user_ratings) == 0:
            return scores

        pairs = []
        for pid, rating in zip(user_ratings['product_id'].astype(str).tolist(), user_ratings['rating'].to_numpy()):
            if pid in self.product_to_idx:
                pairs.append((pid, float(rating)))
        if not pairs:
            return scores

        rated_indices = np.array([self.product_to_idx[pid] for pid, _ in pairs], dtype=np.int32)
        ratings_arr = np.array([r for _, r in pairs], dtype=float)
        rated_features = self.product_features[rated_indices]

        # Candidate features
        valid_candidates = [str(pid) for pid in product_ids if str(pid) in self.product_to_idx]
        if not valid_candidates:
            return scores

        candidate_indices = np.array([self.product_to_idx[pid] for pid in valid_candidates], dtype=np.int32)
        candidate_features = self.product_features[candidate_indices]

        sim = cosine_similarity(candidate_features, rated_features)
        weight_sums = sim.sum(axis=1)
        weighted = sim @ ratings_arr

        mean_rating = float(np.mean(ratings_arr))
        preds = np.where(weight_sums > 0, weighted / weight_sums, mean_rating)
        preds = np.clip(preds, 1.0, 5.0)

        for pid, pred in zip(valid_candidates, preds.tolist()):
            scores[pid] = float(pred)

        return scores

    def recommend_svd(self, user_id: str, candidate_items: List[str], top_n: int, ignore_rated_product_ids: Optional[Set[str]] = None) -> pd.DataFrame:
        """Generate recommendations using SVD collaborative filtering"""
        if not self.models_loaded:
            self.load_models()

        if user_id not in self.user_to_idx:
            return pd.DataFrame()

        user_rated = set(self._get_user_rated_set(user_id))
        if ignore_rated_product_ids:
            user_rated -= set(ignore_rated_product_ids)

        # Filter candidates to unrated products and those present in factor matrix
        filtered_candidates = [str(pid) for pid in candidate_items if str(pid) not in user_rated and str(pid) in self.product_to_idx]
        if not filtered_candidates:
            return pd.DataFrame()

        user_idx = self.user_to_idx[user_id]
        candidate_indices = np.array([self.product_to_idx[pid] for pid in filtered_candidates], dtype=np.int32)

        # Vectorized scoring: dot(user_vec, product_vecs)
        user_vec = self.user_factors[user_idx]
        scores_arr = user_vec @ self.product_factors[candidate_indices].T
        scores_arr = np.clip(scores_arr, 1.0, 5.0)

        n_take = min(int(top_n), int(scores_arr.shape[0]))
        if n_take <= 0:
            return pd.DataFrame()

        # Partial selection for speed
        if scores_arr.shape[0] > n_take:
            top_idx = np.argpartition(-scores_arr, n_take - 1)[:n_take]
            top_idx = top_idx[np.argsort(-scores_arr[top_idx])]
        else:
            top_idx = np.argsort(-scores_arr)

        result = []
        for i in top_idx:
            product_id = filtered_candidates[int(i)]
            score = float(scores_arr[int(i)])
            info = self._product_info_by_id.get(product_id)
            if not info:
                continue
            title = f"{info['brand']} {info['category']}" if pd.notna(info.get('brand')) else info.get('category')
            result.append({
                'product_id': product_id,
                'title': title,
                'category': info.get('category'),
                'brand': info.get('brand'),
                'price': info.get('price'),
                'predicted_rating': round(score, 2)
            })

        return pd.DataFrame(result)

    def recommend_content(self, product_id: str, candidate_items: List[str], top_n: int) -> pd.DataFrame:
        """Generate content-based recommendations similar to a given product"""
        if not self.models_loaded:
            self.load_models()

        if product_id not in self.product_to_idx:
            return pd.DataFrame()

        product_idx = self.product_to_idx[product_id]
        target_features = self.product_features[product_idx]

        # Filter candidates
        candidates = [pid for pid in candidate_items if pid != product_id and pid in self.product_to_idx]

        if not candidates:
            return pd.DataFrame()

        # Compute similarities
        similarities = []
        for candidate_id in candidates:
            candidate_idx = self.product_to_idx[candidate_id]
            candidate_features = self.product_features[candidate_idx]

            similarity = cosine_similarity(target_features.reshape(1, -1),
                                        candidate_features.reshape(1, -1))[0, 0]
            similarities.append((candidate_id, similarity))

        # Sort by similarity (descending)
        similarities.sort(key=lambda x: x[1], reverse=True)

        # Get top N
        top_similar = similarities[:top_n]

        # Create result DataFrame
        result = []
        for product_id, similarity in top_similar:
            product_info = self.products_df[self.products_df['product_id'] == product_id]
            if not product_info.empty:
                info = product_info.iloc[0]
                title = f"{info['brand']} {info['category']}" if pd.notna(info['brand']) else info['category']
                result.append({
                    'product_id': product_id,
                    'title': title,
                    'category': info['category'],
                    'brand': info['brand'],
                    'price': info['price'],
                    'similarity_score': round(similarity, 3)
                })

        return pd.DataFrame(result)

    def _generate_candidates(self, user_id: str, last_viewed_product_id: Optional[str] = None) -> List[str]:
        """Generate candidate products for recommendation"""
        candidate_size = self.config['recommendation']['candidate_size']

        if last_viewed_product_id and last_viewed_product_id in self.product_to_idx:
            # Use same category products as candidates
            viewed_category = self.products_df[self.products_df['product_id'] == last_viewed_product_id]['category'].iloc[0]
            category_products = self.products_df[self.products_df['category'] == viewed_category]['product_id'].tolist()
            candidates = category_products[:candidate_size]
        else:
            # Use popular products as candidates
            if not self._popular_product_ids:
                self._rebuild_popularity_cache()
            candidates = self._popular_product_ids[:candidate_size]

        return candidates

    def recommend_products(
        self,
        user_id: str,
        last_viewed_product_id: Optional[str] = None,
        top_n: int = 10,
        ignore_rated_product_ids: Optional[Set[str]] = None,
        candidate_items: Optional[List[str]] = None,
        candidate_size: Optional[int] = None,
    ) -> pd.DataFrame:
        """
        Main recommendation function implementing hybrid logic

        Args:
            user_id: User ID to generate recommendations for
            last_viewed_product_id: Last product viewed (for content-based boost)
            top_n: Number of recommendations to return

        Returns:
            DataFrame with recommended products
        """
        if not self.models_loaded:
            self.load_models()

        # Cold-start handling
        if user_id not in self.user_to_idx:
            print(f"User {user_id} not found. Returning popular products.")
            return self.get_popular_products(top_n)

        # Candidate products (allow override for evaluation/collab)
        if candidate_items is not None:
            candidates = candidate_items
        else:
            if candidate_size is not None:
                # Temporary override (do not mutate config)
                prev_size = self.config['recommendation']['candidate_size']
                self.config['recommendation']['candidate_size'] = int(candidate_size)
                try:
                    candidates = self._generate_candidates(user_id, last_viewed_product_id)
                finally:
                    self.config['recommendation']['candidate_size'] = prev_size
            else:
                candidates = self._generate_candidates(user_id, last_viewed_product_id)

        if not candidates:
            return self.get_popular_products(top_n)

        # Get SVD recommendations (scored for all candidates)
        svd_recs = self.recommend_svd(user_id, candidates, len(candidates), ignore_rated_product_ids=ignore_rated_product_ids)

        if svd_recs.empty:
            return self.get_popular_products(top_n)

        # Hybrid scoring
        collaborative_weight = self.config['hybrid']['collaborative_weight']
        content_weight = self.config['hybrid']['content_weight']

        # Content scoring is the expensive part; limit it to the best collaborative candidates.
        # This keeps quality while making evaluation much faster.
        content_eval_limit = int(self.config.get('hybrid', {}).get('content_eval_limit', 200))

        final_scores = []
        svd_rows = svd_recs.head(content_eval_limit)
        candidate_ids = svd_rows['product_id'].astype(str).tolist()
        content_scores = self._predict_content_scores_batch(
            user_id=user_id,
            product_ids=candidate_ids,
            ignore_rated_product_ids=ignore_rated_product_ids,
        )

        for _, row in svd_rows.iterrows():
            product_id = str(row['product_id'])
            svd_score = float(row['predicted_rating'])
            content_score = float(content_scores.get(product_id, 3.0))
            hybrid_score = (collaborative_weight * svd_score) + (content_weight * content_score)
            final_scores.append((product_id, hybrid_score))

        # Sort by hybrid score
        final_scores.sort(key=lambda x: x[1], reverse=True)

        # Get top N
        top_scores = final_scores[:top_n]

        # Create final result
        result = []
        for product_id, score in top_scores:
            product_info = self.products_df[self.products_df['product_id'] == product_id]
            if not product_info.empty:
                info = product_info.iloc[0]
                title = f"{info['brand']} {info['category']}" if pd.notna(info['brand']) else info['category']
                result.append({
                    'product_id': product_id,
                    'title': title,
                    'category': info['category'],
                    'brand': info['brand'],
                    'price': info['price'],
                    'predicted_rating': round(score, 2)
                })

        return pd.DataFrame(result)


# Global engine instance
_engine = None

def get_engine() -> RecommendationEngine:
    """Get or create the global recommendation engine"""
    global _engine
    if _engine is None:
        _engine = RecommendationEngine()
    return _engine

def load_models() -> None:
    """Load all models and data"""
    get_engine().load_models()

def get_popular_products(top_n: int = 10) -> pd.DataFrame:
    """Get popular products"""
    return get_engine().get_popular_products(top_n)

def recommend_svd(user_id: str, candidate_items: List[str], top_n: int, ignore_rated_product_ids: Optional[Set[str]] = None) -> pd.DataFrame:
    """SVD-based recommendations"""
    return get_engine().recommend_svd(user_id, candidate_items, top_n, ignore_rated_product_ids=ignore_rated_product_ids)

def recommend_content(product_id: str, candidate_items: List[str], top_n: int) -> pd.DataFrame:
    """Content-based recommendations"""
    return get_engine().recommend_content(product_id, candidate_items, top_n)

def recommend_products(
    user_id: str,
    last_viewed_product_id: Optional[str] = None,
    top_n: int = 10,
    ignore_rated_product_ids: Optional[Set[str]] = None,
    candidate_items: Optional[List[str]] = None,
    candidate_size: Optional[int] = None,
) -> pd.DataFrame:
    """Main recommendation function"""
    return get_engine().recommend_products(
        user_id,
        last_viewed_product_id=last_viewed_product_id,
        top_n=top_n,
        ignore_rated_product_ids=ignore_rated_product_ids,
        candidate_items=candidate_items,
        candidate_size=candidate_size,
    )


def main():
    """Command line interface"""
    parser = argparse.ArgumentParser(description='E-commerce Product Recommendations')
    parser.add_argument('--user_id', type=str, required=True, help='User ID for recommendations')
    parser.add_argument('--last_viewed', type=str, help='Last viewed product ID')
    parser.add_argument('--top_n', type=int, default=10, help='Number of recommendations')

    args = parser.parse_args()

    # Load models
    load_models()

    # Generate recommendations
    recommendations = recommend_products(
        user_id=args.user_id,
        last_viewed_product_id=args.last_viewed,
        top_n=args.top_n
    )

    # Display results
    if recommendations.empty:
        print("No recommendations available.")
    else:
        print(f"\nTop {len(recommendations)} Recommendations for User {args.user_id}:")
        print("=" * 80)

        for idx, row in recommendations.iterrows():
            print(f"{idx+1}. {row['title'][:70]}{'...' if len(row['title']) > 70 else ''}")
            print(f"   Category: {row['category']}")
            print(f"   Brand: {row['brand']}")
            print(f"   Price: ${row['price']:.2f}")
            # Handle different column names for popular vs predicted recommendations
            if 'predicted_rating' in row:
                print(f"   Predicted Rating: {row['predicted_rating']:.2f}/5.0")
            elif 'mean_rating' in row:
                print(f"   Average Rating: {row['mean_rating']:.2f}/5.0 ({int(row['rating_count'])} reviews)")
            print()


if __name__ == "__main__":
    main()