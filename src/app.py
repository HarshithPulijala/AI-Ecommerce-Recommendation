"""
Flask Web Application for E-commerce Recommendation System
Provides REST API endpoints for product recommendations
"""

from flask import Flask, request, jsonify, send_from_directory
from flask_cors import CORS
import pandas as pd
import numpy as np
import logging
import os
import sys
import random
import gc
from pathlib import Path
from datetime import datetime

# Add parent directory to path for imports
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.recommend import RecommendationEngine, load_models, get_engine, recommend_products

# Initialize Flask app
app = Flask(__name__, static_folder='../static', static_url_path='/static')
CORS(app)

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# Global recommendation engine instance
recommendation_engine = None
models_status = {
    'loaded': False,
    'error': None,
    'timestamp': None,
    'loading': False
}

# Flag to ensure we only load once
_initialization_started = False


def initialize_engine():
    """Initialize the recommendation engine"""
    global recommendation_engine, models_status, _initialization_started
    
    if _initialization_started:
        return
    
    _initialization_started = True
    models_status['loading'] = True
    
    try:
        logger.info("Initializing recommendation engine...")
        recommendation_engine = get_engine()
        load_models()
        
        # Force garbage collection to free memory after loading
        gc.collect()
        
        models_status['loaded'] = True
        models_status['loading'] = False
        models_status['timestamp'] = datetime.now().isoformat()
        logger.info("✓ Recommendation engine initialized successfully")
    except Exception as e:
        logger.error(f"Failed to initialize recommendation engine: {str(e)}", exc_info=True)
        models_status['loaded'] = False
        models_status['loading'] = False
        models_status['error'] = str(e)
        _initialization_started = False


def ensure_initialized():
    """Ensure models are initialized, trigger loading if needed"""
    if not models_status['loaded'] and not models_status['loading']:
        try:
            initialize_engine()
        except Exception as e:
            logger.error(f"Initialization failed: {str(e)}")


# ==================== API ENDPOINTS ====================

@app.route('/health', methods=['GET'])
def health_check():
    """Health check endpoint - returns system status"""
    return jsonify({
        'status': 'healthy',
        'timestamp': datetime.now().isoformat(),
        'models_loaded': models_status['loaded'],
        'models_error': models_status['error'],
        'initialized_at': models_status['timestamp']
    }), 200


@app.route('/api/health', methods=['GET'])
def api_health():
    """API health endpoint"""
    return health_check()


@app.route('/api/recommend', methods=['POST'])
def get_recommendations():
    """
    Get product recommendations for a user
    
    Request JSON:
    {
        "user_id": "string",
        "top_n": int (optional, default: 10),
        "exclude_purchased": bool (optional, default: true)
    }
    
    Response:
    {
        "success": bool,
        "user_id": string,
        "top_n": int,
        "recommendations": [
            {
                "rank": int,
                "product_id": string,
                "title": string,
                "category": string,
                "brand": string,
                "price": float,
                "predicted_rating": float,
                "url": string
            }
        ],
        "processing_time_ms": float
    }
    """
    # Trigger lazy loading if needed
    ensure_initialized()
    
    start_time = datetime.now()
    
    try:
        # Parse request
        data = request.get_json()
        
        if not data:
            return jsonify({
                'success': False,
                'error': 'Request body must be JSON'
            }), 400
        
        # Validate required fields
        user_id = data.get('user_id')
        if not user_id:
            return jsonify({
                'success': False,
                'error': 'user_id is required'
            }), 400
        
        # Get optional parameters
        top_n = data.get('top_n', 10)
        exclude_purchased = data.get('exclude_purchased', True)
        
        # Validate parameters
        if not isinstance(top_n, int) or top_n < 1 or top_n > 100:
            return jsonify({
                'success': False,
                'error': 'top_n must be an integer between 1 and 100'
            }), 400
        
        # Ensure engine is loaded
        if not models_status['loaded'] or recommendation_engine is None:
            return jsonify({
                'success': False,
                'error': 'Recommendation engine not ready',
                'details': models_status['error'],
                'loading': models_status['loading']
            }), 503
        
        logger.info(f"Generating {top_n} recommendations for user: {user_id}")
        
        # Check if user exists in the system
        user_exists = False
        if recommendation_engine and hasattr(recommendation_engine, 'user_to_idx'):
            user_exists = user_id in recommendation_engine.user_to_idx
        logger.info(f"User {user_id} exists in system: {user_exists}")
        
        # Get recommendations using the engine's method
        # For invalid users, this will automatically return popular products
        recommendations_df = recommendation_engine.recommend_products(
            user_id=user_id,
            top_n=top_n,
            ignore_rated_product_ids=None if not exclude_purchased else set()
        )
        
        # Check if recommendations were found
        if recommendations_df is None or recommendations_df.empty:
            return jsonify({
                'success': True,
                'user_id': str(user_id),
                'top_n': top_n,
                'valid_user': user_exists,
                'recommendations': [],
                'message': 'Invalid User ID. No recommendations available.' if not user_exists else 'No recommendations found for this user'
            }), 200
        
        # Format recommendations for response
        recommendations = []
        for idx, row in recommendations_df.iterrows():
            rec = {
                'rank': idx + 1,
                'product_id': str(row['product_id']),
                'title': str(row['title']),
                'category': str(row['category']),
                'brand': str(row['brand']) if pd.notna(row['brand']) else 'Unknown',
                'price': float(row['price']) if pd.notna(row['price']) else 0.0,
                'predicted_rating': float(row.get('predicted_rating', 0)),
                'url': str(row.get('url', ''))
            }
            recommendations.append(rec)
        
        processing_time = (datetime.now() - start_time).total_seconds() * 1000
        
        # Prepare response message
        if not user_exists:
            message = f"⚠️ Invalid User ID. Showing top {len(recommendations)} popular products instead."
        else:
            message = f"Personalized recommendations for user {user_id}"
        
        return jsonify({
            'success': True,
            'user_id': str(user_id),
            'valid_user': user_exists,
            'top_n': top_n,
            'recommendations': recommendations,
            'total_recommendations': len(recommendations),
            'message': message,
            'processing_time_ms': round(processing_time, 2)
        }), 200
        
    except ValueError as e:
        logger.error(f"ValueError in recommendations: {str(e)}")
        return jsonify({
            'success': False,
            'error': 'Invalid user ID or parameters',
            'details': str(e)
        }), 400
    except Exception as e:
        logger.error(f"Error generating recommendations: {str(e)}", exc_info=True)
        return jsonify({
            'success': False,
            'error': 'Failed to generate recommendations',
            'details': str(e)
        }), 500


@app.route('/api/product/<product_id>', methods=['GET'])
def get_product(product_id):
    """
    Get detailed information about a specific product
    
    Response:
    {
        "success": bool,
        "product": {
            "product_id": string,
            "title": string,
            "category": string,
            "brand": string,
            "price": float,
            "mean_rating": float,
            "rating_count": int,
            "url": string
        }
    }
    """
    ensure_initialized()
    
    try:
        if not models_status['loaded'] or recommendation_engine is None:
            return jsonify({
                'success': False,
                'error': 'Recommendation engine not ready'
            }), 503
        
        # Get product from products_df
        products_df = recommendation_engine.products_df
        product = products_df[products_df['product_id'] == product_id]
        
        if product.empty:
            return jsonify({
                'success': False,
                'error': f'Product {product_id} not found'
            }), 404
        
        product_info = product.iloc[0]
        
        return jsonify({
            'success': True,
            'product': {
                'product_id': str(product_info.get('product_id', product_id)),
                'title': str(product_info.get('title', 'Unknown')),
                'category': str(product_info.get('category', 'Unknown')),
                'brand': str(product_info.get('brand', 'Unknown')),
                'price': float(product_info.get('price', 0)),
                'mean_rating': float(product_info.get('mean_rating', 0)),
                'rating_count': int(product_info.get('rating_count', 0)),
                'url': str(product_info.get('url', ''))
            }
        }), 200
    
    except Exception as e:
        logger.error(f"Error fetching product {product_id}: {str(e)}")
        return jsonify({
            'success': False,
            'error': 'Failed to fetch product details',
            'details': str(e)
        }), 500


@app.route('/api/users/sample', methods=['GET'])
def get_sample_users():
    """
    Get sample user IDs for testing
    
    Query params:
        limit: number of users to return (default: 5, max: 20)
    
    Response:
    {
        "success": bool,
        "users": ["user_id1", "user_id2", ...],
        "total_available": int
    }
    """
    ensure_initialized()
    
    try:
        limit = request.args.get('limit', 5, type=int)
        limit = min(max(limit, 1), 20)  # Clamp between 1 and 20
        
        if not models_status['loaded'] or recommendation_engine is None:
            return jsonify({
                'success': False,
                'error': 'Recommendation engine not ready'
            }), 503
        
        # Get unique user IDs from interactions
        interactions_df = recommendation_engine.interactions_df
        unique_users = interactions_df['user_id'].unique()
        
        # Get sample users (those with more interactions for better recommendations)
        user_counts = interactions_df['user_id'].value_counts()
        active_users = user_counts[user_counts >= 2].index.tolist()
        
        if not active_users:
            active_users = unique_users.tolist()
        
        # Randomly sample users each time
        if len(active_users) > limit:
            # Create a copy as list and shuffle for true randomness
            users_list = list(active_users)
            random.shuffle(users_list)
            sample = users_list[:limit]
        else:
            sample = active_users[:limit]
        
        return jsonify({
            'success': True,
            'users': [str(u) for u in sample],
            'total_available': len(unique_users),
            'active_users': len(active_users)
        }), 200
    
    except Exception as e:
        logger.error(f"Error fetching sample users: {str(e)}")
        return jsonify({
            'success': False,
            'error': 'Failed to fetch sample users',
            'details': str(e)
        }), 500


@app.route('/api/stats', methods=['GET'])
def get_stats():
    """Get system statistics and model information"""
    ensure_initialized()
    
    try:
        if not models_status['loaded'] or recommendation_engine is None:
            return jsonify({
                'success': False,
                'error': 'Recommendation engine not ready'
            }), 503
        
        stats = {
            'total_users': int(recommendation_engine.interactions_df['user_id'].nunique()),
            'total_products': int(recommendation_engine.products_df.shape[0]),
            'total_interactions': int(len(recommendation_engine.interactions_df)),
            'models_initialized_at': models_status['timestamp']
        }
        
        return jsonify({
            'success': True,
            'stats': stats
        }), 200
    
    except Exception as e:
        logger.error(f"Error fetching stats: {str(e)}")
        return jsonify({
            'success': False,
            'error': 'Failed to fetch statistics',
            'details': str(e)
        }), 500


# ==================== STATIC FILES ====================

@app.route('/', methods=['GET'])
def root():
    """Root endpoint - confirms service is running"""
    return jsonify({
        'status': 'ok',
        'message': 'AI E-commerce Recommendation API is running',
        'version': '1.0',
        'endpoints': {
            'health': '/api/health',
            'recommend': '/api/recommend (POST)',
            'product': '/api/product/<id>',
            'sample_users': '/api/users/sample',
            'stats': '/api/stats',
            'ui': '/ui'
        }
    }), 200


@app.route('/ui', methods=['GET'])
@app.route('/ui/', methods=['GET'])
def serve_index():
    """Serve the main HTML page"""
    static_dir = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), 'static')
    return send_from_directory(static_dir, 'index.html')


@app.route('/<path:path>', methods=['GET'])
def serve_static(path):
    """Serve static files (CSS, JS, etc.)"""
    return send_from_directory('../static', path)


# ==================== ERROR HANDLERS ====================

@app.errorhandler(404)
def not_found(error):
    """Handle 404 errors"""
    return jsonify({
        'success': False,
        'error': 'Endpoint not found',
        'path': request.path
    }), 404


@app.errorhandler(500)
def internal_error(error):
    """Handle 500 errors"""
    logger.error(f"Internal server error: {str(error)}", exc_info=True)
    return jsonify({
        'success': False,
        'error': 'Internal server error'
    }), 500


# ==================== STARTUP ====================

if __name__ == '__main__':
    try:
        # Start Flask app (models will load on first request)
        port = int(os.environ.get('PORT', 5000))
        debug = os.environ.get('FLASK_ENV') == 'development'
        
        logger.info(f"Starting Flask app on port {port}")
        
        # Keep the app running
        app.run(
            host='0.0.0.0',
            port=port,
            debug=False,
            threaded=True,
            use_reloader=False
        )
    except KeyboardInterrupt:
        logger.info("Server shutting down...")
    except Exception as e:
        logger.error(f"Failed to start application: {str(e)}", exc_info=True)
        import traceback
        traceback.print_exc()
        sys.exit(1)
