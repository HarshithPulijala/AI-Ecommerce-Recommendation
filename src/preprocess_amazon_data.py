"""
Preprocessing Script for Amazon Reviews 2023 Dataset

This script converts Amazon Reviews 2023 JSONL files to CSV format
matching the schema required for data_preparation.py

Usage:
    python src/preprocess_amazon_data.py

Before running:
    1. Download review file: review_categories/[Category].jsonl.gz
    2. Download metadata file: meta_categories/meta_[Category].jsonl.gz
    3. Extract .gz files to get .jsonl files
    4. Place extracted .jsonl files in data/raw/
    5. Update CATEGORY_NAME below to match your downloaded category
"""

import pandas as pd
import json
import gzip
from pathlib import Path

# ================================
# CONFIGURATION
# ================================
BASE_DIR = Path(__file__).parent.parent
RAW_DATA_DIR = BASE_DIR / "data" / "raw"

# Update this to match your downloaded category
CATEGORY_NAME = "Appliances"  # Change to: "Amazon_Fashion", "Baby_Products", etc.

# Input file paths (JSONL format, can be .gz or .jsonl)
REVIEW_FILE = RAW_DATA_DIR / f"{CATEGORY_NAME}.jsonl"  # Or .jsonl.gz
META_FILE = RAW_DATA_DIR / f"meta_{CATEGORY_NAME}.jsonl"  # Or .jsonl.gz

# Output file paths (CSV format)
OUTPUT_USERS = RAW_DATA_DIR / "users.csv"
OUTPUT_PRODUCTS = RAW_DATA_DIR / "products.csv"
OUTPUT_INTERACTIONS = RAW_DATA_DIR / "interactions.csv"

print("=" * 60)
print("Amazon Reviews 2023 - Data Preprocessing")
print("=" * 60)
print(f"Category: {CATEGORY_NAME}")
print(f"Review file: {REVIEW_FILE}")
print(f"Metadata file: {META_FILE}")
print()


# ================================
# HELPER FUNCTION: Load JSONL
# ================================
def load_jsonl(file_path):
    """
    Load JSONL file (supports both .jsonl and .jsonl.gz)
    
    Args:
        file_path: Path to JSONL file
        
    Returns:
        List of dictionaries
    """
    data = []
    
    # Check if file is gzipped
    if str(file_path).endswith('.gz'):
        open_func = gzip.open
        mode = 'rt'
    else:
        open_func = open
        mode = 'r'
    
    print(f"Loading {file_path.name}...")
    with open_func(file_path, mode, encoding='utf-8') as f:
        for line_num, line in enumerate(f, 1):
            try:
                data.append(json.loads(line.strip()))
            except json.JSONDecodeError as e:
                print(f"Warning: Skipped line {line_num} due to JSON error: {e}")
                continue
    
    print(f"Loaded {len(data)} records")
    return data


# ================================
# 1. LOAD REVIEW DATA
# ================================
print("Step 1: Loading review data...")
print("-" * 60)

if not REVIEW_FILE.exists():
    # Try .gz version
    REVIEW_FILE = RAW_DATA_DIR / f"{CATEGORY_NAME}.jsonl.gz"
    if not REVIEW_FILE.exists():
        raise FileNotFoundError(
            f"Review file not found. Expected: {RAW_DATA_DIR / CATEGORY_NAME}.jsonl or .jsonl.gz\n"
            f"Please download and extract the review file first."
        )

reviews_data = load_jsonl(REVIEW_FILE)

# Extract reviews to DataFrame
reviews_list = []
for review in reviews_data:
    reviews_list.append({
        'user_id': review.get('user_id', ''),
        'product_id': review.get('parent_asin', review.get('asin', '')),  # Use parent_asin if available
        'rating': review.get('rating', None),
        'timestamp': review.get('timestamp', None)
    })

reviews_df = pd.DataFrame(reviews_list)
print(f"Extracted {len(reviews_df)} reviews")
print()


# ================================
# 2. LOAD PRODUCT METADATA
# ================================
print("Step 2: Loading product metadata...")
print("-" * 60)

if not META_FILE.exists():
    # Try .gz version
    META_FILE = RAW_DATA_DIR / f"meta_{CATEGORY_NAME}.jsonl.gz"
    if not META_FILE.exists():
        raise FileNotFoundError(
            f"Metadata file not found. Expected: {RAW_DATA_DIR / f'meta_{CATEGORY_NAME}'}.jsonl or .jsonl.gz\n"
            f"Please download and extract the metadata file first."
        )

meta_data = load_jsonl(META_FILE)

# Extract product metadata to DataFrame
products_list = []
for item in meta_data:
    # Extract description (convert list to string)
    description = item.get('description', [])
    if isinstance(description, list):
        description = ' '.join(str(d) for d in description if d)
    elif not description:
        description = ''
    
    # Extract brand from details
    details = item.get('details', {})
    brand = details.get('Brand', details.get('brand', 'Unknown'))
    
    # Extract category (use main_category or first category)
    category = item.get('main_category', '')
    if not category and item.get('categories'):
        category = item.get('categories', [{}])[0] if item.get('categories') else ''
        if isinstance(category, dict):
            category = category.get('name', '')
    
    products_list.append({
        'product_id': item.get('parent_asin', ''),
        'category': category,
        'brand': brand,
        'price': item.get('price', None),
        'description': description
    })

products_df = pd.DataFrame(products_list)
print(f"Extracted {len(products_df)} products")
print()


# ================================
# 3. CREATE USERS DATAFRAME
# ================================
print("Step 3: Creating users dataset...")
print("-" * 60)

# Extract unique users from reviews
# Note: Amazon dataset doesn't have user metadata, so we create minimal user info
unique_users = reviews_df['user_id'].unique()

# Create users dataframe with minimal fields
# For real data, you might need to infer or use default values
users_list = []
for user_id in unique_users:
    users_list.append({
        'user_id': user_id,
        'age': None,  # Not available in Amazon dataset
        'gender': None,  # Not available in Amazon dataset
        'location': None  # Not available in Amazon dataset
    })

users_df = pd.DataFrame(users_list)
print(f"Created {len(users_df)} user records")
print("Note: Age, gender, and location are not available in Amazon dataset")
print("These fields will be handled by data_preparation.py (filled or dropped)")
print()


# ================================
# 4. CREATE INTERACTIONS DATAFRAME
# ================================
print("Step 4: Creating interactions dataset...")
print("-" * 60)

# Use reviews_df but rename to match expected schema
interactions_df = reviews_df.copy()

# Convert timestamp from unix to readable format (optional, keep as-is for now)
# Your data_preparation.py will handle timestamp conversion if needed

print(f"Created {len(interactions_df)} interaction records")
print()


# ================================
# 5. CLEAN PRODUCT DATA
# ================================
print("Step 5: Cleaning product data...")
print("-" * 60)

# Remove products without product_id
initial_count = len(products_df)
products_df = products_df[products_df['product_id'].notna() & (products_df['product_id'] != '')]
print(f"Removed {initial_count - len(products_df)} products without product_id")

# Remove duplicates
initial_count = len(products_df)
products_df = products_df.drop_duplicates(subset=['product_id'])
print(f"Removed {initial_count - len(products_df)} duplicate products")

print(f"Final products count: {len(products_df)}")
print()


# ================================
# 6. FILTER INTERACTIONS (Keep only valid product_ids)
# ================================
print("Step 6: Filtering interactions...")
print("-" * 60)

valid_product_ids = set(products_df['product_id'].unique())
initial_count = len(interactions_df)
interactions_df = interactions_df[interactions_df['product_id'].isin(valid_product_ids)]
print(f"Filtered to {len(interactions_df)} interactions with valid products (removed {initial_count - len(interactions_df)})")
print()


# ================================
# 7. FILTER USERS (Keep only users with interactions)
# ================================
print("Step 7: Filtering users...")
print("-" * 60)

valid_user_ids = set(interactions_df['user_id'].unique())
initial_count = len(users_df)
users_df = users_df[users_df['user_id'].isin(valid_user_ids)]
print(f"Filtered to {len(users_df)} users with interactions (removed {initial_count - len(users_df)})")
print()


# ================================
# 8. SAVE CSV FILES
# ================================
print("Step 8: Saving CSV files...")
print("-" * 60)

users_df.to_csv(OUTPUT_USERS, index=False)
print(f"Saved: {OUTPUT_USERS} ({len(users_df)} rows)")

products_df.to_csv(OUTPUT_PRODUCTS, index=False)
print(f"Saved: {OUTPUT_PRODUCTS} ({len(products_df)} rows)")

interactions_df.to_csv(OUTPUT_INTERACTIONS, index=False)
print(f"Saved: {OUTPUT_INTERACTIONS} ({len(interactions_df)} rows)")

print()


# ================================
# SUMMARY
# ================================
print("=" * 60)
print("Preprocessing Complete!")
print("=" * 60)
print(f"\nOutput files saved to: {RAW_DATA_DIR}")
print(f"  - users.csv: {len(users_df)} users")
print(f"  - products.csv: {len(products_df)} products")
print(f"  - interactions.csv: {len(interactions_df)} interactions")
print("\nNext step: Run data_preparation.py to clean and create user-item matrix")
print("=" * 60)

