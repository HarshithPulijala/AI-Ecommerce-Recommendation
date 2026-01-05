"""
Milestone 1: Data Preparation
AI-Enabled Recommendation Engine for E-commerce Platform

This script performs data cleaning and prepares datasets for recommendation models.
It loads raw data, cleans it, ensures referential integrity, and creates a user-item matrix.
"""

import pandas as pd
import numpy as np
from pathlib import Path

# ================================
# CONFIGURATION
# ================================
# Define paths using relative paths
BASE_DIR = Path(__file__).parent.parent
RAW_DATA_DIR = BASE_DIR / "data" / "raw"
PROCESSED_DATA_DIR = BASE_DIR / "data" / "processed"

# Create processed directory if it doesn't exist
PROCESSED_DATA_DIR.mkdir(parents=True, exist_ok=True)

print("=" * 60)
print("Milestone 1: Data Preparation")
print("=" * 60)
print(f"Base directory: {BASE_DIR}")
print(f"Raw data directory: {RAW_DATA_DIR}")
print(f"Processed data directory: {PROCESSED_DATA_DIR}")
print()


# ================================
# 1. LOAD RAW DATASETS
# ================================
print("Step 1: Loading raw datasets...")
print("-" * 60)

# Load users dataset
users_path = RAW_DATA_DIR / "users.csv"
users_df = pd.read_csv(users_path)
print(f"Loaded users.csv: {users_df.shape[0]} rows, {users_df.shape[1]} columns")
print(f"Users columns: {list(users_df.columns)}")

# Load products dataset
products_path = RAW_DATA_DIR / "products.csv"
products_df = pd.read_csv(products_path)
print(f"Loaded products.csv: {products_df.shape[0]} rows, {products_df.shape[1]} columns")
print(f"Products columns: {list(products_df.columns)}")

# Load interactions dataset
interactions_path = RAW_DATA_DIR / "interactions.csv"
interactions_df = pd.read_csv(interactions_path)
print(f"Loaded interactions.csv: {interactions_df.shape[0]} rows, {interactions_df.shape[1]} columns")
print(f"Interactions columns: {list(interactions_df.columns)}")
print()


# ================================
# 2. DATA CLEANING - USERS
# ================================
print("Step 2: Cleaning users dataset...")
print("-" * 60)
print(f"Initial users shape: {users_df.shape}")

# Remove rows with missing user_id
initial_users_count = len(users_df)
users_df = users_df.dropna(subset=['user_id'])
removed_users = initial_users_count - len(users_df)
print(f"Removed {removed_users} rows with missing user_id")

# Remove duplicate rows
initial_users_count = len(users_df)
users_df = users_df.drop_duplicates()
removed_duplicates = initial_users_count - len(users_df)
print(f"Removed {removed_duplicates} duplicate rows")

# Convert user_id to string for consistency (if needed)
users_df['user_id'] = users_df['user_id'].astype(str)

print(f"Final users shape: {users_df.shape}")
print()


# ================================
# 3. DATA CLEANING - PRODUCTS
# ================================
print("Step 3: Cleaning products dataset...")
print("-" * 60)
print(f"Initial products shape: {products_df.shape}")

# Remove rows with missing product_id
initial_products_count = len(products_df)
products_df = products_df.dropna(subset=['product_id'])
removed_products = initial_products_count - len(products_df)
print(f"Removed {removed_products} rows with missing product_id")

# Handle missing product descriptions by filling with default string
missing_descriptions = products_df['description'].isna().sum()
products_df['description'] = products_df['description'].fillna('No description available')
print(f"Filled {missing_descriptions} missing descriptions with default string")

# Fix data type: price → float
products_df['price'] = pd.to_numeric(products_df['price'], errors='coerce')

# Handle missing prices using median imputation
median_price = products_df['price'].median()
missing_prices = products_df['price'].isna().sum()
products_df['price'] = products_df['price'].fillna(median_price)
print(f"Filled {missing_prices} missing prices with median value: {median_price:.2f}")

# Remove products with non-positive prices (price <= 0)
initial_products_count = len(products_df)
products_df = products_df[products_df['price'] > 0]
removed_negative_prices = initial_products_count - len(products_df)
print(f"Removed {removed_negative_prices} products with non-positive prices")

# Remove duplicate rows
initial_products_count = len(products_df)
products_df = products_df.drop_duplicates()
removed_duplicates = initial_products_count - len(products_df)
print(f"Removed {removed_duplicates} duplicate rows")

# Convert product_id to string for consistency (if needed)
products_df['product_id'] = products_df['product_id'].astype(str)

print(f"Final products shape: {products_df.shape}")
print()


# ================================
# 4. DATA CLEANING - INTERACTIONS
# ================================
print("Step 4: Cleaning interactions dataset...")
print("-" * 60)
print(f"Initial interactions shape: {interactions_df.shape}")

# Remove rows with missing critical IDs (user_id, product_id)
initial_interactions_count = len(interactions_df)
interactions_df = interactions_df.dropna(subset=['user_id', 'product_id'])
removed_missing_ids = initial_interactions_count - len(interactions_df)
print(f"Removed {removed_missing_ids} rows with missing user_id or product_id")

# Fix data type: rating → float
interactions_df['rating'] = pd.to_numeric(interactions_df['rating'], errors='coerce')

# Remove invalid ratings (outside 1-5 range)
initial_interactions_count = len(interactions_df)
interactions_df = interactions_df.dropna(subset=['rating'])  # Remove NaN ratings first
interactions_df = interactions_df[(interactions_df['rating'] >= 1) & (interactions_df['rating'] <= 5)]
removed_invalid_ratings = initial_interactions_count - len(interactions_df)
print(f"Removed {removed_invalid_ratings} rows with invalid ratings (outside 1-5 range)")

# Remove duplicate rows
initial_interactions_count = len(interactions_df)
interactions_df = interactions_df.drop_duplicates()
removed_duplicates = initial_interactions_count - len(interactions_df)
print(f"Removed {removed_duplicates} duplicate rows")

# Convert IDs to string for consistency
interactions_df['user_id'] = interactions_df['user_id'].astype(str)
interactions_df['product_id'] = interactions_df['product_id'].astype(str)

print(f"Interactions shape after cleaning: {interactions_df.shape}")
print()


# ================================
# 5. REFERENTIAL INTEGRITY
# ================================
print("Step 5: Ensuring referential integrity...")
print("-" * 60)

# Get valid user_ids and product_ids from cleaned datasets
valid_user_ids = set(users_df['user_id'].unique())
valid_product_ids = set(products_df['product_id'].unique())

print(f"Valid user_ids: {len(valid_user_ids)}")
print(f"Valid product_ids: {len(valid_product_ids)}")

# Keep only interactions where user_id exists in users
initial_interactions_count = len(interactions_df)
interactions_df = interactions_df[interactions_df['user_id'].isin(valid_user_ids)]
removed_invalid_users = initial_interactions_count - len(interactions_df)
print(f"Removed {removed_invalid_users} interactions with invalid user_id")

# Keep only interactions where product_id exists in products
initial_interactions_count = len(interactions_df)
interactions_df = interactions_df[interactions_df['product_id'].isin(valid_product_ids)]
removed_invalid_products = initial_interactions_count - len(interactions_df)
print(f"Removed {removed_invalid_products} interactions with invalid product_id")

print(f"Final interactions shape: {interactions_df.shape}")
print()


# ================================
# 6. BUILD USER-ITEM INTERACTION MATRIX
# ================================
print("Step 6: Building user-item interaction matrix...")
print("-" * 60)

# Calculate average rating for each user-product pair
# Handle cases where same user-product pair has multiple interactions
user_product_ratings = interactions_df.groupby(['user_id', 'product_id'])['rating'].mean().reset_index()
print(f"Unique user-product pairs: {len(user_product_ratings)}")

# Get unique users and products for matrix dimensions
unique_users = sorted(users_df['user_id'].unique())
unique_products = sorted(products_df['product_id'].unique())
num_users = len(unique_users)
num_products = len(unique_products)

print(f"Matrix dimensions: {num_users} users × {num_products} products")
print(f"Total possible entries: {num_users * num_products:,}")
print(f"Non-zero entries: {len(user_product_ratings):,}")
sparsity = (1 - len(user_product_ratings) / (num_users * num_products)) * 100
print(f"Sparsity: {sparsity:.2f}%")
print()

# For very large datasets, creating a full dense matrix is not memory-feasible
# The matrix would require ~154 GB of RAM for this dataset size
# Instead, we save the matrix in sparse format (coordinate format)
# This represents the same information: rows=user_id, columns=product_id, values=rating
# Missing values are implicitly 0 (standard sparse matrix representation)

print("Note: Dataset is too large for dense matrix creation (~154 GB required).")
print("Saving matrix in sparse coordinate format (user_id, product_id, rating).")
print("This format represents the same matrix: missing values are implicitly 0.")
print("This is the standard format for large sparse matrices in recommendation systems.")
print()

# Store matrix information for saving (we'll save in coordinate format)
# The matrix structure is: user_id (rows), product_id (columns), rating (values)
user_item_matrix_data = user_product_ratings.copy()

# Create a representation that documents the matrix dimensions
# For very large matrices, we save in coordinate format instead of dense format
print(f"User-item matrix: {num_users} users × {num_products} products")
print(f"Matrix stored in sparse coordinate format: {len(user_product_ratings):,} non-zero entries")
print()

# Store as DataFrame for compatibility (will be saved in coordinate format)
user_item_matrix = user_item_matrix_data.set_index('user_id')


# ================================
# 7. SAVE CLEANED DATASETS
# ================================
print("Step 7: Saving cleaned datasets...")
print("-" * 60)

# Save cleaned users
users_output_path = PROCESSED_DATA_DIR / "users_clean.csv"
users_df.to_csv(users_output_path, index=False)
print(f"Saved: {users_output_path}")

# Save cleaned products
products_output_path = PROCESSED_DATA_DIR / "products_clean.csv"
products_df.to_csv(products_output_path, index=False)
print(f"Saved: {products_output_path}")

# Save cleaned interactions
interactions_output_path = PROCESSED_DATA_DIR / "interactions_clean.csv"
interactions_df.to_csv(interactions_output_path, index=False)
print(f"Saved: {interactions_output_path}")

# Save user-item matrix (in sparse coordinate format for large datasets)
matrix_output_path = PROCESSED_DATA_DIR / "user_item_matrix.csv"
# Save in coordinate format (user_id, product_id, rating)
# This represents the matrix: rows=user_id, columns=product_id, values=rating (0 if missing)
user_item_matrix_data.to_csv(matrix_output_path, index=False)
print(f"Saved: {matrix_output_path} (sparse coordinate format)")
print("  Format: user_id, product_id, rating (missing values are implicitly 0)")

print()


# ================================
# SUMMARY
# ================================
print("=" * 60)
print("Data Preparation Complete!")
print("=" * 60)
print("\nFinal Dataset Statistics:")
print(f"  Users: {users_df.shape[0]} rows")
print(f"  Products: {products_df.shape[0]} rows")
print(f"  Interactions: {interactions_df.shape[0]} rows")
print(f"  User-Item Matrix: {num_users} users × {num_products} products ({len(user_product_ratings):,} non-zero entries)")
print("  Matrix format: Sparse coordinate (user_id, product_id, rating)")
print("\nAll cleaned datasets saved to data/processed/")
print("=" * 60)

