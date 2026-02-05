# AI-Enabled Recommendation Engine for E-commerce Platform

A machine learning-based recommendation system that provides personalized product recommendations to users based on their interaction history and product metadata.

## 📋 Project Overview

This project implements a comprehensive recommendation engine for an e-commerce platform using Amazon Reviews 2023 dataset. The system processes user interactions, product metadata, and builds recommendation models to suggest relevant products to users.

## 🎯 Milestones

### ✅ Milestone 1: Data Preparation (COMPLETED)
- Data collection and cleaning
- Handling data inconsistencies
- Building user-item interaction matrix
- Dataset: Amazon Reviews 2023 - Appliances Category
  - 2.1M ratings
  - 1.7M users
  - 94K products

### ✅ Milestone 2: Model Development (COMPLETED - 18 Jan 2026)
- Implementation of hybrid recommendation algorithms (Collaborative Filtering + Content-Based)
- Model training and optimization using SVD and TF-IDF
- Performance evaluation with MAE/RMSE metrics
- Models saved and ready for deployment

### ✅ Milestone 3: Evaluation and Refinement (COMPLETED - 28 Jan 2026)
- Comprehensive model evaluation using Precision@K, Recall@K, and F1-score metrics
- Hyperparameter tuning for SVD components and hybrid model weights
- Model refinement for improved recommendation accuracy
- Testing of different recommendation scenarios
- Final tuned models with optimized performance

### ✅ Milestone 4: Web App Development & Deployment (COMPLETED - 5 Feb 2026)
- Built Flask REST API backend with 6 functional endpoints
- Created responsive web interface with HTML/CSS/JavaScript
- Integrated recommendation engine with web application
- Comprehensive test suite implemented
- Deployment configuration for multiple platforms (Render.com, PythonAnywhere, AWS)
- Production-ready with Gunicorn and Docker support

## 📁 Project Structure

```
AI -ecommerce/
├── data/
│   ├── raw/              # Raw input data (CSV files)
│   └── processed/        # Cleaned and processed data
│       ├── users_clean.csv
│       ├── products_clean.csv
│       ├── interactions_clean.csv
│       ├── user_item_matrix.csv
│       └── model_evaluation_results.csv
├── models/               # Trained model artifacts
│   ├── svd_model.pkl
│   ├── user_factors.pkl
│   ├── product_factors.pkl
│   └── ...
├── src/
│   ├── data_preparation.py      # Main data cleaning script
│   ├── model_training.py        # Model training and evaluation
│   ├── recommend.py             # Recommendation inference engine
│   └── preprocess_amazon_data.py  # JSONL to CSV converter
├── config.yaml                  # Configuration file
├── requirements.txt             # Python dependencies
├── notebooks/            # Jupyter notebooks for analysis
└── README.md
```

## 🚀 Getting Started

### Prerequisites

- Python 3.8 or higher
- Required packages (see `requirements.txt`)

### Installation

1. Clone the repository:
   ```bash
   git clone https://github.com/YOUR_USERNAME/AI-ecommerce-recommendation-engine.git
   cd AI-ecommerce-recommendation-engine
   ```

2. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```

2. Download the dataset:
   - Go to [Amazon Reviews 2023](https://amazon-reviews-2023.github.io/)
   - Download Appliances category:
     - Review file: `Appliances.jsonl.gz`
     - Metadata file: `meta_Appliances.jsonl.gz`
   - Extract and place in `data/raw/` directory

3. Run data preprocessing (if needed):
   ```bash
   python src/preprocess_amazon_data.py
   ```

4. Run data preparation:
   ```bash
   python src/data_preparation.py
   ```

5. Train the recommendation models:
   ```bash
   python src/model_training.py
   ```

## � Web Application

### Running the Web App Locally

**Windows:**
```bash
launch.bat
python -m flask run --port 5000
# Visit: http://localhost:5000
```

**macOS/Linux:**
```bash
bash launch.sh
python -m flask run --port 5000
# Visit: http://localhost:5000
```

### Web App Features
- 🎨 Modern, responsive user interface
- ⚡ Real-time API integration
- 📱 Mobile-friendly design
- 🔍 Sample user suggestions
- 📊 Product recommendations in grid layout
- ✅ Input validation and error handling

### API Endpoints
- `GET /api/health` - System health check
- `POST /api/recommend` - Get recommendations for a user
- `GET /api/product/<id>` - Product details
- `GET /api/users/sample` - Sample user IDs
- `GET /api/stats` - System statistics

### Testing the Web App
```bash
python test_webapp.py
```

### Deploying the Web App
For production deployment instructions, see [DEPLOYMENT.md](DEPLOYMENT.md)

Supported platforms:
- 🟢 **Render.com** (Recommended - Free tier available)
- 🟡 **PythonAnywhere** (Simple Python hosting)
- 🔵 **AWS EC2** (Full control, scalable)

## �🎯 Generate Recommendations

After training the models, you can generate personalized product recommendations:

### Command Line Usage
```bash
# Generate recommendations for a specific user
python src/recommend.py --user_id A123456789 --top_n 5

# Include last viewed product for better recommendations
python src/recommend.py --user_id A123456789 --last_viewed B0123456789 --top_n 10
```

### Python API Usage
```python
from src.recommend import recommend_products

# Load models and generate recommendations
recommendations = recommend_products(
    user_id="A123456789",
    last_viewed_product_id="B0123456789",  # Optional
    top_n=5
)

# Display results
for idx, row in recommendations.iterrows():
    print(f"{idx+1}. {row['title'][:50]}...")
    print(f"   Predicted Rating: {row['predicted_rating']:.2f}/5.0")
    print(f"   Price: ${row['price']:.2f}")
    print()
```

### Recommendation Types
- **Collaborative Filtering**: Based on similar users' preferences
- **Content-Based**: Based on product features and user history
- **Hybrid**: Combines both approaches (recommended)

### Cold-Start Handling
- **New Users**: Returns popular products
- **New Products**: Uses content-based similarity
- **Missing Data**: Graceful fallback to available methods

### Source
- **Dataset**: Amazon Reviews 2023
- **Category**: Appliances
- **Total Reviews**: 2,105,948
- **Users**: 1,755,732
- **Products**: 94,327

### Data Files

#### Raw Data (`data/raw/`)
- `users.csv` - User information
- `products.csv` - Product metadata (category, brand, price, description)
- `interactions.csv` - User-product interactions with ratings

#### Processed Data (`data/processed/`)
- `users_clean.csv` - Cleaned user data
- `products_clean.csv` - Cleaned product data with imputed missing values
- `interactions_clean.csv` - Cleaned interactions with valid ratings (1-5)
- `user_item_matrix.csv` - User-item interaction matrix (sparse coordinate format)

## 🔧 Data Processing

### Data Cleaning Steps

1. **Missing Value Handling**
   - Removed rows with missing critical IDs
   - Filled missing product descriptions with default string
   - Imputed missing prices using median value

2. **Data Validation**
   - Removed invalid ratings (outside 1-5 range)
   - Removed products with non-positive prices
   - Removed duplicate entries

3. **Referential Integrity**
   - Ensured all interactions reference valid users and products
   - Filtered out orphaned records

4. **Matrix Creation**
   - Built user-item interaction matrix
   - Used sparse coordinate format for memory efficiency
   - Matrix dimensions: 1,755,732 users × 94,327 products

## 📈 Features

- **Comprehensive Data Cleaning**: Handles missing values, duplicates, and data inconsistencies
- **Referential Integrity**: Ensures data consistency across all datasets
- **Memory Efficient**: Uses sparse matrix format for large datasets
- **Scalable**: Handles millions of interactions efficiently
- **Well Documented**: Clean, commented code for easy understanding

## 🛠️ Usage

### Data Preparation

```python
# Run the main data preparation script
python src/data_preparation.py
```

This script will:
1. Load raw datasets
2. Clean and validate data
3. Create user-item interaction matrix
4. Save processed files to `data/processed/`

### Output

After running the script, you'll have:
- Cleaned datasets ready for model training
- User-item interaction matrix in sparse format
- All data quality checks passed

## 📝 Data Schema

### Users
- `user_id`: Unique user identifier
- `age`: User age (if available)
- `gender`: User gender (if available)
- `location`: User location (if available)

### Products
- `product_id`: Unique product identifier
- `category`: Product category
- `brand`: Product brand
- `price`: Product price (USD)
- `description`: Product description

### Interactions
- `user_id`: User identifier
- `product_id`: Product identifier
- `rating`: Rating (1-5)
- `timestamp`: Interaction timestamp

## 🔬 Technical Details

- **Language**: Python 3.8+
- **Libraries**: pandas, numpy
- **Data Format**: CSV files
- **Matrix Format**: Sparse coordinate format (user_id, product_id, rating)
- **Sparsity**: ~99.9987% (standard for recommendation systems)

## 📊 Dataset Statistics

- **Total Interactions**: 2,105,948
- **Unique Users**: 1,755,732
- **Unique Products**: 94,327
- **Matrix Non-zero Entries**: 2,103,990
- **Average Ratings per User**: ~1.2
- **Average Ratings per Product**: ~22.3

 

 

 

## 🙏 Acknowledgments

- Amazon Reviews 2023 Dataset by McAuley Lab
- Dataset source: https://amazon-reviews-2023.github.io/

 

---

**Note**: Large data files (CSV files >100MB) are not included in the repository due to GitHub file size limits. Please download the dataset from the official source and place it in the `data/raw/` directory.

