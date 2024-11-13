# Import required libraries
import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error, r2_score
import matplotlib.pyplot as plt
import seaborn as sns
from datetime import datetime, timedelta
import warnings
warnings.filterwarnings('ignore', category=FutureWarning)


# Set random seed for reproducibility
np.random.seed(42)

# Generate sample e-commerce dataset
def create_sample_dataset(n_records=1000):
    """
    Create a sample e-commerce dataset with:
    - Customer demographics
    - Purchase history
    - Product categories
    - Time-based features
    """
    # Create date range
    start_date = datetime(2023, 1, 1)
    dates = [start_date + timedelta(days=x) for x in range(n_records)]
    
    # Generate customer IDs and demographics
    customer_ids = np.random.randint(1000, 9999, n_records)
    # Changed to float64 to allow NaN values
    ages = np.random.normal(35, 12, n_records).astype(np.float64)
    ages = np.clip(ages, 18, 80)
    
    # Generate purchase data
    purchase_amounts = np.random.lognormal(4, 0.5, n_records)
    categories = ['Electronics', 'Clothing', 'Books', 'Home', 'Food']
    product_categories = np.random.choice(categories, n_records)
    
    # Generate behavioral data
    time_on_site = np.random.exponential(5, n_records)
    items_viewed = np.random.poisson(10, n_records)
    previous_purchases = np.random.poisson(5, n_records)
    
    # Create customer segments
    segments = np.random.choice(['New', 'Regular', 'VIP'], n_records, 
                              p=[0.3, 0.5, 0.2])
    
    # Create DataFrame first
    df = pd.DataFrame({
        'date': dates,
        'customer_id': customer_ids,
        'age': ages,
        'purchase_amount': purchase_amounts,
        'product_category': product_categories,
        'time_on_site': time_on_site,
        'items_viewed': items_viewed,
        'previous_purchases': previous_purchases,
        'customer_segment': segments
    })
    
    # Introduce missing values after DataFrame creation
    df.loc[np.random.choice(len(df), 50), 'age'] = np.nan
    df.loc[np.random.choice(len(df), 30), 'time_on_site'] = np.nan
    
    return df

# Create and save dataset
df = create_sample_dataset(1000)
df.to_csv('ecommerce_data.csv', index=False)

"""
Collaboration Exercise: E-commerce Purchase Prediction

Team Task: Build a model to predict purchase amounts based on customer data.
Each team member will work on different aspects of the analysis.

Instructions:
1. Fork this notebook to your feature branch
2. Complete your assigned tasks
3. Commit changes with meaningful messages
4. Create pull request for review
5. Address any merge conflicts that arise

Tips:
- Run all cells before starting work
- Clear outputs before committing
- Document your code
- Write unit tests for functions
"""

# Task 1: Data Preprocessing
# Team Member 1: Complete this section
def preprocess_data(df):
    """Preprocess the raw e-commerce data."""
    # Create a copy to avoid warnings
    df = df.copy()
    
    # Convert date to datetime
    df['date'] = pd.to_datetime(df['date'])
    
    # Handle missing values without inplace
    df['age'] = df['age'].fillna(df['age'].mean())
    df['time_on_site'] = df['time_on_site'].fillna(df['time_on_site'].median())
    
    # Create time-based features
    df['day_of_week'] = df['date'].dt.dayofweek
    df['month'] = df['date'].dt.month
    df['is_weekend'] = df['date'].dt.dayofweek.isin([5, 6]).astype(int)
    
    # Encode categorical variables
    df = pd.get_dummies(df, columns=['product_category', 'customer_segment'], drop_first=True)
    
    return df

# Task 2: Feature Engineering
# Team Member 2: Complete this section
def engineer_features(df):
    """Create new features for the model."""
    df = df.copy()
    
    # Customer behavior metrics
    df['items_per_minute'] = df['items_viewed'] / df['time_on_site']
    df['avg_time_per_item'] = df['time_on_site'] / df['items_viewed']
    
    # Customer value metrics
    df['purchase_per_previous'] = df['purchase_amount'] / (df['previous_purchases'] + 1)
    
    # Interaction terms
    df['age_purchase_interaction'] = df['age'] * df['previous_purchases']
    
    # Time-based features
    df['month_sin'] = np.sin(2 * np.pi * df['month']/12)
    df['month_cos'] = np.cos(2 * np.pi * df['month']/12)
    
    return df

# Task 3: Model Development
# Team Member 3: Complete this section
def train_model(X, y):
    """Train the prediction model."""
    # Split data
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    
    # Scale features
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    
    # Convert to DataFrame to preserve feature names
    X_train_scaled = pd.DataFrame(X_train_scaled, columns=X_train.columns)
    X_test_scaled = pd.DataFrame(X_test_scaled, columns=X_test.columns)
    
    # Train model
    model = RandomForestRegressor(n_estimators=100, random_state=42)
    model.fit(X_train_scaled, y_train)
    
    # Calculate score
    test_score = model.score(X_test_scaled, y_test)
    
    return model, test_score, X_train.columns  # Return feature names


# Task 4: Model Evaluation
# Team Member 4: Complete this section
def evaluate_model(model, X_test, y_test, feature_names):
    """Evaluate model performance."""
    # Make predictions
    y_pred = model.predict(X_test)
    
    # Calculate metrics
    mse = mean_squared_error(y_test, y_pred)
    r2 = r2_score(y_test, y_pred)
    
    # Feature importance
    feature_importance = pd.DataFrame({
        'feature': feature_names,
        'importance': model.feature_importances_
    }).sort_values('importance', ascending=False)
    
    # Visualize results
    plt.figure(figsize=(10, 6))
    plt.scatter(y_test, y_pred, alpha=0.5)
    plt.plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], 'r--', lw=2)
    plt.xlabel('Actual Purchase Amount')
    plt.ylabel('Predicted Purchase Amount')
    plt.title('Model Predictions vs Actual Values')
    plt.tight_layout()
    
    return {
        'mse': mse,
        'r2': r2,
        'feature_importance': feature_importance
    }

"""
Integration Test Section:
Run this section after completing individual tasks to verify everything works together.
"""

def main():
    # Load data
    df = pd.read_csv('ecommerce_data.csv')
    
    # Preprocess
    df_processed = preprocess_data(df)
    
    # Engineer features
    df_featured = engineer_features(df_processed)
    
    # Prepare for modeling
    X = df_featured.drop(['purchase_amount', 'date', 'customer_id'], axis=1)
    y = df_featured['purchase_amount']
    
    # Train model
    model, test_score, feature_names = train_model(X, y)  # Get feature names
    
    # Get X_test and y_test for evaluation
    _, X_test, _, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    
    # Scale X_test
    scaler = StandardScaler()
    X_test_scaled = pd.DataFrame(scaler.fit_transform(X_test), columns=X_test.columns)
    
    # Evaluate
    metrics = evaluate_model(model, X_test_scaled, y_test, feature_names)
    
    return metrics

# Additional helper functions for testing
def test_preprocessing(df):
    """Test preprocessing function"""
    df_processed = preprocess_data(df)
    assert df_processed.isnull().sum().sum() == 0, "Missing values remain"
    assert pd.api.types.is_datetime64_any_dtype(df_processed['date']), "Date not converted"
    return "Preprocessing tests passed"

def test_feature_engineering(df):
    """Test feature engineering function"""
    df_featured = engineer_features(df)
    assert len(df_featured.columns) > len(df.columns), "No new features created"
    return "Feature engineering tests passed"

def test_model_training(X, y):
    """Test model training function"""
    model, score = train_model(X, y)
    assert isinstance(score, float), "Invalid test score"
    assert score > 0, "Invalid model performance"
    return "Model training tests passed"

"""
Example Git Workflow:

# Create feature branch
git checkout -b feature/preprocessing

# Make changes to preprocessing function
# Test changes
test_preprocessing(df)

# Commit changes
git add .
git commit -m "Add preprocessing function with missing value handling"

# Push changes
git push origin feature/preprocessing

# Create pull request
# Review code
# Merge after approval
"""

if __name__ == "__main__":
    print("Starting e-commerce analysis...")
    metrics = main()
    print("Analysis complete.")
    print(f"Model performance: {metrics}")