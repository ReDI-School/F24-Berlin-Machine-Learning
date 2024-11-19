import pandas as pd
import numpy as np
from sklearn.model_selection import StratifiedKFold, train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import OneHotEncoder
from category_encoders import TargetEncoder
from sklearn.metrics import f1_score, classification_report
from skopt import BayesSearchCV
from scipy.stats import zscore
from tqdm import tqdm

# Load datasets
def load_data(train_path, prediction_path):
    train_data = pd.read_csv(train_path, sep=';')
    prediction_data = pd.read_csv(prediction_path, sep=';')
    return train_data, prediction_data

# Data Preprocessing Functions
def fix_data_manipulations(data):
    """
    Apply various data cleaning and transformation steps.
    """
    # Ensure that 'repair_cost' values are positive
    if 'repair_cost' in data.columns:
        data['repair_cost'] = data['repair_cost'].abs()

    #  Remove extraneous spaces   
    data = data.apply(lambda x: x.str.strip() if x.dtype == "object" else x)
    # if 'issue' in data.columns:
    #    data['issue'] = data['issue'].str.strip().apply(lambda x: ' '.join(x.split()))

    # Handle unrealistic 'runned_miles' values
    if 'runned_miles' in data.columns:
        data['runned_miles'] = data['runned_miles'].replace(9999999, np.nan).fillna(data['runned_miles'].median())

    # Fix 'Fuel_type' column
    if 'Fuel_type' in data.columns:
        data['Fuel_type'] = data['Fuel_type'].replace('still_Diesel_but_you_found_an_easteregg', 'Diesel')

    return data

def do_encoding(X_train, X_test, target_column):
    """
    Apply target encoding for categorical features with more than 15 categories
    and one-hot encoding for features with 15 or fewer categories.
    """
    categorical_columns = X_train.select_dtypes(include=['object']).columns
    onehot_encoder = OneHotEncoder(drop='first', sparse_output=False, handle_unknown="ignore")
    target_encoder = TargetEncoder()

    for col in categorical_columns:
        num_categories = X_train[col].nunique()

        if num_categories > 10:
            # Apply target encoding for columns with more than 15 categories
            X_train[col] = target_encoder.fit_transform(X_train[col], X_train[target_column])
            X_test[col] = target_encoder.transform(X_test[col])
        else:
            # Apply one-hot encoding for columns with 15 or fewer categories
            onehot_encoder.fit(X_train[[col]])
            X_train_encoded = onehot_encoder.transform(X_train[[col]])
            X_test_encoded = onehot_encoder.transform(X_test[[col]])

            # Convert to DataFrame and add to the original DataFrame
            X_train_encoded_df = pd.DataFrame(X_train_encoded, columns=onehot_encoder.get_feature_names_out([col]))
            X_test_encoded_df = pd.DataFrame(X_test_encoded, columns=onehot_encoder.get_feature_names_out([col]))

            # Remove original column and add encoded columns
            X_train = X_train.drop(col, axis=1).join(X_train_encoded_df)
            X_test = X_test.drop(col, axis=1).join(X_test_encoded_df)

    return X_train, X_test

# Missing Data Handling Function
def handle_missing_data(train_df, test_df=None, activate_imputation=True):
    """
    Handles missing data using imputation or dropping rows with missing values.
    """
    if activate_imputation:
        for column in train_df.columns:
            if train_df[column].dtype in [np.float64, np.int64]:
                mean_value = train_df[column].median()
                train_df[column] = train_df[column].fillna(mean_value)
                if test_df is not None:
                    test_df[column] = test_df[column].fillna(mean_value)
                elif train_df[column].dtype == 'object':
                    most_common_category = train_df[column].mode()[0]  # Get the most frequent category
                    train_df[column] = train_df[column].fillna(most_common_category)
                    if test_df is not None:
                        test_df[column] = test_df[column].fillna(most_common_category)
    else:
        train_df = train_df.dropna(axis=0)
        if test_df is not None:
            test_df = test_df.dropna(axis=0)

    return train_df, test_df if test_df is not None else train_df

# Outlier Detection and Removal
def filter_outliers(X_train, X_test, y_train, y_test):
    """
    Detects and removes outliers based on z-scores.
    """

    numeric_cols_train = X_train.select_dtypes(include=[np.number])
    numeric_cols_test = X_test.select_dtypes(include=[np.number])

    z_scores_train = np.abs(zscore(numeric_cols_train))
    z_scores_test = np.abs(zscore(numeric_cols_test))

    X_train = X_train[(z_scores_train < 5).all(axis=1)]
    y_train = y_train.loc[X_train.index]

    X_test = X_test[(z_scores_test < 5).all(axis=1)]
    y_test = y_test.loc[X_test.index]

    return X_train, X_test, y_train, y_test

# Model Training with Bayesian Optimization
def train_model(X_train, y_train, n_iterations=10):
    """
    Train a RandomForestClassifier using Bayesian optimization for hyperparameter tuning,
    with steps to reduce overfitting.
    """
    

    rf = RandomForestClassifier(class_weight='balanced')

    # Initialize progress bar
    progress_bar = tqdm(total=n_iterations, desc="BayesSearchCV Progress")

    # Callback function to update the progress bar
    def progress_callback(res):
        progress_bar.update(1)

    param_space = {
        'bootstrap': [True],
        'max_depth': (1, 10),  # Limit max depth to reduce overfitting
        'n_estimators': (5, 100),  # Reduce upper limit for simpler models
        'min_samples_split': (15, 100),  # Start higher for better generalization
        #'min_samples_leaf': (0.001, 0.01),  # Avoid overly specific splits
       
    }

    # Set up BayesSearchCV
    opt = BayesSearchCV(
        rf,
        param_space,
        n_iter=n_iterations,
        n_jobs=-1,
        verbose=2,
        random_state=42
    )
    opt.fit(X_train, y_train, callback=progress_callback)

    # Close the progress bar when done
    progress_bar.close()

    # Print the best model
    print(f"Best Estimator: {opt.best_estimator_}")
    print(f"Best Parameters: {opt.best_params_}")
    return opt
     

# Model Evaluation
def evaluate_model(clf, X_test, y_test):
    """
    Evaluates the model using F1 score.
    """
    y_pred = clf.predict(X_test)
    f1 = f1_score(y_test, y_pred, average='macro')
    print("Macro F1 Score:", f1)
    #print(classification_report(y_test, y_pred))

# Main execution pipeline
def main():

    # TODO nfold cross validation
    # a lot of chatgpt is used for this code. There might be mistakes Timo did not see, but score is okay.

    # Step 1: Load data
    train_data, prediction_data = load_data("PreProject/train_labeled.csv", "PreProject/prediction_labeled.csv")

    # Step 2. Separate features and target label
    X_train = train_data
    y_train = train_data['Label']  # Store the target labels separately

    # Step 3. Split data: Create training and testing datasets
    X_train, X_test, y_train, y_test = train_test_split(X_train, y_train, test_size=0.2, random_state=42)

    # Step 4: Remove outliers
    X_train, X_test, y_train, y_test = filter_outliers(X_train, X_test, y_train, y_test)

    # Step 5. Apply encoding function
    X_train, X_test = do_encoding(X_train, X_test, target_column='Label')

    # Step 6. Drop 'Label' from X_train and X_test after encoding
    X_train = X_train.drop(columns=['Label'])
    X_test = X_test.drop(columns=['Label'])

    # Step 7: Handle missing data
    X_train, X_test = handle_missing_data(X_train, X_test)

    # Step 8: Train model
    n_iterations = 20
    clf = train_model(X_train, y_train, n_iterations)
    
    # Step 9: Evaluate model
    print("Train:")
    evaluate_model(clf,X_train, y_train,)
    print("Test:")
    evaluate_model(clf, X_test, y_test)

if __name__ == "__main__":
    main()
