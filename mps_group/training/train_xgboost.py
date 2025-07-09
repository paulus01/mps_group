import pandas as pd
import xgboost as xgb
import os

if __name__ == "__main__":
    # Load the preprocessed data
    df = pd.read_csv("/opt/ml/input/data/train/iris_cleaned.csv")
    print("DEBUG: Columns in dataframe:", df.columns.tolist())
    print("DEBUG: First few rows:\n", df.head())
    
    # Prepare features and target
    X = df.drop(["target", "class"], axis=1)
    y = df["target"]
    
    # Train XGBoost model
    model = xgb.XGBClassifier(
        objective='multi:softprob',
        num_class=3,
        random_state=42
    )
    model.fit(X, y)
    
    # Save model in binary format for XGBoost built-in inference
    model.save_model("/opt/ml/model/model.tar.gz")
    print("Model saved successfully at /opt/ml/model/model.tar.gz")