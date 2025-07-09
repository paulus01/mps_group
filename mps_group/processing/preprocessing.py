"""
Iris Dataset Preprocessing for SageMaker
"""

import pandas as pd
import os


def main():
    """Process the Iris dataset for ML training"""
    
    # Paths
    input_path = "/opt/ml/processing/input/data/iris.csv"
    output_path = "/opt/ml/processing/output/"
    
    print("Starting Iris dataset preprocessing...")
    
    # Load data
    try:
        df = pd.read_csv(input_path, header=None)
        print(f"Data loaded: {df.shape}")
    except FileNotFoundError:
        raise FileNotFoundError(f"Input file not found: {input_path}")
    
    # Apply column names
    df.columns = ["sepal_length", "sepal_width", "petal_length", "petal_width", "class"]
    print(f"Columns: {list(df.columns)}")
    
    # Basic feature engineering
    df["sepal_area"] = df["sepal_length"] * df["sepal_width"]
    df["petal_area"] = df["petal_length"] * df["petal_width"]
    print("Created area features")
    
    # Create numeric target
    df["target"] = df["class"].astype("category").cat.codes
    print(f"Target classes: {df['class'].unique()}")
    print(f"Target values: {df['target'].unique()}")
    
    # Save processed data
    os.makedirs(output_path, exist_ok=True)
    output_file = os.path.join(output_path, "iris_cleaned.csv")
    df.to_csv(output_file, index=False)
    
    print(f"Processing completed. Output: {output_file}")
    print(f"Final dataset shape: {df.shape}")


if __name__ == "__main__":
    main()
