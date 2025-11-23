import boto3
import pandas as pd
import numpy as np

# This is when I was trouble shooting an error about the size of the embedding

#list variables

NEW_BUCKET = "top-country-track-classifier"
OLD_BUCKET = "ucwdc-country-classifier"

s3 = boto3.client('s3')
response = s3.get_object(Bucket=OLD_BUCKET, Key='combined_tables_with_features.csv')

df = pd.read_csv(response["Body"])

# If the relevant column is called 'Feature' or 'features'
feature_col = "Feature"  # or whatever your column is named

# Check shape of one example
print(f"combined_tables_with_features.csv feature shape: {np.asarray(df[feature_col].iloc[0]).shape}")

# Check value counts for feature lengths
lengths = df[feature_col].apply(lambda x: np.asarray(x).size)
print(lengths.value_counts().head())