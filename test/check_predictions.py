import boto3
import pandas as pd
import io

# S3 config
BUCKET_NAME = "top-country-track-classifier"
KEY = "track_metadata_with_predictions.pkl"

# Create S3 client and download object
s3 = boto3.client("s3")
response = s3.get_object(Bucket=BUCKET_NAME, Key=KEY)

# Load the pickle data into a DataFrame
body = response['Body'].read()
df = pd.read_pickle(io.BytesIO(body))


print(df[["prediction","prediction_proba"]].head())

# Print the head
for row in df[["prediction","prediction_proba"]]:
    print(row)

