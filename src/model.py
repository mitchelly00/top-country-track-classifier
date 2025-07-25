import boto3
import pandas as pd
import io
import lightgbm as lgb
import numpy as np

MODEL_S3_URL = "https://ucwdc-country-classifier.s3.us-east-1.amazonaws.com/final_models/lightgbm_model.txt"
PKL_S3_URL = "https://top-country-track-classifier.s3.us-east-1.amazonaws.com/metadata/track_metadata_combined.pkl"
BUCKET_NAME = "top-country-track-classifier"
OUTPUT_PKL_PRED_S3 = "track_metadata_with_predictions.pkl"

def download_file_from_s3_url(url):
    """Download a file given a public S3 URL."""
    import requests
    response = requests.get(url)
    if response.status_code != 200:
        raise RuntimeError(f"Failed to download from {url}")
    return response.content

def load_model():
    print("[INFO] Downloading and loading LightGBM model...")
    model_bytes = download_file_from_s3_url(MODEL_S3_URL)
    model_str = model_bytes.decode("utf-8")
    model = lgb.Booster(model_str=model_str)
    print("[INFO] Model loaded.")
    return model

def load_dataframe():
    print("[INFO] Downloading and loading DataFrame...")
    df_bytes = download_file_from_s3_url(PKL_S3_URL)
    df = pd.read_pickle(io.BytesIO(df_bytes))
    print("[INFO] DataFrame loaded with shape:", df.shape)
    return df

def prepare_input(df):
    if "combined" not in df.columns:
        raise KeyError("The column 'combined' is not in the DataFrame.")
    combined_features = df["combined"].tolist()
    X = np.vstack(combined_features)
    return X

def upload_dataframe_as_pickle_to_s3(df, bucket=BUCKET_NAME, s3_key=OUTPUT_PKL_PRED_S3):
    s3 = boto3.client('s3')
    pickle_buffer = io.BytesIO()
    df.to_pickle(pickle_buffer)
    pickle_buffer.seek(0)
    s3.put_object(Bucket=bucket, Key=s3_key, Body=pickle_buffer)
    print(f"[INFO] Uploaded DataFrame to s3://{bucket}/{s3_key}")

def main():
    model = load_model()
    df = load_dataframe()
    X = prepare_input(df)

    print("[INFO] Running predictions...")
    probs = model.predict(X)
    preds = np.argmax(probs, axis=1) if probs.ndim > 1 else (probs > 0.5).astype(int)

    df["prediction"] = preds
    df["prediction_proba"] = probs.tolist() if probs.ndim > 1 else probs

    print("[INFO] Sample predictions:")
    print(df[["prediction", "prediction_proba"]].head())

    # Optionally: save locally or upload back to S3
    upload_dataframe_as_pickle_to_s3(df)
    print("[INFO] Predictions saved to 'track_metadata_with_predictions.pkl'")

if __name__ == "__main__":
    main()
