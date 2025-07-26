import boto3
import pandas as pd
import numpy as np
import io

def inspect_feature_shapes(bucket_name, prefix=""):
    s3 = boto3.client("s3")

    paginator = s3.get_paginator("list_objects_v2")
    pages = paginator.paginate(Bucket=bucket_name, Prefix=prefix)

    print(f"Scanning bucket: {bucket_name}\n")

    for page in pages:
        for obj in page.get("Contents", []):
            key = obj["Key"]
            if not (key.endswith(".pkl") or key.endswith(".csv")):
                continue

            print(f"🔍 {key}")
            try:
                response = s3.get_object(Bucket=bucket_name, Key=key)
                body = response["Body"].read()

                if key.endswith(".pkl"):
                    df = pd.read_pickle(io.BytesIO(body))
                elif key.endswith(".csv"):
                    df = pd.read_csv(io.BytesIO(body))
                else:
                    continue

                for target in ["feature", "embedding", "combined"]:
                    match_col = next((col for col in df.columns if col.lower() == target), None)

                    if match_col:
                        item = df[match_col].iloc[0]
                        if isinstance(item, str):
                            try:
                                item = np.fromstring(item.strip("[]"), sep=' ')
                            except:
                                item = item  # leave as-is if parsing fails
                        else:
                            item = np.array(item)

                        shape = item.shape if hasattr(item, "shape") else len(item)
                        print(f"   ✅ {match_col} shape: {shape}")
                    else:
                        print(f"   ⚠️  No '{target}' column found.")

                print()

            except Exception as e:
                print(f"   ❌ Error: {e}\n")

inspect_feature_shapes(bucket_name="ucwdc-country-classifier")
print("--------------------------------------------")
inspect_feature_shapes(bucket_name="top-country-track-classifier")

