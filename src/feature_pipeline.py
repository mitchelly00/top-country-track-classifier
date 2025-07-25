import boto3
import librosa
import numpy as np
import pandas as pd
import io
import soundfile as sf
import subprocess
import gc
from tqdm import tqdm
import openl3
from kapre.time_frequency import STFT


# ---------- CONFIG ----------
BUCKET_NAME = "top-country-track-classifier"
INPUT_CSV_KEY = "metadata/track_metadata.csv"
OUTPUT_PKL_LOCAL = "track_metadata_features.pkl"
OUTPUT_PKL_S3 = "metadata/track_metadata_features.pkl"
OUTPUT_PKL_EMBED_S3 = "metadata/track_metadata_feature_embedding.pkl"
# ----------------------------
def mp3_to_wav_bytes(mp3_bytes):
    ffmpeg_cmd = [
        'ffmpeg', '-i', 'pipe:0',
        '-f', 'wav', 'pipe:1'
    ]
    process = subprocess.Popen(ffmpeg_cmd, stdin=subprocess.PIPE, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
    wav_bytes, err = process.communicate(mp3_bytes)
    if process.returncode != 0:
        raise RuntimeError(f"ffmpeg error: {err.decode()}")
    return io.BytesIO(wav_bytes)


def extract_features_from_s3(key, bucket=BUCKET_NAME):
    s3 = boto3.client('s3')

    try:
        response = s3.get_object(Bucket=bucket, Key=key)
        audio_bytes_io = io.BytesIO(response['Body'].read())
        audio_bytes_io.seek(0)

        wav_io = mp3_to_wav_bytes(audio_bytes_io.getvalue())
        wav_io.seek(0)

        y, sr = librosa.load(wav_io, sr=None)

        # Feature extraction
        mfcc_mean = np.mean(librosa.feature.mfcc(y=y, sr=sr, n_mfcc=13), axis=1)
        chroma_mean = np.mean(librosa.feature.chroma_stft(y=y, sr=sr), axis=1)
        contrast_mean = np.mean(librosa.feature.spectral_contrast(y=y, sr=sr), axis=1)
        zcr_mean = np.mean(librosa.feature.zero_crossing_rate(y))
        tempo, _ = librosa.beat.beat_track(y=y, sr=sr)

        # Combine
        features = np.hstack([mfcc_mean, chroma_mean, contrast_mean, zcr_mean, tempo])

        del y
        gc.collect()
        return features

    except Exception as e:
        print(f"[ERROR] Failed to extract features for {key}: {e}")
        return None


def load_metadata(bucket=BUCKET_NAME, key=INPUT_CSV_KEY):
    s3 = boto3.client('s3')
    obj = s3.get_object(Bucket=bucket, Key=key)
    df = pd.read_csv(io.BytesIO(obj['Body'].read()))
    return df


# def upload_to_s3(file_path, bucket=BUCKET_NAME, s3_key=OUTPUT_PKL_S3):
#     s3 = boto3.client('s3')
#     s3.upload_file(file_path, bucket, s3_key)
#     print(f"[INFO] Uploaded {file_path} to s3://{bucket}/{s3_key}")


def upload_dataframe_as_pickle_to_s3(df, bucket=BUCKET_NAME, s3_key=OUTPUT_PKL_S3):
    s3 = boto3.client('s3')
    pickle_buffer = io.BytesIO()
    df.to_pickle(pickle_buffer)
    pickle_buffer.seek(0)
    s3.put_object(Bucket=bucket, Key=s3_key, Body=pickle_buffer)
    print(f"[INFO] Uploaded DataFrame to s3://{bucket}/{s3_key}")

# embeddings

class STFTPatched(STFT):
    pass

# Global model (loaded per subprocess)
_model = None

def init_worker():
    global _model
    _model = openl3.models.load_audio_embedding_model(
        input_repr="mel256",
        content_type="music",
        embedding_size=512
    )


def extract_openl3_embedding_from_s3(key: str) -> np.ndarray:
    s3 = boto3.client("s3")
    response = s3.get_object(Bucket=BUCKET_NAME, Key=key)
    mp3_bytes = response['Body'].read()

    wav_io = mp3_to_wav_bytes(mp3_bytes)
    wav_io.seek(0)
    audio, sr = sf.read(wav_io)

    # Use model loaded in init_worker
    global _model
    emb, ts = openl3.get_audio_embedding(audio, sr, model=_model, verbose=0)
    return emb.mean(axis=0)

def process_key(key):
    try:
        emb = extract_openl3_embedding_from_s3(key)
        return (key, emb)
    except Exception as e:
        print(f"Failed to process {key}: {e}")
        return (key, None)

def main():
    print("[INFO] Starting feature extraction pipeline...")

    #load model
    df = load_metadata()
    df = df.head().copy()

    #add features
    feature_list = []

    for _, row in tqdm(df.iterrows(), total=len(df), desc="Extracting features"):
        key = row['s3_key']
        features = extract_features_from_s3(key)
        feature_list.append(features)

    df['feature'] = feature_list

    upload_dataframe_as_pickle_to_s3(df)
    print("[INFO] Feature extraction complete and uploaded.")

    #add embeddings
    # Load model once for multiple calls if needed
    model = openl3.models.load_audio_embedding_model(input_repr="mel256",
                                                    content_type="music",
                                                    embedding_size=512)
    # Extract embeddings sequentially
    embeddings = []
    for k in tqdm(df["s3_key"].tolist(), desc="Extracting Embeddings"):
        try:
            emb = extract_openl3_embedding_from_s3(k, model)
        except Exception as e:
            print(f"Failed to process {k}: {e}")
            emb = None
        embeddings.append(emb)
    
    df["embedding"] = embeddings

    upload_dataframe_as_pickle_to_s3(df, s3_key=OUTPUT_PKL_EMBED_S3)



if __name__ == "__main__":
    main()
