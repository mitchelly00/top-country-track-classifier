import boto3
import librosa
import numpy as np
import pandas as pd
import io
import soundfile as sf
import subprocess
import gc
from tqdm import tqdm


def mp3_to_wav_bytes(mp3_bytes):
    ffmpeg_cmd = [
        'ffmpeg', '-i', 'pipe:0',  # input from stdin
        '-f', 'wav', 'pipe:1'      # output WAV to stdout
    ]
    process = subprocess.Popen(ffmpeg_cmd, stdin=subprocess.PIPE, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
    wav_bytes, err = process.communicate(mp3_bytes)
    if process.returncode != 0:
        raise RuntimeError(f"ffmpeg error: {err.decode()}")
    return io.BytesIO(wav_bytes)



def extract_features_from_s3(key, bucket = "top-country-track-classifier"):
    s3 = boto3.client('s3')

    # Download mp3 file into memory
    response = s3.get_object(Bucket=bucket, Key=key)
    audio_bytes_io = io.BytesIO(response['Body'].read())
    audio_bytes_io.seek(0)

    try:
        # Convert mp3 bytes to wav bytes using ffmpeg subprocess
        wav_io = mp3_to_wav_bytes(audio_bytes_io.getvalue())
        wav_io.seek(0)
    except Exception as e:
        print(f"Error converting mp3 to wav: {e}")
        return None

    try:
        # Load audio with librosa from wav bytes
        y, sr = librosa.load(wav_io, sr=None)
    except Exception as e:
        print(f"Error loading audio with librosa: {e}")
        return None

    # 1. MFCC (13)
    mfcc = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=13)
    mfcc_mean = np.mean(mfcc, axis=1)

    # 2. Chroma (12)
    chroma = librosa.feature.chroma_stft(y=y, sr=sr)
    chroma_mean = np.mean(chroma, axis=1)

    # 3. Spectral Contrast (7)
    contrast = librosa.feature.spectral_contrast(y=y, sr=sr)
    contrast_mean = np.mean(contrast, axis=1)

    # 4. Zero Crossing Rate (1)
    zcr = librosa.feature.zero_crossing_rate(y)
    zcr_mean = np.mean(zcr)

    # 5. Tempo (1)
    tempo, _ = librosa.beat.beat_track(y=y, sr=sr)

    # Combine all features into one vector
    features = np.hstack([mfcc_mean, chroma_mean, contrast_mean, zcr_mean, tempo])

    del y, mfcc, chroma, contrast, zcr  # delete large arrays
    gc.collect()
    
    return features


#added to S3

key = "metadata/track_metadata.csv"
s3 = boto3.client('s3')

obj = s3.get_object(Bucket='top-country-track-classifier', Key=key)
df = pd.read_csv(io.BytesIO(obj['Body'].read()))
# df_found = df[df['Found']==True]

feature_list = []

for index, row in tqdm(df.iterrows(), total=len(df), desc="Extracting features"):
    song_key = row['s3_key']

    try:
        feature = extract_features_from_s3(song_key)
    except Exception as e:
        print(f"Error extracting features from {song_key}: {e}")
        feature = []

    feature_list.append(feature)

df["feature"] = feature_list

file_path = './track_metadata_features.pkl'
df.to_pickle(file_path)


#added to S3
object_name = 'metadata/track_metadata_features.pkl'  # S3 
s3.upload_file(file_path,'top-country-track-classifier', object_name)
print("Exported to S3")