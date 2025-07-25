import requests
import boto3
import pandas as pd
from io import BytesIO
import io

# Playlist URL
playlist_url = "https://api.deezer.com/playlist/4138363542"

# AWS S3
bucket_name = "top-country-track-classifier"
s3 = boto3.client('s3')

# Get playlist info
response = requests.get(playlist_url)
playlist_data = response.json()

# Tracklist
tracks = playlist_data['tracks']['data']

# Data for the dataframe
rows = []

for index, track in enumerate(tracks, start=1):
    title = track['title']
    artist = track['artist']['name']
    preview_url = track['preview']

    if not preview_url:
        print(f"Skipping: {title} by {artist} (no preview)")
        continue

    filename = f"{index:02d}_{artist}_{title}.mp3".replace(" ", "_").replace("/", "_")
    s3_key = f"previews/{filename}"

    # Download preview
    audio = requests.get(preview_url)
    if audio.status_code == 200:
        s3.upload_fileobj(BytesIO(audio.content), bucket_name, s3_key)
        print(f"Uploaded: {filename}")
    else:
        print(f"Failed to download preview for {title} by {artist}")
        continue

    # Add row
    rows.append({
        "order": index,
        "artist": artist,
        "title": title,
        "s3_key": s3_key
    })

def upload_df_to_s3(df, key, bucket=bucket_name):
    s3 = boto3.client("s3")
    csv_buffer = io.StringIO()
    df.to_csv(csv_buffer, index=False)
    s3.put_object(Bucket=bucket, Key=key, Body=csv_buffer.getvalue())

def main():
    # Create dataframe
    df = pd.DataFrame(rows)

    #upload dataframe
    upload_df_to_s3(df, "metadata/track_metadata.csv")
    print("uploaded to S3")

if __name__ == "__main__":
    main()
