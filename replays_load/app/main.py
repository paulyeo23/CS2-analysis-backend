import os

from google.cloud import storage
import psycopg2
import csv
import io

BUCKET_NAME = "replays-preprocess"
DB_PASSWORD = os.environ["DB_PASSWORD"]
DB_HOST = os.environ["DB_HOST"]
DB_NAME = os.environ["DB_NAME"]
DB_USER = os.environ["DB_USER"]

def get_csv_files():
    client = storage.Client()
    bucket = client.bucket(BUCKET_NAME)
    return [blob for blob in bucket.list_blobs() if blob.name.endswith("ticks.csv")]

def load_csv_to_sql(blob):
    conn = psycopg2.connect(
        host=DB_HOST,
        dbname=DB_NAME,
        user=DB_USER,
        password=DB_PASSWORD,
    )
    cursor = conn.cursor()

    print(f"processing {blob.name}...")
    data = blob.download_as_text()
    reader = io.StringIO(data)
 
    cursor.copy_expert(
        "COPY ticks FROM STDIN WITH CSV HEADER",
        reader
    )

    conn.commit()
    cursor.close()
    conn.close()

    print(f"Loaded {blob.name} via COPY")

def run():
    blobs = get_csv_files()
    for blob in blobs:
        load_csv_to_sql(blob)

if __name__ == "__main__":
    run()