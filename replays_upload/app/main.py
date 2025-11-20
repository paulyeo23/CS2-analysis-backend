from google.cloud import storage
from google.cloud.storage.transfer_manager import upload_many_from_filenames
from pathlib import Path
import sys


def upload_to_cloud(bucket_name, source_directory, destination_blob, workers=8):
    client = storage.Client()

    bucket = client.bucket(bucket_name)
    blob = bucket.blob(destination_blob)

    # create list of files to upload
    directory_as_path_obj = Path(source_directory)
    paths = directory_as_path_obj.rglob("*")
    

    file_paths = [path for path in paths if path.is_file()]
    relative_paths = [path.relative_to(source_directory) for path in file_paths]
    string_paths = [str(path) for path in relative_paths]

    
    # # Upload a single file
    # print(f"Uploading {local_file_path} → gs://{bucket_name}/{destination_blob}")
    # blob.upload_from_filename(local_file_path)

    # print("upload complete!")

    # upload files to cloud bucket
    results = upload_many_from_filenames(
        bucket, string_paths, source_directory=source_directory, max_workers=workers
    )

    for name, result in zip(string_paths, results):
        if isinstance(result, Exception):
            print(f"Failed to upload {name} due to exception: {result}")
        else:
            print(f"Uploaded {name} to {bucket.name}.")

if __name__ == "__main__":
    
    if len(sys.argv) < 2:
        raise ValueError("Usage: python main.py <FOLDER_PATH>")
    
    bucket_name = "replays-preprocess"
    destination_blob_name = "uploaded-kills.csv"
    source_directory = sys.argv[1]

    upload_to_cloud(bucket_name, source_directory, destination_blob_name)
