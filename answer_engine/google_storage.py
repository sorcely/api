from google.cloud import storage
import os

# Connects to the google cloud instance which our model is stored on
# And returns a google cloud bucket object which can be used to do CRUD operations
def connect_google_cloud_bucket(bucket_name, base_path):
    # Bucket_name: the name of the bucket to open
    # base_path: the path to the google storage api credential file

    # Checks the pass it correct
    path = os.path.join(base_path, 'bert-qa-bucket.json')
    if path.startswith('/'):
        path = path[1:] # Removes the first character

    # Connection to GCS (Google Storage Bucket)
    storage_client  = storage.Client.from_service_account_json(json_credentials_path=path)
    bucket = storage_client.get_bucket(bucket_name)

    return bucket

# Uploads a file to bucket/storage_name
# Using the Google Cloud Storage API
# bucket is a pre initialized google cloud object
# And filename is the full path to the file you want to get uploaded
def upload_file_to_google_storage(bucket, storage_name, file_name):
    blob = bucket.blob(storage_name)

    # Uploads the file using the specified path
    blob.upload_from_filename(file_name)
    print(f'File {file_name} uploaded to {storage_name}')

# Downloads a file from storage_path to output_dir from bucket
def download_file_from_google_storage(bucket, storage_name, output_dir):
    # Creates a connection between specified storage and server
    blob = bucket.blob(storage_name)

    # Downloads each file
    blob.download_to_filename(output_dir)

def delete_file_from_google_storage(bucket, storage_name):
    # Get the blob for deletion
    blob = bucket.blob(storage_name)

    # Delete the blob file
    blob.delete()
