from google.cloud import storage
from google.cloud.storage.bucket import Bucket
import os

def connect_google_cloud_bucket(bucket_name:str, base_path:str) -> Bucket:
    '''
    Connects to the google cloud instance which our model is stored on
    and returns a google cloud bucket object which can be used to do CRUD operations
    
    ### Args ###
    bucket_name (:obj: `str`)
        * the name of the bucket to open
    base_path (:: `str`) 
        * the path to the google storage api credential file
    '''

    # Checks the pass it correct
    path = os.path.join(base_path, 'bert-qa-bucket.json')
    if path.startswith('/'):
        path = path[1:] # Removes the first character

    # Connection to GCS (Google Storage Bucket)
    storage_client  = storage.Client.from_service_account_json(json_credentials_path=path)
    bucket = storage_client.get_bucket(bucket_name)

    return bucket

def upload_file_to_google_storage(bucket:Bucket, storage_name:str, file_name:str):
    '''
    Uploads a file to bucket/storage_name using the Google Cloud Storage API
    bucket is a pre initialized google cloud object
    And filename is the full path to the file you want to get uploaded

    ### Args ###
    bucket (:obj: `google.cloud.storage.bucket.Bucket`)
        * The google bucket connection object
    storage_name (:obj: `str`)
        * The name of the path in the Google Cloud Storage Bucket
    file_name (:obj: `str`) 
        * The name of the file we want to upload
    '''
    blob = bucket.blob(storage_name)

    # Uploads the file using the specified path
    blob.upload_from_filename(file_name)
    print(f'File {file_name} uploaded to {storage_name}')

def download_file_from_google_storage(bucket:Bucket, storage_name:str, output_dir:str):
    '''
    Downloads a file from storage_path to output_dir from bucket

    ### Args ###
    bucket (:obj: `google.cloud.storage.bucket.Bucket`)
        * The google bucket connection object
    storage_name (:obj: `str`)
        * The name of the path in the Google Cloud Storage Bucket
    output_dir (:obj: `str`) 
        * The path where we download the `storage_name` file
    '''

    # Creates a connection between specified storage and server
    blob = bucket.blob(storage_name)

    # Downloads each file
    blob.download_to_filename(output_dir)

def delete_file_from_google_storage(bucket:Bucket, storage_name:str):
    '''
    Deletes the specified file in google storage bucket

    ### Args ###
    bucket (:obj: `google.cloud.storage.bucket.Bucket`)
        * The google bucket connection object
    storage_name (:obj: `str`)
        * The name of the path in the Google Cloud Storage Bucket
    '''

    # Get the blob for deletion
    blob = bucket.blob(storage_name)

    # Delete the blob file
    blob.delete()
