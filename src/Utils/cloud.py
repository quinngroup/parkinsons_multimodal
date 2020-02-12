from google.cloud import storage

"""
Class that handles interfacing with Google Cloud storage buckets which contain PPMI
image data that will be used for models.
"""
class CController:

    """
    Contstructor. Needs to have key to access bucket and the name of the bucket itself.
    """
    def __init__(self, key_path, bucket_name):

        self.key_path = key_path
        self.bucket_name = bucket_name

        # Requests access
        self.storage_client = storage.Client.from_service_account_json(self.key_path)

        self.bucket = self.storage_client.bucket(self.bucket_name)

    """
    Downloads file (blob) from the filename in the bucket to target destination locally.
    """
    def download_blob(self, filename, destination):
    
        blob = self.bucket.blob(filename)
        blob.download_to_filename(destination)

    """
    Creates iterator (python generator) which can be used to get information about all data
    in the bucket.
    """
    def get_blob_iterator(self, prefix="raw_data/", delimiter=""):

        return self.storage_client.list_blobs(
            self.bucket_name, 
            prefix=prefix, 
            delimiter=delimiter)
