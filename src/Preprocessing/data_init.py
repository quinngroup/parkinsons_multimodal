from glob import glob
import pandas as pd
import os, sys
from Utils.cloud import CController

# Will be initialized to a CController object
cloud = None

data_info_path = "data_info.csv"
organized_dir_path = "organized_data/"

"""
Only method that should be imported from this python file. Handles creation of necessary
directories for the project, downloading information about the data, and the data itself.
Also reorganizes the data by removing intermediate folders and grouping my patients and modalities.
Maintains data_info.csv which contains important information about the dataset.

Data will be stored as
/organized_data/[patientid]/[modality]/[imagename].nii

Information about the data will be at
/data_info.csv
"""
def organized_data_download(key_path, bucket_name):

    global cloud 

    cloud = CController(key_path, bucket_name)

    __make_initial_dirs()
    __download_data_info()

    df = pd.read_csv(data_info_path)

    __make_patient_dirs(df)
    __download_images(df)

"""
Downloads only the .csv file from cloud bucket. There should be only one in the bucket
and it is a separate download from the data. This file contains information about the data
that was downloaded (patient ids, modalities, control groups, etc.). This file will be further
exapnded to also associate an image path with each image logged in this file and will be used
to setup data loaders.
"""
def __download_data_info():

    # If a info file has already been downloaded
    if os.path.exists(data_info_path):
        print("[INFO] Existing data_info.csv found, will not redownload")
        return

    num_found = 0
    info_file_name = ""

    # Iterates through all files in initial directory in cloud bucket. There should
    # only be one .csv file
    for file in cloud.get_blob_iterator(delimiter="/"):

        if file.name[-4:] == ".csv":
            info_file_name = file.name
            num_found += 1

    # Error logging
    if num_found == 0:
        print("[WARNING] No info file found on cloud bucket, exiting")
        exit()

    if num_found > 1:
        print("[WARNING] More than one info file found")

    # Downloads the data_info.csv file
    cloud.download_blob(info_file_name, data_info_path)

"""
Creates directory /organized_data which will contain data downloaded from cloud after
going through organization.
"""
def __make_initial_dirs():

    if not os.path.exists(organized_dir_path):

        print("[INFO] Creating folder organized_data")
        os.makedirs(organized_dir_path)

"""
Makes a directory for each unique patient ID. 
Places them under /organized_data/[patientid].
"""
def __make_patient_dirs(df):

    num_dirs_created = 0

    for patient_id in df.Subject.unique():

            if not os.path.exists(organized_dir_path + str(patient_id)):

                    os.makedirs(organized_dir_path + str(patient_id))
                    num_dirs_created += 1

    print("[INFO] Created %d patient directories" % num_dirs_created)

"""
Extracts necessary information for each file from its filepath as it was downloaded from PPMI.
"""
def __parse_filepath(filepath):

    split_file = filepath.split("/")

    patient_id = split_file[2]
    modality = split_file[3]
    filename = split_file[-1]

    # Cutting out unique id for each file. This will be matched with entries in data_info.csv
    image_id = filename.split("_")[-1][1 : ].split('.')[0]

    return patient_id, modality, filename, int(image_id)

"""
Downloads all images from cloud bucket and places them into appropriate patient folder.
If images have been already downloaded, will not redownload them. If necessary, creates 
folder for modalities under each patient. Adds path to each file to data_info.csv.
"""
def __download_images(df):

    num_images = len(df.index)
    img_id_to_path = {}

    # Iterating over all files in cloud bucket
    for image_number, file in enumerate(cloud.get_blob_iterator()):

            # Filtering out the csv file
            if file.name.split(".")[-1] == "csv":
                    continue

            patient_id, modality, filename, image_id  = __parse_filepath(file.name)

            target_download_dir = organized_dir_path + "%s/%s/" % (patient_id, modality)

            # Matching the file's unique id with its entry in data_info for easier access
            img_id_to_path[image_id] = target_download_dir + filename

            # If image has already been downloaded, don't redownload it
            if os.path.exists(target_download_dir + filename):
                continue

            # Making directories for modalities for each patient if needed
            if not os.path.exists(target_download_dir):
                    os.makedirs(target_download_dir)

            # Logging
            sys.stdout.write("\r[INFO] Downloading file %d/%d" % (image_number, num_images))
            sys.stdout.flush()

            # Downloading image from bucket to appropriate directory
            cloud.download_blob(file.name, target_download_dir + filename)

    # Updating paths to all downloaded file in data_info.csv
    df["Path"] = df["Image Data ID"].map(img_id_to_path)

    df.to_csv(data_info_path)

    print()
