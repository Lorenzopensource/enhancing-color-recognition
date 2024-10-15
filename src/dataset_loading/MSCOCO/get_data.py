import requests
import os
from zipfile import ZipFile, BadZipFile

# For this project I want to get access to MSCOCO Images, Segmentation data and captions.
# I will use the MSCOCO API to download the data.
# The data will be downloaded and extracted to the data directory.

#/synthetic_dataset_from_object_recoloring                # Root directory
#│
#├── /src
#|   ├── /dataset_loading
#|      ├── /MSCOCO
#│          ├── get_data.p                      # This script
#|          ├── /data                           # Subdirectory for storing downloaded data
#│                ├── /annotations                  # For annotations of both sets
#│                ├── /train2017                    # For training images
#│                └── /val2017                      # For validation images
#│
#├── ...   
#├── requirements.txt                
#└── README.md  

# If called directly, the script will generate all the directories above and download all the data (around 28 GB) and extract it.
# Import the specific functions to download just a part of the dataset



def ensure_directories_exist():
    """Create necessary directories if they do not exist."""
    os.makedirs("data/annotations", exist_ok=True)

def download_file(url, local_filename):
    """Download a file from a URL and save it locally."""
    response = requests.get(url, stream=True)
    if response.status_code == 200:
        with open(local_filename, 'wb') as f:
            f.write(response.content)
        print(f"Downloaded {local_filename}")
    else:
        print(f"Failed to download {url}. Status code: {response.status_code}")


def extract_zip_file(zip_path, extract_path):
    """Extract a ZIP file to the specified directory."""
    try:
        with ZipFile(zip_path) as zfile:
            # If there’s a top-level folder inside the zip, it avoids duplicate directories
            members = zfile.namelist()
            root_folder = os.path.commonprefix(members)
            if root_folder and root_folder.endswith("/"):
                # Unzip content into the parent directory instead of recreating it
                zfile.extractall(extract_path)
            else:
                zfile.extractall(extract_path)
        os.remove(zip_path)  # Remove the ZIP file after extraction
        print(f"Extracted {zip_path} to {extract_path}")
    except BadZipFile as e:
        print("Error:", e)

def download_annotations():
    """Download and extract MSCOCO annotations."""
    ensure_directories_exist()  # Ensure directories exist
    url = "http://images.cocodataset.org/annotations/annotations_trainval2017.zip"
    local_filename = "coco_ann2017.zip"
    download_file(url, local_filename)
    extract_zip_file(local_filename, "data")

def download_train_images():
    """Download and extract MSCOCO training images."""
    ensure_directories_exist()  # Ensure directories exist
    url = "http://images.cocodataset.org/zips/train2017.zip"
    local_filename = "train2017.zip"
    download_file(url, local_filename)
    extract_zip_file(local_filename, "data")

def download_val_images():
    """Download and extract MSCOCO validation images."""
    ensure_directories_exist()  # Ensure directories exist
    url = "http://images.cocodataset.org/zips/val2017.zip"
    local_filename = "val2017.zip"
    download_file(url, local_filename)
    extract_zip_file(local_filename, "data")

if __name__ == "__main__":
    download_annotations() # 74,4 MB
    #download_train_images()
    download_val_images() # 825,1 MB
