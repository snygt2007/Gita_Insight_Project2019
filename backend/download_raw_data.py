from zipfile import ZipFile
import shutil
import os
from google_drive_downloader import GoogleDriveDownloader as gdd

# Download the raw logo images after obtaining the permission from the dataset owners
gdd.download_file_from_google_drive(file_id='10XwxP9Ob8Sz8qDxwqRNA9mhvTiYkTdwx',
                                    dest_path='./data/Raw_data.zip',
                                    unzip=True)

