from zipfile import ZipFile
import shutil
import os
from google_drive_downloader import GoogleDriveDownloader as gdd


gdd.download_file_from_google_drive(file_id='',
                                    dest_path='./data/Raw_data.zip',
                                    unzip=True)

