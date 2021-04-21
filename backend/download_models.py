from zipfile import ZipFile
import shutil
import os
from google_drive_downloader import GoogleDriveDownloader as gdd
# Loads pretrained model that was trained using supervised module
gdd.download_file_from_google_drive(file_id='1klb3iZ7HTJdZER87S9n0d5LYJMw0XtfM',dest_path='./models/filename_6272019_v1_search.joblib')
