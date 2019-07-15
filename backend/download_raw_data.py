import zipfile
import os


directory_raw = './data/'
if not os.path.exists(directory_raw):
        os.makedirs(directory_raw)
zip_ref = zipfile.ZipFile('./data/Raw_data.zip', 'r')
zip_ref.extractall(directory_raw)
zip_ref.close()
print("Raw data created")

