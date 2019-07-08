from zipfile import ZipFile
import shutil
import os



with ZipFile('./data/Raw_data.zip', 'r') as zipObj:
   # Extract all the contents of zip file in current directory
   zipObj.extractall('./data')



# attribution: https://www.pythoncentral.io/how-to-recursively-copy-a-directory-folder-in-python/
def copyDirectory(src, dest):
    try:
        shutil.copytree(src, dest)
    # Directories are the same
    except shutil.Error as e:
        print('Directory not copied. Error: %s' % e)
    # Any error saying that the directory doesn't exist
    except OSError as e:
        print('Directory not copied. Error: %s' % e)





