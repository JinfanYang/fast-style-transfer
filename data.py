from __future__ import print_function
import zipfile
import os

train_zip = 'images/train2014.zip'
    
# extract train2014.zip to train_data
train_folder = 'images/train_images'
if not os.path.isdir(train_folder):
    print(train_folder + ' not found, extracting ' + train_zip)
    zip_ref = zipfile.ZipFile(train_zip, 'r')
    zip_ref.extractall(folder)
    zip_ref.close()
