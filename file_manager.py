import os
import tarfile
import pickle

__author__ = "Marcin Stachowiak"
__version__ = "1.0"
__email__ = "marcin.stachowiak.ms@gmail.com"

def check_if_file_exists(path):
    return(os.path.exists(path))

def create_dir_if_not_exists(path):
    if not os.path.exists(path):
        os.makedirs(path)

def unpack(path):
    folder, _ = os.path.split(path)
    tar = tarfile.open(path, "r:gz")
    tar.extractall(folder)
    tar.close()
    print('File %s unpacked.' % path)

def unpickle(file):
    with open(file, 'rb') as fo:
        dict = pickle.load(fo, encoding='bytes')
    return dict