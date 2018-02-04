import os
import tarfile
import pickle

def check_if_file_exists(path):
    return(os.path.exists(path))

def create_dir_if_not_exists(path):
    if not os.path.exists(path):
        os.makedirs(path)

def unpack_tar_gz(path):
    folder, _ = os.path.split(path)
    if path.endswith("tar.gz"):
        tar = tarfile.open(path, "r:gz")
        tar.extractall(folder)
        tar.close()
        print('File %s unpacked.' % path)
    else:
        raise ValueError('Unexpected file! %s' % path)

def unpickle(file):
    with open(file, 'rb') as fo:
        dict = pickle.load(fo, encoding='bytes')
    return dict