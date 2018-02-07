import requests
import file_manager
import os
import sys

__author__ = "Marcin Stachowiak"
__version__ = "1.0"
__email__ = "marcin.stachowiak.ms@gmail.com"

def _download_from_url(source_url,target_dir):
    _, filename = os.path.split(source_url)
    target_file=os.path.join(target_dir,filename)
    bar_length=60

    file_manager.create_dir_if_not_exists(target_dir)

    with open(target_file, "wb") as f:
        r = requests.get(source_url, stream=True)
        total_length = r.headers.get('content-length')

        if total_length is None:
            f.write(r.content)
        else:
            iter = 0
            total_length = int(total_length)
            for data in r.iter_content(chunk_size=2048):
                iter += len(data)
                f.write(data)
                download_percent=(iter / total_length)*100
                done = int(bar_length * download_percent/100)
                print("\r%d%% - [%s%s]" % (download_percent,'=' * done, ' ' * (bar_length - done)))
                sys.stdout.flush()

def download_from_url_if_not_exists(source_url,target_dir):
    _, filename = os.path.split(source_url)
    target_file = os.path.join(target_dir, filename)
    if not file_manager.check_if_file_exists(target_dir):
        _download_from_url(source_url,target_dir)
    else:
        print('File %s already exists.' % target_file)

