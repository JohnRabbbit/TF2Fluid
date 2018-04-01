import pdb
import os
import sys
import shutil
import requests
import hashlib
import numpy as np

__all__ = [
    "DIR_NAME",
    "IMG_SHAPE",
    "LBL_COUNT",
    "download_data",
    "color_preprocessing",
]

MD5 = "c58f30108f718f92721af3b95e74349a"
URL = "http://www.cs.toronto.edu/~kriz/cifar-10-python.tar.gz"

DIR_NAME = "cifar-10-batches-py"
FNAME = "cifar-10-python.tar.gz"
IMG_SHAPE = [3, 32, 32]
LBL_COUNT = 10


def md5file(fname):
    hash_md5 = hashlib.md5()
    f = open(fname, "rb")
    for chunk in iter(lambda: f.read(4096), b""):
        hash_md5.update(chunk)
    f.close()
    return hash_md5.hexdigest()


def download_data():
    filename = os.path.join(DIR_NAME, FNAME)

    download = False
    if not os.path.exists(DIR_NAME): os.makedirs(DIR_NAME)

    if os.path.exists(filename) and md5file(filename) == MD5:
        download = False
        print("DataSet aready exists!")
    else:
        download = True

    if download:
        print "Cache file %s is not found, download from %s." % (DIR_NAME, URL)

        r = requests.get(URL, stream=True)
        total_length = r.headers.get("content-length")

        if total_length is None:
            with open(filename, "w") as f:
                shutil.copyfileobj(r.raw, f)
        else:
            with open(filename, "w") as f:
                dl = 0
                total_length = int(total_length)
                for data in r.iter_content(chunk_size=4096):
                    dl += len(data)
                    f.write(data)
                    done = int(50 * dl / total_length)
                    sys.stdout.write("\r[%s%s]" % ("=" * done,
                                                   " " * (50 - done)))
                    sys.stdout.flush()

    return filename


def color_preprocessing(image_data):
    image_data = image_data.astype('float32')
    for i in range(IMG_SHAPE[0]):
        image_data[:, i, :, :] = (
            image_data[:, i, :, :] - np.mean(image_data[:, i, :, :])) / np.std(
                image_data[:, i, :, :])
    return image_data


if __name__ == "__main__":
    download_data()
