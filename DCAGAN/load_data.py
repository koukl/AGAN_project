import os
import numpy as np
import config
from os import listdir
from PIL import Image

def load_data():
    if config.data == 'mnist':
        return load_mnist()
    elif config.data == 'lsun':
        return load_lsun()
    else:
        raise ValueError('Invalid data specification.')

def load_lsun():
    X = np.zeros((0, config.image_len, config.image_len, 3))
    count = 0
    for f in listdir(config.data_dir):
        filename = os.path.join(config.data_dir, f)
        im = Image.open(filename)
        im_w, im_h = im.size
        # find smaller dimension and resize to config.image_len, keeping aspect ratio same
        # then take centre crop
        # (width, height) order for resize, size etc.
        len = config.image_len
        if im_w < im_h:
            im = im.resize((config.image_len, config.image_len * im_h/im_w))
            L = im.size[1]  # L is the longer side (after resizing)
            im = im.crop((0, (L-len)/2, len, (L-len)/2+len))
        else:
            im = im.resize((config.image_len * im_w / im_h, config.image_len))
            L = im.size[0]  # L is the longer side (after resizing)
            im = im.crop(((L-len)/2, 0, (L-len)/2+len, len))
        im = np.array(im)
        X = np.append(X, np.array([im]), axis=0)
        count += 1
        if count == config.train_size:
            break
    return X / 255.


def load_mnist():
    y_dim = 10
    data_dir = os.path.join(config.data_dir, "mnist")

    fd = open(os.path.join(data_dir, 'train-images-idx3-ubyte'))
    loaded = np.fromfile(file=fd, dtype=np.uint8)
    trX = loaded[16:].reshape((60000, 28, 28, 1)).astype(np.float)

    fd = open(os.path.join(data_dir, 'train-labels-idx1-ubyte'))
    loaded = np.fromfile(file=fd, dtype=np.uint8)
    trY = loaded[8:].reshape((60000)).astype(np.float)

    fd = open(os.path.join(data_dir, 't10k-images-idx3-ubyte'))
    loaded = np.fromfile(file=fd, dtype=np.uint8)
    teX = loaded[16:].reshape((10000, 28, 28, 1)).astype(np.float)

    fd = open(os.path.join(data_dir, 't10k-labels-idx1-ubyte'))
    loaded = np.fromfile(file=fd, dtype=np.uint8)
    teY = loaded[8:].reshape((10000)).astype(np.float)

    trY = np.asarray(trY)
    teY = np.asarray(teY)

    X = np.concatenate((trX, teX), axis=0)
    y = np.concatenate((trY, teY), axis=0).astype(np.int)

    X = X[np.where(y == config.extract_digit)[0]]
    y = y[np.where(y == config.extract_digit)[0]]


    seed = 547
    np.random.seed(seed)
    np.random.shuffle(X)
    X = X[:config.train_size]

    return X / 255.