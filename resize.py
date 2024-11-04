import matplotlib.pyplot as plt
import numpy as np
import os
import imageio

# image shape
HEIGHT = 96
WIDTH = 96
DEPTH = 3

# size of a single image in bytes
SIZE = HEIGHT * WIDTH * DEPTH

DATA_PATH = './data/stl10_binary/train_X.bin'


def plot_image(image):
    """
    :param image: the image to be plotted in a 3-D matrix format
    :return: None
    """
    plt.imshow(image)
    plt.show()


def read_all_images(path_to_data):
    """
    :param path_to_data: the file containing the binary images from the STL-10 dataset
    :return: an array containing all the images
    """

    with open(path_to_data, 'rb') as f:
        # read whole file in uint8 chunks
        everything = np.fromfile(f, dtype=np.uint8)

        # We force the data into 3x96x96 chunks, since the
        # images are stored in "column-major order", meaning
        # that "the first 96*96 values are the red channel,
        # the next 96*96 are green, and the last are blue."
        # The -1 is since the size of the pictures depends
        # on the input file, and this way numpy determines
        # the size on its own.

        images = np.reshape(everything, (-1, 3, 96, 96))

        # Now transpose the images into a standard image format
        # readable by, for example, matplotlib.imshow
        # You might want to comment this line or reverse the shuffle
        # if you will use a learning algorithm like CNN, since they like
        # their channels separated.
        images = np.transpose(images, (0, 3, 2, 1))
        return images

def crop_image21_9(image):
     start = (image.shape[0] - image.shape[0]*  9 // 21) // 2
     return image[start:image.shape[0] - start]

def crop_image16_9(image):
     start = (image.shape[1] - image.shape[0] * 16 // 9) // 2
     return image[:, start:image.shape[1] - start]

def mirror_16_9_to_21_9(image, original_shape):
     to_add = (original_shape[1] - image.shape[1]) // 2
     return np.concatenate([np.flip(image[:, 0:to_add + 1], axis=1), image, np.flip(image[:, image.shape[1] - to_add: image.shape[1] - 1], axis=1)], axis=1)
     

with open(DATA_PATH) as f:
    images = read_all_images(DATA_PATH)

    os.makedirs('datasets/original_21_9', exist_ok=True)
    os.makedirs('datasets/generated_21_9', exist_ok=True)

    for idx, image in enumerate(images):
        original_img_21_9 = crop_image21_9(image)
        original_img_16_9 = crop_image16_9(original_img_21_9)
        generated_img_21_9 = mirror_16_9_to_21_9(original_img_16_9, image.shape)

        print('Processing image ', idx)
        imageio.imwrite(f'datasets/original_21_9/{idx}.png', original_img_21_9)
        imageio.imwrite(f'datasets/generated_21_9/{idx}.png', generated_img_21_9)
        