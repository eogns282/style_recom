import cv2
import numpy as np
from keras.preprocessing import image
from keras.applications.vgg19 import preprocess_input


def make_hist(img):
    unique, counts = np.unique(np.reshape(img, [-1, 3]), axis=0, return_counts=True)
    origin_hist = np.reshape(sorted(counts, reverse=True), [-1, 1])
    return origin_hist


def hist_inform(img):
    w, h = img.shape[0], img.shape[1]

    origin_hist = make_hist(img)
    img_num = origin_hist.shape[0]

    img_hist = (origin_hist).ravel().astype('float32')
    nor_img_hist = img_hist / (w * h * 0.001)

    return nor_img_hist, img_num


def load_image(img_dir):
    img = image.load_img(img_dir, target_size=(224, 224))
    x = image.img_to_array(img)
    x = make_gray_image(x)
    x = np.expand_dims(x, axis=0)
    x = preprocess_input(x)
    return x


def make_gray_image(img, save=False, save_dir=None):
    gray_image = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    h = gray_image.shape[0]
    w = gray_image.shape[1]

    RGB = np.zeros((h, w, 3))
    RGB[:, :, 0] = gray_image
    RGB[:, :, 1] = gray_image
    RGB[:, :, 2] = gray_image
    if save==True:
        cv2.imwrite(save_dir + 'gray_result.jpg', RGB)
    return RGB