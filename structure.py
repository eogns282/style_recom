from keras.applications.vgg19 import VGG19
from keras.preprocessing import image
from keras.applications.vgg19 import preprocess_input
from keras.models import Model
import numpy as np
import json
import cv2


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
	
def Recomm(content_image, style_dict, top_n = 5, reverse=False):
    '''
    :param content_image: content image, str type (ex. './data/content/parents.jpg')
    :param style_dict: dictionary of style image's feature, str type, json file (ex. './style_feature.json')
    :param top_n: num of recommended style image, int type (ex. top_n = 5)
    :return: ordered list of style image name (ex. ['123.jpg', '3.jpg', '24.jpg', '31.jpg', 120.jpg'])
    '''

    base_model = VGG19(weights='imagenet')
    model = Model(inputs=base_model.input, outputs=base_model.get_layer('fc2').output)

    x = load_image(content_image)
    content_features = model.predict(x)[0].tolist()

    features_dict_base = json.loads(open(style_dict).read())
    iter_base = features_dict_base.keys()

    result_dict = {}
    for style_image in iter_base:
        dist = np.sqrt(np.sum(np.square(np.subtract(content_features, features_dict_base[style_image]))))
        result_dict[style_image] = dist

    final_result = sorted(result_dict, key=result_dict.get, reverse=reverse)[:top_n]

    return final_result