from keras.applications.vgg19 import VGG19
from keras.models import Model
import numpy as np
import json
from utils import load_image


def compare_fn(content_image, struc_dict_base):

    base_model = VGG19(weights='imagenet')
    model = Model(inputs=base_model.input, outputs=base_model.get_layer('fc2').output)

    x = load_image(content_image)
    content_features = model.predict(x)[0].tolist()

    iter_base = struc_dict_base.keys()

    temp_dict = {}
    for style_image in iter_base:
        dist = np.sqrt(np.sum(np.square(np.subtract(content_features, struc_dict_base[style_image]))))
        temp_dict[style_image] = dist

    return temp_dict


def structure_rank(content_image, style_dict, top_n = 5, reverse=False):

    features_dict_base = json.loads(open(style_dict).read())
    result_dict = compare_fn(content_image, features_dict_base)
    final_result = sorted(result_dict, key=result_dict.get, reverse=reverse)[:top_n]

    return final_result
